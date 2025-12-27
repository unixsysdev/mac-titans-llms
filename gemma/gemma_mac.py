import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import Optional, Tuple, List

class MemoryMLP(nn.Module):
    """
    The Neural Memory module for Titans MAC.
    Essentially a deep MLP that learns to compress context.
    """
    def __init__(self, hidden_size: int, memory_size: int = 128, layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        ])
        
        # self.retrieve_proj = nn.Linear(hidden_size, hidden_size) # Unused in current prototype

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def update_memory(self, inputs, loss):
        # allow_unused=True handles cases where some params might not be in the graph
        grads = torch.autograd.grad(loss, self.parameters(), retain_graph=True, allow_unused=True)
        for param, grad in zip(self.parameters(), grads):
            if grad is not None:
                # SAFE UPDATE: Use .data to avoid autograd tracking this step as part of the graph
                # or use torch.no_grad() context which we are not in. 
                # Since we are in enable_grad(), we must use .data for the manual update.
                param.data = param.data - (1e-4 * grad) 

class TitansMACLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.memory_mlp = MemoryMLP(hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        # Threshold needs to be higher since MSE on high-dim vectors is large
        self.surprise_threshold = 50.0 

    def forward(self, current_hidden_states: torch.Tensor, prev_memory_state: Optional[torch.Tensor] = None):
        """
        Args:
            current_hidden_states: [Batch, Seq, Dim] - input from Gemma layer
            prev_memory_state: [Batch, MemSize, Dim] - previous memory context
        """
        # We need to compute gradients for the memory module even during inference (TTT)
        with torch.enable_grad():
            # Debugging: check if we are trapped in inference mode
            if torch.is_inference_mode_enabled():
                print("WARNING: Inference mode detected! Gradients cannot be computed.")
            
            # Detach input and ensure it's a leaf tensor for local graph
            inp = current_hidden_states.detach()
            # If in inference mode, we might need to clone to escape it for the new tensor
            if torch.is_inference_mode_enabled():
                 inp = inp.clone()
            
            inp.requires_grad_(True) 
        
            # 1. Retrieve context / Compute outputs
            memory_output = self.memory_mlp(inp)
            
            # 2. Compute "Surprise" / Perplexity
            # Reconstruction objective
            reconstruction = memory_output
            loss = F.mse_loss(reconstruction, inp)
            
            # Debugging check
            if not loss.requires_grad:
                # pass 
                # changing to avoid spam if it happens often, or keep one time warning
                pass

            # 3. Gating logic
            is_surprising = loss > self.surprise_threshold
            
            if is_surprising:
                # Trigger memory update (Test-Time Training)
                print(f"  [MAC] Surprise! (Loss: {loss.item():.4f}) -> Updating Memory")
                self.memory_mlp.update_memory(inp, loss)
            
        # 4. Return detached output to merge back into main model flow
        return memory_output.detach()

class GemmaWithTitans:
    def __init__(self, model_name: str = "google/gemma-3n-E4B", device: str = "cuda"):
        self.device = device
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.config = self.model.config
        
        # MAC Logic: "Silent Learning" until context overflow
        self.total_tokens_processed = 0
        self.start_memory_injection_threshold = 32000 # Default to 32k as requested
        # For testing, you might want to set this lower manually

        
        # Robust Layer Discovery
        self.layers_list = None
        
        # 1. Try standard paths
        potential_paths = [
            ["model", "layers"],        # Standard Llama/Gemma
            ["model", "language_model", "layers"], # Gemma 3n (Multimodal)
            ["model", "decoder", "layers"], # T5 style
            ["transformer", "h"],       # GPT-2/BLOOM
            ["layers"],                 # Some variants
            ["model", "blocks"],        # MPT/Dolly
        ]
        
        for path in potential_paths:
            curr = self.model
            valid = True
            for part in path:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    valid = False
                    break
            
            if valid and isinstance(curr, (nn.ModuleList, list, tuple)):
                self.layers_list = curr
                print(f"Found layers at path: {'.'.join(path)}")
                break
        
        # 2. Fallback: Search for the largest ModuleList
        if self.layers_list is None:
            print("Standard layer paths failed. Inspecting model structure...")
            max_len = 0
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList):
                    # Heuristic: The backbone usually has the most layers
                    if len(module) > max_len:
                        max_len = len(module)
                        self.layers_list = module
                        print(f"Heuristic found candidate layers at: {name} (Size: {len(module)})")

        if self.layers_list is None:
            # Inspection Dump
            print("\nCRITICAL: Could not find 'layers'. Dumping model structure:")
            print(self.model)
            raise AttributeError("Could not identify the Transformer blocks (layers) in the model.")

        if self.layers_list is None:
            # Inspection Dump
            print("\nCRITICAL: Could not find 'layers'. Dumping model structure:")
            print(self.model)
            raise AttributeError("Could not identify the Transformer blocks (layers) in the model.")

        self.insertion_layer_idx = len(self.layers_list) // 2
        
        # Robust Hidden Size Discovery
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            # Try text_config for multimodal models
            if hasattr(self.config, "text_config"):
                hidden_size = getattr(self.config.text_config, "hidden_size", None)
            
            # Try d_model (T5/older/other standards)
            if hidden_size is None:
                hidden_size = getattr(self.config, "d_model", None)
                
        if hidden_size is None:
             # Fallback: Check the first layer's input dimension logic
             # This is a bit hacky but works for Linear layers
             first_layer = self.layers_list[0]
             # Gemma3nTextDecoderLayer -> self_attn -> q_proj
             if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
                 hidden_size = first_layer.self_attn.q_proj.in_features
                 print(f"Inferred hidden_size from layer structure: {hidden_size}")

        if hidden_size is None:
            raise AttributeError("Could not determine 'hidden_size' from config or model structure.")

        self.mac_layer = TitansMACLayer(hidden_size).to(device)
        
        self._register_hooks()
        self.memory_context = None

    def _register_hooks(self):
        def pre_hook(module, input):
            hidden_states = input[0]
            mem_out = self.mac_layer(hidden_states)
            # Update global token counter (approximate based on batch size 1)
            # We use a simple attribute on the wrapper class to track global state
            self.total_tokens_processed += hidden_states.shape[1]
            
            # Silent MAC logic
            if self.total_tokens_processed > self.start_memory_injection_threshold:
                if not getattr(self, "mac_active_message_shown", False):
                    print(f"\n\033[92m>>> MAC MEMORY INJECTION ACTIVATED (Tokens > {self.start_memory_injection_threshold}) <<<\033[0m")
                    self.mac_active_message_shown = True
                    
                # Active Phase: Memory helps generating
                # Reduced residual scale to stop random init from breaking generation
                combined = hidden_states + 0.001 * mem_out
                return (combined, *input[1:])
            else:
                # Silent Phase: Memory learns but stays quiet
                # We do NOT add mem_out to the residual stream
                # print("  [MAC] Silent Mode - Input unchanged") # Uncomment for verbose debugging
                return input # Pass through unchanged

        layer = self.layers_list[self.insertion_layer_idx]
        layer.register_forward_pre_hook(pre_hook)

    def generate(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
             outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0])

if __name__ == "__main__":
    print("Run interact_mac.py to use this model.")
