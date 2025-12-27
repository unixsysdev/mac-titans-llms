import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, TextStreamer
from typing import Optional, Tuple, List
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
                # TUNED: Conservative learning rate for stable training
                # 0.01 was causing loss explosion to inf
                param.data = param.data - (1e-4 * grad)

    def reset_memory(self):
        """Reset memory weights to initial random state"""
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias) 

class TitansMACLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.memory_mlp = MemoryMLP(hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        # TUNED: Threshold based on observed loss values
        # Phase 2 losses are 20-50, so 25.0 should trigger updates on novel content
        self.surprise_threshold = 25.0 

    def forward(self, current_hidden_states: torch.Tensor, prev_memory_state: Optional[torch.Tensor] = None):
        """
        Args:
            current_hidden_states: [Batch, Seq, Dim] - input from Qwen layer
            prev_memory_state: [Batch, MemSize, Dim] - previous memory context
        """
        # DEBUG: Log EVERY call to see if forward is running
        print(f"  [MAC FORWARD] Called! shape: {current_hidden_states.shape}")

        # We need to compute gradients for the memory module even during inference (TTT)
        with torch.enable_grad():
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

            # DEBUG: Log EVERY loss during Phase 2 (first 10 chunks) to see what's happening
            if not hasattr(self, '_log_count'):
                self._log_count = 0
            self._log_count += 1
            if self._log_count <= 10:  # Log first 10 forward passes
                print(f"  [MAC DEBUG] Loss: {loss.item():.2f} (threshold: {self.surprise_threshold})")

            # 3. Gating logic
            is_surprising = loss > self.surprise_threshold

            if is_surprising:
                # TUNED: Log all surprising events for debugging (with threshold 100.0)
                print(f"  [MAC] Surprise! (Loss: {loss.item():.2f}) -> Updating Memory")
                self.memory_mlp.update_memory(inp, loss)
            
        # 4. Return detached output to merge back into main model flow
        return memory_output.detach()

class QwenWithTitans:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", device: str = "cuda"):
        # UPDATED: Using Qwen3 4B Instruct Model (non-Thinking, stable)
        
        self.device = device 
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Loading model {model_name}...")
        # device_map="cuda" is REQUIRED for FP8 models - .to(device) doesn't work
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
        )
        self.config = self.model.config
        
        # MAC Logic: "Silent Learning" until context overflow
        self.total_tokens_processed = 0
        self.start_memory_injection_threshold = 32000 
        
        # Qwen usually puts layers in model.layers
        self.layers_list = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.layers_list = self.model.model.layers
        elif hasattr(self.model, "layers"):
             self.layers_list = self.model.layers
        
        if self.layers_list is None:
             raise AttributeError("Could not find 'layers' in Qwen model.")

        print(f"Found {len(self.layers_list)} layers.")
        self.insertion_layer_idx = len(self.layers_list) // 2
        
        # Hidden size
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None:
            # Fallback
            hidden_size = self.layers_list[0].self_attn.q_proj.in_features
            
        print(f"Hidden Size: {hidden_size}")

        # Get model dtype to ensure MAC layer matches
        model_dtype = next(self.model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")

        self.mac_layer = TitansMACLayer(hidden_size).to(device=device, dtype=model_dtype)
        self._register_hooks()

    def reset_mac_memory(self):
        """Reset the MAC memory to initial random state"""
        print("[MAC] Memory reset to initial state")
        self.mac_layer.memory_mlp.reset_memory()
        # Also reset the debug counter
        if hasattr(self.mac_layer, '_log_count'):
            delattr(self.mac_layer, '_log_count')

    def _register_hooks(self):
        def pre_hook(module, input):
            hidden_states = input[0]

            # DEBUG: Log hook calls
            if torch.rand(1).item() < 0.001:  # Log 0.1% of the time
                print(f"  [HOOK] Called! tokens shape: {hidden_states.shape}")

            mem_out = self.mac_layer(hidden_states)

            self.total_tokens_processed += hidden_states.shape[1]

            # Silent MAC logic
            if self.total_tokens_processed > self.start_memory_injection_threshold:
                if not getattr(self, "mac_active_message_shown", False):
                    print(f"\n\033[92m>>> MAC MEMORY INJECTION ACTIVATED (Tokens > {self.start_memory_injection_threshold}) <<<\033[0m")
                    self.mac_active_message_shown = True

                # TUNED: Gentler injection scale (0.1 instead of 1.0)
                # 1.0 was too strong and disrupted generation
                combined = hidden_states + 0.1 * mem_out
                return (combined, *input[1:])
            else:
                # BEFORE injection threshold: Only compute MAC (for learning), don't inject
                # This allows MAC to learn from the beginning even before injection starts
                return input

        layer = self.layers_list[self.insertion_layer_idx]
        layer.register_forward_pre_hook(pre_hook)

    def generate(self, prompt: str, max_new_tokens: int = 50):
        # Qwen Instruct uses chat templates natively
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Stream output so we see the "Thinking" process live
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
             generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
