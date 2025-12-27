import torch
from gemma_mac import GemmaWithTitans
import time

def run_needle_test():
    print("--- Gemma + Titans MAC: Needle in a Haystack Test ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    # 1. Load Model
    model_name = "google/gemma-3n-E4B"
    try:
        wrapper = GemmaWithTitans(model_name, device=device)
        print("Model Loaded Successfully!")
        
        # We want the threshold to be 4k (as requested) so we can prove it learns even when "silent"
        # and retrieves after the window is exceeded.
        wrapper.start_memory_injection_threshold = 4000
        print(f"Memory Injection Threshold: {wrapper.start_memory_injection_threshold}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Prepare Data
    needle = "The secret password to the vault is BLUEBERRY."
    query = "What is the secret password to the vault?"
    
    # --- Control Test (Sanity Check) ---
    print("\n--- Phase 0: Control Test (Short Context) ---")
    control_prompt = f"<start_of_turn>user\n{needle} {query}<end_of_turn>\n<start_of_turn>model\n"
    control_response = wrapper.generate(control_prompt, max_new_tokens=20)
    print(f"Control Prompt: {needle} {query}")
    print(f"Control Response: {control_response.strip()}")
    if "BLUEBERRY" in control_response.upper():
        print("\033[92mCONTROL PASSED: Model can retrieve password from short context.\033[0m")
    else:
        print("\033[91mCONTROL FAILED: Model cannot even answer the question in short context!\033[0m")
        return

    # A short filler sentence ~10 tokens
    filler = "The quick brown fox jumps over the lazy dog. " 
    
    print("\n--- Phase 1: Constructing Haystack ---")
    # We want total > 4k tokens (e.g. 6k).
    # Needle is at 0.
    input_text = needle + " " + (filler * 600) # Reduced multiplier for ~5-6k tokens
    
    # Tokenize to check length
    tokens = wrapper.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    num_tokens = tokens.shape[1]
    print(f"Total Sequence Length: {num_tokens} tokens")
    
    if num_tokens < 4100:
        print("Warning: Sequence might be too short to push needle out of standard attention window (assuming ~4k).")
        print("Adding more text...")
        input_text += (filler * 200)
        tokens = wrapper.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        num_tokens = tokens.shape[1]
        print(f"New Sequence Length: {num_tokens} tokens")

    # 3. Feed the Haystack (Training the Memory)
    print("\n--- Phase 2: Feeding Haystack (Learning) ---")
    print("Feeding text in chunks of 1024 tokens to trigger sequential memory updates...")
    
    chunk_size = 1024
    start_time = time.time()
    
    wrapper.model.eval() # Ensure model is in eval mode (our MAC handles gradients internally)
    
    # We manually feed chunks to the model to trigger the hooks
    # We do not generate, just forward pass.
    # NOTE: We need to use wrapper.model() directly
    
    with torch.no_grad(): # Main model no_grad, MAC handles its own grad
        for i in range(0, num_tokens, chunk_size):
            chunk = tokens[:, i : i + chunk_size]
            
            # Forward pass triggers the hook -> MAC update
            # We ignore output, we just want the side effect (Memory Update)
            wrapper.model(chunk)
            
            processed = min(i + chunk_size, num_tokens)
            print(f"Processed {processed}/{num_tokens} tokens...", end="\r")

    print(f"\nDistractor Injection Complete. Time: {time.time() - start_time:.2f}s")
    print(f"MAC State: Tokens Processed = {wrapper.total_tokens_processed}")
    
    # 4. The Query (Retrieval)
    print("\n--- Phase 3: The Query (Retrieval) ---")
    query = "What is the secret password to the vault?"
    
    # Explicitly force chat template for the query
    prompt = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
    
    print(f"Prompt: {query}")
    print("Generating response...")
    
    # We use the wrapper.generate which handles the threshold logic (now likely Active)
    response = wrapper.generate(prompt, max_new_tokens=20)
    
    print("-" * 40)
    print(f"FULL RESPONSE:\n{response}")
    print("-" * 40)
    
    if "BLUEBERRY" in response.upper():
        print("\n\033[92mSUCCESS: The Needle was found!\033[0m")
    else:
        print("\n\033[91mFAILURE: The Needle was lost.\033[0m")

if __name__ == "__main__":
    run_needle_test()
