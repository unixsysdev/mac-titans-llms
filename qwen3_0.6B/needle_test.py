import torch
from qwen_mac import QwenWithTitans
import time

def run_needle_test():
    print("--- Qwen 3 + Titans MAC: Needle in a Haystack Test ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    # 1. Load Model
    # Try Qwen 3 if available, otherwise 2.5
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
    try:
        wrapper = QwenWithTitans(model_name, device=device)
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

    # A more natural filler text (Alice in Wonderland snippet) to avoid "repetition collapse"
    filler = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: "
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "
        "'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), "
        "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her. "
    )
    
    print("\n--- Phase 1: Constructing Haystack ---")
    # We want total > 4k tokens (e.g. 6k).
    # Needle is at 0.
    # Adjust multiplier based on new filler length (~100 chars / ~25 tokens per filler?)
    # Old filler was ~10 tokens * 600 = 6000. New filler is ~120 tokens.
    # So we need roughly 50 repetitions to hit 6000.
    input_text = needle + " " + (filler * 60) 
    
    # Tokenize to check length
    tokens = wrapper.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    num_tokens = tokens.shape[1]
    print(f"Total Sequence Length: {num_tokens} tokens")
    
    if num_tokens < 4100:
        print("Warning: Sequence might be too short. Adding more text...")
        input_text += (filler * 30)
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
    
    if "BLUEBERRY" in response.upper() or "VAULT" in response.upper():
         pass # Check logic above
    
    # --- Phase 4: Long Context Control (Native Qwen Capabilities) ---
    print("\n--- Phase 4: Long Context Control (Native Qwen Check) ---")
    print("Resetting model and testing if Qwen can find the needle natively (Limit > 6k)...")
    
    # Re-init wrapper to clear memory state
    try:
        wrapper_control = QwenWithTitans(model_name, device=device)
        # Set threshold to INFINITY so MAC never injects (Native behavior)
        wrapper_control.start_memory_injection_threshold = 999999
        
        # Feed Haystack
        wrapper_control.model.eval()
        with torch.no_grad():
            for i in range(0, num_tokens, chunk_size):
                chunk = tokens[:, i : i + chunk_size]
                wrapper_control.model(chunk)
                
        # Query
        print("Generating control response (MAC Disabled)...")
        control_long_response = wrapper_control.generate(prompt, max_new_tokens=20)
        
        print("-" * 40)
        print(f"CONTROL (MAC DISABLED) RESPONSE:\n{control_long_response}")
        print("-" * 40)
        
        if "BLUEBERRY" in control_long_response.upper():
            print("\033[93mBASELINE RESULT: Qwen natively found the needle (Expected, since context < 32k).\033[0m")
            print("This confirms our 'MAC Test' is simulating a forgetting scenario by artificially limiting the threshold.")
        else:
            print("\033[91mBASELINE RESULT: Qwen failed natively. The task is genuinely hard.\033[0m")
            
    except Exception as e:
        print(f"Control test failed: {e}")

if __name__ == "__main__":
    run_needle_test()
