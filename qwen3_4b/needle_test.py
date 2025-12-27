import torch
from qwen_mac import QwenWithTitans
import time

def run_needle_test():
    print("--- Qwen 3 (4B) + Titans MAC: Needle in a Haystack Test ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    # 1. Load Model
    model_name = "Qwen/Qwen3-4B-Instruct-2507" 
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
    # Saftey-friendly Needle to avoid model refusals
    needle = "The giant's favorite color is MAGENTA."
    query = "What is the giant's favorite color?"
    
    # --- Control Test (Sanity Check) ---
    print("\n--- Phase 0: Control Test (Short Context) ---")
    control_prompt = f"{needle} {query}"
    control_response = wrapper.generate(control_prompt, max_new_tokens=50)
    print(f"Control Prompt: {control_prompt}")
    print(f"Control Response:\n{control_response.strip()}")
    if "MAGENTA" in control_response.upper():
        print("\033[92mCONTROL PASSED: Model can retrieve fact from short context.\033[0m")
    else:
        print("\033[91mCONTROL FAILED: Model cannot even answer the question in short context!\033[0m")
        # return

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
    print("Resetting MAC memory to fresh state before feeding haystack...")
    wrapper.reset_mac_memory()
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
    query = "What is the giant's favorite color?"

    # Plain text prompt - generate() handles chat template
    prompt = query

    print(f"Prompt: {prompt}")
    print("Generating response...")
    
    # We use the wrapper.generate which handles the threshold logic (now likely Active)
    response = wrapper.generate(prompt, max_new_tokens=20)
    
    print("-" * 40)
    print(f"FULL RESPONSE:\n{response}")
    print("-" * 40)
    
    if "MAGENTA" in response.upper():
         print("\n\033[92mSUCCESS: The Needle was found!\033[0m")
    else:
         print("\n\033[91mFAILURE: The Needle was lost.\033[0m")
    
    # --- Phase 4: Long Context Control (Native Qwen Capabilities) ---
    print("\n--- Phase 4: Long Context Control (Native Qwen Check) ---")
    print("Testing if Qwen can find the needle when context is IN the prompt (baseline capability)...")

    # CORRECT BASELINE TEST: Include context IN the prompt
    # This tests if the model can retrieve when context is part of the generation prompt
    prompt_with_context = input_text + "\n\n" + query

    print(f"Total context in prompt: {num_tokens} tokens")
    print("Generating baseline response (context in prompt, MAC disabled)...")

    # Disable MAC for pure baseline
    wrapper.start_memory_injection_threshold = 999999999
    control_long_response = wrapper.generate(prompt_with_context, max_new_tokens=50)

    print("-" * 40)
    print(f"CONTROL (CONTEXT IN PROMPT) RESPONSE:\n{control_long_response}")
    print("-" * 40)

    if "MAGENTA" in control_long_response.upper():
        print("\033[92mBASELINE PASSED: Qwen can retrieve when context is in prompt (up to ~8k tokens).\033[0m")
    else:
        print("\033[91mBASELINE FAILED: Qwen cannot retrieve even with context in prompt.\033[0m")

if __name__ == "__main__":
    run_needle_test()
