"""
Experiment 6: MAC Injection Diagnostic Test

Directly tests if MAC is actually injecting into the model.
This is a sanity check to verify the implementation is working.
"""

import torch
from qwen_mac import QwenWithTitans

def test_mac_injection():
    print("="*70)
    print("EXPERIMENT 6: MAC Injection Diagnostic Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # Simple test text
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = wrapper.tokenizer(test_text, return_tensors="pt").to(device)
    input_ids = tokens['input_ids']

    print("="*70)
    print("TEST 1: Verify MAC Forward Pass is Called")
    print("="*70)

    wrapper.reset_mac_memory()

    # Capture whether MAC forward is called
    print("\nProcessing text through model...")
    print("Watch for '[MAC FORWARD]' messages below:\n")

    with torch.no_grad():
        output = wrapper.model(input_ids)

    print("\n✅ If you see '[MAC FORWARD]' messages above, MAC is being called")

    print("\n" + "="*70)
    print("TEST 2: Check Injection Status")
    print("="*70)

    print(f"\nCurrent injection threshold: {wrapper.start_memory_injection_threshold} tokens")
    print(f"Current tokens processed: {wrapper.total_tokens_processed}")
    print(f"Injection active: {wrapper.total_tokens_processed > wrapper.start_memory_injection_threshold}")

    print("\n" + "="*70)
    print("TEST 3: Process Enough Tokens to Activate Injection")
    print("="*70)

    wrapper.reset_mac_memory()

    # Create a long text to pass the 4000 token threshold
    long_text = "The quick brown fox jumps over the lazy dog. " * 500  # ~6000 tokens
    long_tokens = wrapper.tokenizer(long_text, return_tensors="pt").to(device)
    long_input_ids = long_tokens['input_ids']
    long_token_count = long_input_ids.shape[1]

    print(f"\nProcessing {long_token_count} tokens...")
    print(f"This should pass the {wrapper.start_memory_injection_threshold} token threshold\n")

    chunk_size = 1024
    with torch.no_grad():
        for i in range(0, long_token_count, chunk_size):
            chunk = long_input_ids[:, i:i+chunk_size]
            _ = wrapper.model(chunk)
            tokens_so_far = wrapper.total_tokens_processed
            injection_active = tokens_so_far > wrapper.start_memory_injection_threshold
            print(f"  Processed {tokens_so_far} tokens - Injection: {'ACTIVE' if injection_active else 'INACTIVE'}")

    print(f"\n✅ Injection should now be ACTIVE (tokens > {wrapper.start_memory_injection_threshold})")

    print("\n" + "="*70)
    print("TEST 4: Generate Text With and Without MAC")
    print("="*70)

    test_prompt = "Once upon a time"

    # Without MAC (reset)
    wrapper.reset_mac_memory()
    print("\nGenerating WITHOUT MAC (memory reset):")
    response_no_mac = wrapper.generate(test_prompt, max_new_tokens=30)
    print(f"Response: {response_no_mac}")

    # With MAC (after learning)
    wrapper.reset_mac_memory()
    learning_text = "Pirate speak: Ahoy matey! Shiver me timbers! " * 100
    learning_tokens = wrapper.tokenizer(learning_text, return_tensors="pt").to(device)

    print(f"\nLearning pirate text ({learning_tokens['input_ids'].shape[1]} tokens)...")
    with torch.no_grad():
        # Process enough to activate injection
        for i in range(0, min(learning_tokens['input_ids'].shape[1], 5000), 1024):
            chunk = learning_tokens['input_ids'][:, i:i+1024]
            _ = wrapper.model(chunk)

    print(f"Tokens processed: {wrapper.total_tokens_processed}")
    print(f"Injection active: {wrapper.total_tokens_processed > wrapper.start_memory_injection_threshold}")

    print("\nGenerating WITH MAC (after learning pirate text):")
    response_with_mac = wrapper.generate(test_prompt, max_new_tokens=30)
    print(f"Response: {response_with_mac}")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print(f"\nWithout MAC: {response_no_mac}")
    print(f"With MAC:    {response_with_mac}")

    if response_no_mac != response_with_mac:
        print("\n✅ SUCCESS: MAC is affecting generation!")
        print("   The responses are different, proving MAC injection works.")
    else:
        print("\n⚠️ SAME RESPONSE: MAC may not be affecting generation")
        print("   Either injection isn't working or effect is too subtle")

    print("\n" + "="*70)

    return {
        'no_mac_response': response_no_mac,
        'with_mac_response': response_with_mac,
        'responses_differ': response_no_mac != response_with_mac
    }

if __name__ == "__main__":
    test_mac_injection()
