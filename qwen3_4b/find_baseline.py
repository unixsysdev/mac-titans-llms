import torch
from qwen_mac import QwenWithTitans

def test_context_length(context_length_tokens):
    """Test if model can retrieve needle at specific context length"""
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Testing Context Length: {context_length_tokens} tokens")
    print(f"{'='*60}")

    wrapper = QwenWithTitans(model_name, device=device)

    # Disable MAC for pure baseline test
    wrapper.start_memory_injection_threshold = 999999999

    needle = "The giant's favorite color is MAGENTA."
    query = "What is the giant's favorite color?"

    # Create filler text
    filler = (
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: "
        "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "
        "'and what is the use of a book,' thought Alice 'without pictures or conversation?' "
        "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), "
        "whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, "
        "when suddenly a White Rabbit with pink eyes ran close by her. "
    )

    # Build haystack with needle at start
    # Each filler ~120 tokens, so we repeat as needed
    filler_repeats = max(1, int((context_length_tokens - 10) / 120))
    input_text = needle + " " + (filler * filler_repeats)

    # Tokenize to verify length
    tokens = wrapper.tokenizer(input_text, return_tensors="pt").to(device)
    actual_length = tokens.shape[1]
    print(f"Target: {context_length_tokens} tokens | Actual: {actual_length} tokens")

    # Feed the haystack
    print(f"Feeding {actual_length} tokens to model...")
    wrapper.model.eval()
    with torch.no_grad():
        wrapper.model(tokens)

    # Query
    print(f"Query: {query}")
    response = wrapper.generate(query, max_new_tokens=50)

    # Check result
    success = "MAGENTA" in response.upper()
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n{status}")
    print(f"Response: {response}")
    print(f"Needle found: {success}")

    # Clean up to free GPU memory
    del wrapper
    torch.cuda.empty_cache()

    return success, actual_length, response

if __name__ == "__main__":
    print("Baseline Context Length Test for Qwen3-4B-Instruct")
    print("Finding the maximum context length for reliable needle retrieval...")

    # Test various context lengths
    test_lengths = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    results = []
    for length in test_lengths:
        try:
            success, actual_length, response = test_context_length(length)
            results.append({
                'target': length,
                'actual': actual_length,
                'success': success,
                'response': response
            })
        except Exception as e:
            print(f"ERROR at {length} tokens: {e}")
            results.append({
                'target': length,
                'actual': None,
                'success': False,
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Target':<10} {'Actual':<10} {'Result':<10}")
    print(f"{'-'*40}")

    for r in results:
        actual = str(r.get('actual', 'N/A'))
        result = "✅ PASS" if r.get('success') else "❌ FAIL"
        print(f"{r['target']:<10} {actual:<10} {result:<10}")

    # Find the crossover point
    successful_lengths = [r['actual'] for r in results if r.get('success') and r.get('actual')]
    failed_lengths = [r['actual'] for r in results if not r.get('success') and r.get('actual')]

    if successful_lengths and failed_lengths:
        max_success = max(successful_lengths)
        min_fail = min(failed_lengths)
        print(f"\n🎯 Baseline Crossover Point:")
        print(f"   Maximum working context: {max_success} tokens")
        print(f"   First failing context: {min_fail} tokens")
        print(f"   Estimated limit: ~{max_success} tokens")
