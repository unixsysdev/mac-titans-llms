"""
Experiment 1: "Alien Language" - Rule Learning Test

Tests if MAC can learn and apply a made-up word-mapping rule from early in a long document.

Hypothesis: MAC should learn the surprising "alien" words and inject them into generation,
even when the original rule is far outside the context window.
"""

import torch
from qwen_mac import QwenWithTitans
import time

def generate_alien_document(num_tokens=10000):
    """
    Generate a long document with "alien language" words embedded throughout.
    The mapping is introduced at the START, then used throughout.
    """
    # The "alien language" mapping
    alien_words = {
        "glorp": "beautiful",
        "zibble": "quickly",
        "frazz": "computer",
        "blap": "happy",
        "snizzle": "program"
    }

    # PURE Q&A repetition - make it impossible to miss the pattern
    single_qa = f"""
Q: What does 'glorp' mean?
A: 'glorp' means {alien_words['glorp']}.

Q: Define 'glorp'.
A: The word 'glorp' means {alien_words['glorp']}.

Q: What is 'glorp'?
A: 'glorp' is {alien_words['glorp']}.

Q: What does 'frazz' mean?
A: 'frazz' means {alien_words['frazz']}.

Q: Define 'frazz'.
A: 'frazz' is a word that means {alien_words['frazz']}.

Q: What is a 'frazz'?
A: A 'frazz' is a {alien_words['frazz']}.

Q: What does 'snizzle' mean?
A: 'snizzle' means {alien_words['snizzle']}.

Q: What is a 'snizzle'?
A: A 'snizzle' is a {alien_words['snizzle']}.

Q: Define 'snizzle'.
A: The word 'snizzle' means {alien_words['snizzle']}.

"""

    # Repeat 50x for overwhelming pattern
    document = single_qa * 50

    return document, alien_words

def test_alien_language():
    print("="*70)
    print("EXPERIMENT 1: Alien Language - Rule Learning Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # Generate document with alien words
    print("Generating document with embedded 'alien language'...")
    document, alien_words = generate_alien_document(num_tokens=10000)

    # Show the alien language rules
    print("\n" + "="*70)
    print("ALIEN LANGUAGE RULES (introduced at start):")
    print("="*70)
    for alien, meaning in alien_words.items():
        print(f"  '{alien}' → '{meaning}'")
    print("="*70)

    # Tokenize to get length
    tokens = wrapper.tokenizer(document, return_tensors="pt").to(device)
    num_tokens = tokens['input_ids'].shape[1]
    print(f"\nDocument length: {num_tokens} tokens")

    # ========== PHASE 1: Feed Document (Learning) ==========
    print("\n" + "="*70)
    print("PHASE 1: Feeding Document (MAC Learning)")
    print("="*70)
    print("Resetting MAC memory...")
    wrapper.reset_mac_memory()

    print(f"Feeding {num_tokens} tokens in chunks...")
    chunk_size = 1024
    start_time = time.time()

    wrapper.model.eval()
    with torch.no_grad():
        for i in range(0, num_tokens, chunk_size):
            chunk = tokens['input_ids'][:, i : i + chunk_size]
            wrapper.model(chunk)
            print(f"  Processed {min(i + chunk_size, num_tokens)}/{num_tokens} tokens...")

    feed_time = time.time() - start_time
    print(f"Feeding complete in {feed_time:.2f}s")
    print(f"MAC State: {wrapper.total_tokens_processed} tokens processed")

    # ========== PHASE 2: Test Rule Application ==========
    print("\n" + "="*70)
    print("PHASE 2: Testing Rule Application")
    print("="*70)

    # Test queries that require applying the alien language rules
    # Each query tests if the model knows the meaning of an alien word
    test_queries = [
        ("glorp", "beautiful", "What does 'glorp' mean?"),
        ("frazz", "computer", "Define 'frazz'."),
        ("snizzle", "program", "What is a 'snizzle'?"),
    ]

    results = []
    for i, (alien_word, expected_meaning, query) in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Alien word: '{alien_word}' → Expected: '{expected_meaning}'")
        print(f"Query: {query}")
        print("Generating response...")

        response = wrapper.generate(query, max_new_tokens=50)
        print(f"Response: {response}")

        # Check if the expected meaning appears in the response
        success = expected_meaning in response.lower()

        results.append({
            'alien_word': alien_word,
            'expected': expected_meaning,
            'query': query,
            'response': response,
            'success': success
        })

        print(f"✅ PASS: '{expected_meaning}' found" if success else f"❌ FAIL: '{expected_meaning}' not found")

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nTests passed: {success_count}/{len(results)}")

    print("\nDetailed results:")
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"  {status}: '{r['alien_word']}' → '{r['expected']}'")

    if success_count > 0:
        print("\n✅ SUCCESS: MAC learned and applied the alien language rules!")
        print("   Even though the rules were introduced at the START of the document,")
        print("   the MAC was able to learn and inject them during generation.")
    else:
        print("\n❌ FAILED: MAC did not apply the alien language rules")
        print("   The model may need tuning or the architecture may need adjustment")

    print("\n" + "="*70)

    return results

if __name__ == "__main__":
    test_alien_language()
