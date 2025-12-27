"""
Experiment 2: "Style Transfer" - Pattern Injection Test

Tests if MAC can learn and replay a specific writing style from a long document,
even when generating completely different content.

Hypothesis: MAC should learn the "surprising" pirate style and inject it into
unrelated generation tasks, maintaining the style across context boundaries.
"""

import torch
from qwen_mac import QwenWithTitans
import time

def generate_pirate_document(num_tokens=10000):
    """
    Generate a long document in 17th-century pirate speak.
    The style should be consistent and distinctive.
    """

    # Short, punchy pirate phrases to repeat
    pirate_phrases = """
    Ahoy there, matey! Shiver me timbers and blow me down!
    Arrr, me hearties, the sea be a cruel mistress indeed.
    Fair winds to ye, wherever the sails may take ye!
    The cap'n be a grizzled old sea dog with a white beard.
    Hoist the colors and ready the cannons, we sail at dawn!
    A pirate's life be freedom, gold, and adventure on the high seas.
    Ye scurvy dogs better swab the deck or walk the plank!
    Thar be treasure buried on that island, mark me words.
    The bosun cracked his whip and the crew jumped to it.
    We set sail with the wind in our sails and gold in our eyes.
    """

    # Repeat 80x for massive repetition
    pirate_text = pirate_phrases * 80

    return pirate_text

def test_style_transfer():
    print("="*70)
    print("EXPERIMENT 2: Style Transfer - Pattern Injection Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # Generate pirate-style document
    print("Generating pirate-style document...")
    pirate_document = generate_pirate_document()

    # Tokenize to get length
    tokens = wrapper.tokenizer(pirate_document, return_tensors="pt").to(device)
    num_tokens = tokens['input_ids'].shape[1]
    print(f"Document length: {num_tokens} tokens")

    # Show a sample of the style
    print("\n" + "="*70)
    print("STYLE SAMPLE (Pirate Speak):")
    print("="*70)
    print(pirate_document[:500] + "...")
    print("="*70)

    # ========== PHASE 1: Feed Document (Learning Style) ==========
    print("\n" + "="*70)
    print("PHASE 1: Feeding Pirate Document (MAC Style Learning)")
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

    # ========== PHASE 2: Test Style Injection ==========
    print("\n" + "="*70)
    print("PHASE 2: Testing Style Injection")
    print("="*70)
    print("Testing if MAC injects pirate style into UNRELATED topics...")
    print("="*70)

    # Test queries about completely different topics
    test_queries = [
        "What is a computer program?",
        "How does photosynthesis work?",
        "Describe a sunny day.",
        "Write a weather report."
    ]

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        print("Generating response...")

        response = wrapper.generate(query, max_new_tokens=80)
        print(f"\nResponse:\n{response}")

        # Check for pirate style indicators
        pirate_indicators = [
            "ahoy", "matey", "arr", "ye", "har", "pirate", "sea", "ship",
            "cap'n", "bosun", "cutlass", "treasure", "rum"
        ]

        style_score = sum(1 for indicator in pirate_indicators
                         if indicator in response.lower())

        results.append({
            'query': query,
            'response': response,
            'style_score': style_score
        })

        print(f"\nPirate style indicators found: {style_score}")

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    avg_style_score = sum(r['style_score'] for r in results) / len(results)
    print(f"\nAverage pirate style indicators per response: {avg_style_score:.1f}")

    high_style_count = sum(1 for r in results if r['style_score'] >= 3)
    print(f"Responses with strong pirate style (≥3 indicators): {high_style_count}/{len(results)}")

    if avg_style_score >= 3:
        print("\n✅ SUCCESS: MAC successfully injected pirate style!")
        print("   Even though the queries were about COMPLETELY different topics,")
        print("   the MAC maintained the writing style from the 10k-token document.")
    elif avg_style_score >= 1:
        print("\n⚠️ PARTIAL SUCCESS: Some style injection detected")
        print("   The MAC shows some ability to maintain style, but needs tuning.")
    else:
        print("\n❌ FAILED: MAC did not inject the pirate style")
        print("   The model reverted to normal style for unrelated queries.")

    print("\n" + "="*70)

    return results

if __name__ == "__main__":
    test_style_transfer()
