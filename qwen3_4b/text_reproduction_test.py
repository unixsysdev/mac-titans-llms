"""
Experiment 4: Text Reproduction - Direct Reconstruction Test

Tests if MAC can help reproduce text it just learned.
This is the SIMPLEST possible test for reconstruction-based MAC.

Hypothesis: MAC should help reproduce recently learned text better than baseline,
because it explicitly learns to reconstruct hidden states.
"""

import torch
from qwen_mac import QwenWithTitans
import time

def test_text_reproduction():
    print("="*70)
    print("EXPERIMENT 4: Text Reproduction - Direct Reconstruction Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # A distinctive paragraph to memorize
    target_text = """
    The crimson dragonfly danced above the emerald pond at sunset.
    Its wings reflected golden light like scattered jewels in the water.
    Three silver fish watched from below, dreaming of impossible flights.
    """

    print("="*70)
    print("TARGET TEXT TO REPRODUCE:")
    print("="*70)
    print(target_text.strip())
    print("="*70)

    # Tokenize
    tokens = wrapper.tokenizer(target_text, return_tensors="pt").to(device)
    num_tokens = tokens['input_ids'].shape[1]
    print(f"Target text length: {num_tokens} tokens")

    # ========== TEST 1: Baseline (No Learning) ==========
    print("\n" + "="*70)
    print("TEST 1: Baseline - No MAC Learning")
    print("="*70)

    wrapper.reset_mac_memory()

    # DON'T include target text in prompt - test recall from MAC
    prompt_baseline = "What text comes after 'The crimson dragonfly'?"
    print(f"Prompt: '{prompt_baseline}'")
    print("\nGenerating response...")

    baseline_response = wrapper.generate(prompt_baseline, max_new_tokens=50)
    print(f"\nBaseline Response:\n{baseline_response}")

    # Calculate overlap
    baseline_words = set(baseline_response.lower().split())
    target_words = set(target_text.lower().split())
    baseline_overlap = len(baseline_words & target_words) / len(target_words) * 100

    print(f"\nWord overlap with target: {baseline_overlap:.1f}%")

    # ========== TEST 2: With MAC Learning ==========
    print("\n" + "="*70)
    print("TEST 2: With MAC - After Learning Phase")
    print("="*70)

    # Reset and learn
    wrapper.reset_mac_memory()
    print("Feeding target text to MAC...")

    wrapper.model.eval()
    with torch.no_grad():
        _ = wrapper.model(tokens['input_ids'])

    print(f"MAC processed {num_tokens} tokens")

    # Now test recall WITHOUT text in prompt
    prompt_mac = "What text comes after 'The crimson dragonfly'?"
    print(f"\nPrompt: '{prompt_mac}'")
    print("Generating response...")

    mac_response = wrapper.generate(prompt_mac, max_new_tokens=50)
    print(f"\nMAC Response:\n{mac_response}")

    # Calculate overlap
    mac_words = set(mac_response.lower().split())
    mac_overlap = len(mac_words & target_words) / len(target_words) * 100

    print(f"\nWord overlap with target: {mac_overlap:.1f}%")

    # ========== TEST 3: Continuation ==========
    print("\n" + "="*70)
    print("TEST 3: Text Continuation - Complete the Pattern")
    print("="*70)

    # Give first sentence, see if MAC continues with the rest
    prompt_continuation = "The crimson dragonfly danced above the emerald pond at sunset."

    print(f"Prompt: '{prompt_continuation}'")
    print("Expected: Should continue with 'Its wings reflected...'")

    # Reset and learn
    wrapper.reset_mac_memory()
    with torch.no_grad():
        _ = wrapper.model(tokens['input_ids'])

    print("\nGenerating completion...")
    continuation = wrapper.generate(prompt_continuation, max_new_tokens=50)
    print(f"Continuation: {continuation}")

    # Check if it contains key distinctive words from the original text
    distinctive_words = ["wings", "reflected", "golden", "jewels", "silver", "fish", "dreaming"]
    continuation_distinctive = [w for w in distinctive_words if w in continuation.lower()]
    continuation_score = len(continuation_distinctive)

    print(f"\nDistinctive words from original found: {continuation_distinctive}")
    print(f"Score: {continuation_score}/{len(distinctive_words)} words")

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nTest 1 - Baseline word overlap: {baseline_overlap:.1f}%")
    print(f"Test 2 - MAC word overlap: {mac_overlap:.1f}%")
    print(f"Test 3 - Continuation distinctive words: {continuation_score}/{len(distinctive_words)}")

    mac_improvement = mac_overlap - baseline_overlap

    print(f"\nMAC improvement over baseline (recall): {mac_improvement:+.1f}%")

    if mac_improvement > 10:
        print("\n✅ SUCCESS: MAC improves text recall!")
        print("   Reconstruction-based memory helps retain and retrieve content.")
    elif mac_improvement > 0:
        print("\n⚠️ MARGINAL: MAC shows slight improvement")
    elif continuation_score >= 3:
        print("\n⚠️ PARTIAL: MAC shows improvement in continuation task")
        print(f"   Found {continuation_score} distinctive words from original")
    else:
        print("\n❌ FAILED: MAC does not improve recall or continuation")

    print("\n" + "="*70)

    return {
        'baseline_overlap': baseline_overlap,
        'mac_overlap': mac_overlap,
        'continuation_score': continuation_score,
        'improvement': mac_improvement
    }

if __name__ == "__main__":
    test_text_reproduction()
