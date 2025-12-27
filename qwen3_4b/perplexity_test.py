"""
Experiment 5: Perplexity Reduction - Core MAC Capability Test

Tests if MAC reduces perplexity on text it has learned.
This is the FUNDAMENTAL test of reconstruction-based learning.

Hypothesis: MAC should reduce perplexity (loss) on text it has seen,
because it learns to reconstruct familiar patterns.
"""

import torch
from qwen_mac import QwenWithTitans
import time
from tqdm import tqdm

def calculate_perplexity(wrapper, input_ids, max_length=512):
    """Calculate perplexity for a given text."""
    wrapper.model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        # Process in chunks
        for i in range(0, input_ids.shape[1], max_length):
            chunk = input_ids[:, i:i+max_length]

            # Get outputs
            outputs = wrapper.model(chunk, labels=chunk)
            loss = outputs.loss

            total_loss += loss.item() * chunk.shape[1]
            total_tokens += chunk.shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss

def test_perplexity_reduction():
    print("="*70)
    print("EXPERIMENT 5: Perplexity Reduction - Core MAC Capability Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # Create three texts:
    # 1. Learning text - what MAC will see during training
    # 2. Similar text - same style/pattern, different content
    # 3. Different text - completely different style

    learning_text = """
    The purple butterfly flew over the golden mountains at dawn.
    The silver hawk soared through the crystal clouds at noon.
    The blue whale swam across the emerald ocean at midnight.
    The red fox ran along the sandy beach at sunset.
    The green snake slithered across the rocky path at twilight.
    """ * 20  # Repeat 20x for strong learning signal

    similar_text = """
    The yellow bee buzzed around the orange flowers at morning.
    The white owl glided above the gray forest at evening.
    The pink salmon jumped up the river falls at afternoon.
    """

    different_text = """
    Computer programming is the process of creating instructions for machines.
    Software developers write code using programming languages like Python.
    The code is compiled or interpreted to execute on specific hardware platforms.
    Modern applications use frameworks and libraries for rapid development.
    """

    print("="*70)
    print("TEXT SAMPLES:")
    print("="*70)
    print(f"\n1. Learning text (repeated 20x):")
    print(f"   {learning_text[:100]}...")
    print(f"\n2. Similar text (same pattern):")
    print(f"   {similar_text.strip()}")
    print(f"\n3. Different text (unrelated):")
    print(f"   {different_text.strip()}")
    print("="*70)

    # Tokenize
    learning_tokens = wrapper.tokenizer(learning_text, return_tensors="pt").to(device)
    similar_tokens = wrapper.tokenizer(similar_text, return_tensors="pt").to(device)
    different_tokens = wrapper.tokenizer(different_text, return_tensors="pt").to(device)

    learning_len = learning_tokens['input_ids'].shape[1]
    similar_len = similar_tokens['input_ids'].shape[1]
    different_len = different_tokens['input_ids'].shape[1]

    print(f"\nLearning text: {learning_len} tokens")
    print(f"Similar text: {similar_len} tokens")
    print(f"Different text: {different_len} tokens")

    # ========== BASELINE: No MAC Learning ==========
    print("\n" + "="*70)
    print("BASELINE: Perplexity WITHOUT MAC Learning")
    print("="*70)

    wrapper.reset_mac_memory()

    print("\nCalculating baseline perplexities...")
    ppl_similar_baseline, loss_similar_baseline = calculate_perplexity(wrapper, similar_tokens['input_ids'])
    ppl_different_baseline, loss_different_baseline = calculate_perplexity(wrapper, different_tokens['input_ids'])

    print(f"Similar text perplexity (baseline): {ppl_similar_baseline:.2f}")
    print(f"Different text perplexity (baseline): {ppl_different_baseline:.2f}")

    # ========== WITH MAC: After Learning ==========
    print("\n" + "="*70)
    print("WITH MAC: Perplexity AFTER Learning Phase")
    print("="*70)

    wrapper.reset_mac_memory()

    print(f"\nFeeding learning text to MAC ({learning_len} tokens)...")
    chunk_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, learning_len, chunk_size), desc="Learning"):
            chunk = learning_tokens['input_ids'][:, i:i+chunk_size]
            _ = wrapper.model(chunk)

    print("Calculating perplexities with MAC...")
    ppl_similar_mac, loss_similar_mac = calculate_perplexity(wrapper, similar_tokens['input_ids'])
    ppl_different_mac, loss_different_mac = calculate_perplexity(wrapper, different_tokens['input_ids'])

    print(f"\nSimilar text perplexity (with MAC): {ppl_similar_mac:.2f}")
    print(f"Different text perplexity (with MAC): {ppl_different_mac:.2f}")

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nSimilar Text (same pattern as learning):")
    print(f"  Baseline perplexity: {ppl_similar_baseline:.2f}")
    print(f"  MAC perplexity:       {ppl_similar_mac:.2f}")
    ppl_similar_change = ((ppl_similar_mac - ppl_similar_baseline) / ppl_similar_baseline) * 100
    print(f"  Change:              {ppl_similar_change:+.1f}%")

    print(f"\nDifferent Text (unrelated to learning):")
    print(f"  Baseline perplexity: {ppl_different_baseline:.2f}")
    print(f"  MAC perplexity:       {ppl_different_mac:.2f}")
    ppl_different_change = ((ppl_different_mac - ppl_different_baseline) / ppl_different_baseline) * 100
    print(f"  Change:              {ppl_different_change:+.1f}%")

    # Interpret results
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if ppl_similar_change < -5 and ppl_different_change > -2:
        print("\n✅ SUCCESS: MAC reduces perplexity on similar text!")
        print("   Lower perplexity = better = MAC learned the pattern.")
        print(f"   Similar text improved by {abs(ppl_similar_change):.1f}%")
    elif ppl_similar_change < 0:
        print(f"\n⚠️ PARTIAL: Slight perplexity reduction on similar text ({ppl_similar_change:.1f}%)")
        print("   MAC shows weak pattern learning")
    elif ppl_different_change < -5 and ppl_similar_change > -2:
        print("\n⚠️ INTERESTING: MAC reduces perplexity more on unrelated text")
        print("   May indicate general compression rather than pattern learning")
    else:
        print("\n❌ FAILED: MAC does not reduce perplexity")
        print("   Reconstruction learning is not effective")

    print("\n" + "="*70)

    return {
        'ppl_similar_baseline': ppl_similar_baseline,
        'ppl_similar_mac': ppl_similar_mac,
        'ppl_different_baseline': ppl_different_baseline,
        'ppl_different_mac': ppl_different_mac,
    }

if __name__ == "__main__":
    test_perplexity_reduction()
