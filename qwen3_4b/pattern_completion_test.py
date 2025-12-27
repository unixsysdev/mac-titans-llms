"""
Experiment 3: Pattern Completion - Reconstruction Test

Tests if MAC can learn and complete repetitive patterns.
This plays to MAC's strength: learning to reconstruct/reproduce patterns.

Hypothesis: MAC should learn repetitive patterns and complete them when prompted,
because reconstruction objective works for pattern reproduction (not association).
"""

import torch
from qwen_mac import QwenWithTitans
import time

def generate_pattern_document():
    """
    Generate a document with a very specific, repetitive pattern.
    Pattern: "The magic number is [X]. The square is [X*X]."
    """

    pattern_lines = []
    for i in range(1, 51):
        square = i * i
        line = f"The magic number is {i}. The square is {square}."
        pattern_lines.append(line)

    # Create 3 paragraphs, each with the number-square pattern
    document = "\n".join(pattern_lines) + "\n\n"
    document += "\n".join(pattern_lines) + "\n\n"
    document += "\n".join(pattern_lines)

    return document

def test_pattern_completion():
    print("="*70)
    print("EXPERIMENT 3: Pattern Completion - Reconstruction Test")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"\nLoading {model_name}...")
    wrapper = QwenWithTitans(model_name, device=device)
    print("Model loaded!\n")

    # Generate pattern document
    print("Generating document with repetitive pattern...")
    pattern_document = generate_pattern_document()

    # Show a sample
    print("\n" + "="*70)
    print("PATTERN SAMPLE (First 3 lines):")
    print("="*70)
    lines = pattern_document.split('\n')
    for line in lines[:3]:
        print(f"  {line}")
    print("  ...")
    print("="*70)

    # Tokenize
    tokens = wrapper.tokenizer(pattern_document, return_tensors="pt").to(device)
    num_tokens = tokens['input_ids'].shape[1]
    print(f"\nDocument length: {num_tokens} tokens")

    # ========== PHASE 1: Feed Pattern (Learning) ==========
    print("\n" + "="*70)
    print("PHASE 1: Feeding Pattern Document (MAC Learning)")
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

    # ========== PHASE 2: Test Pattern Completion ==========
    print("\n" + "="*70)
    print("PHASE 2: Testing Pattern Completion")
    print("="*70)
    print("Testing if MAC completes the learned pattern...")
    print("="*70)

    # Test prompts that follow the pattern
    test_prompts = [
        ("The magic number is 51.", "51"),
        ("The magic number is 99.", "99"),
        ("The magic number is 100.", "100"),
    ]

    results = []

    for prompt, expected_num in test_prompts:
        print(f"\n--- Test Prompt: '{prompt}' ---")
        print("Generating completion (expecting square calculation)...")

        response = wrapper.generate(prompt, max_new_tokens=30)
        print(f"Response: {response}")

        # Check if response contains the correct square
        expected_square = int(expected_num) * int(expected_num)

        # Look for the square number in the response
        has_correct_square = str(expected_square) in response

        # Also check if it mentions "square" or pattern keywords
        pattern_keywords = ["square", "The square is", "equals"]
        has_pattern = any(keyword in response.lower() for keyword in pattern_keywords)

        success = has_correct_square and has_pattern

        results.append({
            'prompt': prompt,
            'expected': f"{expected_num} → {expected_square}",
            'response': response,
            'has_correct_square': has_correct_square,
            'has_pattern': has_pattern,
            'success': success
        })

        if success:
            print(f"✅ SUCCESS: Correctly computed {expected_num}² = {expected_square}")
        elif has_pattern and not has_correct_square:
            print(f"⚠️ PARTIAL: Has pattern but wrong or missing square")
        else:
            print(f"❌ FAIL: No pattern completion")

    # ========== CONTROL TEST (No MAC) ==========
    print("\n" + "="*70)
    print("CONTROL TEST: Baseline (No MAC Learning)")
    print("="*70)
    print("Resetting MAC and testing WITHOUT learning phase...")
    print("="*70)

    wrapper.reset_mac_memory()

    control_prompt = "The magic number is 51."
    print(f"\nPrompt: '{control_prompt}'")
    print("Generating response...")

    control_response = wrapper.generate(control_prompt, max_new_tokens=30)
    print(f"Response: {control_response}")

    control_has_square = "2601" in control_response  # 51² = 2601
    control_has_pattern = any(kw in control_response.lower() for kw in pattern_keywords)

    print(f"\nControl test:")
    if control_has_square and control_has_pattern:
        print("  ✅ Baseline also completes pattern (model knows math)")
    else:
        print("  ❌ Baseline doesn't complete pattern")

    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nMAC Test: {success_count}/{len(results)} successful completions")

    print("\nDetailed results:")
    for r in results:
        status = "✅ PASS" if r['success'] else ("⚠️ PARTIAL" if r['has_pattern'] else "❌ FAIL")
        print(f"  {status}: {r['prompt']} → {r['expected']}")

    if success_count >= len(results) - 1:
        print("\n✅ SUCCESS: MAC learned and completed the pattern!")
        print("   The reconstruction objective works for pattern reproduction.")
    elif success_count > 0:
        print("\n⚠️ PARTIAL SUCCESS: MAC shows some pattern learning")
    else:
        print("\n❌ FAILED: MAC did not complete the pattern")
        if not control_has_square:
            print("   Note: Neither MAC nor baseline completed the pattern")
        else:
            print("   Note: Baseline completed pattern too - model knows math independently")

    print("\n" + "="*70)

    return results

if __name__ == "__main__":
    test_pattern_completion()
