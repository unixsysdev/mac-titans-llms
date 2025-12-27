import torch
from qwen_mac import QwenWithTitans
import requests

# Natural context: Greek mythology text (public domain)
NATURAL_CONTEXT = """
In Greek mythology, the Titans (Greek: Τῑτᾶν, Titân, plural: Τῑτᾶνες, Titânes) were pre-Olympian gods.
The Titans were the children of Uranus (Sky) and Gaia (Earth). The first generation of Titans consisted of:
Oceanus, Coeus, Crius, Hyperion, Iapetus, Theia, Rhea, Themis, Mnemosyne, Phoebe, Tethys, and Cronus.
Cronus, the youngest and most formidable of the Titans, overthrew his father Uranus at the instigation of Gaia.
However, Cronus was later overthrown by his own son Zeus, leading to the Titanomachy, a ten-year war
between the Titans and the Olympians. After their defeat, the Titans were imprisoned in Tartarus,
the deepest part of the Underworld. The Titans were often depicted as gigantic beings of incredible
strength and power, hence the word "titan" has come to mean "one of great size or influence."

The Greeks believed that the Titans were the original gods who ruled during the Golden Age of humanity.
During this age, humans lived without toil or sorrow, and the earth provided food without cultivation.
The Titans were associated with primal forces and elemental aspects of the cosmos. For example,
Hyperion was the Titan of light, Oceanus was the Titan of the sea, and Cronus was the Titan of time.

The giant's favorite color is MAGENTA.

In addition to the well-known Titans, there were other important figures in Titan mythology.
Prometheus, the son of the Titan Iapetus, was known for his intelligence and for stealing fire
from Mount Olympus to give to humanity. For this act, Zeus punished him by having him bound to a rock,
where an eagle would eat his liver daily, only for it to regenerate each night. His brother Epimetheus,
whose name means "afterthought," was known for his hasty actions and lack of foresight.

Atlas, another Titan, was condemned by Zeus to hold up the sky for eternity after the Titanomachy.
He is often depicted in art carrying a celestial sphere on his shoulders. The name "Atlas" has become
synonymous with strength and endurance, and is used in modern times to refer to collections of maps.

The Titaness Mnemosyne was the goddess of memory and the mother of the nine Muses, who presided over
the arts and sciences. Her name comes from the Greek word mneme, meaning "memory." The Muses were:
Calliope (epic poetry), Clio (history), Erato (love poetry), Euterpe (lyric poetry), Melpomene
(tragedy), Polyhymnia (sacred poetry), Terpsichore (dance), Thalia (comedy), and Urania (astronomy).

The Titans continue to influence modern culture and literature. The word "titan" is used to describe
someone of exceptional strength or ability. The elements titanium and titanate are named after them.
The moon of Saturn named Titan is the largest moon of Saturn and the second-largest natural satellite
in the Solar System. In popular culture, the Titans appear in various films, books, and video games,
often depicted as powerful beings of enormous size.

The influence of Titan mythology extends beyond just the name. Many modern stories draw parallels
between the Titans and their conflict with the Olympians, using it as a metaphor for generational
conflict or the struggle between old and new gods. The Titans represent the primal, chaotic forces
of nature that must be tamed and ordered by the newer, more civilized Olympian gods.

The legacy of the Titans endures in our language, our science, and our imagination. They remind us
of a time when the world was young, when gods walked the earth, and when humanity lived in harmony
with the divine. Though they were defeated by the Olympians, the Titans remain an important part
of Greek mythology and continue to captivate audiences around the world with their stories of
power, rebellion, and the eternal struggle between order and chaos.
"""

def test_context_length(target_tokens):
    """Test if model can retrieve needle at specific context length"""
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"Testing Context Length: ~{target_tokens} tokens")
    print(f"{'='*70}")

    try:
        wrapper = QwenWithTitans(model_name, device=device)

        # Disable MAC for pure baseline test
        wrapper.start_memory_injection_threshold = 999999999

        query = "What is the giant's favorite color?"

        # Build context by repeating the natural text
        context_text = NATURAL_CONTEXT
        repeats = max(1, int(target_tokens / 1500))  # ~1500 tokens per repeat
        full_context = (context_text * repeats)

        # Tokenize to check length
        tokens = wrapper.tokenizer(full_context, return_tensors="pt").to(device)
        actual_length = tokens['input_ids'].shape[1]
        print(f"Target: ~{target_tokens} tokens | Actual: {actual_length} tokens")

        # Feed the context
        print(f"Feeding {actual_length} tokens to model...")
        wrapper.model.eval()
        with torch.no_grad():
            wrapper.model(tokens['input_ids'])

        # Query
        print(f"\nQuery: {query}")
        response = wrapper.generate(query, max_new_tokens=50)

        # Check result
        success = "MAGENTA" in response.upper()
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n{status}")
        print(f"Response: {response.strip()}")
        print(f"Needle found: {success}")

        # Clean up
        del wrapper
        torch.cuda.empty_cache()

        return success, actual_length, response

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None, str(e)

if __name__ == "__main__":
    print("Baseline Context Length Test for Qwen3-4B-Instruct")
    print("Using natural Greek mythology context with needle in the middle")
    print("="*70)

    # Test progressively larger contexts
    test_lengths = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000]

    results = []
    for length in test_lengths:
        success, actual_length, response = test_context_length(length)
        results.append({
            'target': length,
            'actual': actual_length,
            'success': success,
            'response': response[:100] if response else None  # Truncate for display
        })
        print("\n" + "-"*70)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Baseline Context Limits")
    print(f"{'='*70}")
    print(f"{'Target':<10} {'Actual':<12} {'Result':<10}")
    print(f"{'-'*40}")

    for r in results:
        actual = str(r.get('actual'))[:10] if r.get('actual') else "ERROR"
        result = "✅ PASS" if r.get('success') else "❌ FAIL"
        print(f"{r['target']:<10} {actual:<12} {result:<10}")

    # Find the crossover point
    successful_lengths = [r['actual'] for r in results if r.get('success') and r.get('actual')]
    failed_lengths = [r['actual'] for r in results if not r.get('success') and r.get('actual')]

    print(f"\n{'='*70}")
    if successful_lengths:
        max_success = max(successful_lengths)
        print(f"✅ Maximum working context: {max_success} tokens")
    else:
        print(f"❌ No context lengths succeeded!")

    if failed_lengths:
        min_fail = min(failed_lengths)
        print(f"❌ Minimum failing context: {min_fail} tokens")

    if successful_lengths and failed_lengths:
        print(f"\n🎯 Estimated baseline limit: ~{max_success} tokens")
        print(f"   The model can NOT retrieve the needle beyond this point.")
