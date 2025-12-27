import torch
from qwen_mac import QwenWithTitans
import sys

def interactive_session():
    print("--- Qwen 3 + Titans MAC Interactive Session ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8" 
    print(f"Loading Model: {model_name}...")
    
    try:
        wrapper = QwenWithTitans(model_name, device=device)
        print("Model Loaded Successfully!")
        
        # Allow user to override threshold for testing
        print("\nDefault Memory Injection Threshold is 32,000 tokens.")
        thresh_input = input("Press ENTER to keep default, or type a number (e.g. 50) to test memory injection sooner: ").strip()
        if thresh_input.isdigit():
            wrapper.start_memory_injection_threshold = int(thresh_input)
            print(f"Threshold set to {wrapper.start_memory_injection_threshold} tokens.")
        else:
            print("Using default threshold: 32,000 tokens.")

        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                if not user_input.strip():
                    continue

                print("Generating...", end="\r")
                
                # QwenWithTitans.generate handles the chat template internally now
                response_text = wrapper.generate(user_input, max_new_tokens=64)
                
                print(f"\nModel:\n{response_text}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
                
    except Exception as e:
        print(f"\nFailed to load model or run generation: {e}")

if __name__ == "__main__":
    interactive_session()
