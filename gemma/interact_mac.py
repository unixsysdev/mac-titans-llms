import torch
from gemma_mac import GemmaWithTitans
import sys

def interactive_session():
    print("--- Gemma + Titans MAC Interactive Session ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")
    
    model_name = "google/gemma-3n-E4B"
    print(f"Loading Model: {model_name}...")
    
    try:
        wrapper = GemmaWithTitans(model_name, device=device)
        print("Model Loaded Successfully!")
        
        # Allow user to override threshold for testing
        print("\nDefault Memory Injection Threshold is 32,000 tokens (Standard Titans behavior).")
        thresh_input = input("Press ENTER to keep 32k, or type a number (e.g. 50) to test memory injection sooner: ").strip()
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
                # Generate
                # Use Few-Shot Prompting to force the Base Model to act like a Chatbot
                # We give it examples of how to behave.
                preamble = (
                    "This is a conversation between a curious User and a helpful AI Assistant.\n\n"
                    "User: Hi\n"
                    "AI: Hello! How can I help you today?\n"
                    "User: What is the capital of France?\n"
                    "AI: The capital of France is Paris.\n"
                )
                prompt = f"{preamble}User: {user_input}\nAI:"
                
                full_response = wrapper.generate(prompt, max_new_tokens=64)
                
                # Clean up: Stripe the preamble and prompt
                response_text = full_response[len(prompt):]
                
                # Stop at the next "User:" to prevent it from ignoring the stop token and simulating the user
                if "User:" in response_text:
                    response_text = response_text.split("User:")[0]
                
                print(f"\nModel:\n{response_text.strip()}\n")
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
                
    except Exception as e:
        print(f"\nFailed to load model or run generation: {e}")
        print("Tip: Ensure the model path 'gemma3n:e4b' is correct and accessible to transformers.")

if __name__ == "__main__":
    interactive_session()
