from transformers import AutoModelForCausalLM, AutoConfig
import torch

def inspect():
    model_name = "google/gemma-3n-E4B"
    print(f"Loading {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nModel Type:", type(model))
    print("\n--- Model Content ---")
    print(model)
    
    print("\n--- Inspecting 'model' attribute ---")
    if hasattr(model, 'model'):
        inner = model.model
        print("Inner Model Type:", type(inner))
        print("Inner Model Attributes:", dir(inner))
        
        # Check for layers
        if hasattr(inner, 'layers'):
            print("inner.layers FOUND")
        else:
            print("inner.layers NOT found")
    else:
        print("No 'model' attribute found on CausalLM.")

if __name__ == "__main__":
    inspect()
