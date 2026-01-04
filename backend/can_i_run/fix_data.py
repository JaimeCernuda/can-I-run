
import json
from pathlib import Path

# Resolve data directory relative to this script
SCRIPT_DIR = Path(__file__).parent
# Try to find data dir (handle both dev layout and installed package)
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
if not DATA_DIR.exists():
    DATA_DIR = SCRIPT_DIR.parent / "data" # Fallback
    
MODELS_JSON_PATH = DATA_DIR / "models.json"

def main():
    with open(MODELS_JSON_PATH, "r") as f:
        data = json.load(f)
    
    valid_models = []
    removed = []
    
    for m in data["models"]:
        name = m["name"]
        num_heads = m.get("num_heads", 0)
        hidden_dim = m.get("hidden_dim", 0)
        num_layers = m.get("num_layers")
        num_kv_heads = m.get("num_kv_heads")
        
        if (num_heads == 0 or hidden_dim == 0 or 
            num_layers is None or num_layers == 0 or
            num_kv_heads is None): # num_kv_heads can be 0? No.
            removed.append(f"{name} (heads={num_heads}, dim={hidden_dim}, layers={num_layers}, kv={num_kv_heads})")
        else:
            valid_models.append(m)
            
    if removed:
        print(f"Removing {len(removed)} invalid models:")
        for r in removed:
            print(f" - {r}")
            
        data["models"] = valid_models
        with open(MODELS_JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print("Updated models.json")
    else:
        print("No invalid models found.")

if __name__ == "__main__":
    main()
