"""
Script to fetch REAL model data from Unsloth's HuggingFace repository.
This script OVERWRITES models.json with fresh data directly from the API.

Usage: python update_models.py
"""

import json
import requests
import re
from pathlib import Path
from typing import Dict, List, Any

# Setup paths relative to this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
MODELS_JSON_PATH = DATA_DIR / "models.json"

UNSLOTH_API_URL = "https://huggingface.co/api/models?author=unsloth&sort=downloads&direction=-1&limit=100"

def fetch_unsloth_models() -> List[Dict]:
    print("Fetching models from Unsloth HF...")
    all_models = []
    # Start with a limit of 100, but use pagination to get everything
    url = UNSLOTH_API_URL
    
    while url:
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                break
                
            all_models.extend(data)
            print(f"Fetched {len(data)} models. Total so far: {len(all_models)}")
            
            # Helper to check for next link
            url = None
            if "Link" in resp.headers:
                links = resp.headers["Link"].split(",")
                for link in links:
                    parts = link.split(";")
                    if len(parts) > 1 and 'rel="next"' in parts[1]:
                        url = parts[0].strip("<> ")
                        break
        except Exception as e:
            print(f"Error fetching page: {e}")
            break
            
    return all_models

def get_readme(model_id: str) -> str:
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        print(f"Error fetching README for {model_id}: {e}")
    return ""

def extract_benchmarks(readme_text: str) -> Dict[str, float]:
    benchmarks = {"mmlu": 0.0, "humaneval": 0.0, "gsm8k": 0.0}
    if not readme_text:
        return benchmarks
        
    # Regex patterns (flexible to catch "MMLU = 85.0", "MMLU: 85%", etc)
    patterns = {
        "mmlu": r"(?:MMLU|Massive Multitask)[^\d\n]*?(\d+(?:\.\d+)?)",
        "humaneval": r"(?:HumanEval|Human\s*Eval)[^\d\n]*?(\d+(?:\.\d+)?)",
        "gsm8k": r"(?:GSM8K|Grade\s*School\s*Math)[^\d\n]*?(\d+(?:\.\d+)?)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, readme_text, re.IGNORECASE)
        if match:
             try:
                 val = float(match.group(1))
                 if val > 100: val = 100.0 # Sanity check
                 benchmarks[key] = val
             except:
                 pass
                 
    return benchmarks

def get_model_config(model_id: str) -> Dict:
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching config for {model_id}: {e}")
    return None

def detect_domains_and_capabilities(config: Dict, model_id: str) -> tuple[List[str], List[str]]:
    domains = ["general"]
    capabilities = []
    
    archs = config.get("architectures", [])
    arch_str = str(archs).lower()
    model_type = config.get("model_type", "").lower()
    
    # Vision Detection
    if "vision" in model_type or "vl" in model_type or "image" in model_type:
         domains.append("vision")
         capabilities.append("vision")
    elif any(x in arch_str for x in ["llava", "qwen2vl", "idefics", "pixtral", "vision"]):
         domains.append("vision")
         capabilities.append("vision")
    
    # Code Detection (heuristic based on name/config)
    if "coder" in model_id.lower() or "code" in model_id.lower():
        domains.append("code")
    
    # MoE Detection (capabilities)
    if config.get("num_local_experts", 0) > 1:
        # MoE is handled as a separate flag, but is a capability too?
        pass

    return list(set(domains)), list(set(capabilities))

def parse_config_to_model_entry(name: str, config: Dict) -> Dict:
    # 0. Flatten Vision Configs (Llama 3.2 Vision, etc store params in text_config)
    eff_config = config
    if "text_config" in config:
        # Merge text_config into root, prioritizing text_config
        # This ensures we get the LLM part's dimensions, not the Vision part's (if any conflict)
        eff_config = {**config, **config["text_config"]}

    # 1. Parameter Extraction
    hidden_size = eff_config.get("hidden_size", eff_config.get("d_model", 0))
    num_layers = eff_config.get("num_hidden_layers", eff_config.get("n_layer", 0))
    num_heads = eff_config.get("num_attention_heads", eff_config.get("n_head", 0))
    # Handle explicit null in config
    num_kv_heads = eff_config.get("num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    vocab_size = eff_config.get("vocab_size", 32000)
    max_position_embeddings = eff_config.get("max_position_embeddings", 2048)
    
    # 2. MoE Logic
    num_experts = eff_config.get("num_local_experts", 0)
    active_experts = eff_config.get("num_experts_per_tok", 0)
    is_moe = num_experts > 1
    
    # 3. Total Parameter Estimation (Billion)
    # Prefer explicit sizing from name if available (more reliable for users)
    total_params_b = 0.0
    size_match = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
    if size_match:
        total_params_b = float(size_match.group(1))
    
    # Fallback to calculation if name parsing fails AND we have config data
    if total_params_b == 0.0 and hidden_size and num_layers:
        # Rough calc: 12 * layers * h^2
        raw_params = (12 * num_layers * (hidden_size ** 2)) + (vocab_size * hidden_size)
        if is_moe:
            # MoE adjustment (crude)
            raw_params = raw_params * num_experts * 0.8 # sparse assumption
        total_params_b = round(raw_params / 1e9, 2)

    # 4. Active Parameter Estimation
    active_params_b = total_params_b
    if is_moe and num_experts > 0 and active_experts > 0:
        # Scale down based on active experts
        # This is a heuristic.
        active_params_b = total_params_b * (active_experts / num_experts)
        active_params_b = round(active_params_b, 2)

    # 5. Domains & Capabilities
    domains, capabilities = detect_domains_and_capabilities(config, name)

    # 6. Construct Entry
    return {
        "name": name.split("/")[-1].replace("-bnb-4bit", "").replace("-GGUF", "").replace("-unsloth", ""), # Clean name
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "is_moe": is_moe,
        "num_experts": num_experts if is_moe else None,
        "num_active_experts": active_experts if is_moe else None,
        "hidden_dim": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "max_context_length": max_position_embeddings,
        "effective_context_length": max_position_embeddings,
        "domains": domains,
        "capabilities": capabilities,
        "benchmarks": { # Placeholder for real data
             "mmlu": 0,
             "humaneval": 0,
             "gsm8k": 0
        },
        "notes": "Auto-imported from Unsloth"
    }

def main():
    print("--- STARTING MODEL REGENERATION ---")
    print("WARNING: This will overwrite models.json with fresh data.")
    
    unsloth_models = fetch_unsloth_models()
    
    valid_models = []
    seen_names = set()
    
    for m in unsloth_models:
        model_id = m["modelId"]
        
        # Filter unwanted formats (GGUF, BNB specific entries if we want base models?)
        # User wants "actual real data". Unsloth repos ARE finetunes/quants usually.
        # We want the "main" entry.
        # Let's keep specific interesting ones but maybe filter out excessive variants if needed.
        # For now, get ALL valid configs.
        
        if "-GGUF" in model_id or ".gguf" in model_id:
             # GGUF repos usually don't have config.json matching the base model architecture 1:1 in the root sometimes?
             # Or they do. Let's try to get config.
             pass

        print(f"Processing: {model_id}")
        config = get_model_config(model_id)
        
        if not config:
            print(f"  -> No config found, skipping.")
            continue
            
        name = model_id.split("/")[-1]
        
        # Parse
        entry = parse_config_to_model_entry(name, config)
        
        # Validate critical fields for calc
        if entry["hidden_dim"] == 0 or entry["num_heads"] == 0 or entry["num_layers"] == 0:
            print(f"  -> Missing critical dimensions, skipping.")
            continue
            
        if entry["name"] in seen_names:
            print(f"  -> Duplicate name, skipping.")
            continue

        # Fetch Benchmarks from README
        readme = get_readme(model_id)
        entry["benchmarks"] = extract_benchmarks(readme)
        if entry["benchmarks"]["mmlu"] > 0:
             print(f"  -> Found Benchmarks: MMLU {entry['benchmarks']['mmlu']}")
            
        valid_models.append(entry)
        seen_names.add(entry["name"])
        print(f"  -> Added {entry['name']} ({entry['total_params_b']}B)")

    # Prepare final JSON structure
    final_data = {
        "models": valid_models
    }
    
    # Save
    with open(MODELS_JSON_PATH, "w") as f:
        json.dump(final_data, f, indent=2)
        
    print(f"--- SUCCESS ---")
    print(f"Wrote {len(valid_models)} models to {MODELS_JSON_PATH}")

if __name__ == "__main__":
    main()
