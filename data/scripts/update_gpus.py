"""
Script to regenerate gpus.json from verified internet sources as defined in OriginalPrompt.md.

Sources:
1. Specs: https://github.com/voidful/gpu-info-api (URL: https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json)
2. Benchmarks: https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference (README.md)
"""

import json
import requests
import re
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
GPUS_JSON_PATH = DATA_DIR / "gpus.json"

URL_GPU_INFO_API = "https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json"
URL_BENCHMARKS = "https://raw.githubusercontent.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference/main/README.md"

def fetch_json(url):
    print(f"Fetching {url}...")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def fetch_text(url):
    print(f"Fetching {url}...")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def parse_vram(val):
    """
    Parse Memory Size field which can be '24576', '12288 24576', etc.
    Returns float GB.
    """
    if not val: return 0.0
    s = str(val).strip()
    
    # Check if multiple values (e.g. "12288 24576"), take the max
    parts = s.split()
    max_mb = 0.0
    for p in parts:
        # cleanup
        clean = re.sub(r"[^\d\.]", "", p)
        if clean:
            try:
                v = float(clean)
                if v > max_mb: max_mb = v
            except:
                pass
                
    # Convert MB to GB
    return max_mb / 1024.0

def parse_bandwidth(val):
    """
    Parse Memory Bandwidth (GB/s).
    """
    if not val: return 0.0
    s = str(val).strip()
    # Handle "448.0 576.0" -> take max
    parts = s.split()
    max_bw = 0.0
    for p in parts:
        clean = re.sub(r"[^\d\.]", "", p)
        try:
           v = float(clean)
           if v > max_bw: max_bw = v
        except:
           pass
    return max_bw

def scrape_benchmarks():
    text = fetch_text(URL_BENCHMARKS)
    benchmarks = {}
    
    # Parse Markdown Table
    # Looking for lines like | 4090 24GB | 145.2 ... |
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "GPU" in line:
            continue
            
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2: continue
        
        name_part = parts[0] # e.g. "4090 24GB"
        
        # Try to extract 4-bit benchmark (usually first col after name or near it)
        # Based on README format: | GPU | 8B Q4_K_M | ...
        try:
            val = parts[1]
            if val.upper() in ["OOM", "-", ""]: continue
            dataset_tps = float(val)
            benchmarks[name_part] = dataset_tps
        except:
            continue
            
    return benchmarks

def get_generation(name):
    n = name.lower()
    if "4090" in n or "4080" in n or "4070" in n: return "Ada Lovelace"
    if "3090" in n or "3080" in n or "3070" in n or "a100" in n: return "Ampere"
    if "2080" in n or "2070" in n or "titan rtx" in n: return "Turing"
    if "1080" in n or "titan x" in n or "p40" in n: return "Pascal"
    if "rx 7" in n: return "RDNA 3"
    if "rx 6" in n: return "RDNA 2"
    if "m3" in n: return "Apple M3"
    if "m2" in n: return "Apple M2"
    if "m1" in n: return "Apple M1"
    if "h100" in n: return "Hopper"
    if "blackwell" in n or "b200" in n: return "Blackwell"
    return "Unknown"

def main():
    raw_data = fetch_json(URL_GPU_INFO_API)
    benchmarks = scrape_benchmarks()
    
    final_gpus = []
    
    print(f"Loaded {len(raw_data)} raw GPU entries. Filtering...")
    
    for key, info in raw_data.items():
        name = info.get("Model name", key) # Sometimes Model name is cleaner
        if not name: name = key
        
        # Filter for relevant cards
        # We want: RTX 20/30/40/50, GTX 16/10, RX 6000/7000, Mac, Data Center (A100, H100)
        n_up = name.upper()
        
        relevant = False
        if "RTX" in n_up: relevant = True
        elif "GTX 16" in n_up or "GTX 108" in n_up or "GTX 107" in n_up: relevant = True
        elif "RX 6" in n_up or "RX 7" in n_up or "RX 8" in n_up: relevant = True # Future proof RX 8?
        elif "M1" in n_up or "M2" in n_up or "M3" in n_up or "M4" in n_up: relevant = True
        elif "A100" in n_up or "H100" in n_up or "A10G" in n_up or "L40" in n_up: relevant = True
        elif "TITAN" in n_up: relevant = True
        
        if not relevant:
            if "4090" in n_up: print(f"DEBUG: 4090 filtered out (not relevant). Name: {name}")
            continue
        
        # Debug Keys for 4090
        if "4090" in n_up:
            print(f"DEBUG: Processing 4090. Name: {name}", flush=True)
            print(f"Keys: {list(info.keys())}", flush=True) 

        # Flexible Key Access
        def get_val(keys_list):
            for k in keys_list:
                if k in info: return info[k]
                # Try stripped/lower match
                for raw_k in info:
                    if raw_k.strip().lower() == k.lower(): return info[raw_k]
            return None

        # Parse data
        def get_val(keys_list):
            for k in keys_list:
                if k in info: return info[k]
                for raw_k in info:
                    if raw_k.strip().lower() == k.lower(): return info[raw_k]
            return None

        mem_raw = get_val(["Memory Size (MiB)", "Memory", "VRAM", "Memory size", "Memory config", "Memory Size (GiB)"])
        bw_raw = get_val(["Memory Bandwidth (GB/s)", "Bandwidth"])
        
        # Check if we got GiB
        vram_gb = 0.0
        used_key = ""
        # Find which key actually worked for debug/logic
        for k in ["Memory Size (MiB)", "Memory", "VRAM", "Memory size", "Memory config", "Memory Size (GiB)"]:
             if k in info: used_key = k; break
        
        if used_key == "Memory Size (GiB)":
            # Start with direct parse
             s = str(mem_raw).strip()
             clean = re.sub(r"[^\d\.]", "", s.split()[0]) # take first number
             try: vram_gb = float(clean)
             except: vram_gb = 0.0
        else:
             vram_gb = parse_vram(mem_raw)

        bw_gbps = parse_bandwidth(bw_raw)
        
        if vram_gb < 6: 
            continue # Skip low end
        
        # Vendor
        vendor = "NVIDIA"
        if "AMD" in n_up or "RADEON" in n_up or "RX" in n_up: vendor = "AMD"
        elif "INTEL" in n_up or "ARC" in n_up: vendor = "Intel"
        elif "APPLE" in n_up or "M1" in n_up or "M2" in n_up or "M3" in n_up or "M4" in n_up: vendor = "Apple"
        
        # DEBUG: Check if we are seeing non-NVIDIA at all
        if vendor != "NVIDIA":
             print(f"DEBUG: Found {vendor} GPU: {name}", flush=True)

        # Match benchmark
        
        # Match benchmark
        # Simple match: check if key parts of name exist in benchmark entry
        # e.g. "GeForce RTX 4090" -> match "4090"
        tps = None
        
        # Create search token
        token = name.lower().replace("geforce", "").replace("rtx", "").replace("radeon", "").replace("apple", "").strip().split()[0]
        # For macs: "m1 max" -> token "m1"? No, need smarter match for Macs.
        
        if vendor == "Apple":
            token = name.lower() # use full name for mac fuzzy match
            
        for b_name, b_val in benchmarks.items():
            if token in b_name.lower():
                # specific check for variants like Ti, Super
                if "ti" in name.lower() and "ti" not in b_name.lower(): continue
                if "super" in name.lower() and "super" not in b_name.lower(): continue
                tps = b_val
                break
        
        # Correct Brand Name
        # The API sometimes returns "NVIDIA GeForce RTX 4090" or just "GeForce RTX 4090"
        # We prefer clean names.
        clean_name = name
        if vendor == "NVIDIA" and not name.startswith("NVIDIA"):
            # clean_name = f"NVIDIA {name}" # actually UI handles vendor usually, but keeping name clean is good
            pass
            
        final_gpus.append({
            "name": clean_name,
            "vendor": vendor,
            "vram_gb": vram_gb,
            "memory_bandwidth_gbps": bw_gbps,
            "fp16_tflops": 0, # Not reliably in this dataset
            "generation": get_generation(name),
            "baseline_tps_8b_q4": tps
        })
        
    # Deduplication (some entries might be duplicates with different keys)
    # Use name + vram as unique key
    unique_map = {}
    for g in final_gpus:
        k = f"{g['name']}_{g['vram_gb']}"
        unique_map[k] = g
        
    sorted_gpus = sorted(unique_map.values(), key=lambda x: (x['vendor'], x['vram_gb']), reverse=True)
    
    data = {"gpus": sorted_gpus}
    with open(GPUS_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved {len(sorted_gpus)} GPUs to {GPUS_JSON_PATH}")

if __name__ == "__main__":
    main()
