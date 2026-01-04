"""
KV Cache Calculation Functions

The KV (Key-Value) cache stores attention key and value vectors for all
previous tokens, enabling efficient autoregressive generation.

Formula:
    KV Cache (GB) = 2 × Layers × KV_Heads × Head_Dim × Context × Batch × 2 bytes
                    ─────────────────────────────────────────────────────────────
                                         1,073,741,824

Where:
    2           = Key AND Value caches
    Layers      = Number of transformer layers (e.g., 80 for Llama-70B)
    KV_Heads    = Number of key-value heads (may be < attention heads due to GQA)
    Head_Dim    = Hidden dimension ÷ Number of attention heads
    Context     = Sequence length (e.g., 8192 tokens)
    Batch       = Batch size (typically 1 for interactive use)
    2 bytes     = FP16 storage per value

Example: Llama-3.1-70B at 8K context
    = 2 × 80 layers × 8 KV heads × 128 head_dim × 8192 tokens × 1 × 2
    = 2.68 GB

Note on Grouped Query Attention (GQA):
    Many modern models use GQA where num_kv_heads < num_heads.
    For example, Llama-3.1-70B has 64 attention heads but only 8 KV heads,
    reducing KV cache size by 8x while maintaining performance.
"""

from .models import Model


# Common context length positions for the slider
CONTEXT_POSITIONS = [
    2048,     # 2K
    4096,     # 4K
    8192,     # 8K
    16384,    # 16K
    32768,    # 32K
    65536,    # 64K
    131072,   # 128K
    262144,   # 256K
    524288,   # 512K
    1048576,  # 1M
]


def calculate_head_dim(model: Model) -> int:
    """
    Calculate the dimension of each attention head.

    Head dimension = Hidden dimension ÷ Number of attention heads

    Args:
        model: Model specification

    Returns:
        Head dimension size
    """
    return model.hidden_dim // model.num_heads


def calculate_kv_cache(
    model: Model,
    context_length: int,
    batch_size: int = 1,
    bytes_per_value: int = 2  # FP16
) -> float:
    """
    Calculate the KV cache size for a given context length.

    The KV cache grows linearly with context length. This is often the
    limiting factor for long-context inference.

    Args:
        model: Model specification
        context_length: Number of tokens in the context window
        batch_size: Number of sequences being processed (default 1)
        bytes_per_value: Bytes per cached value (2 for FP16, 1 for INT8 KV cache)

    Returns:
        KV cache size in GB
    """
    head_dim = calculate_head_dim(model)

    # Calculate bytes needed for KV cache
    bytes_required = (
        2 *                      # K and V
        model.num_layers *
        model.num_kv_heads *
        head_dim *
        context_length *
        batch_size *
        bytes_per_value
    )

    return bytes_required / (1024 ** 3)


def get_available_context_positions(
    model: Model,
    all_positions: list[int] = None
) -> list[dict]:
    """
    Get available context positions for the slider, considering model limits.

    Returns positions marked as:
    - available: Context is within max_context_length
    - warning: Context is between effective and max (may have degraded performance)
    - unavailable: Context exceeds max_context_length

    Args:
        model: Model specification (or None for all positions available)
        all_positions: List of context positions to check (defaults to CONTEXT_POSITIONS)

    Returns:
        List of dicts with position, available, and warning flags
    """
    if all_positions is None:
        all_positions = CONTEXT_POSITIONS

    results = []
    for pos in all_positions:
        if model is None:
            results.append({
                "position": pos,
                "available": True,
                "warning": False,
                "label": format_context_length(pos)
            })
        else:
            available = pos <= model.max_context_length
            warning = available and pos > model.effective_context_length
            results.append({
                "position": pos,
                "available": available,
                "warning": warning,
                "label": format_context_length(pos)
            })

    return results


def format_context_length(tokens: int) -> str:
    """
    Format a context length for display.

    Args:
        tokens: Number of tokens

    Returns:
        Human-readable string (e.g., "8K", "128K", "1M")
    """
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"
    else:
        return str(tokens)


def estimate_kv_cache_warning(
    kv_cache_gb: float,
    total_vram_gb: float,
    threshold_percent: float = 50.0
) -> dict:
    """
    Check if KV cache is taking up too much VRAM.

    A warning is issued when KV cache exceeds the threshold percentage
    of total VRAM, as this leaves less room for model weights.

    Args:
        kv_cache_gb: KV cache size in GB
        total_vram_gb: Total GPU VRAM in GB
        threshold_percent: Warning threshold as percentage

    Returns:
        Dict with warning status and details
    """
    percent = (kv_cache_gb / total_vram_gb) * 100 if total_vram_gb > 0 else 0

    return {
        "kv_cache_gb": round(kv_cache_gb, 2),
        "percent_of_vram": round(percent, 1),
        "exceeds_threshold": percent > threshold_percent,
        "threshold_percent": threshold_percent,
        "message": (
            f"KV cache uses {percent:.1f}% of VRAM (>{threshold_percent}%)"
            if percent > threshold_percent else None
        )
    }
