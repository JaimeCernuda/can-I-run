"""
Manual model-to-leaderboard name mappings.

These mappings handle cases where automatic matching fails due to naming
conventions between Unsloth model names and leaderboard entries.

Format: {unsloth_model_name: leaderboard_model_id}
"""

# Mapping from Unsloth/common model names to Open LLM Leaderboard model IDs
MANUAL_MODEL_MAPPINGS: dict[str, str] = {
    # Llama 3.x series
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama-3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Meta-Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    # Qwen series
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-14B-Instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "QwQ-32B-Preview": "Qwen/QwQ-32B-Preview",
    # Mistral series
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    # Gemma series
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    # Phi series
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Phi-3-small-8k-instruct": "microsoft/Phi-3-small-8k-instruct",
    "Phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    # DeepSeek series
    "DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "DeepSeek-Coder-V2-Lite-Instruct": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    # Yi series
    "Yi-1.5-6B-Chat": "01-ai/Yi-1.5-6B-Chat",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
    "Yi-1.5-34B-Chat": "01-ai/Yi-1.5-34B-Chat",
    # Command R series
    "c4ai-command-r-v01": "CohereForAI/c4ai-command-r-v01",
    "c4ai-command-r-plus": "CohereForAI/c4ai-command-r-plus",
}


def get_manual_mappings() -> dict[str, str]:
    """Get the manual model mappings dictionary."""
    return MANUAL_MODEL_MAPPINGS.copy()


def lookup_manual_mapping(model_name: str) -> str | None:
    """
    Look up a model name in manual mappings.

    Args:
        model_name: Model name to look up

    Returns:
        Leaderboard model ID if found, None otherwise
    """
    # Try exact match first
    if model_name in MANUAL_MODEL_MAPPINGS:
        return MANUAL_MODEL_MAPPINGS[model_name]

    # Try without common suffixes
    cleaned = model_name
    for suffix in ["-bnb-4bit", "-GGUF", "-unsloth", "-bnb-8bit"]:
        cleaned = cleaned.replace(suffix, "")

    if cleaned in MANUAL_MODEL_MAPPINGS:
        return MANUAL_MODEL_MAPPINGS[cleaned]

    return None
