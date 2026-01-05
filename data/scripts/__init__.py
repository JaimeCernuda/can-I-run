"""
Data update scripts for can-I-run project.

Usage:
    uv run python -m data.scripts [--only gpus|models|quants] [--dry-run] [-v]
"""

from .update_gpus import main as update_gpus
from .update_models import main as update_models
from .update_quants import main as update_quants

__all__ = ["update_gpus", "update_models", "update_quants"]
