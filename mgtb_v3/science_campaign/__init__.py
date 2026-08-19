"""Config-driven, resumable scientific ablation campaigns."""

from .config import load_campaign, resolve_variant

__all__ = ["load_campaign", "resolve_variant"]
