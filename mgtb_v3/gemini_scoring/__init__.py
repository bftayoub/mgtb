"""Blind, resumable Gemini scoring for the frozen Omni-MATH campaign."""

from .controls import deterministic_verdict
from .schema import validate_judge_response

__all__ = ["deterministic_verdict", "validate_judge_response"]
