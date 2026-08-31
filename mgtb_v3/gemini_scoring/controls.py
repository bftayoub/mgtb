from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from mgtb_v3.eval.math500 import normalize_math_answer


_COMPLEX_MARKERS = re.compile(
    r"(?:\\(?:equiv|pmod|mod|in|cup|cap|setminus|forall|exists|mapsto)|"
    r"[\[\]{}(),;]|(?:^|\W)(?:iff|or|and)(?:$|\W)|[a-zA-Z])"
)
_PLAIN_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_PLAIN_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_LATEX_FRACTION = re.compile(r"^\\(?:frac|dfrac|tfrac)\{([+-]?\d+)\}\{([+-]?\d+)\}$")


def _strip_scalar(text: str) -> str:
    value = str(text).strip().rstrip(".").strip()
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left):-len(right)].strip()
            break
    return value.replace("−", "-").replace("\\,", "").strip()


def simple_number(text: str | None) -> Fraction | None:
    """Parse only unambiguous scalar integers, decimals and numeric fractions."""
    if text is None:
        return None
    value = _strip_scalar(text)
    match = _PLAIN_FRACTION.fullmatch(value) or _LATEX_FRACTION.fullmatch(value)
    if match:
        denominator = int(match.group(2))
        return None if denominator == 0 else Fraction(int(match.group(1)), denominator)
    if not _PLAIN_NUMBER.fullmatch(value):
        return None
    try:
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def is_complex_answer(text: str | None) -> bool:
    return bool(text and _COMPLEX_MARKERS.search(_strip_scalar(text)))


def deterministic_verdict(candidate: str | None, reference: str | None) -> dict[str, Any]:
    """Apply conservative pre-API rules; ``verdict=None`` means send to Gemini."""
    if candidate is None or not str(candidate).strip():
        return {"verdict": "FALSE", "rule": "empty_or_invalid_extraction", "certain": True}
    if reference is None or not str(reference).strip():
        return {"verdict": "FALSE", "rule": "empty_or_invalid_reference", "certain": True}

    normalized_candidate = normalize_math_answer(candidate)
    normalized_reference = normalize_math_answer(reference)
    if normalized_candidate is None:
        return {"verdict": "FALSE", "rule": "empty_or_invalid_extraction", "certain": True}
    if normalized_reference is None:
        return {"verdict": "FALSE", "rule": "empty_or_invalid_reference", "certain": True}

    # Exact equality is safe even for symbolic answers: no equivalence inference is made.
    if normalized_candidate == normalized_reference:
        return {
            "verdict": "TRUE", "rule": "certain_normalized_equality", "certain": True,
            "normalized_candidate": normalized_candidate,
            "normalized_reference": normalized_reference,
        }

    candidate_number = simple_number(candidate)
    reference_number = simple_number(reference)
    if candidate_number is not None and reference_number is not None:
        return {
            "verdict": "FALSE", "rule": "different_simple_numbers", "certain": True,
            "normalized_candidate": str(candidate_number),
            "normalized_reference": str(reference_number),
        }

    # Never infer inequality for structured mathematical objects.
    return {
        "verdict": None, "rule": "requires_mathematical_judge", "certain": False,
        "normalized_candidate": normalized_candidate,
        "normalized_reference": normalized_reference,
        "complex": is_complex_answer(candidate) or is_complex_answer(reference),
    }
