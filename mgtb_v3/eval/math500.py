from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


ANSWER_MARKER = "####"
DEFAULT_DATASET_NAME = "HuggingFaceH4/MATH-500"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_PROMPT_STYLE = "math500_cot"

BOXED_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
TEXT_RE = re.compile(r"\\(?:text|mathrm)\s*\{([^{}]*)\}")


def load_math500_items(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    split: str = "test",
    limit: int | None = 500,
    seed: int | None = 0,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "MATH-500 loading requires the optional eval dependency. "
            'Install it with: pip install -e ".[eval]"'
        ) from exc

    load_args = [dataset_name]
    if dataset_config:
        load_args.append(dataset_config)
    dataset = load_dataset(*load_args, split=split)
    if seed is not None:
        dataset = dataset.shuffle(seed=int(seed))
    if limit is not None:
        n = min(max(int(limit), 0), len(dataset))
        dataset = dataset.select(range(n))

    items: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        problem = str(row["problem"])
        answer = str(row["answer"])
        items.append(
            {
                "id": row.get("unique_id", f"math500-{split}-{index:06d}"),
                "dataset": "math500",
                "split": split,
                "question": problem,
                "answer": answer,
                "reference_answer": answer,
                "subject": row.get("subject"),
                "level": row.get("level"),
                "prompt": format_math500_prompt(problem, prompt_style),
            }
        )
    return items


def format_math500_prompt(problem: str, prompt_style: str = DEFAULT_PROMPT_STYLE) -> str:
    if prompt_style != DEFAULT_PROMPT_STYLE:
        raise ValueError(f"unknown MATH-500 prompt_style {prompt_style!r}; expected {DEFAULT_PROMPT_STYLE!r}")
    return (
        "Solve the following math problem step by step. "
        f"Finish with a line in the form {ANSWER_MARKER} <answer>.\n\n"
        f"Problem: {problem.strip()}\n\n"
        "Solution:"
    )


def extract_model_answer(text: str | None) -> str | None:
    if text is None:
        return None
    output = str(text).strip()
    if not output:
        return None
    if ANSWER_MARKER in output:
        return _first_nonempty_line(output.rsplit(ANSWER_MARKER, 1)[-1])

    boxed = _last_boxed_content(output)
    if boxed:
        return boxed

    for marker in ["final answer is", "answer is", "therefore"]:
        lower = output.lower()
        if marker in lower:
            return _first_nonempty_line(output[lower.rfind(marker) + len(marker) :])
    return _first_nonempty_line(output.splitlines()[-1])


def score_math500(text: str | None, reference_answer: str | None) -> dict[str, Any]:
    reference = normalize_math_answer(reference_answer)
    prediction_raw = extract_model_answer(text)
    prediction = normalize_math_answer(prediction_raw)
    correct = reference is not None and prediction is not None and reference == prediction
    return {
        "reference_answer": reference,
        "prediction_answer": prediction,
        "correct": 1.0 if correct else 0.0,
        "answer_extraction_ok": prediction is not None,
        "reference_extraction_ok": reference is not None,
    }


def normalize_math_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    text = _strip_math_delimiters(text)
    boxed = _last_boxed_content(text)
    if boxed:
        text = boxed
    text = text.strip().rstrip(".")
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    text = TEXT_RE.sub(r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    previous = None
    while previous != text:
        previous = text
        text = FRAC_RE.sub(r"\1/\2", text)
        text = SQRT_RE.sub(r"sqrt(\1)", text)
        text = re.sub(r"\{([^{}]*)\}", r"\1", text)
    text = text.replace("^\\circ", "").replace("\\circ", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("−", "-")
    text = re.sub(r"\s+", "", text)
    text = text.strip("$").strip()
    if not text:
        return None
    numeric = _normalize_plain_decimal(text)
    return numeric if numeric is not None else text.lower()


def _first_nonempty_line(text: str) -> str | None:
    for line in str(text).splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def _strip_math_delimiters(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped[1:-1].strip()
    if stripped.startswith("\\(") and stripped.endswith("\\)"):
        return stripped[2:-2].strip()
    if stripped.startswith("\\[") and stripped.endswith("\\]"):
        return stripped[2:-2].strip()
    return stripped


def _last_boxed_content(text: str) -> str | None:
    matches = list(BOXED_RE.finditer(text))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        for pos in range(start, len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    content = text[start:pos].strip()
                    return content or None
    return None


def _normalize_plain_decimal(text: str) -> str | None:
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")
