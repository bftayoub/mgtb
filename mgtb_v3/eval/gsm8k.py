from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


ANSWER_MARKER = "####"
DEFAULT_PROMPT_STYLE = "gsm8k_cot"
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:[ ,]\d{3})+)|\d+)(?:\.\d+)?(?!\w)")


def load_gsm8k_items(
    dataset_name: str = "gsm8k",
    dataset_config: str | None = "main",
    split: str = "test",
    limit: int | None = 100,
    seed: int | None = 0,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "GSM8K loading requires the optional eval dependency. "
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
        question = str(row["question"])
        answer = str(row["answer"])
        reference_answer = extract_reference_answer(answer)
        items.append(
            {
                "id": row.get("id", f"{dataset_name}-{split}-{index:06d}"),
                "dataset": "gsm8k",
                "split": split,
                "question": question,
                "answer": answer,
                "reference_answer": reference_answer,
                "prompt": format_gsm8k_prompt(question, prompt_style),
            }
        )
    return items


def format_gsm8k_prompt(question: str, prompt_style: str = DEFAULT_PROMPT_STYLE) -> str:
    if prompt_style != DEFAULT_PROMPT_STYLE:
        raise ValueError(f"unknown GSM8K prompt_style {prompt_style!r}; expected {DEFAULT_PROMPT_STYLE!r}")
    return (
        "Solve the following grade-school math problem step by step. "
        f"Finish with a line in the form {ANSWER_MARKER} <answer>.\n\n"
        f"Question: {question.strip()}\n\n"
        "Solution:"
    )


def extract_reference_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = str(answer)
    if ANSWER_MARKER in text:
        numbers = _extract_numbers(text.rsplit(ANSWER_MARKER, 1)[-1])
        return numbers[0] if numbers else None
    numbers = _extract_numbers(text)
    return numbers[-1] if numbers else None


def extract_model_answer(text: str | None) -> str | None:
    if text is None:
        return None
    output = str(text)
    if ANSWER_MARKER in output:
        numbers = _extract_numbers(output.rsplit(ANSWER_MARKER, 1)[-1])
        if numbers:
            return numbers[0]
    numbers = _extract_numbers(output)
    return numbers[-1] if numbers else None


def score_gsm8k(text: str | None, reference_answer: str | None) -> dict[str, Any]:
    reference = extract_reference_answer(reference_answer)
    prediction = extract_model_answer(text)
    correct = False
    if reference is not None and prediction is not None:
        correct = _to_decimal(reference) == _to_decimal(prediction)
    return {
        "reference_answer": reference,
        "prediction_answer": prediction,
        "correct": 1.0 if correct else 0.0,
        "answer_extraction_ok": prediction is not None,
        "reference_extraction_ok": reference is not None,
    }


def _extract_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    for match in NUMBER_RE.finditer(text):
        canonical = _canonical_number(match.group(0))
        if canonical is not None:
            numbers.append(canonical)
    return numbers


def _canonical_number(raw: str) -> str | None:
    compact = raw.strip().replace(",", "").replace(" ", "")
    if compact.startswith("+"):
        compact = compact[1:]
    try:
        return _decimal_to_string(Decimal(compact))
    except InvalidOperation:
        return None


def _to_decimal(value: str) -> Decimal:
    return Decimal(value.replace(",", "").replace(" ", ""))


def _decimal_to_string(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")
