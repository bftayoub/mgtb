from mgtb_v3.eval.math500 import (
    extract_model_answer,
    format_math500_prompt,
    normalize_math_answer,
    score_math500,
)


def test_format_math500_prompt_requests_marker():
    prompt = format_math500_prompt("Compute $2+2$.")
    assert "#### <answer>" in prompt
    assert "Compute $2+2$." in prompt


def test_extract_model_answer_prefers_marker_and_boxed_answer():
    assert extract_model_answer("Reasoning\n#### \\frac{14}{3}") == "\\frac{14}{3}"
    assert extract_model_answer("Thus $\\boxed{3\\sqrt{13}}$.") == "3\\sqrt{13}"


def test_normalize_math_answer_handles_common_latex_forms():
    assert normalize_math_answer("\\frac{14}{3}") == "14/3"
    assert normalize_math_answer("$\\boxed{3\\sqrt{13}}$") == "3sqrt(13)"
    assert normalize_math_answer("\\text{Evelyn}") == "evelyn"
    assert normalize_math_answer("90^\\circ") == "90"


def test_score_math500_exact_normalized_answer():
    score = score_math500("We compute the result.\n#### \\boxed{\\frac{14}{3}}", "\\frac{14}{3}")
    assert score["prediction_answer"] == "14/3"
    assert score["reference_answer"] == "14/3"
    assert score["correct"] == 1.0
    assert score["answer_extraction_ok"] is True


def test_score_math500_marks_missing_prediction():
    score = score_math500("", "42")
    assert score["prediction_answer"] is None
    assert score["reference_answer"] == "42"
    assert score["correct"] == 0.0
    assert score["answer_extraction_ok"] is False
