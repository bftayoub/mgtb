from mgtb_v3.eval.gsm8k import extract_model_answer, extract_reference_answer, score_gsm8k


def test_reference_answer_extraction_normalizes_numbers():
    assert extract_reference_answer("Reasoning here. #### 42") == "42"
    assert extract_reference_answer("Reasoning here. #### -3") == "-3"
    assert extract_reference_answer("Reasoning here. #### 1,234") == "1234"
    assert extract_reference_answer("Reasoning here. #### 1 234.50") == "1234.5"


def test_model_answer_extraction_prefers_final_marker():
    text = "We considered 12 and 19 first.\n#### -7.0\n"
    assert extract_model_answer(text) == "-7"


def test_model_answer_extraction_uses_last_number_without_marker():
    text = "First compute 12. Then the final answer is 42."
    assert extract_model_answer(text) == "42"


def test_model_answer_extraction_can_fail_cleanly():
    assert extract_model_answer("There is no numeric final answer here.") is None


def test_score_gsm8k_exact_numeric_match():
    score = score_gsm8k("The result is #### 1,234.0", "Detailed solution #### 1234")
    assert score["prediction_answer"] == "1234"
    assert score["reference_answer"] == "1234"
    assert score["correct"] == 1.0
    assert score["answer_extraction_ok"] is True


def test_score_gsm8k_marks_missing_prediction():
    score = score_gsm8k("I cannot solve this.", "Detailed solution #### 5")
    assert score["prediction_answer"] is None
    assert score["reference_answer"] == "5"
    assert score["correct"] == 0.0
    assert score["answer_extraction_ok"] is False
