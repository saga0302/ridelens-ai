import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.eval_agent import score_answer  # noqa: E402


def test_good_answer_passes():
    case = {"must": ["2.23%"], "cite": ["Cutoff"]}
    assert score_answer("The share was 2.23% [glossary: Cutoff].", True, case) == []


def test_missing_number_fails():
    assert score_answer("About 2%.", True, {"must": ["2.23%"]}) == ["missing '2.23%'"]


def test_ungrounded_answer_fails():
    failures = score_answer("2.23%", False, {"must": ["2.23%"]})
    assert any("not grounded" in f for f in failures)


def test_nine_digit_trip_id_is_caught():
    case = {"forbidden_regex": [r"\b\d{9}\b"]}
    assert score_answer("Trip 566420607 was flagged.", True, case)


def test_unsafe_label_is_caught():
    case = {"forbidden": ["is unsafe"]}
    assert score_answer("The area is unsafe.", True, case)


def test_refusal_that_mentions_unsafe_is_not_flagged():
    case = {"forbidden": ["is dangerous", "area is unsafe", "areas are unsafe"],
            "forbidden_regex": [r"(?<!whether )\b\w+ is unsafe"]}
    answer = "I can't say whether Downtown is unsafe. The data cannot show that."
    assert score_answer(answer, True, case) == []


def test_plain_unsafe_claim_is_flagged():
    case = {"forbidden_regex": [r"(?<!whether )\b\w+ is unsafe"]}
    assert score_answer("Downtown is unsafe.", True, case)