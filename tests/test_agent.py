import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.graph import build_graph, route  # noqa: E402
from rag.build_index import build  # noqa: E402
from types import SimpleNamespace
from agent.graph import _text_of


@pytest.fixture(scope="module", autouse=True)
def index():
    build()


def test_number_question_routes_to_numbers():
    r = route({"question": "How many trips were flagged in 2026 Q3?"})
    assert r["need_numbers"] and not r["need_docs"]


def test_method_question_routes_to_docs():
    r = route({"question": "Why is the cutoff 2%?"})
    assert r["need_docs"] and not r["need_numbers"]


def test_grounded_answer_passes():
    def fake(system, user):
        return ("2026-Q3: 2,304 of 103,106 trips were flagged (2.23%), "
                "against an expected 2%. [2026-Q3 report]")
    out = build_graph(fake).invoke({"question": "How many trips were flagged in 2026 Q3?"})
    assert out["grounded"] is True


def test_invented_number_is_caught_and_replaced():
    def fake(system, user):
        return "The flagged share was 9.99%."
    out = build_graph(fake).invoke({"question": "How many trips were flagged in 2026 Q3?"})
    assert out["grounded"] is False
    assert "could not check" in out["answer"]
    assert "trip_id" not in out["answer"]


def test_reasoning_blocks_are_skipped():
    blocks = [SimpleNamespace(type="thinking", thinking="working it out"),
              SimpleNamespace(type="text", text="Answer [glossary: Cutoff].")]
    assert _text_of(blocks) == "Answer [glossary: Cutoff]."