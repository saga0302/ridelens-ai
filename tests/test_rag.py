import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag.build_index import build  # noqa: E402
from rag.retrieve import search  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def index():
    build()


def test_index_has_sections():
    assert build() > 10


def test_cutoff_question_finds_the_cutoff_explanation():
    results = search("why is the cutoff 2 percent?", n=3)
    assert any("percentile" in r["text"] or "2%" in r["text"] for r in results)


def test_results_say_where_they_came_from():
    for r in search("what is a flagged trip?", n=2):
        assert r["source"] in ("README.md", "docs/glossary.md")
        assert r["heading"]