import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.quarter_tools import (EXAMPLE_FIELDS, MAX_EXAMPLES, get_backtest,  # noqa: E402
                                 get_fault_comparison, get_flagged_examples,
                                 get_gate_result, get_report, get_trip_summary,
                                 list_quarters, _parse_quarter)


@pytest.mark.parametrize("bad", ["2026Q3", "2026-Q5", "26-Q3", "", None])
def test_bad_quarter_is_rejected(bad):
    with pytest.raises(ValueError):
        _parse_quarter(bad)


def test_list_quarters_covers_2019_to_2026_q3():
    quarters = list_quarters()
    assert quarters[0]["quarter"] == "2019-Q1"
    assert quarters[-1]["quarter"] == "2026-Q3"
    assert len(quarters) == 31


def test_gate_result_for_2026_q3_passed():
    assert get_gate_result("2026-Q3")["gate_passed"] is True


def test_missing_report_returns_a_clear_error():
    r = get_report("2019-Q1")
    assert "error" in r and "comparison.run_report 2019 1" in r["error"]


def test_trip_summary_matches_saved_report():
    s = get_trip_summary("2026-Q3")
    assert s["expected_share"] == 0.02
    assert 0 < s["flagged_share"] < 0.05


def test_examples_are_capped_and_have_no_ids():
    ex = get_flagged_examples("2026-Q3", n=500)["examples"]
    assert len(ex) <= MAX_EXAMPLES
    for e in ex:
        assert set(e) <= set(EXAMPLE_FIELDS)
        assert "trip_id" not in e


def test_fault_comparison_has_all_eight_faults():
    fc = get_fault_comparison("2026-Q3")
    assert len(fc["recall_by_fault"]) == 8
    assert "isolation_forest" in fc["clean_false_positive_rate"]


def test_backtest_summary():
    b = get_backtest()
    assert b["quarters_tested"] == 15
    assert b["fired"] == 2