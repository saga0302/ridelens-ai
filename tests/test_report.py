import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from comparison.report import assemble_report  # noqa: E402

GATE = {"gate_passed": True, "gate_failures": []}


def test_flagged_share_comes_from_trip_counts():
    r = assemble_report(2026, 3, GATE, n_trips=1000, n_flagged=25, examples=[], same_quarter=[])
    assert r["trip_level"]["flagged_share"] == 0.025
    assert r["quarter"] == "2026-Q3"


def test_gate_failures_are_data_quality_not_alerts():
    gate = {"gate_passed": False, "gate_failures": [{"check": "duration_in_range", "detail": "x"}]}
    r = assemble_report(2026, 3, gate, n_trips=1000, n_flagged=20, examples=[], same_quarter=[])
    assert r["data_quality"]["gate_passed"] is False
    assert "alerts" not in r


def test_same_quarter_flags_do_not_change_trip_level_count():
    rows = [{"metric": "m", "new": 5.0, "past_median": 1.0, "past_min": 0.9,
             "past_max": 1.1, "outside_past_range": True, "beyond_loo_cutoff": True}]
    r = assemble_report(2026, 3, GATE, n_trips=1000, n_flagged=20, examples=[], same_quarter=rows)
    assert r["trip_level"]["flagged"] == 20
    assert r["same_quarter"][0]["beyond_loo_cutoff"] is True