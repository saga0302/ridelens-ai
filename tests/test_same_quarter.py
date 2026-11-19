import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from comparison.same_quarter import compare_summaries  # noqa: E402


def past_years(values):
    return pd.DataFrame({"m": values}, index=list(range(2019, 2019 + len(values))))


def test_value_inside_past_spread_is_not_flagged():
    past = past_years([100, 102, 98, 101, 99, 100, 103])
    report = compare_summaries({"m": 101}, past)
    assert not report.loc[0, "flagged"]


def test_value_far_outside_past_spread_is_flagged():
    past = past_years([100, 102, 98, 101, 99, 100, 103])
    report = compare_summaries({"m": 150}, past)
    assert report.loc[0, "flagged"]


def test_cutoff_is_largest_leave_one_year_out_gap():
    values = [100, 110, 90, 100, 100, 100, 100]
    past = past_years(values)
    report = compare_summaries({"m": 100}, past)
    expected = max(
        abs(v - pd.Series(values).drop(i).median()) / pd.Series(values).drop(i).median()
        for i, v in enumerate(values)
    )
    assert report.loc[0, "cutoff"] == pytest.approx(expected)


def test_too_few_past_years_raises():
    past = past_years([100, 101])
    with pytest.raises(ValueError):
        compare_summaries({"m": 100}, past)

def test_tiny_shares_do_not_produce_huge_cutoffs():
    past = past_years([0.0, 0.0, 0.001, 0.0, 0.0, 0.002, 0.0])
    report = compare_summaries({"m": 0.0}, past)
    assert report.loc[0, "cutoff"] < 1
    assert not report.loc[0, "flagged"]