"""Fault-injection tests for the quarterly quality gate.

Each test breaks a clean copy of Q3 2026 in one specific way and checks
that the matching hard check fails, and that no unrelated check fails.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quality_gate import EXPECTED_COLUMNS, run_checks  # noqa: E402

CLEAN_PATH = "/home/claude/metro-trips-2026-q3.csv"


@pytest.fixture(scope="module")
def clean():
    return pd.read_csv(CLEAN_PATH)


def results_by_name(df):
    return {name: ok for name, ok, _ in run_checks(df, 2026, 3)}


def assert_only_fails(df, expected_failure):
    results = results_by_name(df)
    failed = {name for name, ok in results.items() if not ok}
    assert failed == {expected_failure}, f"expected only {expected_failure} to fail, got {failed}"


def test_clean_file_passes_everything(clean):
    results = results_by_name(clean)
    assert all(results.values()), [k for k, v in results.items() if not v]


def test_missing_column_fails_schema_without_crashing(clean):
    df = clean.drop(columns=["duration"])
    results = results_by_name(df)  # must not raise
    assert results["schema_matches"] is False
    # Column-dependent checks also fail, since there is nothing to validate.
    assert results["duration_in_range"] is False


def test_reordered_columns_fail_schema(clean):
    df = clean[list(reversed(EXPECTED_COLUMNS))]
    assert_only_fails(df, "schema_matches")


def test_unparseable_timestamp_fails(clean):
    df = clean.copy()
    df.loc[0, "start_time"] = "not a date"
    assert_only_fails(df, "timestamps_parse")


def test_duplicate_trip_id_fails(clean):
    df = pd.concat([clean, clean.head(5)], ignore_index=True)
    assert_only_fails(df, "no_duplicate_trip_ids")


@pytest.mark.parametrize("bad_duration", [0, 1441])
def test_duration_outside_publisher_rules_fails(clean, bad_duration):
    df = clean.copy()
    df.loc[10, "duration"] = bad_duration
    assert_only_fails(df, "duration_in_range")


def test_unknown_pass_type_fails(clean):
    df = clean.copy()
    df.loc[10, "passholder_type"] = "Student Pass"
    assert_only_fails(df, "pass_types_known")


def test_truncated_quarter_fails_coverage(clean):
    ts = pd.to_datetime(clean["start_time"])
    df = clean[ts < pd.Timestamp("2026-08-15")].copy()
    assert_only_fails(df, "quarter_fully_covered")


def test_missing_end_coordinates_above_limit_fails(clean):
    df = clean.copy()
    n = int(len(df) * 0.10)
    df.loc[df.index[:n], "end_lat"] = None
    results = results_by_name(df)
    assert not results["null_rate_end_lat"]


def test_small_null_rate_is_only_soft(clean):
    # Current file has 3.3% missing end coordinates, under the 5% limit.
    results = results_by_name(clean)
    assert results["null_rate_end_lat"]
