"""Quarterly quality gate for LA Metro Bike Share trip files.

Each check returns (name, passed, detail). No model is involved: every
decision here is plain code so it can be tested and explained.
"""
import sys
from datetime import date

import pandas as pd

EXPECTED_COLUMNS = [
    "trip_id", "duration", "start_time", "end_time", "start_station",
    "start_lat", "start_lon", "end_station", "end_lat", "end_lon",
    "bike_id", "plan_duration", "trip_route_category", "passholder_type",
    "bike_type",
]
# "Flex Pass" is a real product in 2017-2019 files; "Testing" rows are dropped
# by normalize.py before the gate runs, so they are not valid here.
VALID_PASS_TYPES = {"Monthly Pass", "Walk-up", "One Day Pass", "Annual Pass", "Flex Pass"}
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"  # e.g. "7/1/2026 0:05"
MAX_NULL_RATE = 0.05
MIN_DURATION, MAX_DURATION = 1, 1440


def quarter_bounds(year: int, q: int):
    start = date(year, 3 * (q - 1) + 1, 1)
    end = date(year + (q == 4), (3 * q) % 12 + 1, 1) - pd.Timedelta(days=1)
    return pd.Timestamp(start), pd.Timestamp(end)


def _missing(df, cols):
    return [c for c in cols if c not in df.columns]


def run_checks(df: pd.DataFrame, year: int, q: int):
    results = []

    cols_ok = list(df.columns) == EXPECTED_COLUMNS
    missing = _missing(df, EXPECTED_COLUMNS)
    results.append(("schema_matches", cols_ok,
                    "all 15 expected columns in order" if cols_ok
                    else f"missing {missing}, found {list(df.columns)}"))

    # Checks that need specific columns report a failure instead of crashing.
    ts_missing = _missing(df, ["start_time", "end_time"])
    if ts_missing:
        results.append(("timestamps_parse", False, f"missing columns {ts_missing}"))
        results.append(("quarter_fully_covered", False, "no timestamps to check"))
    else:
        start = pd.to_datetime(df["start_time"], format=TIMESTAMP_FORMAT, errors="coerce")
        end = pd.to_datetime(df["end_time"], format=TIMESTAMP_FORMAT, errors="coerce")
        bad_ts = int(start.isna().sum() + end.isna().sum())
        results.append(("timestamps_parse", bad_ts == 0, f"{bad_ts} unparseable"))

        qs, qe = quarter_bounds(year, q)
        if start.notna().any():
            first, last = start.min(), start.max()
            covered = first.normalize() <= qs + pd.Timedelta(days=1) and \
                last.normalize() >= qe - pd.Timedelta(days=7)
            coverage_days = (last.normalize() - first.normalize()).days + 1
            expected_days = (qe - qs).days + 1
            results.append(("quarter_fully_covered", bool(covered),
                            f"{first.date()} to {last.date()} "
                            f"({coverage_days} of {expected_days} days)"))
        else:
            results.append(("quarter_fully_covered", False, "no parseable timestamps"))

    if "trip_id" in df.columns:
        dup = int(df["trip_id"].duplicated().sum())
        results.append(("no_duplicate_trip_ids", dup == 0, f"{dup} duplicates"))
    else:
        results.append(("no_duplicate_trip_ids", False, "missing column trip_id"))

    if "duration" in df.columns:
        bad_dur = int((~df["duration"].between(MIN_DURATION, MAX_DURATION)).sum())
        results.append(("duration_in_range", bad_dur == 0,
                        f"{bad_dur} outside {MIN_DURATION}-{MAX_DURATION} min"))
    else:
        results.append(("duration_in_range", False, "missing column duration"))

    if "passholder_type" in df.columns:
        bad_pass = int((~df["passholder_type"].dropna().isin(VALID_PASS_TYPES)).sum())
        results.append(("pass_types_known", bad_pass == 0,
                        f"{bad_pass} unknown pass types"))
    else:
        results.append(("pass_types_known", False, "missing column passholder_type"))

    for col in ["end_lat", "end_lon", "start_lat", "start_lon", "passholder_type"]:
        if col not in df.columns:
            results.append((f"null_rate_{col}", False, f"missing column {col}"))
            continue
        rate = float(df[col].isna().mean())
        results.append((f"null_rate_{col}", rate <= MAX_NULL_RATE,
                        f"{rate:.1%} missing (limit {MAX_NULL_RATE:.0%})"))

    return results


if __name__ == "__main__":
    path, year, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    df = pd.read_csv(path)
    results = run_checks(df, year, q)
    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)
