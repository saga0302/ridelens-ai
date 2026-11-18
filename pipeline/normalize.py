"""Map every era of LA Metro Bike Share trip files onto one canonical schema.

The files do not share one format across 2017-2026:
  - column names: 2017 Q1 uses station IDs; 2021 Q2 adds station-name columns
  - bike_type: missing before late 2018
  - timestamps: "M/D/YYYY H:MM" in some quarters, "YYYY-MM-DD HH:MM:SS" in others
  - pass types: "Flex Pass" (2017-2019), "Testing" (2019-2022 test records),
    empty values (2020+)

Everything that is changed is recorded in the returned report so it can be
shown in the dashboard instead of being silently fixed.
"""
import csv
import io

import pandas as pd

CANONICAL_COLUMNS = [
    "trip_id", "duration", "start_time", "end_time", "start_station",
    "start_lat", "start_lon", "end_station", "end_lat", "end_lon",
    "bike_id", "plan_duration", "trip_route_category", "passholder_type",
    "bike_type",
]
RENAME = {"start_station_id": "start_station", "end_station_id": "end_station"}
TIMESTAMP_OUT = "%m/%d/%Y %H:%M"
TEST_PASS_TYPE = "Testing"


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse both observed formats; return canonical strings (NaT -> None)."""
    raw = series.astype(str).str.strip()
    iso = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mdy = pd.to_datetime(raw, format="%m/%d/%Y %H:%M", errors="coerce")
    parsed = iso.fillna(mdy)
    return parsed.dt.strftime(TIMESTAMP_OUT)


def normalize_frame(df: pd.DataFrame):
    """Return (canonical_df, report). Input is a raw DataFrame from one file."""
    report = {"rows_in": int(len(df)), "renamed": {}, "dropped_columns": [],
              "added_columns": [], "testing_rows_dropped": 0}

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for old, new in RENAME.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
            report["renamed"][old] = new

    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra:
        df = df.drop(columns=extra)
        report["dropped_columns"] = extra

    if "bike_type" not in df.columns:
        df["bike_type"] = None
        report["added_columns"].append("bike_type (all missing in source)")

    if "passholder_type" in df.columns:
        is_test = df["passholder_type"].astype(str).str.strip() == TEST_PASS_TYPE
        report["testing_rows_dropped"] = int(is_test.sum())
        df = df.loc[~is_test]

    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = _parse_timestamps(df[col])

    df = df[CANONICAL_COLUMNS]
    report["rows_out"] = int(len(df))
    return df, report


def load_quarter(path: str):
    """Read one raw trip CSV (any era) and return (canonical_df, report)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    # numeric columns come back as strings; convert the ones the gate and model need
    out, report = normalize_frame(df)
    for col in ["duration", "start_lat", "start_lon", "end_lat", "end_lon"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    report["source"] = path
    return out, report
