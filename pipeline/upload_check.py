"""Check an uploaded trip file and score it without changing the training data.

Uses saved summaries, so it runs without the raw trip files:
  - Only 2026 Q3 and later are accepted.
  - The file must cover one complete quarter (trips on its last day).
  - If the file matches the saved checksum for that quarter, the saved report is shown.
  - Otherwise the file passes the quality gate and is scored with the saved model.
  - Nothing is saved or added to training.
"""
import hashlib
import json
import os
import sys
import tempfile

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from comparison.report import assemble_report, same_quarter_rows, score_trips, trip_examples  # noqa: E402
from comparison.same_quarter import FIRST_YEAR, compare_summaries, summarise  # noqa: E402
from pipeline.normalize import load_quarter  # noqa: E402
from pipeline.quality_gate import run_checks  # noqa: E402

FIRST_ACCEPTED = (2026, 3)
SUMMARIES = os.path.join(ROOT, "data", "summaries")
MIN_PAST = 3


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _saved_past(year, q):
    """Saved summaries of earlier same-quarter years: one row per year."""
    rows = {}
    for y in range(FIRST_YEAR, year):
        path = os.path.join(SUMMARIES, f"{y}-Q{q}.json")
        if os.path.exists(path):
            rows[y] = _load_json(path)["summary"]
    return pd.DataFrame(rows).T


def _saved_profile(year, q):
    path = os.path.join(SUMMARIES, "profiles", f"{year}-Q{q}.json")
    if not os.path.exists(path):
        return None
    p = _load_json(path)
    return (pd.Series(p["median"], dtype=float),
            pd.Series(p["iqr"], dtype=float),
            pd.Series(p["pass_shares"], dtype=float))


def _rejected(message):
    return {"status": "rejected", "message": message}


def _check(path):
    df, _ = load_quarter(path)
    start = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M")
    periods = start.dt.to_period("Q")
    main = periods.value_counts().idxmax()
    year, q = main.year, main.quarter

    if (year, q) < FIRST_ACCEPTED:
        return _rejected(f"Only 2026-Q3 and later can be uploaded. This file is {year}-Q{q}.")
    if periods.nunique() > 1:
        return _rejected("This file covers more than one quarter. Upload one quarter at a time.")

    stored_path = os.path.join(SUMMARIES, f"{year}-Q{q}.json")
    stored = _load_json(stored_path) if os.path.exists(stored_path) else None
    if stored and stored["sha256"] == _sha256(path):
        report_path = os.path.join(ROOT, "results", "reports", f"report_{year}Q{q}.json")
        if os.path.exists(report_path):
            report = _load_json(report_path)
            report["status"] = "stored"
            return report

    last_day = pd.Timestamp(year, 3 * q, 1) + pd.offsets.MonthEnd(0)
    if start.max().normalize() < last_day:
        return _rejected(f"{year}-Q{q} is not complete yet (last trip "
                         f"{start.max():%b %d}). Upload it again after {last_day:%b %d, %Y}.")

    checks = run_checks(df, year, q)
    failed = [(n, d) for n, ok, d in checks if not ok]
    if failed:
        return {"status": "gate_failed", "quarter": f"{year}-Q{q}",
                "failures": [{"check": n, "detail": d} for n, d in failed]}

    model_path = os.path.join(ROOT, "models", f"season_q{q}_for_{year}Q{q}.joblib")
    if not os.path.exists(model_path):
        return _rejected(f"No saved model for {year}-Q{q} yet. Run "
                         f"python -m training.save_latest_model {year} {q} first.")
    bundle = joblib.load(model_path)

    past = _saved_past(year, q)
    if len(past) < MIN_PAST:
        return _rejected(f"Not enough earlier {year - FIRST_YEAR}-year history for Q{q}.")
    compare_df = compare_summaries(summarise(df), past)

    gate = {"gate_passed": True, "gate_failures": []}
    index, scores, flagged = score_trips(df, bundle)
    examples = trip_examples(df, index, scores, flagged, _saved_profile(year, q))
    result = assemble_report(year, q, gate, len(index), int(flagged.sum()),
                             examples, same_quarter_rows(compare_df))
    result["status"] = "differs_from_stored" if stored else "new"
    return result


def check_upload(content: bytes):
    """content: the uploaded file's bytes. Returns a dict with a 'status' key."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        return _check(path)
    finally:
        os.remove(path)