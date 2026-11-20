"""Check an uploaded trip file and score it without changing the training data.

Rules:
  - Only 2026 Q3 and later are accepted.
  - The file must cover one complete quarter (trips on its last day).
  - If the file matches our copy of that quarter, show the saved report.
  - If it differs from our copy, or is a new quarter, score it with the
    saved model for that quarter.
  - Nothing is added to training data or data/manifest.json.
"""
import glob
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
from comparison.run_same_quarter import load  # noqa: E402
from comparison.same_quarter import FIRST_YEAR, compare_quarter  # noqa: E402
from comparison.reasons import profile  # noqa: E402
from pipeline.normalize import load_quarter  # noqa: E402
from pipeline.quality_gate import run_checks  # noqa: E402

FIRST_ACCEPTED = (2026, 3)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rejected(message):
    return {"status": "rejected", "message": message}


def _check(path):
    df, _ = load_quarter(path)
    start = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M")
    periods = start.dt.to_period("Q")
    main = periods.value_counts().idxmax()
    year, q = main.year, main.quarter

    if (year, q) < FIRST_ACCEPTED:
        return _rejected(f"Only 2026-Q3 and later can be uploaded. This file is "
                         f"{year}-Q{q}.")
    if periods.nunique() > 1:
        return _rejected("This file covers more than one quarter. Upload one quarter at a time.")

    stored = glob.glob(os.path.join(ROOT, "data", "raw", f"{year}-Q{q}", "*.csv"))
    if stored and _sha256(stored[0]) == _sha256(path):
        report_path = os.path.join(ROOT, "results", "reports", f"report_{year}Q{q}.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
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

    frames = {}
    for y in range(FIRST_YEAR, year + 1):
        if glob.glob(os.path.join(ROOT, "data", "raw", f"{y}-Q{q}", "*.csv")):
            frames[(y, q)] = load(y, q) if y < year else df
    gate = {"gate_passed": True, "gate_failures": []}
    index, scores, flagged = score_trips(df, bundle)
    prof = profile({k: v for k, v in frames.items() if k[1] == q and k[0] < year})
    examples = trip_examples(df, index, scores, flagged, prof)
    same = same_quarter_rows(compare_quarter(frames, year, q))
    result = assemble_report(year, q, gate, len(index), int(flagged.sum()), examples, same)
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