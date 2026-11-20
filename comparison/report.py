"""Assemble one quarter's report for the explanation step.

Alerts come only from the trip-level model. Same-quarter numbers are information:
they show where this quarter sits against past years, with no alert attached.
"""
import json
import os

import joblib
import numpy as np

from comparison.same_quarter import compare_quarter
from training.train_season_matched import FLAG_RATE, features
from comparison.reasons import profile, reasons

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_COLUMNS = ["start_time", "duration", "passholder_type"]
N_EXAMPLES = 10


def load_gate(year, q, manifest_path=os.path.join(ROOT, "data", "manifest.json")):
    """Data-quality result for one quarter, from the gate manifest."""
    with open(manifest_path) as f:
        for entry in json.load(f):
            if entry["year"] == year and entry["quarter"] == q:
                return {"gate_passed": entry["gate_passed"],
                        "gate_failures": entry["gate_failures"]}
    return {"gate_passed": None, "gate_failures": []}


def score_trips(df, bundle):
    """Index, Isolation Forest scores and flags for every trip in df."""
    X = features(df)
    scores = bundle["model"].score_samples(bundle["scaler"].transform(X[bundle["features"]]))
    return X.index, scores, scores < bundle["threshold"]


def trip_examples(df, index, scores, flagged, prof=None):
    """The most unusual flagged trips, for the explanation step."""
    flagged_idx = index[flagged]
    order = np.argsort(scores[flagged])[:N_EXAMPLES]
    chosen = flagged_idx[order]
    rows = df.loc[chosen, EXAMPLE_COLUMNS].astype(str).to_dict("records")
    if prof is not None:
        for row, idx in zip(rows, chosen):
            row["reasons"] = "; ".join(reasons(df, idx, prof))
    return rows


def same_quarter_rows(compare_df):
    """Same-quarter table as information: range check, and the loo cutoff marked as info only."""
    rows = []
    for r in compare_df.to_dict("records"):
        rows.append({
            "metric": r["metric"],
            "new": float(r["new"]),
            "past_median": float(r["past_median"]),
            "past_min": float(r["past_min"]),
            "past_max": float(r["past_max"]),
            "outside_past_range": bool(r["new"] < r["past_min"] or r["new"] > r["past_max"]),
            "beyond_loo_cutoff": bool(r["flagged"]),  # information only, not an alert
        })
    return rows


def assemble_report(year, q, gate, n_trips, n_flagged, examples, same_quarter):
    if n_trips == 0:
        raise ValueError("no trips in this quarter")
    return {
        "quarter": f"{year}-Q{q}",
        "data_quality": gate,
        "trip_level": {
            "trips": int(n_trips),
            "flagged": int(n_flagged),
            "flagged_share": float(n_flagged) / n_trips,
            "expected_share": FLAG_RATE,
            "examples": examples,
        },
        "same_quarter": same_quarter,
    }


def build_report(frames, year, q, models_dir=os.path.join(ROOT, "models")):
    """frames: {(year, quarter): normalised DataFrame}, including past same-quarter years."""
    df = frames[(year, q)]
    bundle = joblib.load(os.path.join(models_dir, f"season_q{q}_for_{year}Q{q}.joblib"))
    past = {k: v for k, v in frames.items() if k[1] == q and k[0] < year}
    prof = profile(past)
    index, scores, flagged = score_trips(df, bundle)
    examples = trip_examples(df, index, scores, flagged, prof)
    same = same_quarter_rows(compare_quarter(frames, year, q))
    return assemble_report(year, q, load_gate(year, q), len(index),
                           int(flagged.sum()), examples, same)