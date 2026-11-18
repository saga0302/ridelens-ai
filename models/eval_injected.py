"""Injected-fault evaluation: Isolation Forest (season-matched) vs a rule baseline.

For each test quarter:
  - clean sample: 20,000 real trips (false-positive rate = share flagged)
  - fault samples: 2,000 different real trips per fault type, each corrupted in
    one specific way (recall = share flagged)

Caveat: these faults are chosen by us, so this measures how well each method
catches the faults we planted, not how often real anomalies are missed.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_season_matched import discover, features, haversine_km, FEATURES  # noqa: E402

TEST_QUARTERS = [(2026, 3), (2026, 2), (2026, 1), (2025, 4)]
N_CLEAN = 20_000
N_FAULT = 2_000
SEED = 42
MAX_SPEED_KMH = 60.0  # straight-line speed above this is impossible for a bike


def rule_flags(df: pd.DataFrame) -> np.ndarray:
    dur = df["duration"].astype(float)
    bad_duration = (dur < 1) | (dur > 1440) | dur.isna()
    missing = df[["start_lat", "start_lon", "end_lat", "end_lon"]].isna().any(axis=1)
    dist = haversine_km(df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"])
    speed = dist / (dur / 60.0)  # straight-line lower bound on true speed
    impossible = (speed > MAX_SPEED_KMH).fillna(False)
    return (bad_duration | missing | impossible).to_numpy()


def inject(df: pd.DataFrame, fault: str) -> pd.DataFrame:
    out = df.copy()
    if fault == "teleport_end_30km":
        out["end_lat"] = out["end_lat"] + 0.27
        out["end_lon"] = out["end_lon"] + 0.27
    elif fault == "impossible_speed":
        out["duration"] = 1.0
        out["end_lat"] = out["end_lat"] + 0.05
    elif fault == "missing_end_coords":
        out["end_lat"] = np.nan
        out["end_lon"] = np.nan
    elif fault == "duration_out_of_range":
        out["duration"] = 2000.0
    elif fault == "duration_zero":
        out["duration"] = 0.0
    else:
        raise ValueError(fault)
    return out


FAULTS = ["teleport_end_30km", "impossible_speed", "missing_end_coords",
          "duration_out_of_range", "duration_zero"]


def score_if(bundle, df):
    X = features(df)
    scores = bundle["model"].score_samples(bundle["scaler"].transform(X))
    return scores < bundle["threshold"]


def main():
    data = discover()
    results = []
    for ty, tq in TEST_QUARTERS:
        bundle = joblib.load(f"models/season_q{tq}_for_{ty}Q{tq}.joblib")
        test = data[(ty, tq)]
        rng = np.random.default_rng(SEED)

        clean = test.sample(n=min(N_CLEAN, len(test)), random_state=SEED)
        row = {"test": f"{ty}-Q{tq}", "clean_trips": len(clean)}
        row["fp_if"] = float(score_if(bundle, clean).mean())
        row["fp_rules"] = float(rule_flags(clean).mean())

        for fault in FAULTS:
            idx = rng.choice(len(test), size=min(N_FAULT, len(test)), replace=False)
            injected = inject(test.iloc[idx], fault)
            row[f"recall_if__{fault}"] = float(score_if(bundle, injected).mean())
            row[f"recall_rules__{fault}"] = float(rule_flags(injected).mean())
        results.append(row)

        print(f"\n{ty} Q{tq}  clean trips={len(clean):,}")
        print(f"  false-positive rate  IF={row['fp_if']:.2%}  rules={row['fp_rules']:.2%}")
        for fault in FAULTS:
            print(f"  recall  {fault:<24} IF={row[f'recall_if__{fault}']:.2%}"
                  f"  rules={row[f'recall_rules__{fault}']:.2%}")

    with open("models/injected_eval_results.json", "w") as f:
        json.dump({"max_speed_kmh": MAX_SPEED_KMH, "n_clean": N_CLEAN,
                   "n_fault_per_type": N_FAULT, "seed": SEED,
                   "faults": FAULTS, "results": results}, f, indent=2)
    print("\nSaved models/injected_eval_results.json")


if __name__ == "__main__":
    sys.exit(main())
