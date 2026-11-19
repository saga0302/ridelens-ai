"""Season-matched Isolation Forest, following the test plan in the design doc.

For each test quarter T, the model is trained only on the SAME quarter-of-year
from earlier years (back to 2017), then scores T. The score threshold is the
5th percentile of the training scores, so it is fixed before T is seen.

Test plan (latest first):
  2026 Q3  <- train 2017 Q3 .. 2025 Q3
  2026 Q2  <- train 2017 Q2 .. 2025 Q2
  2026 Q1  <- train 2017 Q1 .. 2025 Q1
  2025 Q4  <- train 2017 Q4 .. 2024 Q4   (2026 Q4 not yet published)

Sensitivity: the same models retrained without the 2017-2018 quarters, which
came from a smaller system with different columns.

Limitation: `is_electric` is dropped. bike_type is missing before late 2018, so
it cannot be used consistently across all training years.
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.normalize import load_quarter  # noqa: E402

PASS_TYPES = ["Monthly Pass", "Walk-up", "One Day Pass", "Annual Pass", "Flex Pass"]
FEATURES = ["duration_min", "distance_km", "start_hour", "is_weekend",
            "is_round_trip"] + [f"pass_{p}" for p in PASS_TYPES]
FLAG_RATE = 0.02
TEST_QUARTERS = [(2026, 3), (2026, 2), (2026, 1), (2025, 4)]
FIRST_YEAR = 2019  # 2017-2018 dropped: smaller early system, different flag rates
PRE_2019_CUTOFF = 2019  # sensitivity run drops training quarters before this year


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def features(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M")
    X = pd.DataFrame(index=df.index)
    X["duration_min"] = df["duration"].astype(float)  # source is already minutes (1-1440)
    dist = haversine_km(df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"])
    X["distance_km"] = np.nan_to_num(dist, nan=0.0)  # missing coords -> 0 km
    X["start_hour"] = start.dt.hour.astype(float)
    X["is_weekend"] = (start.dt.dayofweek >= 5).astype(float)
    X["is_round_trip"] = (df["trip_route_category"] == "Round Trip").astype(float)
    for p in PASS_TYPES:
        X[f"pass_{p}"] = (df["passholder_type"] == p).astype(float)
    return X[FEATURES]


def discover():
    """Map (year, q) -> normalized DataFrame, loading each raw file once."""
    cache = {}
    for d in sorted(glob.glob("data/raw/*-Q*")):
        m = re.match(r"data/raw/(\d{4})-Q(\d)$", d)
        if not m:
            continue
        csvs = glob.glob(f"{d}/*.csv")
        df, _ = load_quarter(csvs[0])
        cache[(int(m.group(1)), int(m.group(2)))] = df
    return cache


def fit_model(frames):
    df = pd.concat(frames, ignore_index=True)
    X = features(df)
    scaler = StandardScaler().fit(X)
    model = IsolationForest(n_estimators=200, max_samples=256,
                            random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X))
    return model, scaler


def held_out_threshold(data, keys):
    """Score each training quarter with a model that never saw it.
    The cutoff is the FLAG_RATE quantile of those out-of-sample scores."""
    pooled = []
    for k in keys:
        model, scaler = fit_model([data[j] for j in keys if j != k])
        pooled.append(model.score_samples(scaler.transform(features(data[k]))))
    return float(np.quantile(np.concatenate(pooled), FLAG_RATE))


def train_for(test_year, q, data):
    keys = [(y, q) for y in range(FIRST_YEAR, test_year) if (y, q) in data]
    model, scaler = fit_model([data[k] for k in keys])
    threshold = held_out_threshold(data, keys)
    return {"model": model, "scaler": scaler, "threshold": threshold,
            "train_keys": [f"{y}-Q{qq}" for y, qq in keys]}


def flagged_share(bundle, df):
    X = features(df)
    scores = bundle["model"].score_samples(bundle["scaler"].transform(X))
    return float((scores < bundle["threshold"]).mean()), int(len(df))


def main():
    data = discover()
    results = []
    for (ty, tq) in TEST_QUARTERS:
        test_df = data[(ty, tq)]
        bundle = train_for(ty, tq, data)
        share, n = flagged_share(bundle, test_df)

        # in-sample rates for the training quarters, for reference only
        past = {}
        for k in bundle["train_keys"]:
            y, qq = int(k[:4]), int(k[-1])
            past[k], _ = flagged_share(bundle, data[(y, qq)])

        row = {"test": f"{ty}-Q{tq}", "test_trips": n,
               "train_quarters": bundle["train_keys"],
               "flagged_share_test": round(share, 4),
               "train_quarter_flagged_share_in_sample":
                   {k: round(v, 4) for k, v in past.items()},
               "threshold": bundle["threshold"]}
        results.append(row)
        tag = f"season_q{tq}_for_{ty}Q{tq}"
        joblib.dump({"model": bundle["model"], "scaler": bundle["scaler"],
                     "threshold": bundle["threshold"], "features": FEATURES,
                     "train_keys": bundle["train_keys"]},
                    f"models/{tag}.joblib")

        print(f"\nTEST {ty} Q{tq}  trained on {len(bundle['train_keys'])} same-season quarters"
              f" ({bundle['train_keys'][0]} .. {bundle['train_keys'][-1]})")
        print(f"  cutoff score {bundle['threshold']:.4f} (2nd percentile of held-out scores)")
        print(f"  flagged share (test)  {share:.2%}  of {n:,} trips")

    meta = {"trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn": sklearn.__version__, "features": FEATURES,
            "flag_rate": FLAG_RATE, "n_estimators": 200, "max_samples": 256,
            "random_state": 42, "results": results}
    with open("results/season_matched_results.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("\nSaved results/season_matched_results.json")


if __name__ == "__main__":
    sys.exit(main())

