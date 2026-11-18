"""Train the trip-level Isolation Forest on past quarters; test on a holdout.

Train: 2025 Q3, 2025 Q4, 2026 Q2. Holdout: 2026 Q3 (never seen in training).
The score threshold is fixed from training data only, then applied unchanged
to the holdout, so any drift shows up as a change in the flagged share.
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RAW = "data/raw"
TRAIN = {"2025-Q3": "2025-Q3/metro-trips-2025-q3.csv",
         "2025-Q4": "2025-Q4/metro-trips-2025-q4.csv",
         "2026-Q2": "2026-Q2/metro-trips-2026-q2.csv"}
HOLDOUT = {"2026-Q3": "2026-Q3-copy/metro-trips-2026-q3.csv"}
PASS_TYPES = ["Monthly Pass", "Walk-up", "One Day Pass", "Annual Pass"]
FEATURES = ["duration_min", "distance_km", "start_hour", "is_weekend",
            "is_electric", "is_round_trip"] + [f"pass_{p}" for p in PASS_TYPES]
FLAG_RATE = 0.05  # 5% of training trips define the threshold


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def features(df):
    start = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M")
    X = pd.DataFrame(index=df.index)
    X["duration_min"] = df["duration"].astype(float)
    dist = haversine_km(df["start_lat"], df["start_lon"], df["end_lat"], df["end_lon"])
    X["distance_km"] = np.nan_to_num(dist, nan=0.0)  # missing coords -> 0 km
    X["start_hour"] = start.dt.hour.astype(float)
    X["is_weekend"] = (start.dt.dayofweek >= 5).astype(float)
    X["is_electric"] = (df["bike_type"] == "electric").astype(float)
    X["is_round_trip"] = (df["trip_route_category"] == "Round Trip").astype(float)
    for p in PASS_TYPES:
        X[f"pass_{p}"] = (df["passholder_type"] == p).astype(float)
    return X[FEATURES]


def load(paths):
    frames = []
    for q, rel in paths.items():
        df = pd.read_csv(f"{RAW}/{rel}")
        df["quarter"] = q
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    train_df = load(TRAIN)
    hold_df = load(HOLDOUT)
    X_train = features(train_df)
    X_hold = features(hold_df)

    scaler = StandardScaler().fit(X_train)
    model = IsolationForest(n_estimators=200, max_samples=256,
                            random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X_train))

    # Higher score = more normal in sklearn; flag the lowest 5% of training scores.
    train_scores = model.score_samples(scaler.transform(X_train))
    threshold = float(np.quantile(train_scores, FLAG_RATE))

    def report(df, X):
        scores = model.score_samples(scaler.transform(X))
        flagged = scores < threshold
        out = df.assign(score=scores, flagged=flagged)
        return out.groupby("quarter").agg(
            trips=("flagged", "size"),
            flagged=("flagged", "sum"),
            flagged_share=("flagged", "mean"),
            median_score=("score", "median"),
        ).round(4)

    print("TRAINING QUARTERS (threshold set here)")
    print(report(train_df, X_train).to_string())
    print("\nHOLDOUT QUARTER (threshold unchanged)")
    print(report(hold_df, X_hold).to_string())

    version = "trip_iforest_v1"
    joblib.dump({"model": model, "scaler": scaler, "threshold": threshold,
                 "features": FEATURES}, f"models/{version}.joblib")
    meta = {"version": version,
            "trained_on": list(TRAIN), "holdout": list(HOLDOUT),
            "features": FEATURES, "flag_rate": FLAG_RATE, "threshold": threshold,
            "n_estimators": 200, "max_samples": 256, "random_state": 42,
            "sklearn": sklearn.__version__,
            "saved_at": datetime.now(timezone.utc).isoformat()}
    with open(f"models/{version}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved models/{version}.joblib and .json")


if __name__ == "__main__":
    sys.exit(main())
