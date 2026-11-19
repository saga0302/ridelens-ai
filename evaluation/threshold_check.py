"""Leave-one-quarter-out threshold check. Does not change any saved model.

For each season, every past same-season quarter is scored by a model that
never saw it. Those honest scores define what "normal" looks like. A trip is
flagged when its robust z-score is below a cutoff.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from training.train_season_matched import discover, features  # noqa: E402

FIRST_YEAR = 2019
TEST_QUARTERS = [(2026, 3), (2026, 2), (2026, 1), (2025, 4)]
PERCENTILES = [0.01, 0.02, 0.05, 0.10]


def fit(frames):
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    X = features(df)
    scaler = StandardScaler().fit(X)
    model = IsolationForest(n_estimators=200, max_samples=256,
                            random_state=42, n_jobs=-1)
    model.fit(scaler.transform(X))
    return model, scaler


def score(bundle, df):
    model, scaler = bundle
    return model.score_samples(scaler.transform(features(df)))


def robust_z(scores, ref):
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (scores - med) / mad


def main():
    data = discover()
    for ty, tq in TEST_QUARTERS:
        past = [(y, tq) for y in range(FIRST_YEAR, ty) if (y, tq) in data]

        # honest scores: each past quarter scored by a model trained without it
        held = {}
        for k in past:
            others = [data[j] for j in past if j != k]
            held[k] = score(fit(others), data[k])
        ref = np.concatenate(list(held.values()))

        # the test quarter, scored by a model trained on all past quarters
        test_scores = score(fit([data[k] for k in past]), data[(ty, tq)])

        print(f"\n{ty} Q{tq}  trained on {len(past)} same-season quarters")
        for p in PERCENTILES:
            cut = np.quantile(ref, p)
            z_cut = robust_z(np.array([cut]), ref)[0]
            past_rates = " ".join(
                f"{k[0]}-Q{k[1]}:{(s < cut).mean():.1%}"
                for k, s in held.items())
            test_rate = (test_scores < cut).mean()
            print(f"  lowest {p:.0%} (score cutoff {cut:.4f}, z {z_cut:.2f}): "
                  f"test {test_rate:.2%} | past {past_rates}")

if __name__ == "__main__":
    sys.exit(main())