"""Quarter-level check: does the hour-of-day pattern look like past same-season quarters?

Per trip checks cannot see a whole-quarter timing bug (each trip looks normal),
so this compares the 24-hour trip profile of a quarter with the same season in
earlier years.

Distance = total variation between hour profiles (0 = identical, 1 = disjoint).
Threshold = largest leave-one-out distance among the training quarters, so a
quarter is flagged only if it is less like the others than any past quarter was.

Tests on the 2026 Q3 holdout (and the other test quarters):
  - clean quarter (should NOT flag)
  - start times shifted -6 h for 1%, 5%, 20%, 100% of trips
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_season_matched import discover  # noqa: E402

TEST_QUARTERS = [(2026, 3), (2026, 2), (2026, 1), (2025, 4)]
FIRST_YEAR = 2019
SHIFT_FRACTIONS = [0.01, 0.05, 0.20, 1.00]
SEED = 42


def hour_profile(df: pd.DataFrame) -> np.ndarray:
    hours = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M").dt.hour
    counts = np.bincount(hours, minlength=24).astype(float)
    return counts / counts.sum()


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def shift_fraction(df, frac, rng):
    out = df.copy()
    n = int(round(frac * len(out)))
    idx = rng.choice(len(out), size=n, replace=False)
    shifted = pd.to_datetime(out["start_time"].iloc[idx], format="%m/%d/%Y %H:%M") - pd.Timedelta(hours=6)
    out.iloc[idx, out.columns.get_loc("start_time")] = shifted.dt.strftime("%m/%d/%Y %H:%M").values
    return out


def main():
    data = discover()
    rng = np.random.default_rng(SEED)
    results = []
    for ty, tq in TEST_QUARTERS:
        train_keys = [(y, tq) for y in range(FIRST_YEAR, ty) if (y, tq) in data]
        profiles = {k: hour_profile(data[k]) for k in train_keys}

        # leave-one-out distances within the training quarters
        loo = {}
        for k in train_keys:
            others = np.mean([profiles[j] for j in train_keys if j != k], axis=0)
            loo[f"{k[0]}-Q{k[1]}"] = tvd(profiles[k], others)
        threshold = max(loo.values())
        reference = np.mean(list(profiles.values()), axis=0)

        test = data[(ty, tq)]
        row = {"test": f"{ty}-Q{tq}", "n_train": len(train_keys),
               "threshold": round(threshold, 5),
               "loo_distances": {k: round(v, 5) for k, v in loo.items()},
               "clean_distance": round(tvd(hour_profile(test), reference), 5),
               "clean_flagged": bool(tvd(hour_profile(test), reference) > threshold)}
        row["shifted"] = {}
        for frac in SHIFT_FRACTIONS:
            d = tvd(hour_profile(shift_fraction(test, frac, rng)), reference)
            row["shifted"][f"{int(frac*100)}%"] = {
                "distance": round(d, 5), "flagged": bool(d > threshold)}
        results.append(row)

        print(f"\n{ty} Q{tq}  train={len(train_keys)} same-season quarters"
              f"  threshold={threshold:.4f}")
        print(f"  clean distance {row['clean_distance']:.4f}  flagged={row['clean_flagged']}")
        for k, v in row["shifted"].items():
            print(f"  {k:>4} of trips shifted -6h: distance {v['distance']:.4f}"
                  f"  flagged={v['flagged']}")

    with open("results/quarter_hour_check_results.json", "w") as f:
        json.dump({"first_year": FIRST_YEAR, "shift_fractions": SHIFT_FRACTIONS,
                   "seed": SEED, "results": results}, f, indent=2)
    print("\nSaved results/quarter_hour_check_results.json")


if __name__ == "__main__":
    sys.exit(main())
