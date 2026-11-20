"""Save the model for a quarter that has no data yet (e.g. 2026 Q4).

It is trained on the same quarter from earlier years only, the same way the
test models are, and saved as models/season_q{q}_for_{year}Q{q}.joblib.

Usage (from the project root):  python -m training.save_latest_model 2026 4
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_season_matched import FEATURES, discover, train_for  # noqa: E402


def main(year, q):
    data = discover()
    bundle = train_for(year, q, data)
    out = os.path.join("models", f"season_q{q}_for_{year}Q{q}.joblib")
    joblib.dump({"model": bundle["model"], "scaler": bundle["scaler"],
                 "threshold": bundle["threshold"], "features": FEATURES,
                 "train_keys": bundle["train_keys"]}, out)
    print(f"Saved {out}")
    print(f"trained on {len(bundle['train_keys'])} quarters: "
          f"{bundle['train_keys'][0]} .. {bundle['train_keys'][-1]}")
    print(f"cutoff score {bundle['threshold']:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))