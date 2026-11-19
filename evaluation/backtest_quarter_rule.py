"""Backtest the quarter-level alert rule on past quarters.

Rule under test: alert a quarter if its flagged share is above the highest
out-of-sample flagged share among earlier same-quarter years.

Every share here comes from a model trained only on years before that quarter,
so nothing leaks forward. No past quarter has a labelled anomaly, so a fired
alert counts as a possible false alarm.

Usage (from the project root): python evaluation/backtest_quarter_rule.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # discover() reads data/raw relative to the working directory

from training.train_season_matched import FIRST_YEAR, discover, flagged_share, train_for  # noqa: E402

MIN_TRAIN_QUARTERS = 2  # a quarter is scored only if this many earlier same-season quarters exist
MIN_REFS = 2            # a quarter is compared only if this many earlier shares exist


def main():
    data = discover()

    shares = {}
    for (y, q) in sorted(data):
        n_train = sum(1 for yy in range(FIRST_YEAR, y) if (yy, q) in data)
        if n_train < MIN_TRAIN_QUARTERS:
            continue
        bundle = train_for(y, q, data)
        share, _ = flagged_share(bundle, data[(y, q)])
        shares[(y, q)] = share
        print(f"scored {y}-Q{q}: {share:.2%} (trained on {n_train} earlier quarters)", flush=True)

    rows = []
    for (y, q) in sorted(shares):
        refs = [shares[(yy, q)] for yy in range(FIRST_YEAR, y) if (yy, q) in shares]
        if len(refs) < MIN_REFS:
            continue
        ref_max = max(refs)
        rows.append({
            "quarter": f"{y}-Q{q}",
            "share": round(shares[(y, q)], 5),
            "earlier_max": round(ref_max, 5),
            "earlier_quarters": len(refs),
            "fired": bool(shares[(y, q)] > ref_max),
        })

    print(f"\n{'quarter':<9}{'share':>8}{'earlier max':>13}{'refs':>6}  fired")
    for r in rows:
        print(f"{r['quarter']:<9}{r['share']:>8.2%}{r['earlier_max']:>13.2%}"
              f"{r['earlier_quarters']:>6}  {'YES' if r['fired'] else 'no'}")

    fired = sum(r["fired"] for r in rows)
    # If every quarter were alike, the newest of n+1 quarters is the highest with chance 1/(n+1).
    expected = sum(1 / (r["earlier_quarters"] + 1) for r in rows)
    print(f"\nfired {fired} of {len(rows)} quarters "
          f"(expected by chance if all quarters are alike: {expected:.1f})")

    out_dir = os.path.join(ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "quarter_rule_backtest.json")
    with open(out, "w") as f:
        json.dump({"rule": "fire if share > max of earlier same-quarter out-of-sample shares",
                   "min_train_quarters": MIN_TRAIN_QUARTERS, "min_refs": MIN_REFS,
                   "rows": rows, "fired": fired, "expected_by_chance": expected},
                  f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()