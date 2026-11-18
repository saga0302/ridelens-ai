"""Normalize every raw quarter, run the quality gate, write data/manifest.json."""
import glob
import json
import os
import re
import sys

from normalize import load_quarter
from quality_gate import run_checks

manifest = []
for d in sorted(glob.glob("data/raw/*-Q*")):
    m = re.match(r"data/raw/(\d{4})-Q(\d)$", d)
    if not m:
        continue
    year, q = int(m.group(1)), int(m.group(2))
    csvs = [p for p in glob.glob(f"{d}/*.csv")]
    if not csvs:
        print("NO CSV", d); continue
    df, rep = load_quarter(csvs[0])
    checks = run_checks(df, year, q)
    failed = [(n, det) for n, ok, det in checks if not ok]
    rep.update({"year": year, "quarter": q,
                "gate_passed": not failed,
                "gate_failures": [{"check": n, "detail": det} for n, det in failed]})
    manifest.append(rep)
    status = "PASS" if not failed else "FAIL"
    print(f"{year} Q{q}  {status}  rows {rep['rows_in']:>7} -> {rep['rows_out']:>7}"
          f"  testing_dropped={rep['testing_rows_dropped']}"
          + (f"  renamed={list(rep['renamed'])}" if rep['renamed'] else "")
          + (f"  dropped_cols={len(rep['dropped_columns'])}" if rep['dropped_columns'] else ""))
    for n, det in failed:
        print(f"      - {n}: {det}")

os.makedirs("data", exist_ok=True)
with open("data/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\n{sum(r['gate_passed'] for r in manifest)}/{len(manifest)} quarters pass the gate")
