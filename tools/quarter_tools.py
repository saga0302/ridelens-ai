"""Read-only tools over saved results.

Used by the MCP server, the agent and the dashboard. Every function reads files
under results/ and data/manifest.json, returns plain dicts and lists, and writes nothing.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")
QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")
MAX_EXAMPLES = 10
EXAMPLE_FIELDS = ("start_time", "duration", "passholder_type", "reasons")  # never trip IDs


def _parse_quarter(quarter):
    m = QUARTER_RE.match(quarter or "")
    if not m:
        raise ValueError(f"quarter must look like 2026-Q3, got {quarter!r}")
    return int(m.group(1)), int(m.group(2))


def _report_path(quarter):
    y, q = _parse_quarter(quarter)
    return os.path.join(RESULTS, "reports", f"report_{y}Q{q}.json")


def _missing_report(quarter):
    y, q = _parse_quarter(quarter)
    return {"quarter": quarter,
            "error": f"no saved report; run: python -m comparison.run_report {y} {q}"}


def _load_report(quarter):
    path = _report_path(quarter)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def list_quarters():
    """Every quarter in the gate manifest, with its gate result and whether a report is saved."""
    out = []
    for entry in sorted(_load_manifest(), key=lambda e: (e["year"], e["quarter"])):
        quarter = f"{entry['year']}-Q{entry['quarter']}"
        out.append({"quarter": quarter,
                    "gate_passed": entry["gate_passed"],
                    "report_saved": os.path.exists(_report_path(quarter))})
    return out


def get_gate_result(quarter):
    _parse_quarter(quarter)
    for entry in _load_manifest():
        if f"{entry['year']}-Q{entry['quarter']}" == quarter:
            return {"quarter": quarter,
                    "gate_passed": entry["gate_passed"],
                    "gate_failures": entry["gate_failures"]}
    return {"quarter": quarter, "error": "quarter not in manifest"}


def get_report(quarter):
    report = _load_report(quarter)
    return report if report is not None else _missing_report(quarter)


def get_trip_summary(quarter):
    report = _load_report(quarter)
    if report is None:
        return _missing_report(quarter)
    t = report["trip_level"]
    return {"quarter": quarter, "trips": t["trips"], "flagged": t["flagged"],
            "flagged_share": t["flagged_share"], "expected_share": t["expected_share"]}


def get_same_quarter(quarter):
    report = _load_report(quarter)
    if report is None:
        return _missing_report(quarter)
    return {"quarter": quarter, "metrics": report["same_quarter"]}


def get_flagged_examples(quarter, n=MAX_EXAMPLES):
    report = _load_report(quarter)
    if report is None:
        return _missing_report(quarter)
    n = max(0, min(int(n), MAX_EXAMPLES))
    examples = [{k: e[k] for k in EXAMPLE_FIELDS if k in e}
                for e in report["trip_level"]["examples"][:n]]
    return {"quarter": quarter, "examples": examples}


def get_fault_comparison(quarter):
    """Clean false-positive rates and recall per planted fault, for one test quarter."""
    _parse_quarter(quarter)
    with open(os.path.join(RESULTS, "evaluation", "uncovered_eval_results.json")) as f:
        data = json.load(f)
    for row in data["results"]:
        if row["test"] == quarter:
            clean = {k[len("fp__"):]: v for k, v in row.items() if k.startswith("fp__")}
            recall = {}
            for k, v in row.items():
                if k.startswith("recall__"):
                    _, detector, fault = k.split("__", 2)
                    recall.setdefault(fault, {})[detector] = v
            return {"quarter": quarter, "clean_false_positive_rate": clean,
                    "recall_by_fault": recall}
    return {"quarter": quarter, "error": "no fault comparison for this quarter"}


def get_backtest():
    """Summary of the quarter-level rule backtest (report only, so this is context)."""
    with open(os.path.join(RESULTS, "evaluation", "quarter_rule_backtest.json")) as f:
        data = json.load(f)
    return {"rule": data["rule"], "quarters_tested": len(data["rows"]),
            "fired": data["fired"], "expected_by_chance": data["expected_by_chance"],
            "rows": data["rows"]}