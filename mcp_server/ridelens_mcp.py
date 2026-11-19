"""RideLens MCP server: read-only tools over the saved results.

Run with the stdio transport:  python -m mcp_server.ridelens_mcp
Open the MCP dev inspector:    mcp dev mcp_server/ridelens_mcp.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from tools import quarter_tools as qt  # noqa: E402

mcp = FastMCP("ridelens")


@mcp.tool()
def list_quarters() -> list[dict]:
    """List every quarter in the data, with its data-quality result and whether a report is saved."""
    return qt.list_quarters()


@mcp.tool()
def get_gate_result(quarter: str) -> dict:
    """Data-quality checks for one quarter, such as '2026-Q3'. Lists any failed checks."""
    return qt.get_gate_result(quarter)


@mcp.tool()
def get_report(quarter: str) -> dict:
    """Full saved report for one quarter: data quality, trip-level flags and same-quarter comparison."""
    return qt.get_report(quarter)


@mcp.tool()
def get_trip_summary(quarter: str) -> dict:
    """Trips, flagged trips, flagged share and the 2% expected share for one quarter."""
    return qt.get_trip_summary(quarter)


@mcp.tool()
def get_same_quarter(quarter: str) -> dict:
    """Each metric for one quarter against the same quarter in past years, with past ranges."""
    return qt.get_same_quarter(quarter)


@mcp.tool()
def get_flagged_examples(quarter: str, n: int = 10) -> dict:
    """Up to 10 of the most unusual flagged trips for one quarter, without trip IDs."""
    return qt.get_flagged_examples(quarter, n)


@mcp.tool()
def get_fault_comparison(quarter: str) -> dict:
    """How four detectors did on planted faults for a test quarter: clean false-positive rates and recall per fault."""
    return qt.get_fault_comparison(quarter)


@mcp.tool()
def get_backtest() -> dict:
    """Backtest of the quarter-level rule over 2023-2026: how often it would have fired, and what chance would give."""
    return qt.get_backtest()


if __name__ == "__main__":
    mcp.run()  # stdio transport