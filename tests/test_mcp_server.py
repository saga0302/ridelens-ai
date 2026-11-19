import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp_server.ridelens_mcp import mcp  # noqa: E402

EXPECTED_TOOLS = {
    "list_quarters", "get_gate_result", "get_report", "get_trip_summary",
    "get_same_quarter", "get_flagged_examples", "get_fault_comparison", "get_backtest",
}


def test_all_read_only_tools_are_registered():
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS