"""RideLens AI dashboard: visuals first. Public to read; the chat needs a passcode."""
import hmac
import os
import sys
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

from dashboard.charts import gauge, heatmap, range_chart, trend  # noqa: E402
from dashboard.theme import apply, header, kpi, kpi_grid, trip_cards  # noqa: E402
from rag.build_index import INDEX_DIR, build  # noqa: E402
from tools import quarter_tools as qt  # noqa: E402
from pipeline.upload_check import check_upload  # noqa: E402

CONTACT_EMAIL = "sagarikaraju2000@gmail.com"  
MAX_QUESTIONS = 10
DETECTORS = ["rules", "isolation_forest", "lof", "hbos"]

st.set_page_config(page_title="RideLens AI", page_icon="🚲", layout="wide")
apply()
header("RideLens AI", "LA Metro bike trips, quarter by quarter.")


@st.cache_resource
def get_graph():
    if not os.path.exists(INDEX_DIR):
        build()
    from agent.graph import build_graph
    return build_graph()


def ask(question):
    try:
        return get_graph().invoke({"question": question})["answer"]
    except Exception as e:  # keep the page up if the API or credits fail
        print(f"agent error: {e}")
        return "The assistant is unavailable right now. Please try again later."


saved = [q["quarter"] for q in qt.list_quarters() if q["report_saved"]]
quarter = st.selectbox("Pick a quarter", saved[::-1])

tabs = st.tabs(["Overview", "Compare", "Trips", "Trend", "Methods", "Ask", "About", "Upload"])

with tabs[0]:
    summary = qt.get_trip_summary(quarter)
    gate = qt.get_gate_result(quarter)
    passed = gate["gate_passed"]
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(gauge(summary["flagged_share"]), use_container_width=True,
                        config={"displayModeBar": False})
    with right:
        st.markdown(kpi_grid([
            kpi("Trips", f"{summary['trips']:,}", "🚲", "cyan"),
            kpi("Flagged", f"{summary['flagged']:,}", "🚩", "magenta"),
            kpi("Data checks", "Passed" if passed else "Failed",
                "✅" if passed else "⛔", "lime" if passed else "magenta"),
            kpi("Target", f"{summary['expected_share']:.0%}", "🎯", "cyan"),
        ]), unsafe_allow_html=True)
    if not passed:
        st.error("Failed checks: " + ", ".join(f["check"] for f in gate["gate_failures"]))

with tabs[1]:
    metrics = qt.get_same_quarter(quarter)["metrics"]
    st.plotly_chart(range_chart(metrics), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption("🟢 inside past range   🔴 outside   |  white tick = past median")

with tabs[2]:
    examples = qt.get_flagged_examples(quarter, 10)["examples"]
    st.markdown(trip_cards(examples), unsafe_allow_html=True)
    st.caption("Biggest outliers first. Many are data errors, so treat them as leads.")

with tabs[3]:
    rows = qt.get_backtest()["rows"]
    st.plotly_chart(trend(rows), use_container_width=True, config={"displayModeBar": False})
    st.caption("Each quarter scored by a model trained only on earlier years.")

with tabs[4]:
    fault = qt.get_fault_comparison(quarter)
    if "error" in fault:
        st.info("Method results are saved only for the test quarters.")
    else:
        faults = list(fault["recall_by_fault"])
        z = [[fault["recall_by_fault"][f][d] for f in faults] for d in DETECTORS]
        st.plotly_chart(heatmap(DETECTORS, faults, z), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Share of planted faults each method caught.")

with tabs[5]:
    expected = os.environ.get("APP_PASSCODE", "")
    if not expected:
        st.info("The chat is switched off on this copy.")
    elif not st.session_state.get("unlocked"):
        typed = st.text_input("Passcode", type="password")
        if typed:
            if hmac.compare_digest(typed, expected):
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Not the right passcode.")
        subject = quote("RideLens chat access")
        body = quote("Hi, I'd like access to the RideLens chat. My name: ")
        st.link_button("🔑 Request access",
                       f"mailto:{CONTACT_EMAIL}?subject={subject}&body={body}")
    else:
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("asked", 0)
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])
        left_q = MAX_QUESTIONS - st.session_state.asked
        if left_q <= 0:
            st.warning("Question limit reached. Refresh to start again.")
        elif question := st.chat_input(f"Ask RideLens · {left_q} left"):
            st.session_state.asked += 1
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask(question)
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with tabs[6]:
    st.markdown("""
- 🚲 **What:** checks each quarter of LA Metro bike trips and spots what's unusual.
- 🔎 **Not:** not run by LA Metro. It can't say why a trip happened or judge an area's safety.
- 🗂️ **Data:** public LA Metro trip files, 2019 Q1 to 2026 Q3. No rider information.
""")

with tabs[7]:
    st.markdown("### Upload a quarter")
    st.caption("2026 Q3 onward. One complete quarter per file. Uploads are never added to our data.")
    file = st.file_uploader("Trip CSV", type=["csv"], label_visibility="collapsed")
    if file is not None:
        with st.spinner("Checking and scoring..."):
            result = check_upload(file.getvalue())
        status = result["status"]
        if status == "rejected":
            st.warning(result["message"])
        elif status == "gate_failed":
            names = ", ".join(x["check"] for x in result["failures"])
            st.error(f"{result['quarter']} failed data checks: {names}")
        else:
            if status == "stored":
                st.success(f"{result['quarter']} matches our copy. Showing the saved result.")
            elif status == "differs_from_stored":
                st.warning(f"This file differs from our copy of {result['quarter']}. "
                           "Scored below, not saved.")
            else:
                st.success(f"{result['quarter']} is new. Scored with the model trained on "
                           "earlier years, not added to our data.")
            t = result["trip_level"]
            st.markdown(kpi_grid([
                kpi("Trips", f"{t['trips']:,}", "🚲", "cyan"),
                kpi("Flagged", f"{t['flagged']:,}", "🚩", "magenta"),
                kpi("Share", f"{t['flagged_share']:.2%}", "📊", "lime"),
                kpi("Target", f"{t['expected_share']:.0%}", "🎯", "cyan"),
            ]), unsafe_allow_html=True)
            st.plotly_chart(range_chart(result["same_quarter"]), use_container_width=True,
                config={"displayModeBar": False}, key="upload_range")
            st.markdown(trip_cards(t["examples"]), unsafe_allow_html=True)