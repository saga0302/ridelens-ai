"""Dark neon theme, animations and card helpers for the RideLens dashboard."""
import streamlit as st

NEON = {"cyan": "#00f0ff", "magenta": "#ff2bd6", "lime": "#b6ff3b", "ink": "#e8e8ff"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@400;600;700&display=swap');
.stApp {
  background: radial-gradient(circle at 12% 0%, #1b1c4a 0%, #070712 50%) fixed;
  color: #e8e8ff; font-family: 'Inter', sans-serif;
}
.stApp::before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    linear-gradient(rgba(0,240,255,.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,43,214,.06) 1px, transparent 1px);
  background-size: 42px 42px; animation: drift 28s linear infinite;
}
@keyframes drift { to { background-position: 420px 420px; } }
header[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none !important; }
.block-container { position: relative; z-index: 1; padding-top: 1.5rem; max-width: 1280px; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; }

/* title and animated riding lane */
.neon-title {
  font-family: 'Orbitron', sans-serif; font-size: 2.6rem; font-weight: 800;
  background: linear-gradient(90deg, #00f0ff, #ff2bd6, #b6ff3b);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: glow 3s ease-in-out infinite alternate;
}
@keyframes glow {
  from { filter: drop-shadow(0 0 4px rgba(0,240,255,.4)); }
  to   { filter: drop-shadow(0 0 16px rgba(255,43,214,.6)); }
}
.subtitle { opacity: .8; margin-top: -6px; }
.ride-lane {
  position: relative; height: 58px; overflow: hidden; margin-bottom: 8px;
  border-bottom: 1px dashed rgba(0,240,255,.35);
}
.ride-lane span {
  position: absolute; bottom: 6px; font-size: 34px; left: -60px;
  animation: ride 9s linear infinite;
}
@keyframes ride { to { left: 100%; } }

/* tabs: high contrast, neon when selected */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(0,240,255,.25); }
[data-baseweb="tab"] {
  background: rgba(15,16,32,.9); border-radius: 12px 12px 0 0;
  padding: 10px 18px !important; font-weight: 700; color: #c9c9ff !important;
}
[data-baseweb="tab"] p { color: #c9c9ff !important; font-size: 1rem; }
[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(90deg, #00f0ff, #ff2bd6);
}
[data-baseweb="tab"][aria-selected="true"] p { color: #070712 !important; }
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* selectbox */
[data-baseweb="select"] > div {
  background: #0f1020 !important; border: 1px solid rgba(0,240,255,.5) !important;
  color: #ffffff !important; border-radius: 12px !important;
}
[data-testid="stWidgetLabel"] p { color: #c9c9ff !important; font-weight: 600; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.kpi {
  background: linear-gradient(160deg, rgba(20,22,48,.95), rgba(10,11,26,.95));
  border-radius: 18px; padding: 18px 20px; border: 1px solid rgba(0,240,255,.25);
  box-shadow: 0 0 24px rgba(0,240,255,.12); transition: transform .25s;
  animation: fadeUp .6s ease both;
}
.kpi:hover { transform: translateY(-4px); }
.kpi-kicker { font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; opacity: .7; }
.kpi-label { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; opacity: .7; }
.kpi-value { font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 800; margin-top: 4px; }
.kpi-icon { font-size: 1.6rem; animation: bob 3s ease-in-out infinite; }
.kpi-cyan .kpi-value { color: #00f0ff; }
.kpi-magenta .kpi-value { color: #ff2bd6; }
.kpi-lime .kpi-value { color: #b6ff3b; }
@keyframes bob { 50% { transform: translateY(-4px); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }

/* trip cards */
.trip-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
.trip-card {
  border-radius: 16px; padding: 14px; background: rgba(15,16,32,.85);
  border: 1px solid rgba(255,43,214,.3); animation: fadeUp .6s ease both;
}
.trip-duration { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; color: #00f0ff; margin: 6px 0; }
.chip {
  display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: .8rem;
  background: rgba(182,255,59,.15); color: #b6ff3b;
}
.chip-bad { background: rgba(255,43,214,.15); color: #ff2bd6; }

/* chat bubbles */
[data-testid="stChatMessage"] {
  background: rgba(15,16,32,.7); border: 1px solid rgba(0,240,255,.2); border-radius: 14px;
}
"""


def apply():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(
        '<div class="ride-lane"><span>🚲</span>'
        '<span style="animation-delay:-4.5s">🚴</span></div>'
        f'<div class="neon-title">{title}</div>'
        f'<p class="subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def kpi(label, value, icon, tone="cyan"):
    return (f'<div class="kpi kpi-{tone}"><div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>')


def kpi_grid(cards):
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


def trip_cards(examples):
    cards = "".join(
        f'<div class="trip-card" style="animation-delay:{i * 0.07:.2f}s">'
        f'<div class="kpi-label">{e["start_time"]}</div>'
        f'<div class="trip-duration">{e["duration"]} min</div>'
        f'<span class="chip chip-bad">{e["passholder_type"]}</span>'
        f'<div class="kpi-label" style="margin-top:8px">{e.get("reasons", "")}</div></div>'
        for i, e in enumerate(examples)
    )
    return f'<div class="trip-grid">{cards}</div>'


def fmt_metric(metric, value):
    if metric.startswith("share_"):
        return f"{value:.2%}"
    if metric == "trips":
        return f"{value:,.0f}"
    return f"{value:.2f}"


def pretty_metric(metric):
    return metric.replace("share_", "share of ").replace("pass_", "pass: ").replace("_", " ")