"""Neon charts for the RideLens dashboard."""
import plotly.graph_objects as go

from dashboard.theme import NEON, fmt_metric, pretty_metric

TARGET = 0.02


def style(fig, height=300):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter", color=NEON["ink"]))
    return fig


def gauge(share, target=TARGET):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=share * 100,
        number={"suffix": "%", "valueformat": ".2f",
                "font": {"family": "Orbitron", "size": 44, "color": NEON["cyan"]}},
        gauge={"axis": {"range": [0, 5], "tickcolor": NEON["ink"]},
               "bar": {"color": NEON["cyan"], "thickness": 0.28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, target * 100], "color": "rgba(182,255,59,.18)"},
                         {"range": [target * 100, 5], "color": "rgba(255,43,214,.16)"}],
               "threshold": {"line": {"color": NEON["magenta"], "width": 4},
                             "value": target * 100}},
        title={"text": "Flagged share · target 2%",
               "font": {"size": 15, "color": NEON["ink"]}}))
    return style(fig, 280)


def trend(rows):
    fig = go.Figure(go.Bar(
        x=[r["quarter"] for r in rows], y=[r["share"] for r in rows],
        marker=dict(color=[NEON["magenta"] if r["share"] > TARGET else NEON["cyan"]
                           for r in rows],
                    line=dict(color="rgba(255,255,255,.4)", width=1)),
        hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig.add_hline(y=TARGET, line=dict(color=NEON["lime"], dash="dot"),
                  annotation_text="2% target", annotation_font_color=NEON["lime"])
    fig.update_yaxes(tickformat=".1%", gridcolor="rgba(255,255,255,.06)")
    fig.update_xaxes(showgrid=False)
    return style(fig, 320)


def range_chart(metrics):
    """One row per measure: the past range as a band, the median as a tick,
    and this quarter as a dot (lime inside the range, magenta outside)."""
    fig = go.Figure()
    for i, m in enumerate(metrics):
        lo, hi = m["past_min"], m["past_max"]
        span = (hi - lo) or 1.0

        def norm(v):
            return (v - lo) / span

        fig.add_trace(go.Scatter(x=[0, 1], y=[i, i], mode="lines",
                                 line=dict(color="rgba(0,240,255,.35)", width=12),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[norm(m["past_median"])], y=[i], mode="markers",
                                 marker=dict(symbol="line-ns", size=18, color=NEON["ink"],
                                             line=dict(width=3)),
                                 hoverinfo="skip", showlegend=False))
        inside = not m["outside_past_range"]
        fig.add_trace(go.Scatter(
            x=[norm(m["new"])], y=[i], mode="markers",
            marker=dict(size=16, color=NEON["lime"] if inside else NEON["magenta"],
                        line=dict(color="#070712", width=2)),
            text=[f"{fmt_metric(m['metric'], m['new'])}"],
            hovertemplate="%{text}<extra></extra>", showlegend=False))

    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=list(range(len(metrics))),
                   ticktext=[pretty_metric(m["metric"]) for m in metrics],
                   autorange="reversed", showgrid=False),
        xaxis=dict(range=[-0.25, 1.25], showticklabels=False, zeroline=False, showgrid=False))
    return style(fig, 70 * len(metrics) + 60)


def heatmap(detectors, faults, z):
    fig = go.Figure(go.Heatmap(
        z=z, x=[f.replace("_", " ") for f in faults], y=detectors, zmin=0, zmax=1,
        colorscale=[[0, "#0f1020"], [0.5, NEON["magenta"]], [1, NEON["lime"]]],
        texttemplate="%{z:.0%}",
        hovertemplate="%{y} · %{x}: %{z:.0%}<extra></extra>"))
    return style(fig, 320)