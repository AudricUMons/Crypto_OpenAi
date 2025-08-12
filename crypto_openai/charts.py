import plotly.graph_objects as go
import pandas as pd
from typing import List, Tuple

def render_main_chart(df: pd.DataFrame, series: List[Tuple[str, pd.Series]]):
    fig = go.Figure()
    for name, s in series:
        fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=name))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_trades_chart(df: pd.DataFrame, buys_d, buys_p, sells_d, sells_p, title="Trades"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines", name="Price"))
    if buys_d:
        fig.add_trace(go.Scatter(x=buys_d, y=buys_p, mode="markers", name="BUY", marker_symbol="triangle-up", marker_size=10))
    if sells_d:
        fig.add_trace(go.Scatter(x=sells_d, y=sells_p, mode="markers", name="SELL", marker_symbol="triangle-down", marker_size=10))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=30, b=10))
    return fig
