"""
Plotly Chart Library & Dark Glassmorphic Theme Engine for ECIP Dashboard.
Provides high-performance interactive charts with curated neon color schemes.
Includes Line Charts, Treemaps, Geospatial State Maps, Donut Charts, Bar Charts,
Horizontal Bar Charts, Rating Histograms, Pareto 80/20 Charts, Box Plots, and RFM Heatmaps.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Curated Neon Palette
PALETTE = ["#38bdf8", "#c084fc", "#34d399", "#f43f5e", "#fbbf24", "#06b6d4", "#a855f7", "#38bdf8", "#818cf8"]
COLOR_PRIMARY = "#38bdf8"
COLOR_ACCENT = "#c084fc"
COLOR_SUCCESS = "#34d399"
COLOR_WARNING = "#fbbf24"

def apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies glassmorphic dark styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#f8fafc", family="Segoe UI"),
            x=0.01, y=0.96
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Segoe UI", size=12),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color="#f8fafc")
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, linecolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, linecolor="rgba(255,255,255,0.1)"
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="Segoe UI"
        )
    )
    return fig

def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str = "Monthly Revenue Trend") -> go.Figure:
    """Creates a smooth neon line chart with gradient area fill."""
    if df.empty:
        df = pd.DataFrame({x: ["No Data"], y: [0]})
    fig = px.line(df, x=x, y=y, markers=True, color_discrete_sequence=[COLOR_PRIMARY])
    fig.update_traces(
        line=dict(width=3, shape="spline"),
        marker=dict(size=7, symbol="circle", color="#06b6d4"),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)"
    )
    return apply_dark_theme(fig, title)

def create_customer_growth_chart(df: pd.DataFrame, title: str = "Customer Growth Trend") -> go.Figure:
    """Creates multi-trace Customer Growth Chart (New vs Returning vs Total)."""
    if df.empty:
        df = pd.DataFrame({"Month": ["No Data"], "Total_Customers": [0], "New_Customers": [0], "Returning_Customers": [0]})

    fig = go.Figure()
    if "New_Customers" in df.columns:
        fig.add_trace(go.Bar(x=df["Month"], y=df["New_Customers"], name="New Customers", marker_color="#38bdf8"))
    if "Returning_Customers" in df.columns:
        fig.add_trace(go.Bar(x=df["Month"], y=df["Returning_Customers"], name="Returning Customers", marker_color="#c084fc"))
    if "Total_Customers" in df.columns:
        fig.add_trace(go.Scatter(x=df["Month"], y=df["Total_Customers"], name="Total Active", mode="lines+markers", line=dict(color="#34d399", width=3)))

    fig.update_layout(barmode="stack")
    return apply_dark_theme(fig, title)

def create_treemap_chart(df: pd.DataFrame, path_col: str = "Category", values_col: str = "Revenue", title: str = "Revenue by Category Treemap") -> go.Figure:
    """Creates a Treemap for Category Revenue hierarchy."""
    if df.empty:
        df = pd.DataFrame({path_col: ["No Data"], values_col: [0]})

    if path_col not in df.columns:
        path_col = df.columns[0]
    if values_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        values_col = num_cols[0] if len(num_cols) > 0 else df.columns[-1]

    fig = px.treemap(
        df,
        path=[path_col],
        values=values_col,
        color=values_col,
        color_continuous_scale="Purples"
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return apply_dark_theme(fig, title)

def create_state_map_chart(df: pd.DataFrame, title: str = "Revenue by State Map") -> go.Figure:
    """Creates a Geospatial State Map scatter plot on map projection."""
    if df.empty or "Lat" not in df.columns:
        if not df.empty and "State" in df.columns:
            val_col = "Revenue" if "Revenue" in df.columns else ("Total_Revenue" if "Total_Revenue" in df.columns else ("Customer_Count" if "Customer_Count" in df.columns else df.columns[1]))
            return create_bar_chart(df, x="State", y=val_col, title=title)
        df = pd.DataFrame({"State": ["No Data"], "Total_Revenue": [0], "Lat": [-14.23], "Lon": [-51.92]})

    size_col = None
    for col in ["Total_Revenue", "Revenue", "Customer_Count"]:
        if col in df.columns:
            size_col = col
            break
    if not size_col:
        num_cols = df.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c not in ["Lat", "Lon"]]
        size_col = num_cols[0] if len(num_cols) > 0 else None

    hover_name = "State_Name" if "State_Name" in df.columns else ("State" if "State" in df.columns else None)

    kwargs = {
        "lat": "Lat",
        "lon": "Lon",
        "projection": "mercator"
    }
    if "State" in df.columns:
        kwargs["text"] = "State"
    if hover_name in df.columns:
        kwargs["hover_name"] = hover_name
    if size_col and size_col in df.columns:
        kwargs["size"] = size_col
        kwargs["color"] = size_col
        kwargs["color_continuous_scale"] = "Viridis"

    fig = px.scatter_geo(df, **kwargs)
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=True,
        showland=True,
        landcolor="rgba(15, 23, 42, 0.8)",
        countrycolor="rgba(255, 255, 255, 0.1)",
        bgcolor="rgba(0, 0, 0, 0)"
    )
    return apply_dark_theme(fig, title)

def create_donut_chart(df: pd.DataFrame, names: str, values: str, title: str = "Payment Method Distribution") -> go.Figure:
    """Creates a neon donut chart."""
    if df.empty:
        df = pd.DataFrame({names: ["No Data"], values: [0]})
    fig = px.pie(df, names=names, values=values, hole=0.6, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0f172a", width=2)))
    return apply_dark_theme(fig, title)

def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "Order Status", horizontal: bool = False, orientation: str = None) -> go.Figure:
    """Creates a horizontal or vertical bar chart with rounded corners."""
    if df.empty:
        df = pd.DataFrame({x: ["No Data"], y: [0]})
    is_horiz = horizontal or (orientation == "h")
    if is_horiz:
        fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=[COLOR_ACCENT])
    else:
        fig = px.bar(df, x=x, y=y, color_discrete_sequence=[COLOR_PRIMARY])
    fig.update_traces(marker_line_color="rgba(0,0,0,0)", opacity=0.9)
    return apply_dark_theme(fig, title)

def create_histogram_chart(df: pd.DataFrame, x: str = "Star_Rating", y: str = "Count", title: str = "Customer Ratings Histogram") -> go.Figure:
    """Creates a rating distribution histogram."""
    if df.empty:
        df = pd.DataFrame({x: ["1 Star"], y: [0]})
    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale="Tealgrn")
    fig.update_traces(marker_line_color="rgba(0,0,0,0)", opacity=0.9)
    return apply_dark_theme(fig, title)

def create_pareto_chart(df: pd.DataFrame, x: str = "Customer_Percentile", y: str = "Cumulative_Revenue_Pct", title: str = "Pareto (80/20) Revenue Distribution") -> go.Figure:
    """Creates Pareto cumulative revenue percentage curve."""
    if df.empty:
        df = pd.DataFrame({x: [0, 100], y: [0, 100]})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        mode="lines+markers",
        name="Cumulative Revenue %",
        line=dict(color="#c084fc", width=3, shape="spline"),
        marker=dict(size=6, color="#38bdf8")
    ))
    fig.add_shape(type="line", x0=20, y0=0, x1=20, y1=100, line=dict(color="#f43f5e", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=80, x1=100, y1=80, line=dict(color="#fbbf24", width=1.5, dash="dash"))

    fig.update_layout(
        xaxis_title="Cumulative Customers (%)",
        yaxis_title="Cumulative Revenue (%)",
        yaxis=dict(range=[0, 105])
    )
    return apply_dark_theme(fig, title)

def create_rfm_heatmap(pivot_df: pd.DataFrame, title: str = "RFM Matrix (Recency vs Frequency)") -> go.Figure:
    """Creates 2D Heatmap of Recency Score vs Frequency Score customer counts."""
    if pivot_df.empty:
        pivot_df = pd.DataFrame([[0]*5]*5, index=[f"R{i}" for i in range(5,0,-1)], columns=[f"F{i}" for i in range(1,6)])

    fig = px.imshow(
        pivot_df.values,
        x=pivot_df.columns.tolist(),
        y=pivot_df.index.tolist(),
        color_continuous_scale="Purples",
        text_auto=True
    )
    return apply_dark_theme(fig, title)

def create_3d_pca_scatter(df: pd.DataFrame, x: str = "PC1", y: str = "PC2", z: str = "PC3", color: str = "Cluster_Name", title: str = "3D PCA Customer Cluster Plot") -> go.Figure:
    """Creates interactive 3D PCA scatter plot."""
    if df.empty or x not in df.columns:
        df = pd.DataFrame({x: [0], y: [0], z: [0], color: ["Cluster 1"]})
    fig = px.scatter_3d(
        df, x=x, y=y, z=z, color=color,
        color_discrete_sequence=PALETTE,
        opacity=0.85
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return apply_dark_theme(fig, title)

def create_box_plot(df: pd.DataFrame, x: str, y: str, title: str = "CLV Distribution by Spending Tier") -> go.Figure:
    """Creates a distribution box plot."""
    if df.empty or x not in df.columns or y not in df.columns:
        df = pd.DataFrame({x: ["Tier 1"], y: [100]})
    fig = px.box(df, x=x, y=y, color=x, color_discrete_sequence=PALETTE)
    return apply_dark_theme(fig, title)

def create_pie_chart(df: pd.DataFrame, names: str, values: str, title: str = "Distribution") -> go.Figure:
    """Creates a pie chart."""
    return create_donut_chart(df, names, values, title)

def create_treemap(df: pd.DataFrame, path: Any, values: str, title: str = "Treemap") -> go.Figure:
    """Creates a multi-level or single-level treemap chart."""
    if df.empty:
        df = pd.DataFrame({"Category": ["No Data"], "Revenue": [0]})
        path = ["Category"]
        values = "Revenue"

    if isinstance(path, list):
        valid_path = [p for p in path if p in df.columns]
    elif isinstance(path, str) and path in df.columns:
        valid_path = [path]
    else:
        valid_path = [df.columns[0]]

    if not valid_path:
        valid_path = [df.columns[0]]

    if values not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        values = num_cols[0] if len(num_cols) > 0 else df.columns[-1]

    fig = px.treemap(
        df,
        path=valid_path,
        values=values,
        color=values,
        color_continuous_scale="Purples"
    )
    fig.update_traces(textinfo="label+value+percent parent")
    return apply_dark_theme(fig, title)
