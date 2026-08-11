"""
Market Basket Analysis & Product Association Page UI Layout for ECIP Phase 17.
Fully integrated with MBAService backend.
Provides 8 KPIs, Association Rule Explorer, Network Graph, Product Bundles,
Cross-Sell Opportunities, Customer Segment Basket Analysis, Category Co-occurrence Heatmap,
Seasonal Telemetry, Product Search Intelligence, Business Recommendations, and Multi-Format Exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.header import render_top_header
from dashboard.components.ui_elements import render_breadcrumb
from dashboard.components.kpi_cards import render_glass_kpi_card, render_kpi_grid_row
from dashboard.components.filters import (
    render_mba_filter_panel,
    render_mba_search_input
)
from dashboard.components.charts import (
    apply_dark_theme,
    create_bar_chart,
    create_line_chart,
    COLOR_PRIMARY, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING
)
from dashboard.utils.exporter import render_export_buttons
from backend.services.mba_service import mba_service
from utils.logger import setup_logger

logger = setup_logger("ECIP.MBAPage")

def render_mba_layout():
    """Renders complete Market Basket Analysis & Product Association Dashboard Page for Phase 17."""
    render_top_header("Market Basket Analysis & Product Association Dashboard")
    render_breadcrumb(["Home", "Analytics", "Market Basket Analysis"])

    # Load Data Payload via Backend Service
    datasets = mba_service.load_all_mba_datasets()

    # 1. Sidebar Filters
    status_files = mba_service.get_dataset_files_status()
    cats_list = list(datasets.get("master_dataset", pd.DataFrame())["product_category_name_english"].dropna().unique()) if "product_category_name_english" in datasets.get("master_dataset", pd.DataFrame()).columns else []
    
    filter_opts = {
        "categories": cats_list,
        "customer_segments": ["VIP Power Buyers", "Loyal Frequenters", "At-Risk High Rollers", "New Customers"]
    }
    filters = render_mba_filter_panel(filter_opts)

    # 2. Product Search Input
    st.markdown("### 🔍 Search Market Basket Intelligence")
    search_query = render_mba_search_input()

    if search_query.strip():
        _render_product_search_panel(search_query, datasets)
        st.markdown("---")

    # 3. Dataset Status Notice Expander
    with st.expander("ℹ️ Market Basket Datasets & Mining Engine Status", expanded=False):
        status_rows = []
        for key, meta in status_files.items():
            status_rows.append({
                "Dataset Key": key,
                "Status": "🟢 Loaded" if meta["available"] else "🟡 Fallback Active",
                "Size (MB)": meta["size_mb"],
                "Last Modified": meta["last_modified"],
                "File Path": meta["path"]
            })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

    # 4. 8 KPI Cards Grid
    kpis = mba_service.compute_mba_kpis(datasets)
    _render_8_kpi_cards(kpis)

    st.markdown("---")

    # 5. Association Rule Explorer & Scatter Plot
    st.markdown("### 🎯 Association Rule Explorer & Scatter Space")
    _render_association_rule_explorer(datasets, filters)

    st.markdown("---")

    # 6. Product Association Network Graph
    st.markdown("### 🕸️ Product Association Network Graph")
    _render_association_network_graph(datasets, filters)

    st.markdown("---")

    # 7. Product Bundle Analysis
    st.markdown("### 📦 Product Bundle Analysis & Merchandising Strategies")
    _render_product_bundles_section(datasets)

    st.markdown("---")

    # 8. Cross-Sell Opportunities Engine
    st.markdown("### 🔀 Cross-Sell Opportunities Engine")
    _render_cross_sell_section(datasets)

    st.markdown("---")

    # 9. Customer Segment Basket Analysis
    st.markdown("### 👥 Customer Segment Basket Behavior Analysis")
    _render_segment_basket_section(datasets)

    st.markdown("---")

    # 10. Category Co-occurrence Heatmap
    st.markdown("### 🗺️ Category Co-occurrence Heatmap")
    _render_category_heatmap_section(datasets)

    st.markdown("---")

    # 11. Seasonal Basket Telemetry (If available)
    seasonal_data = mba_service.get_seasonal_basket_analysis(datasets.get("master_dataset", pd.DataFrame()))
    if seasonal_data["available"]:
        st.markdown("### 📅 Seasonal Basket Telemetry")
        _render_seasonal_basket_section(seasonal_data)
        st.markdown("---")

    # 12. Business Recommendations & Financial Impact
    st.markdown("### 💡 Data-Driven Merchandising Recommendations & Financial Impact")
    _render_business_recommendations_section(datasets)

    st.markdown("---")

    # 13. Data Export Hub
    st.markdown("### 📥 Multi-Format Market Basket Data Exports")
    _render_export_hub_section(datasets, filters)


def _render_8_kpi_cards(kpis: dict):
    """Renders 8 Glassmorphic Market Basket KPI Cards in 2 rows."""
    k_keys = list(kpis.keys())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = kpis[k_keys[0]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "cyan", k["last_updated"])
    with c2:
        k = kpis[k_keys[1]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "green", k["last_updated"])
    with c3:
        k = kpis[k_keys[2]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "purple", k["last_updated"])
    with c4:
        k = kpis[k_keys[3]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "purple", k["last_updated"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        k = kpis[k_keys[4]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "cyan", k["last_updated"])
    with c6:
        k = kpis[k_keys[5]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "green", k["last_updated"])
    with c7:
        k = kpis[k_keys[6]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "green", k["last_updated"])
    with c8:
        k = kpis[k_keys[7]]
        render_glass_kpi_card(k["title"], k["value"], k["change"], k["is_positive"], k["badge"], k["icon"], "purple", k["last_updated"])


def _render_product_search_panel(query: str, datasets: dict):
    """Renders Product Search Intelligence detail panel."""
    st.markdown(f"#### 🔍 Product Search Results for: `{query}`")
    search_res = mba_service.get_product_search_intelligence(query, datasets)

    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom: 12px;">
            <div style="font-size: 15px; font-weight: 700; color: #38bdf8;">Search Target Category: {search_res['category']}</div>
            <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">
                <b>Total Sales Volume:</b> {search_res['performance']['total_units']} orders | 
                <b>Total Revenue:</b> ${search_res['performance']['total_revenue']:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    rules_df = search_res["strongest_rules"]
    if not rules_df.empty:
        st.markdown("##### Strongest Association Rules for Target Product/Category")
        ant_col = "antecedents_str" if "antecedents_str" in rules_df.columns else "antecedents"
        cons_col = "consequents_str" if "consequents_str" in rules_df.columns else "consequents"
        disp_cols = [c for c in [ant_col, cons_col, "support", "confidence", "lift"] if c in rules_df.columns]
        st.dataframe(rules_df[disp_cols], use_container_width=True)
    else:
        st.info("No direct association rules matched this specific query.")


def _render_association_rule_explorer(datasets: dict, filters: dict):
    """Renders interactive Rule Explorer with text search, sliders, sorting, scatter plot, and bar chart."""
    rules_df = datasets.get("association_rules", pd.DataFrame())
    if rules_df.empty:
        st.info("Association rules mining telemetry syncing...")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        ant_input = st.text_input("Antecedent Product/Cat", "", key="exp_ant")
    with c2:
        cons_input = st.text_input("Consequent Product/Cat", "", key="exp_cons")
    with c3:
        sort_field = st.selectbox("Sort By Metric", ["lift", "confidence", "support"], index=0, key="exp_sort")
    with c4:
        sort_order = st.selectbox("Sort Direction", ["Descending", "Ascending"], index=0, key="exp_dir")

    filtered_rules = mba_service.filter_association_rules(
        rules_df,
        antecedent=ant_input,
        consequent=cons_input,
        category=filters.get("categories"),
        min_support=filters.get("min_support", 0.0),
        min_confidence=filters.get("min_confidence", 0.0),
        min_lift=filters.get("min_lift", 1.0),
        sort_by=sort_field,
        ascending=(sort_order == "Ascending")
    )

    st.markdown(f"Showing **{len(filtered_rules)}** Association Rules matching criteria")

    ant_col = "antecedents_str" if "antecedents_str" in filtered_rules.columns else "antecedents"
    cons_col = "consequents_str" if "consequents_str" in filtered_rules.columns else "consequents"
    disp_cols = [c for c in [ant_col, cons_col, "support", "confidence", "lift", "leverage", "conviction"] if c in filtered_rules.columns]
    
    st.dataframe(filtered_rules[disp_cols].head(100), use_container_width=True)

    # Plotly Scatter Plot (Support vs Confidence; size & color = Lift)
    c_scat, c_bar = st.columns(2)

    with c_scat:
        fig_scatter = px.scatter(
            filtered_rules.head(50),
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_data=[ant_col, cons_col] if ant_col in filtered_rules.columns else None,
            color_continuous_scale="Viridis",
            title="Support vs. Confidence Scatter Space (Size/Color = Lift)"
        )
        apply_dark_theme(fig_scatter, "Support vs. Confidence Scatter Space")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c_bar:
        top_10 = filtered_rules.head(10).copy()
        if ant_col in top_10.columns and cons_col in top_10.columns:
            top_10["rule_label"] = top_10[ant_col].astype(str) + " → " + top_10[cons_col].astype(str)
        else:
            top_10["rule_label"] = "Rule " + top_10.index.astype(str)

        fig_bar = px.bar(
            top_10,
            x="lift",
            y="rule_label",
            orientation="h",
            color="lift",
            color_continuous_scale="Teal",
            title="Top 10 Association Rules by Lift Score"
        )
        apply_dark_theme(fig_bar, "Top 10 Association Rules by Lift Score")
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_association_network_graph(datasets: dict, filters: dict):
    """Renders Product Association Network Graph using Plotly node-edge chart."""
    rules_df = datasets.get("association_rules", pd.DataFrame())
    if rules_df.empty:
        return

    filtered_rules = mba_service.filter_association_rules(
        rules_df,
        min_lift=filters.get("min_lift", 1.2),
        min_confidence=filters.get("min_confidence", 0.0)
    )

    graph_data = mba_service.get_association_network_graph(filtered_rules, top_n=25)
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes or not edges:
        st.info("No association network nodes found matching current filters.")
        return

    # Position nodes in a circular layout
    N = len(nodes)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    node_positions = {nodes[i]["id"]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

    edge_x = []
    edge_y = []
    for edge in edges:
        if edge["source"] in node_positions and edge["target"] in node_positions:
            x0, y0 = node_positions[edge["source"]]
            x1, y1 = node_positions[edge["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(56, 189, 248, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = [node_positions[n["id"]][0] for n in nodes]
    node_y = [node_positions[n["id"]][1] for n in nodes]
    node_text = [n["id"] for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Viridis",
            color=[i for i in range(N)],
            size=18,
            line_width=2,
            line_color="#38bdf8"
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text="<b>Product Association Network Topology Graph</b>", font=dict(color="#f8fafc", size=15)),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_product_bundles_section(datasets: dict):
    """Renders Product Bundle Analysis expander cards."""
    bundles_df = datasets.get("product_bundles", pd.DataFrame())
    if bundles_df.empty:
        st.info("Product bundle recommendations loading...")
        return

    st.markdown(f"Found **{len(bundles_df)}** High-Value Product Bundles")

    for i, (_, b) in enumerate(bundles_df.head(10).iterrows(), 1):
        b_name = b.get("bundle_name", f"Bundle #{i}")
        price = float(b.get("estimated_bundle_price", 49.99))
        rev = float(b.get("projected_revenue_potential", 5000.0))
        lift = float(b.get("lift_score", 2.5))
        conf = float(b.get("confidence_pct", 50.0))
        strat = str(b.get("merchandising_strategy", "Promote bundle at checkout."))

        with st.expander(f"📦 BUNDLE #{i}: {b_name} — Price: ${price:.2f} (Lift: {lift:.2f})", expanded=(i==1)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Primary Category:** `{b.get('primary_category', 'Category A')}`")
                st.markdown(f"**Add-on Category:** `{b.get('addon_category', 'Category B')}`")
            with c2:
                st.markdown(f"**Conversion Confidence:** `{conf:.1f}%`")
                st.markdown(f"**Lift Score:** `{lift:.2f}x`")
            with c3:
                st.markdown(f"💰 **Projected Revenue Potential:** `${rev:,.2f}`")
                st.markdown(f"⭐ **Priority:** `P1 - High Opportunity`")
            st.info(f"💡 **Merchandising Strategy:** {strat}")


def _render_cross_sell_section(datasets: dict):
    """Renders Cross-Sell Opportunities engine section."""
    cross_df = datasets.get("cross_sell_recommendations", pd.DataFrame())
    if cross_df.empty:
        st.info("Cross-sell triggers loading...")
        return

    st.dataframe(cross_df.head(25), use_container_width=True)


def _render_segment_basket_section(datasets: dict):
    """Renders Customer Segment Basket Analysis comparison."""
    seg_df = mba_service.get_customer_segment_basket_analysis(datasets)
    st.dataframe(seg_df, use_container_width=True)


def _render_category_heatmap_section(datasets: dict):
    """Renders Category Co-occurrence Heatmap using Plotly."""
    master_df = datasets.get("master_dataset", pd.DataFrame())
    co_matrix = mba_service.get_category_cooccurrence_matrix(master_df, top_n=10)

    if co_matrix.empty:
        st.info("Category co-occurrence matrix telemetry loading...")
        return

    fig_heat = px.imshow(
        co_matrix,
        text_auto=True,
        color_continuous_scale="Teal",
        title="Top 10 Category Order Co-occurrence Frequency"
    )
    apply_dark_theme(fig_heat, "Category Order Co-occurrence Frequency Matrix")
    st.plotly_chart(fig_heat, use_container_width=True)


def _render_seasonal_basket_section(seasonal_data: dict):
    """Renders Seasonal Basket Telemetry trends."""
    monthly_df = seasonal_data.get("monthly_trend", pd.DataFrame())
    if monthly_df.empty:
        return

    fig_line = px.line(
        monthly_df,
        x="year_month",
        y="avg_items_per_order",
        markers=True,
        title="Monthly Average Basket Size (Items per Order)",
        color_discrete_sequence=[COLOR_PRIMARY]
    )
    apply_dark_theme(fig_line, "Monthly Average Basket Size (Items per Order)")
    st.plotly_chart(fig_line, use_container_width=True)


def _render_business_recommendations_section(datasets: dict):
    """Renders evidence-based merchandising recommendations card grid."""
    recs = mba_service.get_business_recommendations(datasets)

    c1, c2 = st.columns(2)
    for i, r in enumerate(recs):
        target_col = c1 if i % 2 == 0 else c2
        with target_col:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 12px; border-left: 4px solid #34d399;">
                    <div style="font-size: 11px; color: #34d399; font-weight: 700;">{r['type'].upper()}</div>
                    <div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-top: 4px;">{r['title']}</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;"><b>Rationale:</b> {r['rationale']}</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;"><b>Action:</b> {r['action']}</div>
                    <div style="font-size: 12px; color: #38bdf8; font-weight: 700; margin-top: 6px;">📈 {r['estimated_impact']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_export_hub_section(datasets: dict, filters: dict):
    """Renders export hub controls."""
    rules_df = datasets.get("association_rules", pd.DataFrame())
    bundles_df = datasets.get("product_bundles", pd.DataFrame())
    cross_df = datasets.get("cross_sell_recommendations", pd.DataFrame())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 📄 Export Association Rules")
        if not rules_df.empty:
            render_export_buttons(rules_df, "association_rules_export")

    with c2:
        st.markdown("##### 📄 Export Product Bundles")
        if not bundles_df.empty:
            render_export_buttons(bundles_df, "product_bundles_export")

    with c3:
        st.markdown("##### 📄 Export Cross-Sell Opportunities")
        if not cross_df.empty:
            render_export_buttons(cross_df, "cross_sell_export")
