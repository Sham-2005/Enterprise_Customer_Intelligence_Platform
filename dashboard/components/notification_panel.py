"""
Notification Panel Component for ECIP Enterprise Dashboard.
Displays interactive alert feed, model drift notifications, and retention triggers.
"""

import streamlit as st

def render_notification_panel():
    """Renders notification feed card."""
    html = """
    <div class="glass-card">
        <div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 12px;">🔔 Active Enterprise Notifications</div>
        
        <div style="display: flex; gap: 12px; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px; margin-bottom: 10px;">
            <div style="background: rgba(248, 113, 113, 0.2); color: #f87171; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">⚠️</div>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: #f8fafc;">High Risk Customer Spike</div>
                <div style="font-size: 11px; color: #94a3b8;">14 VIP Customers flagged with >85% Churn Probability in last batch.</div>
            </div>
        </div>

        <div style="display: flex; gap: 12px; align-items: flex-start; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px; margin-bottom: 10px;">
            <div style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">ℹ️</div>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: #f8fafc;">Model Version Promoted</div>
                <div style="font-size: 11px; color: #94a3b8;">ChurnClassifier v2.0 promoted to Active in Model Registry.</div>
            </div>
        </div>

        <div style="display: flex; gap: 12px; align-items: flex-start;">
            <div style="background: rgba(52, 211, 153, 0.2); color: #34d399; border-radius: 50%; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">✅</div>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: #f8fafc;">FP-Growth Mining Complete</div>
                <div style="font-size: 11px; color: #94a3b8;">Discovered 142 association rules with average lift of 2.45.</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
