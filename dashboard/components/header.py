"""
Top Navigation Header Bar for ECIP Enterprise Dashboard.
Displays Platform Logo, Project Title, Date & Time.
"""

from datetime import datetime
import streamlit as st

def render_top_header(current_page_title: str = "Executive Dashboard"):
    """Renders executive glassmorphic top header bar."""
    now_str = datetime.now().strftime("%A, %b %d, %Y | %H:%M UTC")

    header_html = f"""
    <div class="glass-header">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 24px; font-weight: 900; background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚡ ECIP PLATFORM
            </div>
            <div style="color: #475569; font-size: 18px;">|</div>
            <div style="font-size: 16px; font-weight: 600; color: #cbd5e1;">
                {current_page_title}
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="font-size: 12px; color: #94a3b8; background: rgba(30, 41, 59, 0.8); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.08);">
                🕒 {now_str}
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
