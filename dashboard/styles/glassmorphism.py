"""
Glassmorphism Enterprise CSS Design System for ECIP Streamlit Dashboard.
Injects custom backdrop-filter blurs, neon accents, dark theme card styles, and animations.
"""

import streamlit as st

def apply_glassmorphism_theme():
    """Injects high-end glassmorphism CSS rules into Streamlit DOM."""
    css_styles = """
    <style>
        /* Global Background & Dark Theme */
        .stApp {
            background: radial-gradient(circle at 10% 20%, #0f172a 0%, #0b0f19 90%);
            color: #f8fafc;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        /* Hide default Streamlit Header & Footer */
        header { visibility: hidden; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }

        /* Glassmorphic Container Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.65) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 22px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
            transition: all 0.3s ease-in-out !important;
            margin-bottom: 18px !important;
        }

        .glass-card:hover {
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15) !important;
            transform: translateY(-2px) !important;
        }

        /* Glassmorphic Top Header Bar */
        .glass-header {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 14px 28px;
            border-radius: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }

        /* Metric & KPI Cards */
        .kpi-title {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
        }

        .kpi-value {
            font-size: 30px;
            font-weight: 800;
            color: #ffffff;
            margin-top: 6px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-positive {
            background: rgba(52, 211, 153, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
        }

        .badge-negative {
            background: rgba(248, 113, 113, 0.15);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
        }

        .badge-cyan {
            background: rgba(6, 182, 212, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
        }

        .badge-purple {
            background: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 700;
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background-color: rgba(11, 15, 25, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* Styled Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 10px !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        .stButton>button:hover {
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.4) !important;
            transform: scale(1.02) !important;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #0284c7;
        }
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)
