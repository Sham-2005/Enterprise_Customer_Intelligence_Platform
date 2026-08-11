"""
Root Streamlit Entrypoint for Cloud Deployments (Streamlit Community Cloud / HuggingFace / Railway).
Routes execution directly to the ECIP Dashboard main application.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dashboard.app import main

if __name__ == "__main__":
    main()
