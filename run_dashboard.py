"""
Launcher script for ECIP Streamlit BI Dashboard.
Runs: streamlit run dashboard/app.py
"""

import sys
import subprocess
from pathlib import Path

import os

def main():
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "dashboard" / "app.py"

    print(f"Launching ECIP Streamlit Dashboard from {app_path}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless=true"]
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main()
