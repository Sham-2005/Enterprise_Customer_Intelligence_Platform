# ECIP Installation & Local Development Setup Guide

## Prerequisites
- Python 3.12+
- Git
- Docker & Docker Compose (Optional for containerized deployment)

---

## Step-by-Step Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/enterprise/ecip-platform.git
   cd ecip-platform
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   ```bash
   cp .env.example .env
   ```

5. **Run Data & Pipeline Orchestration**:
   ```bash
   python run_pipeline.py
   python run_segmentation.py
   python run_churn.py
   python run_clv.py
   python run_recommendations.py
   python run_mba.py
   python run_mlops.py
   ```

6. **Launch Applications**:
   - Backend API: `python -m uvicorn api.app:app --reload`
   - BI Dashboard: `python run_dashboard.py`
