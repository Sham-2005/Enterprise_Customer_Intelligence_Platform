# Enterprise Customer Intelligence Platform (ECIP)

![ECIP License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
App Link: https://enterprisecustomerintelligenceplatform-nbbqhzaxke7erve6dw5rqs.streamlit.app/
**Enterprise Customer Intelligence Platform (ECIP)** is a production-grade, AI-powered customer analytics, lifetime value forecasting, churn prediction, recommendation engine, market basket analysis, and MLOps governance platform built for retail and e-commerce enterprises (similar to Amazon, Shopify, Flipkart, and Walmart).

---

## 🌟 Key Features

* **Customer Analytics & RFM Segmentation**: Computes Recency, Frequency, and Monetary quintiles (1-5) and automatically maps 10+ business RFM segments and AI personas (*VIP Power Buyers*, *Loyal Frequenters*, *At-Risk High Rollers*).
* **AI Churn Risk Intelligence**: Multi-model classification benchmark (**XGBoost**, **LightGBM**, **Random Forest**, **Logistic Regression**) with SMOTE class imbalance handling, 5-tier risk stratification (*Very Low* to *Critical*), SHAP Explainable AI (XAI) interpretations, and automated retention recommendations.
* **Customer Lifetime Value (CLV) Forecasting**: 12-month future CLV regression forecasting, value tier classification (*Platinum*, *Gold*, *Silver*, *Bronze*), and automated revenue growth opportunity identification.
* **Hybrid Recommendation Engine**: Blends Item-Item Collaborative Filtering (Cosine similarity), Content-Based Metadata Filtering (TF-IDF), and Popularity-based Cold-Start fallbacks to serve personalized product recommendations with plain-English explanations.
* **Market Basket Analysis & Association Rules**: Apriori and FP-Growth frequent itemset mining, Support/Confidence/Lift rule evaluation, high-value product bundle generation, and merchandising strategy recommendations.
* **Enterprise MLOps & AI Governance**: Centralized Model Registry, semantic version control (`v1.0`, `v2.0`), experiment tracking logs, Kolmogorov-Smirnov (KS) Data & Concept Drift detection, inference latency monitoring, and audit logging.
* **Executive BI Dashboard**: 14-page interactive Streamlit dashboard featuring custom dark-mode glassmorphic KPI cards, Plotly 2D/3D visualizations, entity lookup search, and CSV/Excel exports.
* **Production REST API**: FastAPI backend providing OpenAPI/Swagger documentation, JWT Authentication, RBAC, health endpoints, and real-time inference routes.

---

## 🏗️ System Architecture Diagram

```
+-----------------------------------------------------------------------+
|                         BUSINESS USERS & CLIENTS                      |
+-----------------------------------------------------------------------+
        |                                                 |
        v                                                 v
+-----------------------+                         +---------------------+
| Streamlit BI Dashboard|                         | REST API Clients    |
| (14 Analytics Pages)  |                         | (FastAPI Swagger)   |
+-----------------------+                         +---------------------+
        |                                                 |
        +-------------------------+-----------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                            FASTAPI BACKEND                            |
|  - JWT Authentication & RBAC        - Global Exception Handling       |
|  - Request/Response Validation     - Async Background Processing     |
+-----------------------------------------------------------------------+
        |                                                 |
        v                                                 v
+-----------------------+                         +---------------------+
| BUSINESS ANALYTICS    |                         | ML PREDICTIVE ENGINE|
| - Data Engineering    |                         | - XGBoost / LightGBM|
| - RFM Segmentation    |                         | - SHAP Explainer    |
| - Market Basket (FP)  |                         | - Hybrid Recommender|
+-----------------------+                         +---------------------+
        |                                                 |
        +-------------------------+-----------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                     ENTERPRISE MLOPS & GOVERNANCE                     |
|  - Model Registry  - Experiment Tracker  - KS Drift Audit  - DB ORM   |
+-----------------------------------------------------------------------+
```

---

## 📂 Project Directory Structure

```
Coustomer churn prediction/
├── config/                  # Configuration Manager & Settings
│   ├── config.yaml          # System parameters & paths
│   └── settings.py          # Dynamic configuration loader
├── backend/                 # Core Business & Machine Learning Logic
│   ├── data/                # Ingestion, Schema Validator, Merger, Feature Engineer
│   ├── analytics/           # RFM, Personas, Market Basket Engine
│   ├── models/              # Churn Classifier, CLV Regressor, Recommender, Risk Engine
│   ├── explainability/      # SHAP Explainability Engine
│   ├── mlops/               # Model Registry, Experiment Tracker, Drift Detector
│   └── db/                  # SQLAlchemy ORM Database Models
├── api/                     # FastAPI Production REST API
│   ├── app.py               # Application entrypoint
│   ├── routes/              # Auth, Churn, CLV, Recommendations, MBA, MLOps routes
│   └── schemas/             # Pydantic DTO contracts
├── dashboard/               # Streamlit Executive BI Dashboard
│   ├── app.py               # Dashboard entrypoint
│   ├── components/          # Reusable KPI cards, Plotly charts, Filters
│   ├── pages/               # 14 Modular Analytics & AI Pages
│   └── utils/               # Data Loader & Export helpers
├── tests/                   # Pytest Automated Test Suite
├── output/                  # Data Warehouse outputs, Models, Reports, Logs
├── run_pipeline.py          # Data Engineering entrypoint
├── run_segmentation.py      # AI Segmentation entrypoint
├── run_churn.py             # Churn Model & SHAP entrypoint
├── run_clv.py               # CLV Regression entrypoint
├── run_recommendations.py   # Hybrid Recommender entrypoint
├── run_mba.py               # Market Basket Analysis entrypoint
├── run_mlops.py             # MLOps & Governance entrypoint
├── run_dashboard.py         # Streamlit launcher script
├── Dockerfile               # Multi-stage container build
├── docker-compose.yml       # Service orchestrator
├── requirements.txt         # Pinned python dependencies
├── README.md                # System Overview
└── SYSTEM_ARCHITECTURE.md   # Architectural Specifications
```

---

## ⚡ Quick Start & Installation

### Option 1: Local Development Setup

1. **Clone & Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Pipeline Stages**:
   ```bash
   python run_pipeline.py         # Execute ETL & Feature Engineering
   python run_segmentation.py     # Execute AI Customer Segmentation
   python run_churn.py            # Train Churn Model & SHAP Explainer
   python run_clv.py              # Train CLV Regressor & Forecasts
   python run_recommendations.py  # Fit Hybrid Recommender System
   python run_mba.py              # Execute Market Basket Analysis
   python run_mlops.py            # Register Models & Run Drift Audit
   ```

3. **Launch FastAPI Backend REST API**:
   ```bash
   python -m uvicorn api.app:app --reload --port 8000
   ```
   * Access Interactive Swagger API Docs: `http://localhost:8000/docs`

4. **Launch BI Dashboard**:
   ```bash
   python run_dashboard.py
   ```
   * Access Streamlit Dashboard: `http://localhost:8501`

---

### Option 2: Docker Container Deployment

```bash
docker-compose up --build
```
* Backend API: `http://localhost:8000`
* BI Dashboard: `http://localhost:8501`

---

## 📊 Phase 11 – Executive Dashboard Backend Integration

The **Executive Dashboard** is powered by a high-performance backend infrastructure designed for enterprise business intelligence, real-time KPI tracking, interactive charts, multi-dimensional filtering, global search, and automated reports export.

### 🔄 Dashboard Data Flow

```
[Processed Datasets / Data Warehouse]
   (master_dataset.csv, customer_metrics.csv, feature_store.csv, predictions)
                            │
                            ▼
                    [DataService]
          (Auto-detection & Graceful Fallback)
                            │
                            ▼
                   [DashboardCache]
        (Thread-safe TTL In-Memory Caching)
                            │
                            ▼
                   [FilterService]
   (Date Range, State, Category, Seller, Payment, Segment)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      [KPIService]  [AnalyticsService] [SearchService]
       (8 KPIs +     (Trend, Growth,    (Customer, Order,
       Baselines)     Treemap, Maps)    Seller, Product)
            │               │               │
            └───────────────┼───────────────┘
                            ▼
              [ExecutiveDashboardBackend]
                   (Unified Facade)
                            │
                            ▼
              [Executive Dashboard UI Page]
              (Streamlit Dark Glassmorphism)
```

### 🧩 Backend Services Architecture

* **`DataService` (`backend/services/data_service.py`)**: Discovers dataset availability, parses date schemas, and provides graceful fallbacks if output CSV files are absent.
* **`FilterService` (`backend/services/filter_service.py`)**: Multi-criteria query engine supporting Date Range, State, Product Category, Seller ID, Payment Method, and Customer Segment.
* **`KPIService` (`backend/services/kpi_service.py`)**: Calculates 8 enterprise C-suite metrics:
  1. *Total Revenue*: Gross product revenue and order volume.
  2. *Total Orders*: Unique customer transaction count.
  3. *Total Customers*: Unique buyer account volume.
  4. *Average Order Value (AOV)*: Mean revenue per fulfilled order.
  5. *Average Customer Rating*: Customer satisfaction (CSAT) 5-star rating baseline.
  6. *Customer Retention Rate (%)*: Percentage of active non-churned repeat purchasers.
  7. *Monthly Revenue Growth (%)*: Month-over-month sales growth trajectory.
  8. *Business Health Score*: Composite 0-100 index weighting Retention (35%), Review CSAT (35%), and Growth (30%).
  - Each KPI calculates *Current Value*, *Previous Period Baseline*, *Percentage Change*, *Trend Arrow (↑/↓)*, and *Timestamp*.
* **`AnalyticsService` (`backend/services/analytics_service.py`)**: Generates structured datasets for Plotly charts:
  - *Revenue Trend*: Monthly, Weekly, and Daily aggregations with spline curve and area fill.
  - *Customer Growth*: Monthly active customer count breakdown (New vs. Returning).
  - *Revenue by Category*: Hierarchical Treemap structure.
  - *Revenue by State*: Geospatial Map of Brazilian states with latitude/longitude centroids.
  - *Payment Method Distribution*: Donut chart breakdown.
  - *Order Status*: Delivery and fulfillment stage distribution.
  - *Top Selling Products*: Top 10 SKUs by revenue horizontal bar chart.
  - *Customer Ratings*: 1 to 5 star rating histogram.
  - *Recent Business Insights*: Automated cards highlighting top revenue category, best state, fastest growing segment, top payment method, top seller, and lowest rated category.
  - *Executive Summary*: Dynamic textual business overview synthesising live calculations.
* **`SearchService` (`backend/services/search_service.py`)**: Universal search engine scanning Customer ID, Order ID, Seller ID, and Product ID to return entity profiles and transaction histories.
* **`ExportService` (`backend/services/export_service.py`)**: Multi-format report export pipeline generating downloadable CSV, Excel (`.xlsx`), and PDF executive reports.
* **`DashboardCache` (`backend/cache/dashboard_cache.py`)**: Thread-safe TTL memory cache with parameter hash generation, cache invalidation, and efficiency stats.
* **`ExecutiveDashboardBackend` (`backend/dashboard/executive_backend.py`)**: Central facade orchestrator.

---

## 👥 Phase 12 – Customer Analytics Module

The **Customer Analytics Module** delivers comprehensive deep-dive intelligence into customer purchasing behavior, demographics, loyalty tiers, revenue contribution (Pareto 80/20 analysis), and recency activity.

### 🧩 Customer Analytics Backend Architecture

```
[Processed Datasets: master_dataset.csv, feature_store.csv, customer_metrics.csv]
                                     │
                                     ▼
                      [CustomerAnalyticsService]
         (Data Ingestion, Multi-Dimensional Filters & Caching)
                                     │
   ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
   ▼               ▼                 ▼                 ▼               ▼
[CustomerKPI] [Demographics]    [Behavior]        [Loyalty]     [RevenueContribution]
 (10 KPIs +   (State/City Maps, (Frequency, AOV,  (VIP/Loyal    (Pareto 80/20,
 Baselines)    Treemaps)         Basket Size)      Histogram)    Top 20 Leaderboard)
   │               │                 │                 │               │
   └───────────────┴─────────────────┼─────────────────┴───────────────┘
                                     ▼
                       [Customer Search & Activity]
                     (Rosters, Recency, Profile Lookups)
                                     │
                                     ▼
                      [Customer Analytics UI Page]
```

### 📊 10 Core Customer KPIs
1. **Total Customers**: Total unique registered buyer account volume.
2. **Active Customers**: Buyers with purchase activity within the last 90 days.
3. **Returning Customers**: Accounts with 2 or more fulfilled orders.
4. **New Customers**: First-time buyer acquisitions in the selected period.
5. **Repeat Purchase Rate (%)**: Percentage of total buyers who place repeat orders.
6. **Average Customer Lifetime Value ($)**: Mean historical/predicted 12-month customer spending.
7. **Average Customer Rating (CSAT)**: Average 1 to 5 star feedback rating provided by customers.
8. **Customer Retention Rate (%)**: Percentage of non-churned active customer accounts.
9. **Average Purchase Frequency**: Mean order count per customer account.
10. **Average Basket Size**: Average order monetary value ($) and item count per transaction.

- *Each KPI includes Current Value, Previous Period Baseline, Percentage Change, Trend Indicator (↑/↓), and Timestamp.*

### 🔍 Customer Profile Search
Allows searching by **Customer ID**, **City**, or **State**. Displays a dedicated Profile Card showing:
- Total Orders & Total Spending
- Average Order Value & Last Purchase Date
- Customer Loyalty Segment & Churn Risk Level
- Predicted 12-Month CLV & Preferred Product Categories
- Preferred Payment Method & Transaction History Roster

### 💡 Core Feature Sections
* **Customer Overview**: Growth trend, new vs. returning stacked trends, active vs. inactive ratio.
* **Customer Demographics**: State geospatial filled map, top cities bar chart, state → city treemap.
* **Purchasing Behavior**: Order frequency distribution, product diversity levels, preferred payment methods, spending quantile tiers.
* **Loyalty Analysis**: Stratification into *VIP Customers*, *High-Value Customers*, *Loyal Customers*, *Occasional Customers*, and *One-Time Buyers*. Loyalty score index histogram (0-100).
* **Revenue Contribution (Pareto 80/20)**: Cumulative revenue curve vs. customer percentile, top 20 customer leaderboard, quantile revenue share.
* **Customer Activity**: Recency distribution buckets, top 50 recently active roster, dormant customers roster (>90 days inactive).
* **Automated Insights**: 6 real-time insight cards.
* **Multi-Format Exports**: CSV, Excel, and PDF downloads.

---

## 🎯 Phase 13 – Customer Segmentation & RFM Intelligence Module

The **Customer Segmentation & RFM Intelligence Module** transforms machine learning clustering models, 2D/3D PCA dimensionality reduction, and RFM analytics into actionable business personas and marketing blueprints.

### 🧩 Segmentation Backend Architecture

```
[Processed Datasets: customer_segments.csv, rfm_scores.csv, feature_store.csv]
                                     │
                                     ▼
                           [SegmentationService]
         (Data Ingestion, Multi-Dimensional Filters & Caching)
                                     │
   ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
   ▼               ▼                 ▼                 ▼               ▼
[SegmentationKPI] [ClusterExplorer]  [RFMDashboard]    [PersonaManager] [MarketingIntelligence]
 (8 KPIs +        (PCA 2D/3D,       (R/F/M Quintiles,  (8 Business      (6 Actionable
 Baselines)        Comparison Matrix) Heatmap Matrix)   Personas)        Campaign Cards)
   │               │                 │                 │               │
   └───────────────┴─────────────────┼─────────────────┴───────────────┘
                                     ▼
                   [Segmentation Search & Profile Lookups]
                 (Customer ID, Cluster Name, Persona Search)
                                     │
                                     ▼
                  [Segmentation & RFM Intelligence UI Page]
```

### 📊 8 Core Segmentation KPIs
1. **Total Customer Segments**: Number of active machine learning clusters / RFM segments.
2. **Total Customers Clustered**: Total buyer accounts evaluated and categorized.
3. **VIP Customers**: High-value account volume in the top revenue cluster (*Champions* / *VIP Power Buyers*).
4. **Loyal Customers**: Account volume in the repeat purchasing cluster (*Loyal Frequent Buyers*).
5. **At-Risk Customers**: Account volume in the churn-prone cluster (*At-Risk High Rollers* / *Need Attention*).
6. **Average Cluster Revenue ($)**: Mean gross sales volume generated per customer cluster.
7. **Average RFM Score**: Composite Recency, Frequency, Monetary 1-5 quintile score index.
8. **Largest Customer Segment**: Name and account volume of the largest customer segment.

- *Each KPI includes Current Value, Previous Period Baseline, Percentage Change, Trend Indicator (↑/↓), and Timestamp.*

### 🔍 Segmentation Search Engine
Allows searching by **Customer ID**, **Cluster Name**, or **Persona**. Displays:
- Customer / Cluster Details & Profile Summary
- Orders, Spending, RFM Quintiles, Historical CLV, and Churn Risk Level
- Automated Marketing Campaign Blueprint & Retention Recommendations
- Roster of sample customers within the segment

### 💡 Core Feature Sections
* **Cluster Overview**: Customer share donut chart, cluster revenue share treemap, average spending/orders/recency bar charts.
* **Interactive Cluster Explorer**: Interactive selector dropdown displaying customer count, revenue, AOV, CSAT, CLV, churn risk, top categories, payment methods, and top states for any cluster.
* **Business Personas Cards**: 8 business personas (*VIP Power Buyers*, *Loyal Frequent Buyers*, *Premium Customers*, *New Customers*, *Occasional Buyers*, *Price Sensitive Customers*, *At-Risk Customers*, *Lost Customers*) with descriptions, revenue contribution, buying behavior, marketing recommendations, and retention strategies.
* **RFM Dashboard**: R, F, M 1-5 quintile distributions, 2D RFM Heatmap matrix (Recency vs. Frequency), RFM score histogram, and 8 RFM segment classifications.
* **PCA Dimensionality Reduction Plots**: Interactive 2D PCA cluster projection plot and 3D PCA scatter plot (PC1 vs PC2 vs PC3) with zoom, hover, and filter.
* **Cluster Comparison Matrix**: Side-by-side comparison across Revenue, Orders, Spending, CLV, Churn Risk, Loyalty Index, and Basket Size.
* **Marketing Intelligence**: 6 automated recommendation cards (*Best Customers to Reward*, *Customers Ready for Upsell*, *Customers Suitable for Cross-Sell*, *High-Risk Customers*, *Discount Campaign Targets*, *Loyalty Program Candidates*).
* **Multi-Format Exports**: CSV, Excel, and PDF downloads.

---

## 🚨 Phase 14 – AI Customer Churn Prediction & Risk Intelligence Dashboard

The **AI Customer Churn Prediction & Risk Intelligence Dashboard** delivers real-time machine learning churn scoring, 5-tier risk stratification, SHAP Explainable AI (XAI) feature attributions, personalized retention blueprints, and batch CSV prediction capabilities.

### 🧩 Churn Dashboard Backend Architecture

```
[Processed Datasets: churn_predictions.csv, feature_store.csv, customer_metrics.csv]
                                     │
                                     ▼
                              [ChurnService]
         (Data Ingestion, Multi-Dimensional Filters & Caching)
                                     │
   ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
   ▼               ▼                 ▼                 ▼               ▼
[ChurnKPIEngine] [RiskClassifier]    [Explainability]  [RetentionIntel] [BatchPredictor]
 (8 KPIs +        (5 Risk Tiers:     (SHAP Attributions (8 Personalized  (Bulk CSV Scoring,
 Baselines)        Very Low→Critical) & Narratives)     Campaign Cards) Download Predictions)
   │               │                 │                 │               │
   └───────────────┴─────────────────┼─────────────────┴───────────────┘
                                     ▼
                      [Churn Search & Account Lookups]
                   (Customer ID, Customer Name, Email Search)
                                     │
                                     ▼
                [AI Churn Prediction & Risk Intelligence UI Page]
```

### 📊 8 Core Churn KPIs
1. **Total Customers**: Total registered buyer accounts evaluated for churn risk.
2. **High-Risk Customers**: Count of accounts in the 60%–80% churn probability tier.
3. **Critical-Risk Customers**: Count of accounts in the 80%–100% churn probability tier.
4. **Average Churn Probability (%)**: System-wide mean predicted probability of churn.
5. **Predicted Churn Rate (%)**: Percentage of customer base with churn probability ≥ 50%.
6. **Retention Success Estimate (%)**: Estimated percentage of high-risk customers recoverable via retention campaigns.
7. **Average Customer Lifetime Value ($)**: Mean 12-month CLV across high-risk accounts.
8. **Estimated Revenue at Risk ($)**: Total sum of spending / CLV across High and Critical risk accounts.

- *Each KPI includes Current Value, Previous Period Baseline, Percentage Change, Trend Indicator (↑/↓), and Timestamp.*

### 🔍 Churn Search Engine
Allows searching by **Customer ID**, **Customer Name**, or **Email**. Displays:
- Complete Customer Profile, Orders, Spending, Segment, and Predicted CLV
- Predicted Churn Probability & 5-Tier Risk Stratification
- SHAP Feature Attributions (Top Positive & Negative Risk Drivers)
- Plain-English Business Explanation Narrative & Targeted Retention Plan
- Interactive Purchase Timeline & Risk Trajectory Curve

### 💡 Core Feature Sections
* **Churn Overview**: Monthly churn trend line chart, risk level donut chart, revenue at risk treemap, churn by state map/bar.
* **High-Risk Customer Explorer**: Interactive dropdown selector providing deep-dive metrics (Orders, Spending, Recency, CSAT, Preferred Payment) for high-risk accounts.
* **Explainable AI (XAI)**: Global SHAP feature importance ranking and local customer attributions explaining *why* a customer is likely to churn.
* **Risk Classification**: 5 Risk Tiers (*Very Low Risk*, *Low Risk*, *Medium Risk*, *High Risk*, *Critical Risk*) distribution and revenue volume at risk.
* **Retention Intelligence**: Personalized campaign cards (*Win-back Campaign*, *Phone Follow-up*, *Personalized Discount*, *Email Campaign*, *Loyalty Reward*, *VIP Upgrade*, *Free Shipping*, *Product Recommendation*) with Priority, Estimated Impact, Expected Revenue Saved ($), and Confidence Score.
* **Customer Timeline**: Purchase history events timeline and historical churn risk trajectory curve.
* **Batch CSV Prediction Engine**: Upload raw customer CSV files, run bulk machine learning scoring, preview results table, and download predictions CSV.
* **Automated Insights**: 5 real-time strategic insight cards.
* **Multi-Format Exports**: CSV, Excel, and PDF report downloads.

---

## 💎 Phase 15 – Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard

The **Customer Lifetime Value (CLV) & Revenue Intelligence Dashboard** forecasts 12-month customer monetary projections, stratifies buyers into 5 value tiers (*Platinum*, *Gold*, *Silver*, *Bronze*, *Standard*), computes multi-horizon revenue forecast curves, delivers SHAP Explainable AI (XAI) feature attributions, and provides strategic opportunity intelligence.

### 🧩 CLV Dashboard Backend Architecture

```
[Processed Datasets: clv_predictions.csv, feature_store.csv, master_dataset.csv]
                                     │
                                     ▼
                               [CLVService]
         (Data Ingestion, Multi-Dimensional Filters & Caching)
                                     │
   ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
   ▼               ▼                 ▼                 ▼               ▼
[CLVKPIEngine]  [ValueClassifier]   [CLVExplainability] [OpportunityIntel] [RevenueForecast]
 (8 KPIs +       (5 Value Tiers:    (SHAP Attributions  (6 Strategic       (Monthly, Quarterly,
 Baselines)       Platinum→Standard) & Summaries)       Expansion Cards)   Annual Horizons)
   │               │                 │                 │               │
   └───────────────┴─────────────────┼─────────────────┴───────────────┘
                                     ▼
                       [CLV Search & Account Lookups]
                   (Customer ID, Customer Name, Segment Search)
                                     │
                                     ▼
                [Customer Lifetime Value & Revenue Intelligence UI Page]
```

### 📊 8 Core CLV KPIs
1. **Total Predicted Customer Lifetime Value ($)**: Sum of 12-month predicted CLV across active accounts.
2. **Average Customer Lifetime Value ($)**: Mean 12-month predicted monetary value per customer.
3. **Highest Value Customer ($)**: Maximum individual account predicted lifetime monetary value.
4. **High-Value Customers**: Account count in Gold and Platinum tiers (CLV ≥ $1,000).
5. **Platinum Customers**: Account count in top elite tier (CLV ≥ $2,500).
6. **Expected Revenue (12 Months) ($)**: Projected gross revenue over next 12 months.
7. **Average Revenue per Customer ($)**: Mean historical gross spending per customer.
8. **Revenue Growth Potential (%)**: Estimated revenue expansion rate from upselling and cross-selling.

- *Each KPI includes Current Value, Previous Period Baseline, Percentage Change, Trend Indicator (↑/↓), and Timestamp.*

### 🔍 CLV Search Engine
Allows searching by **Customer ID**, **Customer Name**, or **Customer Segment**. Displays:
- Complete Customer Profile, Spending, Orders, Loyalty Score, and Value Tier
- Predicted 12-Month CLV & Churn Risk Level
- SHAP Regression Feature Attributions (Top Positive & Drag Drivers)
- Plain-English Business Summary Narrative & Actionable Upsell/VIP Strategy
- Roster of sample accounts within the segment

### 💡 Core Feature Sections
* **CLV Overview**: Value tier revenue contribution treemap, 12-month monthly revenue forecast trend line chart.
* **Customer Value Explorer**: Interactive dropdown selector displaying deep-dive metrics (Predicted CLV, Tier, Spending, Orders, Loyalty, CSAT, Payment) for top accounts.
* **Customer Value Classification**: 5 Value Tiers (*Platinum*, *Gold*, *Silver*, *Bronze*, *Standard*) matrix detailing account counts, total revenue, revenue share %, average spend, and retention rate.
* **Revenue Intelligence**: Top 100 High-Value Customers Leaderboard and Pareto (80/20) cumulative revenue concentration curve.
* **Opportunity Intelligence**: 6 automated recommendation cards (*Upsell Candidates*, *Cross-Sell Candidates*, *VIP Upgrade Candidates*, *Customers Worth Retaining*, *Growing Revenue Accounts*, *Declining Revenue Risks*) with Business Reason, Estimated Impact ($), Priority, and Confidence Score.
* **Explainable AI (XAI)**: Global regression SHAP feature drivers ranking and individual customer attributions explaining *why* a customer has high or low lifetime value.
* **Multi-Horizon Revenue Forecasting**: Monthly, Quarterly, and Annual forecast tables comparing Actual vs Predicted Revenue and variance %.
* **Executive Insights**: 6 automated strategic revenue insight cards.
* **Multi-Format Exports**: CSV, Excel, and PDF report downloads.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](file:///s:/files/Coustomer%20churn%20prediction/LICENSE) file for details.





