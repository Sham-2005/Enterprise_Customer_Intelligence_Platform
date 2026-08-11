# System Architecture Specification - ECIP

## 1. Overview
The **Enterprise Customer Intelligence Platform (ECIP)** is engineered as a decoupled, multi-layered micro-architecture separating Data Ingestion, Business Intelligence Processing, Machine Learning & XAI, REST API Delivery, MLOps Governance, and Interactive UI Dashboard layers.

---

## 2. Layer Definitions

### Data Layer
- **Ingestion & Validation**: Reads Olist relational CSV datasets using encoding fallbacks (`UTF-8`, `ISO-8859-1`). Validates null rates, primary key uniqueness, and monetary constraints.
- **Relational Merger**: Pre-aggregates payments and reviews by order ID before joining customers, orders, items, products, and sellers to prevent Cartesian fan-out joins.
- **Feature Store**: Persists versioned customer-level, product-level, order-level, and seller-level feature matrices to `output/feature_store.csv`.

### Analytics & ML Modeling Layer
- **Customer Segmentation**: `RobustScaler` preprocessing + PCA 3D projection + Unsupervised model benchmark (K-Means, Agglomerative, GMM, DBSCAN) + Rule-based Persona Generator.
- **Churn Intelligence**: SMOTE class balancing + Multi-model classification benchmark (XGBoost, LightGBM, Random Forest, Logistic Regression) + 5-tier risk stratification + SHAP Explainable AI diagnostics.
- **CLV Forecast**: 12-month future value regression modeling + Platinum/Gold/Silver tiering + Revenue Opportunity Engine.
- **Hybrid Recommender**: Collaborative Item-Item Cosine Similarity matrix + Content-Based TF-IDF category vectorization + Cold-Start trending fallback.
- **Market Basket Mining**: Transaction basket one-hot matrix encoding + FP-Growth & Apriori frequent itemset mining + Association rule metrics (Support, Confidence, Lift, Leverage, Conviction).

### API & Service Layer
- Built on FastAPI with async endpoint handlers, Pydantic DTO validation, CORS middleware, JWT bearer token authentication, and OpenAPI Swagger documentation.

### MLOps & Governance Layer
- Centralized Model Registry (`output/models/model_registry.json`) tracking model versions, statuses (`Active`, `Staging`, `Archived`), and rollback pointers.
- Kolmogorov-Smirnov (KS) two-sample feature drift detection.
- Structured inference audit logs (`output/logs/inference_audit.log`).

---

## 3. Phase 16 - AI Recommendation Engine Dashboard Architecture

- **Service Layer (`RecommendationService`)**: Manages dataset discovery (`customer_recommendations.csv`, `recommended_products.csv`, `similar_products.csv`, `cross_sell_products.csv`, `upsell_products.csv`, `trending_products.csv`, `recommendation_metrics.json`), fits `HybridRecommenderEngine`, merges customer context (RFM, CLV, Churn), generates plain-English XAI explanations, computes 8 KPIs, handles multi-criteria interactive filtering, search, and CSV/JSON exports.
- **Cold-Start Fallback Strategy**:
  - *New Customers*: System-wide trending products & category diversity.
  - *New Products*: TF-IDF category content similarity.
  - *Limited History*: RFM segment popularity hybrid blending.
- **Unified Opportunity Matrix**: Merges recommendations with CLV value tiers (Platinum/Gold) and Churn Risk levels (Critical/High) to assign actionable business priorities (`P1 - At-Risk Retention`, `P2 - VIP Cross-Sell`, `P3 - High Value Upsell`, `P4 - Standard`).
- **Dashboard UI (`recommendation_page.py`)**: Enterprise dark glassmorphism layout featuring 8 KPI cards, Customer Recommendation Explorer, Categorized Tabs (Personalized, Similar, Bought Together, Cross-Sell, Upsell, Trending), Product Explorer, Plotly Analytics charts, Cold-Start Strategy Center, Business Intelligence Insights grid, and Multi-Format Exports.

---

## 4. Phase 17 - Market Basket Analysis & Product Association Dashboard Architecture

- **Service Layer (`MBAService`)**: Manages precomputed association outputs (`association_rules.csv`, `product_bundles.csv`, `cross_sell_recommendations.csv`, `basket_statistics.csv`, `mba_metrics.json`), integrates `MarketBasketAnalyzer` for on-the-fly mining fallbacks, computes 8 KPIs, constructs Product Association Network topology graphs, extracts customer segment basket metrics, computes category co-occurrence matrix crosstabs, extracts seasonal basket telemetry, builds business recommendations with financial impact estimates, and provides multi-format data exports.
- **Product Association Network Topology**: Constructs node-edge network graph structures showing antecedents → consequents with Lift score color gradients and interactive filter controls.
- **Customer Segment Basket Analysis**: Cross-analyzes transaction orders against customer RFM segments (VIP, Champions, At-Risk, New) to compute segment-specific basket sizes, top purchased categories, top combinations, and AOVs.
- **Dashboard UI (`mba_page.py`)**: Enterprise dark glassmorphism layout featuring 8 KPI cards, Association Rule Explorer with sorting controls, Support vs. Confidence scatter plot, Product Association Network visualizer, Product Bundle expander cards, Cross-Sell trigger engine, Customer Segment basket comparison, Category Co-occurrence Heatmap, Seasonal Telemetry charts, Product Search detail panel, Business Recommendations grid, and Multi-Format Export Hub.

---

## 5. Phase 18 - MLOps, Model Monitoring & AI Governance Dashboard Architecture

- **Service Layer (`MLOpsService`)**: Connects to `ModelRegistry`, `ExperimentTracker`, `DriftDetector`, `AuditLogger`, `PerformanceMonitor`, and `RetrainingPipeline`. Computes 8 MLOps KPIs, calculates model health classification (🟢 Healthy, 🟡 Warning, 🔴 Critical, ⚪ Not Monitored), executes Version A vs Version B benchmark comparisons, parses Kolmogorov-Smirnov (KS) data drift statistics, logs inference audit trails, builds AI Governance compliance matrices, triggers automated model retraining pipelines, executes version rollbacks, and exports CSV/JSON audit reports.
- **Version A vs Version B Comparison Engine**: Compares performance metrics (ROC-AUC, Precision, Recall, MAE, R², Precision@K, Silhouette), hyperparameters, algorithms, and training dates between any two registered model versions and flags the superior candidate.
- **Role Perspective Control**: Provides role-tailored view perspectives (`ML Engineer`, `Data Scientist`, `Administrator`, `Business Analyst`) to gate destructive lifecycle controls (version rollbacks, retraining triggers).
- **Dashboard UI (`mlops_page.py`)**: Enterprise dark glassmorphism layout featuring 8 KPI cards and 8 modular tabs:
  1. *Overview & System Health*: Health matrix, service status indicators, system alerts.
  2. *Model Registry*: Active versions, registration metadata, rollback controls.
  3. *Performance & Compare*: Model-specific benchmarks & Version A vs B comparison.
  4. *Data Drift Audit*: KS test statistics, p-values, feature drift ranking charts, concept drift status.
  5. *Experiment Tracking*: Experiment leaderboard & hyperparameter inspector.
  6. *Prediction Telemetry*: Latency, volume line charts, error rate monitoring.
  7. *Audit Logs*: Structured, filterable inference audit trail parsed from `inference_audit.log`.
  8. *AI Governance & Retraining*: Compliance matrix badges (`Complete`, `Partial`, `Missing`) and automated retraining pipeline trigger.

---

## 6. Phase 19 - Enterprise Reports & Export Center Architecture

- **Service Layer (`ReportsService`)**: Defines centralized report catalog of 15 enterprise reports across 4 categories (Executive, Customer, AI, Technical). Orchestrates existing analytics and AI services (`DataService`, `KPIService`, `ChurnService`, `CLVService`, `RecommendationService`, `MBAService`, `MLOpsService`), generates multi-format export buffers via `ExportService` (ReportLab formatted PDF, openpyxl/xlsxwriter multi-sheet Excel workbooks, UTF-8 CSVs), builds live pre-download report preview payloads, logs report generations in `output/reports/report_history.json`, and stores output files in `output/reports/{category}/`.
- **Report Storage Layout**:
  - `output/reports/executive/`
  - `output/reports/customer/`
  - `output/reports/churn/`
  - `output/reports/clv/`
  - `output/reports/recommendations/`
  - `output/reports/market_basket/`
  - `output/reports/mlops/`
- **Dashboard UI (`reports_page.py`)**: Enterprise dark glassmorphism layout featuring global report filters (Date Range, State, Customer Segment, Product Category, Seller, Churn Risk, CLV Tier), report category tabs (All Reports, Executive, Customer, AI, Technical, Report History), interactive report cards grid with format badges (`PDF`, `EXCEL`, `CSV`) and status indicators (`Ready`, `Generating`, `Failed`), live Report Preview panel, and Report History archive table.




