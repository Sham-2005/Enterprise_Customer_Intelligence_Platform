# ECIP REST API Specification & Endpoint Guide

The ECIP API provides production-grade REST services for real-time inference, analytics, and MLOps model governance.

## Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoint Reference

### Authentication (`/api/v1/auth`)
* `POST /api/v1/auth/register`: Register new user account.
* `POST /api/v1/auth/login`: Authenticate and receive JWT access token.

### Churn Risk Intelligence (`/api/v1/churn`)
* `GET /api/v1/churn/health`: Service health status.
* `GET /api/v1/churn/model-info`: Retrieve model metrics, ROC-AUC score, and feature importances.
* `POST /api/v1/churn/predict`: Real-time single customer churn inference with SHAP explanation & retention strategy.

### Customer Lifetime Value (`/api/v1/clv`)
* `GET /api/v1/clv/health`: Service health status.
* `GET /api/v1/clv/model-info`: Retrieve CLV model R² score and MAE/RMSE metrics.
* `POST /api/v1/clv/predict`: Real-time 12-month CLV prediction & value tiering.

### Recommendation Engine (`/api/v1/recommendations`)
* `GET /api/v1/recommendations/health`: Service health status.
* `GET /api/v1/recommendations/metrics`: Precision@10, Recall@10, MAP@10 evaluation metrics.
* `POST /api/v1/recommendations/customer`: Returns top $N$ ranked product recommendations with XAI rationale.
* `POST /api/v1/recommendations/similar-products`: Item-to-item similarity lookup.

### Market Basket Analysis (`/api/v1/mba`)
* `GET /api/v1/mba/health`: Service health status.
* `GET /api/v1/mba/rules`: Top association rules (support, confidence, lift).
* `GET /api/v1/mba/bundles`: High-value product bundles with estimated revenue.

### MLOps Governance (`/api/v1/mlops`)
* `GET /api/v1/mlops/health`: System health & uptime metrics.
* `GET /api/v1/mlops/registry`: Central model registry contents.
* `POST /api/v1/mlops/rollback`: Rollback model active version pointer.
* `GET /api/v1/mlops/experiments`: Tracked experiment history.
* `GET /api/v1/mlops/drift`: Kolmogorov-Smirnov feature drift audit.
