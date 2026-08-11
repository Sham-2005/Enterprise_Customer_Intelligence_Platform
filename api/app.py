"""
FastAPI Application Entrypoint for Enterprise Customer Intelligence Platform (ECIP).
Exposes REST API endpoints for Authentication, Churn Risk, CLV, Recommendations, MBA, and MLOps Governance.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.auth_routes import router as auth_router
from api.routes.churn_routes import router as churn_router
from api.routes.clv_routes import router as clv_router
from api.routes.recommendation_routes import router as rec_router
from api.routes.mba_routes import router as mba_router
from api.routes.mlops_routes import router as mlops_router

app = FastAPI(
    title="Enterprise Customer Intelligence Platform (ECIP) REST API",
    description="Production-ready REST API for Authentication, AI Churn Prediction, CLV Forecasting, Recommendations, Market Basket Analysis, and MLOps Governance.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(churn_router)
app.include_router(clv_router)
app.include_router(rec_router)
app.include_router(mba_router)
app.include_router(mlops_router)

@app.get("/")
def root():
    return {
        "platform": "Enterprise Customer Intelligence Platform (ECIP)",
        "status": "Online",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
