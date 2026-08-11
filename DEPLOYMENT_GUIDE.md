# ECIP Production Deployment Guide

## Production Containerized Deployment (Docker Compose)

1. **Verify Environment Configuration**:
   Ensure `.env` contains production secrets and settings:
   ```env
   ENVIRONMENT=production
   DEBUG=False
   JWT_SECRET_KEY=your_production_secret_32bytes
   DATABASE_URL=sqlite:///output/ecip.db
   ```

2. **Launch Container Services**:
   ```bash
   docker-compose up -d --build
   ```

3. **Verify Service Health**:
   - Backend API: `curl http://localhost:8000/api/v1/churn/health`
   - BI Dashboard: `curl http://localhost:8501`

---

## Nginx Reverse Proxy Configuration Example

```nginx
server {
    listen 80;
    server_name ecip.yourcompany.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
