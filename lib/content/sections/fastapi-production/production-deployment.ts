export const productionDeployment = {
  title: 'Production Deployment (Uvicorn, Gunicorn)',
  id: 'production-deployment',
  content: `
# Production Deployment (Uvicorn, Gunicorn)

## Introduction

Deploying FastAPI to production requires proper ASGI server configuration, process management, containerization, and monitoring. Uvicorn serves your FastAPI app, Gunicorn manages worker processes, and Docker/Kubernetes handle orchestration.

**Deployment requirements:**
- **ASGI server**: Uvicorn for async support
- **Process manager**: Gunicorn for multiple workers
- **Containerization**: Docker for consistency
- **Orchestration**: Kubernetes for scaling
- **Monitoring**: Health checks and metrics
- **Zero-downtime**: Blue-green or rolling deployments

In this section, you'll master:
- Uvicorn ASGI server configuration
- Gunicorn with Uvicorn workers
- Docker containerization
- Kubernetes deployment
- Environment configuration
- Health checks
- Monitoring and logging
- Zero-downtime deployments

---

## Uvicorn Basics

### Development Server

\`\`\`bash
# Basic development server with auto-reload
uvicorn main:app --reload

# Custom host and port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# With environment
uvicorn main:app --env-file .env --reload
\`\`\`

### Production Configuration

\`\`\`bash
# Production (single worker)
uvicorn main:app \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --workers 4 \\
  --log-level info \\
  --access-log \\
  --no-use-colors

# With SSL
uvicorn main:app \\
  --host 0.0.0.0 \\
  --port 443 \\
  --ssl-keyfile /path/to/key.pem \\
  --ssl-certfile /path/to/cert.pem
\`\`\`

---

## Gunicorn + Uvicorn Workers

### Why Gunicorn?

\`\`\`
Gunicorn = Process Manager
├── Worker 1 (Uvicorn)
├── Worker 2 (Uvicorn)
├── Worker 3 (Uvicorn)
└── Worker 4 (Uvicorn)

Benefits:
- Graceful restarts (zero downtime)
- Worker health monitoring
- Automatic worker restart on crash
- Load balancing across workers
\`\`\`

### Production Command

\`\`\`bash
# Gunicorn with Uvicorn workers
gunicorn main:app \\
  --workers 4 \\
  --worker-class uvicorn.workers.UvicornWorker \\
  --bind 0.0.0.0:8000 \\
  --timeout 30 \\
  --graceful-timeout 30 \\
  --keep-alive 5 \\
  --access-logfile - \\
  --error-logfile - \\
  --log-level info

# Calculate workers: (2 * CPU_CORES) + 1
# Example: 4 cores = (2 * 4) + 1 = 9 workers
\`\`\`

### Gunicorn Configuration File

\`\`\`python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
keepalive = 5
timeout = 30
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "fastapi-app"

# Server hooks
def on_starting(server):
    print("Gunicorn starting...")

def on_reload(server):
    print("Gunicorn reloading...")

def when_ready(server):
    print("Gunicorn ready!")

def on_exit(server):
    print("Gunicorn shutting down...")
\`\`\`

---

## Docker Deployment

### Dockerfile

\`\`\`dockerfile
# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run with Gunicorn
CMD ["gunicorn", "main:app", "--config", "gunicorn.conf.py"]
\`\`\`

### Docker Compose

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=\${SECRET_KEY}
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
\`\`\`

---

## Kubernetes Deployment

### Deployment YAML

\`\`\`yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
  labels:
    app: fastapi
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
      - name: api
        image: myregistry/fastapi-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  selector:
    app: fastapi
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
\`\`\`

---

## Health Checks

### Health Check Endpoints

\`\`\`python
"""
Production health check endpoints
"""

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health", tags=["health"])
async def health_check():
    """
    Liveness probe: Is the app running?
    Returns 200 if app is alive
    """
    return {"status": "healthy"}

@app.get("/ready", tags=["health"])
async def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe: Is the app ready to serve traffic?
    Checks all dependencies (database, Redis, etc.)
    """
    try:
        # Check database
        await db.execute(text("SELECT 1"))
        
        # Check Redis
        await redis_client.ping()
        
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

@app.get("/startup", tags=["health"])
async def startup_check():
    """
    Startup probe: Has app finished initialization?
    """
    if not app.state.initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "starting"}
        )
    return {"status": "started"}
\`\`\`

---

## Environment Configuration

### Settings Management

\`\`\`python
"""
Production-ready settings with pydantic-settings
"""

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "FastAPI App"
    DEBUG: bool = False
    API_VERSION: str = "v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 0
    
    # Redis
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: list[str] = ["https://example.com"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Monitoring
    SENTRY_DSN: str | None = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
\`\`\`

### Environment Files

\`\`\`bash
# .env.example
APP_NAME="My FastAPI App"
DEBUG=false

DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379/0

SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=https://example.com,https://app.example.com

LOG_LEVEL=INFO
SENTRY_DSN=https://your-sentry-dsn
\`\`\`

---

## Zero-Downtime Deployments

### Rolling Update Strategy

\`\`\`yaml
# Kubernetes rolling update
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 1 extra pod during update
      maxUnavailable: 1  # 1 pod can be unavailable
  template:
    spec:
      containers:
      - name: api
        image: myapp:v2
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Graceful shutdown
\`\`\`

### Blue-Green Deployment

\`\`\`bash
# Deploy new version (green)
kubectl apply -f deployment-green.yaml

# Test green deployment
curl http://green.example.com/health

# Switch traffic from blue to green
kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for issues
# If problems, rollback:
kubectl patch service myapp -p '{"spec":{"selector":{"version":"blue"}}}'
\`\`\`

---

## Monitoring

### Prometheus Metrics

\`\`\`python
"""
Expose Prometheus metrics
"""

from prometheus_client import make_asgi_app, Counter, Histogram
from starlette.middleware.wsgi import WSGIMiddleware

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", WSGIMiddleware(metrics_app))

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
\`\`\`

---

## Nginx Reverse Proxy

### Nginx Configuration

\`\`\`nginx
# nginx.conf
upstream fastapi {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    server_name example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;
    
    # SSL certificates
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
    
    # SSL config
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Proxy to FastAPI
    location / {
        proxy_pass http://fastapi;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://fastapi;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
\`\`\`

---

## Summary

✅ **Uvicorn**: ASGI server for FastAPI  
✅ **Gunicorn**: Process manager with multiple workers  
✅ **Docker**: Containerization with multi-stage builds  
✅ **Kubernetes**: Orchestration with health checks  
✅ **Zero-downtime**: Rolling updates and blue-green  
✅ **Monitoring**: Prometheus metrics and health checks  
✅ **Nginx**: Reverse proxy with SSL and load balancing  

### Production Checklist

**Pre-deployment:**
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Health checks implemented
- [ ] Monitoring configured
- [ ] SSL certificates valid
- [ ] Backup strategy in place

**Deployment:**
- [ ] Use Gunicorn with Uvicorn workers
- [ ] Configure worker count based on CPU
- [ ] Set up health checks (liveness, readiness)
- [ ] Enable logging and metrics
- [ ] Configure graceful shutdown
- [ ] Test rollback procedure

**Post-deployment:**
- [ ] Monitor error rates
- [ ] Check response times
- [ ] Verify health checks passing
- [ ] Test critical endpoints
- [ ] Review logs for errors
- [ ] Confirm zero downtime

### Next Steps

In the final section, we'll explore **FastAPI Best Practices & Patterns**: production-ready project structure, security hardening, and optimization strategies.
`,
};
