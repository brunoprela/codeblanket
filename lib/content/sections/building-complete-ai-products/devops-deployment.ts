export const devopsDeployment = {
  title: 'DevOps & Deployment',
  id: 'devops-deployment',
  content: `
# DevOps & Deployment for AI Applications

## Introduction

Deploying AI applications is fundamentally different from traditional web apps. You need GPU infrastructure, model versioning, expensive compute optimization, and specialized monitoring for LLM-specific metrics like token usage, latency, and quality.

This section covers production deployment strategies for AI applications using Docker, Kubernetes, and cloud platforms.

### Deployment Challenges

**GPU Infrastructure**: Managing expensive GPU instances ($1-5/hour)
**Model Loading**: 10GB+ models take 30-60s to load
**Cold Starts**: Serverless incompatible with large models
**Cost**: $0.001-0.03 per request vs $0.0001 for traditional APIs
**Monitoring**: Need LLM-specific metrics (tokens, quality, hallucinations)

---

## Docker Setup

### Multi-Stage Build for AI App

\`\`\`dockerfile
# Dockerfile for AI backend

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download models (optional, increases image size but faster startup)
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; \\
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \\
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# Stage 3: Application
FROM dependencies AS application

WORKDIR /app

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
\`\`\`

### Docker Compose for Local Development

\`\`\`yaml
# docker-compose.yml

version: '3.8'

services:
  # API server
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/aiapp
    depends_on:
      - redis
      - db
    volumes:
      - ./app:/app/app
      - model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # Celery worker
  worker:
    build: .
    command: celery -A app.tasks worker --loglevel=info --concurrency=2
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/aiapp
    depends_on:
      - redis
      - db
    volumes:
      - ./app:/app/app
      - model_cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # Redis for caching and queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  # PostgreSQL database
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aiapp
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Frontend (Next.js)
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules

volumes:
  redis_data:
  postgres_data:
  model_cache:
\`\`\`

---

## Kubernetes Deployment

### GPU Node Pool Configuration

\`\`\`yaml
# kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-api
  labels:
    app: ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-api
  template:
    metadata:
      labels:
        app: ai-api
    spec:
      # Node selector for GPU nodes
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      
      containers:
      - name: api
        image: gcr.io/project/ai-api:latest
        ports:
        - containerPort: 8000
        
        # Resource requests
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        
        # Environment variables
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        # Volume mounts for model cache
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ai-api-service
spec:
  type: LoadBalancer
  selector:
    app: ai-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
\`\`\`

### Celery Worker Deployment

\`\`\`yaml
# kubernetes/worker-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 5
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      
      containers:
      - name: worker
        image: gcr.io/project/ai-api:latest
        command: ["celery", "-A", "app.tasks", "worker", "--loglevel=info", "--concurrency=2"]
        
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
\`\`\`

---

## Cloud Platform Deployments

### AWS Lambda with GPU (for light models)

\`\`\`python
"""
AWS Lambda handler with container images
"""

import json
from mangum import Mangum
from app.main import app

# FastAPI app wrapped for Lambda
handler = Mangum (app)

# serverless.yml
"""
service: ai-api

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 3008
  timeout: 300
  environment:
    ANTHROPIC_API_KEY: \${env:ANTHROPIC_API_KEY}

functions:
  api:
    handler: handler.handler
    events:
      - httpApi:
          path: /{proxy+}
          method: ANY
    vpc:
      securityGroupIds:
        - sg-xxxxx
      subnetIds:
        - subnet-xxxxx
"""
\`\`\`

### GCP Cloud Run (best for API-based models)

\`\`\`yaml
# cloud-run.yaml

apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ai-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project/ai-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: 4Gi
            cpu: "4"
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic
\`\`\`

### Modal (best for GPU workloads)

\`\`\`python
"""
Modal deployment for GPU-heavy workloads
"""

import modal

stub = modal.Stub("ai-image-gen")

# Define image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "diffusers",
    "transformers",
    "accelerate"
)

@stub.function(
    image=image,
    gpu="T4",  # or "A100", "A10G"
    memory=16384,
    timeout=300,
    container_idle_timeout=60
)
def generate_image (prompt: str) -> bytes:
    """
    Generate image with Stable Diffusion
    """
    from diffusers import StableDiffusionPipeline
    import torch
    import io
    
    # Load model (cached across invocations)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Generate
    image = pipe (prompt).images[0]
    
    # Return as bytes
    buf = io.BytesIO()
    image.save (buf, format='PNG')
    return buf.getvalue()

@stub.local_entrypoint()
def main():
    # Test locally
    result = generate_image.remote("A beautiful sunset")
    print(f"Generated {len (result)} bytes")
\`\`\`

---

## CI/CD Pipeline

### GitHub Actions for Deployment

\`\`\`yaml
# .github/workflows/deploy.yml

name: Deploy AI Application

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  GCP_PROJECT: my-ai-project
  GCP_REGION: us-central1
  IMAGE_NAME: ai-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to GCR
      uses: docker/login-action@v2
      with:
        registry: gcr.io
        username: _json_key
        password: \${{ secrets.GCP_SA_KEY }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          gcr.io/\${{ env.GCP_PROJECT }}/\${{ env.IMAGE_NAME }}:latest
          gcr.io/\${{ env.GCP_PROJECT }}/\${{ env.IMAGE_NAME }}:\${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: \${{ secrets.GCP_SA_KEY }}
        project_id: \${{ env.GCP_PROJECT }}
    
    - name: Deploy to GKE
      run: |
        gcloud container clusters get-credentials ai-cluster --region \${{ env.GCP_REGION }}
        kubectl set image deployment/ai-api ai-api=gcr.io/\${{ env.GCP_PROJECT }}/\${{ env.IMAGE_NAME }}:\${{ github.sha }}
        kubectl rollout status deployment/ai-api
    
    - name: Run smoke tests
      run: |
        kubectl run smoke-test --rm -i --restart=Never --image=curlimages/curl -- \\
          curl -f http://ai-api-service/health || exit 1
\`\`\`

---

## Monitoring & Observability

### Prometheus Metrics

\`\`\`python
"""
Prometheus metrics for AI application
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'model', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'Request duration',
    ['endpoint', 'model']
)

token_count = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: input/output
)

cost_total = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model']
)

active_requests = Gauge(
    'api_active_requests',
    'Currently active requests',
    ['endpoint']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization',
    ['gpu_id']
)

# Middleware
@app.middleware("http")
async def metrics_middleware (request: Request, call_next):
    endpoint = request.url.path
    
    active_requests.labels (endpoint=endpoint).inc()
    start = time.time()
    
    try:
        response = await call_next (request)
        
        duration = time.time() - start
        request_duration.labels(
            endpoint=endpoint,
            model=getattr (request.state, 'model', 'unknown')
        ).observe (duration)
        
        request_count.labels(
            endpoint=endpoint,
            model=getattr (request.state, 'model', 'unknown'),
            status=response.status_code
        ).inc()
        
        return response
        
    finally:
        active_requests.labels (endpoint=endpoint).dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Track LLM usage
def track_llm_usage (model: str, input_tokens: int, output_tokens: int, cost: float):
    token_count.labels (model=model, type="input").inc (input_tokens)
    token_count.labels (model=model, type="output").inc (output_tokens)
    cost_total.labels (model=model).inc (cost)
\`\`\`

### Grafana Dashboard

\`\`\`yaml
# grafana-dashboard.json (simplified)

{
  "dashboard": {
    "title": "AI Application Metrics",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [
          {
            "expr": "rate (api_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Average Latency",
        "targets": [
          {
            "expr": "rate (api_request_duration_seconds_sum[5m]) / rate (api_request_duration_seconds_count[5m])"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "rate (llm_tokens_total[5m])"
          }
        ]
      },
      {
        "title": "Cost per Hour",
        "targets": [
          {
            "expr": "increase (llm_cost_usd_total[1h])"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "gpu_utilization_percent"
          }
        ]
      }
    ]
  }
}
\`\`\`

---

## Conclusion

DevOps for AI applications requires:

1. **GPU Infrastructure**: Docker with CUDA, K8s with GPU nodes
2. **Cost Optimization**: Auto-scaling, spot instances, model caching
3. **Monitoring**: LLM-specific metrics (tokens, cost, quality)
4. **CI/CD**: Automated testing, deployment, rollbacks
5. **Observability**: Prometheus, Grafana, logging

**Best Practices**:
- Use managed services (Modal, Replicate) for GPU workloads
- Cloud Run/Fargate for API-based models
- K8s with GPU nodes for full control
- Always monitor cost per request
- Implement circuit breakers for provider failures

Production AI deployment is expensive but manageable with the right architecture.
`,
};
