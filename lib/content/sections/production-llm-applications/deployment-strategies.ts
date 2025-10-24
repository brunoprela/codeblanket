export const deploymentStrategiesContent = `
# Deployment Strategies

## Introduction

Deploying LLM applications requires careful planning around containers, orchestration, scaling, and zero-downtime updates. This section covers Docker, Kubernetes, cloud deployments, and deployment best practices.

## Docker Containerization

\`\`\`dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

## Docker Compose for Local Development

\`\`\`yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/llm_db
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=llm_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  worker:
    build: .
    command: celery -A app.celery worker --loglevel=info
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

volumes:
  postgres_data:
\`\`\`

## Kubernetes Deployment

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: api
        image: myregistry/llm-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
\`\`\`

## Blue-Green Deployment

\`\`\`python
# Deploy script for zero-downtime updates

def blue_green_deploy():
    """Deploy new version using blue-green strategy."""
    
    # 1. Deploy new version (green)
    deploy_version("green", new_version)
    
    # 2. Run smoke tests on green
    if not smoke_test("green"):
        rollback("green")
        return False
    
    # 3. Switch traffic to green
    switch_traffic_to("green")
    
    # 4. Monitor for issues
    if not monitor_health("green", duration=300):
        switch_traffic_to("blue")
        return False
    
    # 5. Keep blue as backup, green is now live
    return True
\`\`\`

## AWS Deployment

\`\`\`python
# Deploy to AWS ECS with Fargate

import boto3

ecs = boto3.client('ecs')

task_definition = {
    'family': 'llm-api',
    'networkMode': 'awsvpc',
    'requiresCompatibilities': ['FARGATE'],
    'cpu': '1024',
    'memory': '2048',
    'containerDefinitions': [{
        'name': 'api',
        'image': 'myregistry/llm-api:latest',
        'portMappings': [{'containerPort': 8000}],
        'environment': [
            {'name': 'ENV', 'value': 'production'}
        ],
        'secrets': [
            {
                'name': 'OPENAI_API_KEY',
                'valueFrom': 'arn:aws:secretsmanager:us-east-1:123456789:secret:openai-key'
            }
        ]
    }]
}

response = ecs.register_task_definition(**task_definition)
\`\`\`

## CI/CD Pipeline

\`\`\`yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t llm-api:\${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker push myregistry/llm-api:\${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/llm-api api=myregistry/llm-api:\${{ github.sha }}
          kubectl rollout status deployment/llm-api
\`\`\`

## Best Practices

1. **Use containers** for consistent deployments
2. **Implement health checks** for all services
3. **Use blue-green or canary** for zero-downtime
4. **Store secrets securely** (never in code)
5. **Automate deployments** with CI/CD
6. **Monitor deployments** and have rollback plan
7. **Use separate environments** (dev, staging, prod)
8. **Scale horizontally** with load balancers
9. **Version everything** (code, config, infrastructure)
10. **Test deployments** in staging first

Proper deployment strategies ensure your LLM application updates smoothly without downtime or issues.
`;
