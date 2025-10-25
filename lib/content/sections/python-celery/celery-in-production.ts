export const celeryInProduction = {
  title: 'Celery in Production',
  id: 'celery-in-production',
  content: `
# Celery in Production

## Introduction

Deploying Celery to production requires careful attention to **reliability, scalability, security, and monitoring**. This section covers everything you need for a production-ready Celery deployment.

**Production Checklist:**
✅ Process supervisors (auto-restart)
✅ Multiple workers (redundancy)
✅ Monitoring (Flower, Prometheus)
✅ Logging and error tracking
✅ Security (authentication, SSL)
✅ Auto-scaling based on load
✅ Health checks and alerts

---

## Deployment Architectures

### Basic Production Setup

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    Production Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  (Redundancy)   │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘                 │
│        │              │              │                       │
│        └──────────────┼──────────────┘                      │
│                       │                                      │
│                  ┌────▼────┐                                │
│                  │  Redis  │  (Broker + Backend)           │
│                  │ Sentinel│  (High Availability)          │
│                  └────┬────┘                                │
│                       │                                      │
│                  ┌────▼────┐                                │
│                  │ Flower  │  (Monitoring)                 │
│                  └─────────┘                                │
│                                                              │
│  ┌─────────────┐                                           │
│  │ Celery Beat │  (Periodic Tasks)                         │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
\`\`\`

---

## Systemd Services

\`\`\`ini
# /etc/systemd/system/celery-worker.service
[Unit]
Description=Celery Worker
After=network.target redis.service

[Service]
Type=forking
User=celery
Group=celery
WorkingDirectory=/opt/myapp
Environment="CELERY_BROKER_URL=redis://localhost:6379/0"
Environment="CELERY_RESULT_BACKEND=redis://localhost:6379/1"

ExecStart=/opt/myapp/venv/bin/celery -A tasks worker \\
    --loglevel=info \\
    --concurrency=4 \\
    --max-tasks-per-child=1000 \\
    --time-limit=300 \\
    --soft-time-limit=270 \\
    --pidfile=/var/run/celery/worker.pid \\
    --logfile=/var/log/celery/worker.log

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`\`\`

\`\`\`ini
# /etc/systemd/system/celery-beat.service
[Unit]
Description=Celery Beat Scheduler
After=network.target redis.service

[Service]
Type=simple
User=celery
Group=celery
WorkingDirectory=/opt/myapp
Environment="CELERY_BROKER_URL=redis://localhost:6379/0"

ExecStart=/opt/myapp/venv/bin/celery -A tasks beat \\
    --loglevel=info \\
    --pidfile=/var/run/celery/beat.pid \\
    --logfile=/var/log/celery/beat.log

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`\`\`

**Setup:**
\`\`\`bash
# Create directories
sudo mkdir -p /var/run/celery /var/log/celery
sudo chown celery:celery /var/run/celery /var/log/celery

# Enable services
sudo systemctl enable celery-worker celery-beat
sudo systemctl start celery-worker celery-beat

# Check status
sudo systemctl status celery-worker
sudo systemctl status celery-beat
\`\`\`

---

## Docker Deployment

\`\`\`dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create celery user
RUN useradd -m celery
USER celery

# Default command (override in docker-compose)
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
\`\`\`

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  celery_worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      redis:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    restart: unless-stopped

  celery_beat:
    build: .
    command: celery -A tasks beat --loglevel=info
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
    restart: unless-stopped

  flower:
    build: .
    command: celery -A tasks flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FLOWER_BASIC_AUTH=admin:strongpassword
    depends_on:
      - redis
    restart: unless-stopped

volumes:
  redis_data:
\`\`\`

---

## Kubernetes Deployment

\`\`\`yaml
# celery-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
  labels:
    app: celery-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: celery-worker
  template:
    metadata:
      labels:
        app: celery-worker
    spec:
      containers:
      - name: worker
        image: myapp:latest
        command: ["celery", "-A", "tasks", "worker", "--loglevel=info"]
        env:
        - name: CELERY_BROKER_URL
          valueFrom:
            secretKeyRef:
              name: celery-secrets
              key: broker-url
        - name: CELERY_RESULT_BACKEND
          valueFrom:
            secretKeyRef:
              name: celery-secrets
              key: backend-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command: ["celery", "-A", "tasks", "inspect", "ping"]
          initialDelaySeconds: 30
          periodSeconds: 30
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: External
    external:
      metric:
        name: celery_queue_depth
      target:
        type: AverageValue
        averageValue: "1000"
\`\`\`

---

## Production Configuration

\`\`\`python
# celeryconfig.py
from kombu import Queue, Exchange

# Broker
broker_url = 'redis://localhost:6379/0'
broker_connection_retry_on_startup = True
broker_connection_max_retries = 10

# Result Backend
result_backend = 'redis://localhost:6379/1'
result_expires = 3600  # 1 hour

# Task Settings
task_acks_late = True  # Acknowledge after task completes
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1  # One task per worker at a time
worker_max_tasks_per_child = 1000  # Restart worker after 1000 tasks

# Time Limits
task_soft_time_limit = 270  # 4.5 minutes
task_time_limit = 300  # 5 minutes

# Queues
task_queues = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('high_priority', Exchange('high_priority'), routing_key='high_priority'),
    Queue('low_priority', Exchange('low_priority'), routing_key='low_priority'),
)

# Routing
task_routes = {
    'tasks.send_email': {'queue': 'default'},
    'tasks.process_payment': {'queue': 'high_priority'},
    'tasks.generate_report': {'queue': 'low_priority'},
}

# Monitoring
worker_send_task_events = True
task_send_sent_event = True

# Security
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'
\`\`\`

---

## Monitoring and Alerting

\`\`\`python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from celery.signals import task_success, task_failure, task_retry

# Metrics
task_success_counter = Counter('celery_task_success_total', 'Tasks succeeded', ['task_name'])
task_failure_counter = Counter('celery_task_failure_total', 'Tasks failed', ['task_name'])
task_retry_counter = Counter('celery_task_retry_total', 'Tasks retried', ['task_name'])
task_duration = Histogram('celery_task_duration_seconds', 'Task duration', ['task_name'])

@task_success.connect
def task_success_handler(sender=None, **kwargs):
    task_success_counter.labels(task_name=sender.name).inc()

@task_failure.connect
def task_failure_handler(sender=None, **kwargs):
    task_failure_counter.labels(task_name=sender.name).inc()

@task_retry.connect
def task_retry_handler(sender=None, **kwargs):
    task_retry_counter.labels(task_name=sender.name).inc()

# Start Prometheus metrics server
start_http_server(8000)
\`\`\`

---

## Security Best Practices

1. **Broker Authentication**
\`\`\`python
# Redis with password
broker_url = 'redis://:strongpassword@localhost:6379/0'

# RabbitMQ with credentials
broker_url = 'amqp://myuser:mypass@localhost:5672//'
\`\`\`

2. **SSL/TLS**
\`\`\`python
broker_use_ssl = {
    'ssl_cert_reqs': ssl.CERT_REQUIRED,
    'ssl_ca_certs': '/etc/ssl/certs/ca-bundle.crt',
    'ssl_certfile': '/etc/ssl/certs/client-cert.pem',
    'ssl_keyfile': '/etc/ssl/private/client-key.pem',
}
\`\`\`

3. **Input Validation**
\`\`\`python
@app.task
def process_user(user_id: int):
    if not isinstance(user_id, int) or user_id < 1:
        raise ValueError("Invalid user_id")
    # Process...
\`\`\`

---

## Summary

**Production Essentials:**
- Systemd/Docker/K8s deployment
- Multiple workers (redundancy)
- Monitoring (Flower, Prometheus)
- Auto-scaling (HPA)
- Security (auth, SSL)
- Health checks and alerts

**Next Section:** Alternative task queues! 🔄
`,
};
