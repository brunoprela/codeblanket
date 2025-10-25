export const monitoringWithFlower = {
  title: 'Task Monitoring with Flower',
  id: 'monitoring-with-flower',
  content: `
# Task Monitoring with Flower

## Introduction

**Flower** is a real-time web-based monitoring tool for Celery. It provides a beautiful dashboard showing workers, tasks, queues, and execution statistics. In production, Flower is **essential** for:

- Monitoring task execution in real-time
- Debugging failed tasks
- Managing workers (restart, shutdown)
- Analyzing performance metrics
- Detecting bottlenecks and issues

**Without Flower:** You're flying blind—no visibility into what's happening with your 10,000 queued tasks.

**With Flower:** Real-time dashboard showing exactly which tasks are running, which failed, queue depths, worker health, and execution times.

---

## Installation and Basic Setup

\`\`\`bash
# Install Flower
pip install flower

# Basic start (simplest)
celery -A tasks flower

# With custom port
celery -A tasks flower --port=5555

# With broker URL (if not in celery config)
celery -A tasks flower --broker=redis://localhost:6379/0

# With authentication (production)
celery -A tasks flower --basic_auth=admin:strongpassword

# With persistent history (stores task history to disk)
celery -A tasks flower --persistent=True --db=/var/lib/flower/db

# All options
celery -A tasks flower \\
    --port=5555 \\
    --broker=redis://localhost:6379/0 \\
    --basic_auth=admin:strongpassword,user:userpass \\
    --persistent=True \\
    --db=/var/lib/flower/flower.db
\`\`\`

**Access Flower:** Open browser to \`http://localhost:5555\`

---

## Flower Dashboard Overview

### Main Dashboard

When you open Flower, you see:

\`\`\`
╔══════════════════════════════════════════════════════════════╗
║  Flower - Celery Monitoring                                  ║
║  Broker: redis://localhost:6379/0                            ║
╠══════════════════════════════════════════════════════════════╣
║  Workers: 3 online                                           ║
║  Tasks: 1,247 succeeded | 23 failed | 15 active             ║
║  Queue Depth: default (127) | emails (5) | reports (0)     ║
╚══════════════════════════════════════════════════════════════╝

Recent Tasks:
┌──────────────────┬─────────────┬──────────┬──────────┬─────────┐
│ Task Name        │ State       │ Runtime  │ Started  │ Worker  │
├──────────────────┼─────────────┼──────────┼──────────┼─────────┤
│ send_email       │ ✅ SUCCESS  │ 2.3s     │ 10:32:15 │ worker1 │
│ process_payment  │ ⏳ STARTED │ 1.5s     │ 10:32:18 │ worker2 │
│ generate_report  │ ❌ FAILURE │ 5.1s     │ 10:32:10 │ worker3 │
└──────────────────┴─────────────┴──────────┴──────────┴─────────┘
\`\`\`

### Workers View

\`\`\`
Worker: celery@localhost.worker1
Status: ✅ Online
Pool: prefork (4 processes)
CPU: 23%
Memory: 512 MB
Processed: 1,247 tasks

Active Tasks (2):
- process_video [123456] - Running 45s
- send_bulk_email [123457] - Running 12s

Registered Tasks:
- tasks.send_email
- tasks.process_payment
- tasks.generate_report
- tasks.process_video
\`\`\`

### Tasks View

\`\`\`
┌──────────────┬──────────────────────┬──────────┬──────────┬────────────┐
│ UUID         │ Name                 │ State    │ Args     │ Runtime    │
├──────────────┼──────────────────────┼──────────┼──────────┼────────────┤
│ abc-123      │ tasks.send_email     │ SUCCESS  │ (123,)   │ 2.3s       │
│ def-456      │ tasks.process_payment│ FAILURE  │ (456,)   │ 5.1s       │
│ ghi-789      │ tasks.generate_report│ STARTED  │ (789,)   │ 12.5s (...)│
└──────────────┴──────────────────────┴──────────┴──────────┴────────────┘

Click task for details:
- Full traceback (for failures)
- Arguments and kwargs
- Start/end time
- Result/exception
- Retry count
\`\`\`

### Monitor View

Real-time graphs:
- Task completion rate (tasks/minute)
- Success vs failure ratio
- Queue depth over time
- Worker CPU/memory usage

---

## Configuration File

For production, use a configuration file:

\`\`\`python
# flowerconfig.py
"""
Production Flower configuration
"""

# Broker connection
broker_url = 'redis://localhost:6379/0'

# Or for RabbitMQ
# broker_url = 'amqp://guest:guest@localhost:5672//'
# broker_api = 'http://guest:guest@localhost:15672/api/'

# Port
port = 5555

# Authentication (bcrypt hashed passwords)
basic_auth = [
    'admin:$2b$12$P8xW9QxMZvKfZ7RL.dWTruKP8AZN3M5vQFWZLZ4xYZ9xYZ9xYZ9x',  # admin:strongpass
    'viewer:$2b$12$N7xW9QxMZvKfZ7RL.dWTruKP8AZN3M5vQFWZLZ4xYZ9xYZ9xYZ9y'  # viewer:viewerpass
]

# Or simple auth (dev only - not secure for production)
# basic_auth = ['admin:password', 'user:password']

# OAuth2 (Google)
# oauth2_key = 'google_client_id'
# oauth2_secret = 'google_client_secret'
# oauth2_redirect_uri = 'https://flower.company.com/login'

# Persistent task history
persistent = True
db = '/var/lib/flower/flower.db'  # SQLite database for task history

# Task time format
task_time_format = '%Y-%m-%d %H:%M:%S'

# Max number of tasks to keep in memory
max_tasks = 10000

# Auto-refresh interval (seconds)
auto_refresh = True
auto_refresh_interval = 60

# CORS
enable_cors = True
cors_origins = ['https://dashboard.company.com']

# Timezone
timezone = 'US/Eastern'

# SSL
# certfile = '/etc/ssl/certs/flower.crt'
# keyfile = '/etc/ssl/private/flower.key'

# URL prefix (if behind reverse proxy)
# url_prefix = '/flower'

# Enable control (shutdown/restart workers)
# Set to False in production for security
enable_control = False  # Disable shutdown buttons

# Logging
logging = 'INFO'
\`\`\`

**Run with config:**
\`\`\`bash
celery -A tasks flower --conf=flowerconfig.py
\`\`\`

---

## Production Deployment

### Systemd Service

\`\`\`ini
# /etc/systemd/system/celery-flower.service
[Unit]
Description=Flower Celery Monitoring
After=network.target

[Service]
Type=simple
User=celery
Group=celery
WorkingDirectory=/opt/myapp
Environment="CELERY_BROKER_URL=redis://localhost:6379/0"
ExecStart=/opt/myapp/venv/bin/celery -A tasks flower \\
    --port=5555 \\
    --conf=/opt/myapp/flowerconfig.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
\`\`\`

**Enable and start:**
\`\`\`bash
sudo systemctl enable celery-flower
sudo systemctl start celery-flower
sudo systemctl status celery-flower
\`\`\`

### Docker Deployment

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  celery_worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    deploy:
      replicas: 3
  
  flower:
    build: .
    command: celery -A tasks flower --port=5555 --conf=flowerconfig.py
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FLOWER_PORT=5555
      - FLOWER_BASIC_AUTH=admin:strongpassword
    depends_on:
      - redis
    volumes:
      - flower_data:/var/lib/flower

volumes:
  redis_data:
  flower_data:
\`\`\`

**Run:**
\`\`\`bash
docker-compose up -d
# Access at http://localhost:5555
\`\`\`

### Kubernetes Deployment

\`\`\`yaml
# flower-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flower
  labels:
    app: flower
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flower
  template:
    metadata:
      labels:
        app: flower
    spec:
      containers:
      - name: flower
        image: myapp:latest
        command: ["celery", "-A", "tasks", "flower", "--port=5555"]
        ports:
        - containerPort: 5555
        env:
        - name: CELERY_BROKER_URL
          valueFrom:
            secretKeyRef:
              name: celery-secrets
              key: broker-url
        - name: FLOWER_BASIC_AUTH
          valueFrom:
            secretKeyRef:
              name: celery-secrets
              key: flower-auth
        volumeMounts:
        - name: flower-data
          mountPath: /var/lib/flower
      volumes:
      - name: flower-data
        persistentVolumeClaim:
          claimName: flower-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: flower
spec:
  selector:
    app: flower
  ports:
  - port: 5555
    targetPort: 5555
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: flower-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
\`\`\`

**Deploy:**
\`\`\`bash
kubectl apply -f flower-deployment.yaml
kubectl get svc flower  # Get LoadBalancer IP
\`\`\`

---

## Flower API

Flower provides a REST API for programmatic access:

\`\`\`python
"""
Using Flower API programmatically
"""

import requests
import json

FLOWER_URL = 'http://localhost:5555/api'


def get_workers():
    """Get list of workers"""
    response = requests.get (f'{FLOWER_URL}/workers')
    workers = response.json()
    
    for worker_name, info in workers.items():
        print(f"Worker: {worker_name}")
        print(f"  Status: {'Online' if info['status'] else 'Offline'}")
        print(f"  Active tasks: {len (info.get('active', []))}")
        print(f"  Processed: {info['stats']['total']}")
    
    return workers


def get_tasks (limit=100):
    """Get recent tasks"""
    response = requests.get (f'{FLOWER_URL}/tasks', params={'limit': limit})
    tasks = response.json()
    
    for task_id, task_info in tasks.items():
        print(f"Task: {task_info['name']}")
        print(f"  ID: {task_id}")
        print(f"  State: {task_info['state']}")
        print(f"  Runtime: {task_info.get('runtime', 'N/A')}s")
        print()
    
    return tasks


def get_task_info (task_id):
    """Get detailed info for specific task"""
    response = requests.get (f'{FLOWER_URL}/task/info/{task_id}')
    return response.json()


def get_queue_length (queue_name='celery'):
    """Get queue depth"""
    response = requests.get (f'{FLOWER_URL}/queues/length')
    queues = response.json()
    
    for queue, depth in queues['active_queues']:
        if queue == queue_name:
            return depth
    
    return 0


def shutdown_worker (worker_name):
    """Shutdown worker (if enable_control=True)"""
    response = requests.post(
        f'{FLOWER_URL}/worker/shutdown/{worker_name}'
    )
    return response.json()


def restart_worker (worker_name):
    """Restart worker (if enable_control=True)"""
    response = requests.post(
        f'{FLOWER_URL}/worker/pool/restart/{worker_name}'
    )
    return response.json()


# Usage
if __name__ == '__main__':
    print("=== Workers ===")
    workers = get_workers()
    
    print("\\n=== Recent Tasks ===")
    tasks = get_tasks (limit=10)
    
    print("\\n=== Queue Depth ===")
    default_queue_depth = get_queue_length('celery')
    print(f"Default queue: {default_queue_depth} tasks")
\`\`\`

---

## Monitoring and Alerts

\`\`\`python
"""
Auto-monitoring with Flower API
"""

import requests
import time
import logging

logger = logging.getLogger(__name__)

FLOWER_URL = 'http://localhost:5555/api'
ALERT_THRESHOLDS = {
    'queue_depth': 10000,        # Alert if queue > 10K
    'min_workers': 3,             # Alert if workers < 3
    'failure_rate': 0.1,          # Alert if failure rate > 10%
    'avg_runtime': 60,            # Alert if avg runtime > 60s
}


def monitor_celery_health():
    """Continuous monitoring of Celery health"""
    while True:
        try:
            check_queue_depth()
            check_worker_health()
            check_failure_rate()
            check_task_runtime()
        except Exception as e:
            logger.error (f"Monitoring error: {e}")
        
        time.sleep(60)  # Check every minute


def check_queue_depth():
    """Alert if queue depth exceeds threshold"""
    response = requests.get (f'{FLOWER_URL}/queues/length')
    queues = response.json()
    
    for queue_name, depth in queues.get('active_queues', []):
        if depth > ALERT_THRESHOLDS['queue_depth']:
            alert_ops(
                severity='warning',
                title=f'High Queue Depth: {queue_name}',
                message=f'Queue {queue_name} has {depth} tasks (threshold: {ALERT_THRESHOLDS["queue_depth"]})'
            )


def check_worker_health():
    """Alert if insufficient workers online"""
    response = requests.get (f'{FLOWER_URL}/workers')
    workers = response.json()
    
    online_workers = [w for w, info in workers.items() if info.get('status')]
    
    if len (online_workers) < ALERT_THRESHOLDS['min_workers']:
        alert_ops(
            severity='critical',
            title='Insufficient Workers',
            message=f'Only {len (online_workers)} workers online (required: {ALERT_THRESHOLDS["min_workers"]})'
        )


def check_failure_rate():
    """Alert if task failure rate is high"""
    response = requests.get (f'{FLOWER_URL}/tasks', params={'limit': 1000})
    tasks = response.json()
    
    if not tasks:
        return
    
    failed = sum(1 for t in tasks.values() if t.get('state') == 'FAILURE')
    total = len (tasks)
    failure_rate = failed / total if total > 0 else 0
    
    if failure_rate > ALERT_THRESHOLDS['failure_rate']:
        alert_ops(
            severity='warning',
            title='High Task Failure Rate',
            message=f'Failure rate: {failure_rate:.1%} ({failed}/{total} tasks)'
        )


def check_task_runtime():
    """Alert if average task runtime is high"""
    response = requests.get (f'{FLOWER_URL}/tasks', params={'limit': 100})
    tasks = response.json()
    
    runtimes = [t.get('runtime', 0) for t in tasks.values() if t.get('runtime')]
    
    if runtimes:
        avg_runtime = sum (runtimes) / len (runtimes)
        
        if avg_runtime > ALERT_THRESHOLDS['avg_runtime']:
            alert_ops(
                severity='warning',
                title='High Task Runtime',
                message=f'Average task runtime: {avg_runtime:.1f}s (threshold: {ALERT_THRESHOLDS["avg_runtime"]}s)'
            )


def alert_ops (severity: str, title: str, message: str):
    """Send alert to ops team"""
    logger.warning (f"[{severity.upper()}] {title}: {message}")
    
    # Send to Slack
    # slack_webhook (title, message)
    
    # Send to PagerDuty
    # if severity == 'critical':
    #     pagerduty_alert (title, message)
    
    # Send to email
    # send_email (ops_email, title, message)


if __name__ == '__main__':
    monitor_celery_health()
\`\`\`

---

## Auto-Scaling Workers Based on Queue Depth

\`\`\`python
"""
Auto-scale Celery workers based on Flower metrics
"""

import requests
import subprocess
import logging

logger = logging.getLogger(__name__)

FLOWER_URL = 'http://localhost:5555/api'
MIN_WORKERS = 3
MAX_WORKERS = 20
TASKS_PER_WORKER = 100  # Ideal: 100 tasks per worker


class CeleryAutoScaler:
    def __init__(self):
        self.flower_url = FLOWER_URL
        self.min_workers = MIN_WORKERS
        self.max_workers = MAX_WORKERS
        self.tasks_per_worker = TASKS_PER_WORKER
    
    def get_queue_depth (self):
        """Get total queue depth across all queues"""
        response = requests.get (f'{self.flower_url}/queues/length')
        queues = response.json()
        
        total = 0
        for queue_name, depth in queues.get('active_queues', []):
            total += depth
        
        return total
    
    def get_active_workers (self):
        """Get count of online workers"""
        response = requests.get (f'{self.flower_url}/workers')
        workers = response.json()
        
        return len([w for w, info in workers.items() if info.get('status')])
    
    def calculate_needed_workers (self, queue_depth):
        """Calculate optimal worker count"""
        # Need 1 worker per N tasks
        needed = (queue_depth // self.tasks_per_worker) + self.min_workers
        
        # Clamp to min/max
        return max (self.min_workers, min (needed, self.max_workers))
    
    def scale_workers_kubernetes (self, target_replicas):
        """Scale workers in Kubernetes"""
        try:
            subprocess.run([
                'kubectl', 'scale', 'deployment', 'celery-workers',
                '--replicas', str (target_replicas)
            ], check=True)
            logger.info (f"Scaled to {target_replicas} workers")
        except subprocess.CalledProcessError as e:
            logger.error (f"Failed to scale: {e}")
    
    def scale_workers_docker (self, target_replicas):
        """Scale workers in Docker Swarm"""
        try:
            subprocess.run([
                'docker', 'service', 'scale',
                f'myapp_celery_worker={target_replicas}'
            ], check=True)
            logger.info (f"Scaled to {target_replicas} workers")
        except subprocess.CalledProcessError as e:
            logger.error (f"Failed to scale: {e}")
    
    def run_autoscaling (self, platform='kubernetes'):
        """Run auto-scaling loop"""
        while True:
            try:
                queue_depth = self.get_queue_depth()
                current_workers = self.get_active_workers()
                target_workers = self.calculate_needed_workers (queue_depth)
                
                logger.info(
                    f"Queue: {queue_depth}, "
                    f"Workers: {current_workers}, "
                    f"Target: {target_workers}"
                )
                
                if target_workers != current_workers:
                    if platform == 'kubernetes':
                        self.scale_workers_kubernetes (target_workers)
                    elif platform == 'docker':
                        self.scale_workers_docker (target_workers)
            
            except Exception as e:
                logger.error (f"Auto-scaling error: {e}")
            
            time.sleep(60)  # Check every minute


if __name__ == '__main__':
    scaler = CeleryAutoScaler()
    scaler.run_autoscaling (platform='kubernetes')
\`\`\`

**Kubernetes Horizontal Pod Autoscaler (HPA):**

\`\`\`yaml
# celery-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-workers-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-workers
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_depth
        selector:
          matchLabels:
            queue: default
      target:
        type: AverageValue
        averageValue: "1000"
\`\`\`

---

## Security Best Practices

### 1. Authentication

\`\`\`python
# Generate bcrypt password hash
import bcrypt

password = b"strongpassword"
hashed = bcrypt.hashpw (password, bcrypt.gensalt())
print(hashed.decode())  # Use in flowerconfig.py

# flowerconfig.py
basic_auth = [
    'admin:$2b$12$...'  # Hashed password
]
\`\`\`

### 2. NGINX Reverse Proxy

\`\`\`nginx
# /etc/nginx/sites-available/flower
server {
    listen 443 ssl http2;
    server_name flower.company.com;
    
    ssl_certificate /etc/ssl/certs/flower.crt;
    ssl_certificate_key /etc/ssl/private/flower.key;
    
    # Restrict access to internal network
    allow 10.0.0.0/8;
    deny all;
    
    location / {
        proxy_pass http://localhost:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for real-time updates)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
\`\`\`

### 3. OAuth2 (Google)

\`\`\`python
# flowerconfig.py
oauth2_key = 'your-google-client-id'
oauth2_secret = 'your-google-client-secret'
oauth2_redirect_uri = 'https://flower.company.com/login'
oauth2_domains = ['company.com']  # Only allow company emails

# Users will authenticate via Google OAuth
# Only users with @company.com emails can access
\`\`\`

### 4. Read-Only Mode

\`\`\`python
# flowerconfig.py
# Disable worker control (prevent shutdown/restart via UI)
enable_control = False

# Users can view but not control workers
# Critical for production security
\`\`\`

---

## Summary

**Flower Features:**
- Real-time task monitoring
- Worker management and statistics
- Task history with full details
- Queue depth visualization
- REST API for automation

**Production Deployment:**
- Systemd service for auto-restart
- Docker/Kubernetes for containerized deployment
- NGINX reverse proxy for security
- Authentication (basic auth, OAuth2)
- SSL/TLS encryption

**Monitoring and Alerts:**
- API-based health checks
- Queue depth alerts
- Worker health monitoring
- Task failure rate tracking
- Auto-scaling based on metrics

**Security:**
- Authentication required (basic auth or OAuth)
- NGINX reverse proxy with IP whitelist
- SSL/TLS encryption
- Read-only mode (disable controls)
- Network isolation (internal only)

**Next Section:** Redis vs RabbitMQ comparison! ⚖️
`,
};
