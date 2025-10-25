/**
 * Quiz questions for Task Monitoring with Flower section
 */

export const monitoringWithFlowerQuiz = [
  {
    id: 'q1',
    question:
      'Design a comprehensive monitoring strategy for production Celery using Flower, Prometheus, and custom health checks. Include alerts for queue depth, worker health, and task failure rates.',
    sampleAnswer:
      'PRODUCTION MONITORING STRATEGY: **Component 1: Flower (Real-time Dashboard)**: ```bash celery -A tasks flower --port=5555 --basic_auth=admin:pass ``` Monitor: Active tasks, worker status, queue depth, task history. **Component 2: Prometheus Metrics**: ```python from prometheus_client import Counter, Histogram, Gauge task_success = Counter("celery_task_success_total", "Tasks succeeded", ["task_name"]) task_failure = Counter("celery_task_failure_total", "Tasks failed", ["task_name"]) task_duration = Histogram("celery_task_duration_seconds", "Task duration", ["task_name"]) queue_depth = Gauge("celery_queue_depth", "Queue depth", ["queue_name"]) worker_count = Gauge("celery_worker_count", "Active workers") @app.task(bind=True) def monitored_task(self): start = time.time() try: result = process() task_success.labels(task_name=self.name).inc() return result except Exception as exc: task_failure.labels(task_name=self.name).inc() raise finally: duration = time.time() - start task_duration.labels(task_name=self.name).observe(duration) ``` **Component 3: Health Checks**: ```python @app.task def health_check(): try: # Check Redis redis_client.ping() # Check workers inspect = app.control.inspect() active_workers = len(inspect.active_queues() or {}) worker_count.set(active_workers) if active_workers < 3: alert_ops("Only {active_workers} workers active!") # Check queue depth for queue in ["default", "emails", "reports"]: depth = get_queue_depth(queue) queue_depth.labels(queue_name=queue).set(depth) if depth > 10000: alert_ops(f"Queue {queue} depth: {depth}") except Exception as e: alert_ops(f"Health check failed: {e}") # Schedule every 5 minutes app.conf.beat_schedule = {"health-check": {"task": "tasks.health_check", "schedule": 300.0}} ``` **Component 4: Alerts** (Prometheus AlertManager): ```yaml groups: - name: celery_alerts rules: - alert: HighQueueDepth expr: celery_queue_depth > 10000 for: 5m labels: severity: warning annotations: summary: "High queue depth on {{ $labels.queue_name }}" - alert: WorkersDown expr: celery_worker_count < 3 for: 1m labels: severity: critical annotations: summary: "Only {{ $value }} workers active" - alert: HighFailureRate expr: rate(celery_task_failure_total[5m]) > 0.1 for: 2m labels: severity: warning annotations: summary: "Task failure rate > 10%" ``` **Component 5: Grafana Dashboards**: Panels: Worker count (time series), Queue depth per queue (graph), Task success/failure rates (graph), Task duration p50/p95/p99 (heatmap), Active tasks count (gauge). **BENEFITS**: Real-time visibility (Flower), Historical metrics (Prometheus), Automated alerts (AlertManager), Beautiful dashboards (Grafana), Comprehensive health checks.',
    keyPoints: [
      'Flower: Real-time dashboard',
      'Prometheus: Historical metrics',
      'Health checks: Every 5 minutes',
      'Alerts: Queue depth, worker count, failure rate',
      'Grafana: Visualization',
    ],
  },
  {
    id: 'q2',
    question:
      'How do you use Flower API programmatically to auto-scale workers based on queue depth?',
    sampleAnswer:
      'AUTO-SCALING WITH FLOWER API: ```python import requests class CeleryAutoScaler: def __init__(self, flower_url, min_workers=3, max_workers=20): self.flower_url = flower_url self.min_workers = min_workers self.max_workers = max_workers def get_queue_depth(self): """Get total queue depth from Flower""" response = requests.get(f"{self.flower_url}/api/queues/length") queues = response.json() return sum(queues.values()) def get_active_workers(self): """Get active worker count""" response = requests.get(f"{self.flower_url}/api/workers") workers = response.json() return len([w for w in workers.values() if w["status"]]) def calculate_needed_workers(self, queue_depth): """Calculate needed workers based on queue depth""" # Assume 1 worker handles 100 tasks/min needed = (queue_depth // 100) + self.min_workers return min(max(needed, self.min_workers), self.max_workers) def scale_workers(self, target): """Scale workers (Kubernetes example)""" current = self.get_active_workers() if target > current: self.scale_up(target - current) elif target < current: self.scale_down(current - target) def scale_up(self, count): """Add workers""" kubectl_scale(deployment="celery-workers", replicas=f"+{count}") def scale_down(self, count): """Remove workers""" kubectl_scale(deployment="celery-workers", replicas=f"-{count}") def run_autoscaling(self): """Run auto-scaling loop""" while True: queue_depth = self.get_queue_depth() current_workers = self.get_active_workers() target_workers = self.calculate_needed_workers(queue_depth) logger.info(f"Queue: {queue_depth}, Workers: {current_workers}, Target: {target_workers}") if target_workers != current_workers: self.scale_workers(target_workers) time.sleep(60) # Check every minute # Usage scaler = CeleryAutoScaler(flower_url="http://localhost:5555") scaler.run_autoscaling() ``` **KUBERNETES AUTO-SCALING**: ```yaml apiVersion: autoscaling/v2 kind: HorizontalPodAutoscaler metadata: name: celery-workers spec: scaleTargetRef: apiVersion: apps/v1 kind: Deployment name: celery-workers minReplicas: 3 maxReplicas: 20 metrics: - type: External external: metric: name: celery_queue_depth target: type: AverageValue averageValue: "1000" ``` BENEFITS: Auto-scales based on load, Cost optimization (scale down when idle), Prevents queue buildup, Maintains target queue depth.',
    keyPoints: [
      'Flower API: Get queue depth and worker count',
      'Calculate needed workers: queue_depth / tasks_per_worker',
      'Scale command: kubectl scale',
      'Auto-scaling loop: Check every minute',
      'Kubernetes HPA: Built-in auto-scaling',
    ],
  },
  {
    id: 'q3',
    question:
      'Secure Flower in production with authentication, SSL, and reverse proxy setup.',
    sampleAnswer:
      'PRODUCTION FLOWER SECURITY: **1. Basic Authentication**: ```bash celery -A tasks flower --basic_auth=admin:strong_password,user:password ``` Or configuration: ```python # flowerconfig.py basic_auth = ["admin:$2b$12$hash", "user:$2b$12$hash"] # Bcrypt hashes ``` **2. SSL/TLS**: ```bash celery -A tasks flower --certfile=/etc/ssl/certs/flower.crt --keyfile=/etc/ssl/private/flower.key ``` **3. Reverse Proxy (NGINX)**: ```nginx server { listen 443 ssl; server_name flower.company.com; ssl_certificate /etc/ssl/certs/flower.crt; ssl_certificate_key /etc/ssl/private/flower.key; # Restrict access allow 10.0.0.0/8; # Internal network deny all; location / { proxy_pass http://localhost:5555; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; proxy_set_header X-Forwarded-Proto $scheme; # WebSocket support proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; } } ``` **4. OAuth (Google Auth)**: ```python # flowerconfig.py oauth2_key = "google_client_id" oauth2_secret = "google_client_secret" oauth2_redirect_uri = "https://flower.company.com/login" oauth2_domains = ["company.com"] # Restrict to domain ``` **5. Network Isolation**: ```yaml # docker-compose.yml services: flower: networks: - internal # Not exposed to internet ports: - "127.0.0.1:5555:5555" # Only localhost ``` **6. Read-Only Mode**: Disable worker control: ```python # flowerconfig.py enable_control = False # Disable shutdown/restart ``` **PRODUCTION CHECKLIST**: ✅ Basic auth with strong passwords, ✅ SSL/TLS encryption, ✅ Reverse proxy (NGINX), ✅ IP whitelist (internal network), ✅ OAuth for SSO, ✅ Read-only mode (disable controls), ✅ Network isolation (internal only).',
    keyPoints: [
      'Basic auth with bcrypt hashes',
      'SSL/TLS encryption',
      'NGINX reverse proxy with IP whitelist',
      'OAuth for enterprise SSO',
      'Read-only mode (disable worker controls)',
    ],
  },
];
