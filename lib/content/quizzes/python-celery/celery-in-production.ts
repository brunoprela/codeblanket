/**
 * Quiz questions for Celery in Production section
 */

export const celeryInProductionQuiz = [
  {
    id: 'q1',
    question:
      'Design a production Celery deployment on Kubernetes with auto-scaling, monitoring, and high availability.',
    sampleAnswer:
      'PRODUCTION K8S DEPLOYMENT: ```yaml # Worker Deployment apiVersion: apps/v1 kind: Deployment metadata: name: celery-worker spec: replicas: 3 selector: matchLabels: app: celery-worker template: spec: containers: - name: worker image: myapp:latest command: ["celery", "-A", "tasks", "worker"] resources: requests: {memory: "512Mi", cpu: "500m"} limits: {memory: "1Gi", cpu: "1000m"} --- # HorizontalPodAutoscaler apiVersion: autoscaling/v2 kind: HorizontalPodAutoscaler metadata: name: celery-worker-hpa spec: scaleTargetRef: kind: Deployment name: celery-worker minReplicas: 3 maxReplicas: 20 metrics: - type: External external: metric: name: celery_queue_depth target: type: AverageValue averageValue: "1000" --- # Flower Monitoring apiVersion: v1 kind: Service metadata: name: flower spec: selector: app: flower ports: - port: 5555 type: LoadBalancer ``` FEATURES: 3-20 workers (auto-scale), HPA based on queue depth, Flower monitoring (LoadBalancer), Resource limits (prevent OOM), High availability (multi-replica).',
    keyPoints: [
      'K8s Deployment with replicas',
      'HorizontalPodAutoscaler',
      'Flower monitoring service',
      'Resource limits',
      'Auto-scaling on queue depth',
    ],
  },
  {
    id: 'q2',
    question: 'What are critical monitoring metrics for production Celery?',
    sampleAnswer:
      'PRODUCTION METRICS: **1. Queue Depth**: Alert if > 10K, Scale workers if sustained high depth. **2. Worker Health**: Active worker count, CPU/memory per worker, Alert if workers < 3. **3. Task Rates**: Tasks/second (throughput), Success rate (%), Failure rate (%). **4. Latency**: Queue time (task submitted → started), Execution time (task duration), Total time (submitted → complete). **5. Error Rates**: Failed tasks, Retry count, Dead letter queue size. **IMPLEMENTATION**: ```python from prometheus_client import Counter, Histogram, Gauge task_success = Counter("celery_task_success", ["task"]) task_failure = Counter("celery_task_failure", ["task"]) task_duration = Histogram("celery_task_duration_seconds", ["task"]) queue_depth = Gauge("celery_queue_depth", ["queue"]) worker_count = Gauge("celery_worker_count") ``` ALERTS: Queue depth > 10K, Workers < 3, Failure rate > 10%, Task duration p95 > 60s.',
    keyPoints: [
      'Queue depth',
      'Worker health',
      'Task success/failure rates',
      'Latency metrics',
      'Prometheus + Grafana',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a disaster recovery strategy for Celery including broker failure, worker failure, and task loss.',
    sampleAnswer:
      'DISASTER RECOVERY STRATEGY: **1. Broker Failure (Redis/RabbitMQ)**: Setup: Redis Sentinel (HA) or RabbitMQ cluster. Failover: Automatic within seconds. Recovery: Workers reconnect automatically. **2. Worker Failure**: Setup: Run 5+ workers (N+2 redundancy). Detection: Health checks every 30s. Recovery: Systemd/K8s auto-restart. **3. Task Loss**: Prevention: Use RabbitMQ (persistent) or Redis AOF. Detection: Task monitoring + alerting. Recovery: Idempotent tasks (safe to retry). **4. Complete Outage**: Backup: Database task queue backup. Recovery: Replay failed tasks from DB. **5. Data Loss**: Backup: Regular broker snapshots. Recovery: Restore from snapshot + replay. **IMPLEMENTATION**: ```python # Idempotent tasks @app.task def process_order (order_id): if already_processed (order_id): return order = Order.get (order_id) process (order) mark_processed (order_id) # Task backup def backup_failed_tasks(): failed = Task.query.filter_by (status="FAILURE") for task in failed: store_in_db (task) ``` RTO (Recovery Time): < 5 minutes. RPO (Data Loss): < 1 minute (with AOF/RabbitMQ).',
    keyPoints: [
      'Redis Sentinel / RabbitMQ cluster for HA',
      'N+2 worker redundancy',
      'Idempotent tasks',
      'Task backup to database',
      'RTO < 5min, RPO < 1min',
    ],
  },
];
