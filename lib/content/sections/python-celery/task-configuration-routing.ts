export const taskConfigurationRouting = {
  title: 'Task Configuration & Routing',
  id: 'task-configuration-routing',
  content: `
# Task Configuration & Routing

## Introduction

**Task configuration and routing** allows you to control how tasks are executed: which queue they go to, their priority, time limits, serialization, and more. This section covers Celery\'s powerful configuration system for production-grade task management.

**Why Configuration Matters:**
- **Performance**: Optimize worker pools, prefetch settings
- **Reliability**: Set retries, timeouts, acks
- **Scalability**: Route tasks to specialized workers
- **Debugging**: Logging, tracing, monitoring

---

## Celery Configuration System

\`\`\`python
"""
Celery Configuration Methods
"""

from celery import Celery

# Method 1: Direct configuration (app.conf.update)
app = Celery('myapp', broker='redis://localhost:6379/0')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Method 2: Configuration dict
app.conf.update({
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
})

# Method 3: Configuration module (celeryconfig.py)
app.config_from_object('celeryconfig')

# Method 4: Django settings
app.config_from_object('django.conf:settings', namespace='CELERY')
\`\`\`

### Configuration File (celeryconfig.py)

\`\`\`python
"""
celeryconfig.py - Centralized configuration
"""

# Broker settings
broker_url = 'redis://localhost:6379/0'
broker_connection_retry_on_startup = True
broker_connection_retry = True
broker_connection_max_retries = 10

# Result backend settings
result_backend = 'redis://localhost:6379/1'
result_expires = 3600  # 1 hour
result_compression = 'gzip'

# Serialization settings
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']

# Timezone settings
timezone = 'UTC'
enable_utc = True

# Task execution settings
task_track_started = True
task_time_limit = 3600  # Hard limit: 1 hour
task_soft_time_limit = 3000  # Soft limit: 50 minutes
task_acks_late = True  # Acknowledge after completion
task_reject_on_worker_lost = True  # Requeue if worker crashes

# Worker settings
worker_prefetch_multiplier = 1  # Fetch 1 task at a time
worker_max_tasks_per_child = 1000  # Restart after 1000 tasks
worker_disable_rate_limits = False
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'

# Queue settings
task_default_queue = 'default'
task_default_exchange = 'default'
task_default_routing_key = 'default'

# Rate limiting
task_annotations = {
    '*': {'rate_limit': '100/m'},  # All tasks: 100/min
    'tasks.send_email': {'rate_limit': '10/s'},  # Specific task
}

# Task routes
task_routes = {
    'tasks.send_email': {'queue': 'emails'},
    'tasks.process_video': {'queue': 'videos'},
    'tasks.generate_report': {'queue': 'reports'},
}
\`\`\`

---

## Task Routing to Queues

**Why Route Tasks?**
- **Isolation**: Slow tasks don't block fast tasks
- **Specialization**: GPU workers for video, CPU for reports
- **Prioritization**: VIP users get dedicated workers
- **Scaling**: Scale specific queues independently

### Basic Routing

\`\`\`python
"""
Task Routing Configuration
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Define task routes
app.conf.task_routes = {
    # Route by task name
    'tasks.send_email': {'queue': 'emails'},
    'tasks.send_sms': {'queue': 'sms'},
    
    # Wildcard routing
    'tasks.process_*': {'queue': 'processing'},
    
    # Namespace routing
    'analytics.*': {'queue': 'analytics'},
    'reporting.*': {'queue': 'reports'},
    
    # Default catch-all
    '*': {'queue': 'default'},
}

# Define tasks
@app.task (name='tasks.send_email')
def send_email (email: str, subject: str, body: str):
    """Routed to 'emails' queue"""
    pass

@app.task (name='tasks.process_video')
def process_video (video_id: int):
    """Routed to 'processing' queue"""
    pass

@app.task (name='analytics.calculate_metrics')
def calculate_metrics (date: str):
    """Routed to 'analytics' queue"""
    pass
\`\`\`

### Starting Workers for Specific Queues

\`\`\`bash
# Worker 1: Email queue (10 workers, threads for I/O)
celery -A tasks worker \\
    -Q emails \\
    --pool=threads \\
    --concurrency=50 \\
    --hostname=email-worker@%h

# Worker 2: Video processing (4 workers, prefork for CPU)
celery -A tasks worker \\
    -Q videos \\
    --pool=prefork \\
    --concurrency=4 \\
    --hostname=video-worker@%h

# Worker 3: Multiple queues (priority: high,default)
celery -A tasks worker \\
    -Q high,default \\
    --concurrency=8 \\
    --hostname=multi-worker@%h

# Worker 4: All queues except low priority
celery -A tasks worker \\
    -X low-priority \\
    --concurrency=4 \\
    --hostname=main-worker@%h
\`\`\`

---

## Priority Queues

\`\`\`python
"""
Priority Queue Configuration
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Enable priority queues (RabbitMQ only, Redis limited)
app.conf.task_queue_max_priority = 10
app.conf.task_default_priority = 5

# Define priority routes
app.conf.task_routes = {
    'tasks.urgent_*': {'queue': 'high-priority', 'priority': 9},
    'tasks.batch_*': {'queue': 'low-priority', 'priority': 1},
}

# Tasks
@app.task (name='tasks.urgent_notification')
def urgent_notification (user_id: int):
    """High priority task"""
    pass

@app.task (name='tasks.batch_analytics')
def batch_analytics (date: str):
    """Low priority task"""
    pass

# Queue with custom priority
result = urgent_notification.apply_async(
    kwargs={'user_id': 12345},
    priority=9  # Highest priority
)

result = batch_analytics.apply_async(
    kwargs={'date': '2024-01-01'},
    priority=1  # Lowest priority
)
\`\`\`

**Priority Queue Example:**

\`\`\`
Queue: [Task1(p=9)] [Task2(p=7)] [Task3(p=5)] [Task4(p=3)] [Task5(p=1)]
           ↓              ↓          ↓           ↓           ↓
     Urgent(9)     VIP(7)      Normal(5)   Bulk(3)    Cleanup(1)
     
Worker picks highest priority first: Task1 → Task2 → Task3 → Task4 → Task5
\`\`\`

---

## Dynamic Routing

\`\`\`python
"""
Dynamic Task Routing Based on Runtime Conditions
"""

from celery import Celery
from typing import Dict, Any

app = Celery('myapp', broker='redis://localhost:6379/0')

# Custom router function
def route_task (name: str, args: tuple, kwargs: Dict[str, Any],
               options: dict, task=None, **kw):
    """
    Dynamic routing based on task arguments
    
    Returns: {'queue': 'queue_name', 'priority': 0-9}
    """
    # Route based on user tier
    if 'user_id' in kwargs:
        user = get_user (kwargs['user_id'])
        if user.is_vip:
            return {'queue': 'vip', 'priority': 9}
        elif user.is_premium:
            return {'queue': 'premium', 'priority': 7}
        else:
            return {'queue': 'default', 'priority': 5}
    
    # Route based on data size
    if 'data_size' in kwargs:
        if kwargs['data_size'] > 1_000_000:  # > 1MB
            return {'queue': 'large-data', 'priority': 3}
        else:
            return {'queue': 'small-data', 'priority': 7}
    
    # Route based on time of day
    import datetime
    hour = datetime.datetime.now().hour
    if 9 <= hour <= 17:  # Business hours
        return {'queue': 'peak-hours', 'priority': 8}
    else:  # Off-peak
        return {'queue': 'off-peak', 'priority': 5}
    
    # Default
    return {'queue': 'default', 'priority': 5}

# Register router
app.conf.task_routes = (route_task,)

# Example usage
@app.task
def process_order (user_id: int, order_id: int):
    """Dynamically routed based on user tier"""
    pass

# VIP user → 'vip' queue, priority 9
process_order.delay (user_id=12345, order_id=999)

# Regular user → 'default' queue, priority 5
process_order.delay (user_id=67890, order_id=888)
\`\`\`

---

## Task-Specific Configuration

\`\`\`python
"""
Configure individual tasks
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Task with custom settings
@app.task(
    name='tasks.send_email',
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # Retry after 60s
    time_limit=300,  # 5 minutes hard limit
    soft_time_limit=270,  # 4.5 minutes soft limit
    rate_limit='10/s',  # 10 per second
    ignore_result=True,  # Don't store result
    acks_late=True,  # Acknowledge after completion
    reject_on_worker_lost=True,  # Requeue if worker crashes
    autoretry_for=(SMTPException,),  # Auto-retry on SMTP errors
    retry_kwargs={'max_retries': 5},
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes backoff
    retry_jitter=True,  # Add randomness to backoff
)
def send_email (self, email: str, subject: str, body: str):
    """Email task with comprehensive configuration"""
    try:
        smtp_send (email, subject, body)
    except SMTPException as exc:
        raise self.retry (exc=exc)

# Task with compression
@app.task(
    compression='gzip',  # Compress task payload
    serializer='msgpack'  # Fast binary serialization
)
def process_large_data (data: dict):
    """Task handling large payloads"""
    pass

# Task with custom queue
@app.task(
    queue='priority-queue',
    priority=9
)
def urgent_task():
    """Always goes to priority queue"""
    pass
\`\`\`

---

## Task Annotations

\`\`\`python
"""
Task Annotations: Apply settings to multiple tasks
"""

app.conf.task_annotations = {
    # Apply to all tasks
    '*': {
        'rate_limit': '100/m',  # 100 per minute
        'time_limit': 3600,  # 1 hour
        'soft_time_limit': 3000,  # 50 minutes
    },
    
    # Apply to specific task
    'tasks.send_email': {
        'rate_limit': '10/s',
        'time_limit': 300,
        'max_retries': 5,
    },
    
    # Apply to namespace
    'analytics.*': {
        'rate_limit': '50/m',
        'queue': 'analytics',
    },
    
    # Apply to pattern
    'tasks.process_*': {
        'queue': 'processing',
        'acks_late': True,
    },
}
\`\`\`

---

## Rate Limiting

\`\`\`python
"""
Task Rate Limiting
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Global rate limit
app.conf.task_default_rate_limit = '100/m'  # 100 per minute

# Task-specific rate limits
@app.task (rate_limit='10/s')  # 10 per second
def send_email (email: str):
    pass

@app.task (rate_limit='5/m')  # 5 per minute
def call_external_api (endpoint: str):
    pass

@app.task (rate_limit='1/h')  # 1 per hour
def expensive_operation():
    pass

# Rate limit formats
# '10/s' - 10 per second
# '100/m' - 100 per minute
# '1000/h' - 1000 per hour
# '10000/d' - 10,000 per day

# Annotations for rate limiting
app.conf.task_annotations = {
    'tasks.send_email': {'rate_limit': '10/s'},
    'tasks.call_api': {'rate_limit': '100/m'},
    'tasks.expensive': {'rate_limit': '5/h'},
}
\`\`\`

---

## Time Limits

\`\`\`python
"""
Task Time Limits
"""

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

app = Celery('myapp', broker='redis://localhost:6379/0')

# Global time limits
app.conf.task_time_limit = 3600  # 1 hour hard limit
app.conf.task_soft_time_limit = 3000  # 50 minutes soft limit

# Task with time limits
@app.task(
    time_limit=300,  # 5 minutes hard limit (SIGKILL)
    soft_time_limit=270  # 4.5 minutes soft limit (exception)
)
def long_running_task():
    """Task with time limits"""
    try:
        # Long operation
        result = expensive_computation()
        return result
    except SoftTimeLimitExceeded:
        # Clean up before hard limit
        cleanup()
        raise

# Usage in task
@app.task (bind=True, soft_time_limit=600)
def process_data (self):
    """Handle soft time limit"""
    try:
        for item in large_dataset:
            process_item (item)
    except SoftTimeLimitExceeded:
        # Save progress
        save_checkpoint (self.request.id)
        # Re-queue continuation
        process_data_continuation.delay (checkpoint_id=self.request.id)
        raise
\`\`\`

**Time Limit Behavior:**
- **Soft limit**: Raises \`SoftTimeLimitExceeded\` exception (graceful)
- **Hard limit**: Sends SIGKILL to worker process (forced termination)
- Set soft limit < hard limit (gives time for cleanup)

---

## Serialization

\`\`\`python
"""
Task Serialization Configuration
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Available serializers
app.conf.task_serializer = 'json'  # Default, safe
app.conf.result_serializer = 'json'
app.conf.accept_content = ['json']  # Security: only accept JSON

# Other serializers:
# 'json' - Safe, compatible, readable (default)
# 'pickle' - Fast, any Python object (security risk!)
# 'msgpack' - Fast, compact, binary
# 'yaml' - Readable, slow

# Using pickle (security risk in production!)
app.conf.task_serializer = 'pickle'
app.conf.result_serializer = 'pickle'
app.conf.accept_content = ['pickle', 'json']

# Using msgpack (good for performance)
app.conf.task_serializer = 'msgpack'
app.conf.result_serializer = 'msgpack'
app.conf.accept_content = ['msgpack', 'json']

# Per-task serializer
@app.task (serializer='msgpack')
def binary_data_task (data: bytes):
    """Use msgpack for binary data"""
    pass

# Compression
app.conf.result_compression = 'gzip'  # Compress results
app.conf.task_compression = 'gzip'  # Compress tasks

@app.task (compression='gzip')
def large_payload_task (data: dict):
    """Compressed task"""
    pass
\`\`\`

**Serialization Comparison:**

| Serializer | Speed | Size | Safety | Use Case |
|------------|-------|------|--------|----------|
| **json** | Moderate | Large | ✅ Safe | Default, production |
| **pickle** | Fast | Moderate | ❌ Unsafe | Trusted networks only |
| **msgpack** | Very fast | Small | ✅ Safe | High performance |
| **yaml** | Slow | Large | ✅ Safe | Readable configs |

**Security Note:** Never use pickle with untrusted task sources (security vulnerability).

---

## Complete Production Configuration

\`\`\`python
"""
Production-Ready Celery Configuration
"""

# celeryconfig.py

# ======================
# Broker Configuration
# ======================
broker_url = 'redis://localhost:6379/0'
broker_connection_retry_on_startup = True
broker_connection_retry = True
broker_connection_max_retries = 10
broker_heartbeat = 30
broker_transport_options = {
    'visibility_timeout': 3600,  # 1 hour
    'fanout_prefix': True,
    'fanout_patterns': True,
}

# ======================
# Result Backend
# ======================
result_backend = 'redis://localhost:6379/1'
result_expires = 3600  # 1 hour
result_compression = 'gzip'
result_extended = True  # Store task args/kwargs
result_backend_transport_options = {
    'retry_policy': {
        'timeout': 5.0
    }
}

# ======================
# Serialization
# ======================
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
task_compression = 'gzip'

# ======================
# Task Execution
# ======================
task_track_started = True
task_time_limit = 3600  # 1 hour
task_soft_time_limit = 3000  # 50 minutes
task_acks_late = True
task_reject_on_worker_lost = True
task_ignore_result = False
task_store_errors_even_if_ignored = True

# ======================
# Task Routing
# ======================
task_default_queue = 'default'
task_default_exchange = 'default'
task_default_exchange_type = 'direct'
task_default_routing_key = 'default'

task_routes = {
    'tasks.send_email': {'queue': 'emails'},
    'tasks.send_sms': {'queue': 'sms'},
    'tasks.process_video': {'queue': 'videos'},
    'tasks.generate_report': {'queue': 'reports'},
    'analytics.*': {'queue': 'analytics'},
    '*': {'queue': 'default'},
}

# ======================
# Task Priority
# ======================
task_queue_max_priority = 10
task_default_priority = 5

# ======================
# Task Annotations
# ======================
task_annotations = {
    '*': {
        'rate_limit': '100/m',
        'time_limit': 3600,
        'soft_time_limit': 3000,
    },
    'tasks.send_email': {
        'rate_limit': '10/s',
        'max_retries': 3,
    },
    'tasks.call_external_api': {
        'rate_limit': '50/m',
        'max_retries': 5,
    },
}

# ======================
# Worker Configuration
# ======================
worker_prefetch_multiplier = 1  # Fetch 1 task at a time
worker_max_tasks_per_child = 1000  # Restart worker after 1000 tasks
worker_max_memory_per_child = 200000  # 200MB per worker
worker_disable_rate_limits = False
worker_send_task_events = True  # For monitoring
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s'

# ======================
# Beat Configuration (Periodic Tasks)
# ======================
beat_schedule = {
    'cleanup-every-hour': {
        'task': 'tasks.cleanup',
        'schedule': 3600.0,  # Every hour
    },
    'report-every-day': {
        'task': 'tasks.daily_report',
        'schedule': crontab (hour=0, minute=0),  # Midnight
    },
}

# ======================
# Monitoring & Logging
# ======================
task_send_sent_event = True  # Track task lifecycle
worker_send_task_events = True
task_track_started = True

# ======================
# Security
# ======================
accept_content = ['json']  # Only JSON (no pickle)
task_serializer = 'json'
result_serializer = 'json'

# ======================
# Performance
# ======================
broker_pool_limit = 10  # Connection pool size
result_backend_max_retries = 10
redis_max_connections = 50
\`\`\`

---

## Summary

**Key Concepts:**
- **Configuration**: Use celeryconfig.py for centralized settings
- **Routing**: Direct tasks to specific queues for isolation/specialization
- **Priority**: High-priority tasks processed first (RabbitMQ)
- **Rate Limiting**: Control task execution rate
- **Time Limits**: Soft (exception) and hard (SIGKILL) limits
- **Serialization**: JSON (safe), msgpack (fast), pickle (risky)

**Best Practices:**
- Use \`task_acks_late=True\` (acknowledge after completion)
- Set \`worker_max_tasks_per_child=1000\` (prevent memory leaks)
- Enable \`task_reject_on_worker_lost=True\` (requeue on crash)
- Use JSON serialization (security)
- Compress large payloads (gzip)
- Route tasks to specialized workers
- Set reasonable time limits

**Next Section:** Celery Beat for scheduled/periodic tasks! ⏰
`,
};
