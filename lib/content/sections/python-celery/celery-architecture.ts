export const celeryArchitecture = {
  title: 'Celery Architecture',
  id: 'celery-architecture',
  content: `
# Celery Architecture

## Introduction

**Celery** is Python's most popular distributed task queue. It's production-ready, battle-tested at companies like Instagram, Mozilla, and Robinhood. This section covers Celery's architecture, components, and how they work together.

**Why Celery?**
- **Mature**: 10+ years in production
- **Scalable**: Powers systems processing millions of tasks/day
- **Flexible**: Multiple brokers (Redis, RabbitMQ), result backends, serializers
- **Feature-Rich**: Retries, scheduling, rate limiting, monitoring
- **Well-Documented**: Extensive docs and community support

###Architecture Overview

\`\`\`
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Application   │─────▶│  Message Broker  │◀─────│  Celery Worker   │
│  (Producer)     │ Task │ (Redis/RabbitMQ) │ Pull │  (Consumer)      │
│                 │      │                  │      │                  │
│ • Flask/Django  │      │ • Stores tasks   │      │ • Executes tasks │
│ • FastAPI       │      │ • Queues         │      │ • Returns result │
│ • Scripts       │      │ • Routing        │      │ • Process pool   │
└─────────────────┘      └──────────────────┘      └──────────────────┘
        │                                                    │
        │                                                    ▼
        │                                            ┌──────────────────┐
        └───────────────────────────────────────────▶│  Result Backend  │
                                                     │  (Redis/DB)      │
                                                     │                  │
                                                     │ • Task results   │
                                                     │ • Task states    │
                                                     │ • Meta data      │
                                                     └──────────────────┘
\`\`\`

---

## Celery Components

### 1. **Celery Client** (Producer)

The client is your application code that **creates and queues tasks**.

\`\`\`python
"""
Celery Client: Creates tasks
"""

from celery import Celery

# Initialize Celery app
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Define task
@app.task
def add(x, y):
    return x + y

# CLIENT CODE: Queue task (producer)
def client_code():
    """This runs in your web application"""
    
    # Method 1: delay() - simple
    result = add.delay(4, 6)
    print(f"Task queued: {result.id}")
    
    # Method 2: apply_async() - advanced options
    result = add.apply_async(
        args=(4, 6),
        countdown=60,      # Run after 60 seconds
        expires=3600,      # Expire if not run within 1 hour
        retry=True,        # Enable retries
        priority=9         # High priority (0-9)
    )
    
    return result.id
\`\`\`

**Client Responsibilities:**
- Define tasks with \`@app.task\`
- Queue tasks with \`.delay()\` or \`.apply_async()\`
- Retrieve task results
- Check task status

---

### 2. **Message Broker**

The broker is the **queue manager** that stores and routes tasks.

#### **Broker Options:**

| Feature | Redis | RabbitMQ | SQS (AWS) |
|---------|-------|----------|-----------|
| **Speed** | Very fast (in-memory) | Fast | Moderate |
| **Persistence** | Optional (AOF/RDB) | Yes (disk) | Yes (managed) |
| **Max message size** | 512MB | Unlimited | 256KB |
| **Features** | Simple, lightweight | Advanced routing | Managed, scalable |
| **Reliability** | Good | Excellent | Excellent |
| **Use case** | Most applications | Complex routing | AWS-centric |

#### **Redis as Broker:**

\`\`\`python
"""
Celery with Redis broker
"""

from celery import Celery

app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',    # Redis DB 0 for broker
    backend='redis://localhost:6379/1'    # Redis DB 1 for results
)

# Redis broker configuration
app.conf.broker_connection_retry_on_startup = True
app.conf.broker_connection_retry = True
app.conf.broker_connection_max_retries = 10

# Redis visibility timeout (task claim duration)
app.conf.broker_transport_options = {
    'visibility_timeout': 3600,  # 1 hour
    'fanout_prefix': True,
    'fanout_patterns': True,
}
\`\`\`

**Redis Pros:**
- ✅ Simple setup (\`pip install redis\`)
- ✅ Very fast (in-memory)
- ✅ Good for most use cases
- ✅ Easy to debug (redis-cli)

**Redis Cons:**
- ❌ Less reliable than RabbitMQ (in-memory)
- ❌ No advanced routing features
- ❌ Can lose tasks if Redis crashes (unless AOF enabled)

#### **RabbitMQ as Broker:**

\`\`\`python
"""
Celery with RabbitMQ broker
"""

from celery import Celery

app = Celery(
    'myapp',
    broker='amqp://user:pass@localhost:5672//',
    backend='rpc://'  # RPC result backend
)

# RabbitMQ configuration
app.conf.task_acks_late = True  # Acknowledge task after completion
app.conf.task_reject_on_worker_lost = True  # Requeue if worker dies
app.conf.broker_heartbeat = 30  # Connection heartbeat
app.conf.broker_connection_retry_on_startup = True
\`\`\`

**RabbitMQ Pros:**
- ✅ Most reliable (persistent, disk-backed)
- ✅ Advanced features (exchanges, routing, priority)
- ✅ Battle-tested at scale
- ✅ Better monitoring (management UI)

**RabbitMQ Cons:**
- ❌ More complex setup
- ❌ Slightly slower than Redis
- ❌ Requires separate service

---

### 3. **Celery Worker** (Consumer)

Workers **pull tasks from the queue and execute them**.

\`\`\`python
"""
Running Celery Workers
"""

# Basic worker
celery -A myapp worker --loglevel=info

# Worker with options
celery -A myapp worker \\
    --loglevel=info \\
    --concurrency=4 \\           # 4 parallel worker processes
    --pool=prefork \\             # Process pool (default)
    --queues=default,high-priority \\  # Listen to specific queues
    --hostname=worker1@%h \\      # Worker hostname
    --max-tasks-per-child=1000    # Restart worker after 1000 tasks
\`\`\`

#### **Worker Pools:**

Celery supports different execution pools:

**1. prefork (default)** - Multiprocessing

\`\`\`bash
celery -A myapp worker --pool=prefork --concurrency=4
\`\`\`

- ✅ CPU-bound tasks
- ✅ True parallelism (bypasses GIL)
- ✅ Process isolation (crashes don't affect others)
- ❌ Higher memory usage (each process = separate memory)

**2. threads** - Threading

\`\`\`bash
celery -A myapp worker --pool=threads --concurrency=100
\`\`\`

- ✅ I/O-bound tasks
- ✅ Lower memory (shared memory)
- ❌ GIL limitations (no true parallelism)
- ✅ Good for network requests, file I/O

**3. gevent** - Greenlets (lightweight threads)

\`\`\`bash
celery -A myapp worker --pool=gevent --concurrency=1000
\`\`\`

- ✅ Massive concurrency (1000s of tasks)
- ✅ Very low memory
- ✅ Perfect for I/O-bound tasks (HTTP calls, database)
- ❌ Requires gevent-compatible libraries

**4. solo** - Single thread (debugging)

\`\`\`bash
celery -A myapp worker --pool=solo
\`\`\`

- ✅ Simple debugging
- ❌ No concurrency (sequential execution)
- ✅ Development only

---

### 4. **Result Backend**

Stores task **results** and **state** for later retrieval.

\`\`\`python
"""
Result Backend Configuration
"""

from celery import Celery

# Redis result backend
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'  # Separate DB for results
)

# Database result backend
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='db+postgresql://user:pass@localhost/celery_results'
)

# Disable result backend (if you don't need results)
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend=None  # No result storage
)

# Result backend configuration
app.conf.result_expires = 3600  # Results expire after 1 hour
app.conf.result_compression = 'gzip'  # Compress results
app.conf.result_serializer = 'json'  # JSON serialization
\`\`\`

#### **Backend Options:**

| Backend | Speed | Persistence | Use Case |
|---------|-------|-------------|----------|
| **Redis** | Very fast | Memory + AOF | Most applications |
| **Database** | Moderate | Persistent | Long-term storage |
| **RPC** | Fast | No | Temporary results |
| **None** | N/A | No | Fire-and-forget tasks |

#### **Retrieving Results:**

\`\`\`python
"""
Checking task results
"""

# Queue task
result = add.delay(4, 6)

# Check status
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE, RETRY

# Wait for result (blocking)
output = result.get(timeout=10)  # Waits up to 10 seconds
print(output)  # 10

# Check if ready (non-blocking)
if result.ready():
    print("Task complete!")
    print(result.result)  # 10
else:
    print("Task still running...")

# Check if successful
if result.successful():
    print(f"Result: {result.result}")
elif result.failed():
    print(f"Error: {result.traceback}")
\`\`\`

---

## Complete Architecture Example

\`\`\`python
"""
Complete Celery Architecture Example
"""

# ============================================
# FILE: tasks.py (Task definitions)
# ============================================

from celery import Celery
import time
import requests

# Initialize Celery
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # Hard limit: 1 hour
    task_soft_time_limit=3000,  # Soft limit: 50 minutes
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
    worker_max_tasks_per_child=1000,  # Restart after 1000 tasks
)

# Task 1: Simple addition
@app.task(name='tasks.add')
def add(x, y):
    """Simple CPU-bound task"""
    return x + y

# Task 2: Email sending
@app.task(name='tasks.send_email', bind=True, max_retries=3)
def send_email(self, email: str, subject: str, body: str):
    """I/O-bound task with retries"""
    try:
        # Simulate email sending
        response = requests.post(
            'https://api.mailgun.net/v3/domain/messages',
            data={'to': email, 'subject': subject, 'text': body}
        )
        response.raise_for_status()
        return f"Email sent to {email}"
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

# Task 3: Long-running report generation
@app.task(name='tasks.generate_report', bind=True)
def generate_report(self, user_id: int):
    """Long-running task with progress updates"""
    total_steps = 10
    
    for step in range(total_steps):
        time.sleep(5)  # Simulate work
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': step + 1, 'total': total_steps, 'percent': (step + 1) / total_steps * 100}
        )
    
    return {'status': 'complete', 'report_url': f'/reports/{user_id}.pdf'}


# ============================================
# FILE: app.py (Web application - Producer)
# ============================================

from flask import Flask, request, jsonify
from tasks import add, send_email, generate_report
from celery.result import AsyncResult

app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add_numbers():
    """Queue simple addition task"""
    x = request.json['x']
    y = request.json['y']
    
    result = add.delay(x, y)
    
    return jsonify({
        'task_id': result.id,
        'status': 'queued'
    }), 202

@app.route('/send-email', methods=['POST'])
def queue_email():
    """Queue email task"""
    email = request.json['email']
    subject = request.json['subject']
    body = request.json['body']
    
    result = send_email.delay(email, subject, body)
    
    return jsonify({
        'task_id': result.id,
        'status': 'queued'
    }), 202

@app.route('/generate-report', methods=['POST'])
def queue_report():
    """Queue report generation"""
    user_id = request.json['user_id']
    
    result = generate_report.delay(user_id)
    
    return jsonify({
        'task_id': result.id,
        'status': 'queued'
    }), 202

@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Check task status"""
    result = AsyncResult(task_id, app=add.app)
    
    if result.state == 'PENDING':
        response = {
            'state': result.state,
            'status': 'Task not found or pending...'
        }
    elif result.state == 'PROGRESS':
        response = {
            'state': result.state,
            'current': result.info.get('current', 0),
            'total': result.info.get('total', 1),
            'percent': result.info.get('percent', 0)
        }
    elif result.state == 'SUCCESS':
        response = {
            'state': result.state,
            'result': result.result
        }
    else:  # FAILURE
        response = {
            'state': result.state,
            'error': str(result.info)
        }
    
    return jsonify(response)


# ============================================
# RUNNING THE SYSTEM
# ============================================

# Terminal 1: Start Redis
# $ redis-server

# Terminal 2: Start Celery worker
# $ celery -A tasks worker --loglevel=info --concurrency=4

# Terminal 3: Start Flask app
# $ python app.py

# Terminal 4: Test with curl
# $ curl -X POST http://localhost:5000/add -H "Content-Type: application/json" -d '{"x": 5, "y": 3}'
# Response: {"task_id": "abc123", "status": "queued"}

# $ curl http://localhost:5000/task-status/abc123
# Response: {"state": "SUCCESS", "result": 8}
\`\`\`

---

## Task Routing

Route tasks to **specific queues** for prioritization or specialization.

\`\`\`python
"""
Task Routing Configuration
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Define task routes
app.conf.task_routes = {
    'tasks.send_email': {'queue': 'emails'},
    'tasks.process_video': {'queue': 'videos'},
    'tasks.generate_report': {'queue': 'reports'},
    'tasks.*': {'queue': 'default'},  # Catch-all
}

# Queue priorities
app.conf.task_queue_max_priority = 10
app.conf.task_default_priority = 5

# Tasks
@app.task(name='tasks.send_email')
def send_email(email): pass

@app.task(name='tasks.process_video')
def process_video(video_id): pass

# Start workers for specific queues
# Worker 1: Emails only (10 workers)
# $ celery -A tasks worker -Q emails --concurrency=10

# Worker 2: Videos only (2 workers, GPU instances)
# $ celery -A tasks worker -Q videos --concurrency=2

# Worker 3: Reports only (5 workers)
# $ celery -A tasks worker -Q reports --concurrency=5

# Worker 4: Default queue
# $ celery -A tasks worker -Q default --concurrency=4
\`\`\`

**Benefits of Routing:**
- **Specialization**: GPU workers for video, high-CPU workers for reports
- **Prioritization**: Critical tasks in high-priority queue
- **Isolation**: Slow tasks don't block fast tasks
- **Scaling**: Scale specific queues independently

---

## Installation & Setup

\`\`\`bash
# Install Celery
pip install celery

# Install Redis (broker + backend)
pip install redis

# Or install RabbitMQ support
pip install celery[amqp]

# Install with all extras
pip install celery[redis,auth,msgpack]
\`\`\`

\`\`\`python
"""
Minimal Celery Setup
"""

# tasks.py
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def add(x, y):
    return x + y

# Run worker
# $ celery -A tasks worker --loglevel=info

# Queue task (Python shell or script)
# from tasks import add
# result = add.delay(4, 6)
# print(result.get())  # 10
\`\`\`

---

## Celery Architecture Summary

**4 Core Components:**

1. **Client (Producer)**
   - Your application code
   - Queues tasks with \`.delay()\` or \`.apply_async()\`

2. **Message Broker**
   - Redis or RabbitMQ
   - Stores and routes tasks

3. **Worker (Consumer)**
   - Background processes
   - Executes tasks
   - Multiple worker pools (prefork, threads, gevent)

4. **Result Backend**
   - Redis, database, or RPC
   - Stores task results and states

**Data Flow:**
1. Client creates task → Sends to broker
2. Broker stores task in queue
3. Worker pulls task from broker
4. Worker executes task
5. Worker stores result in backend
6. Client retrieves result from backend

**Next Section:** We'll write your first Celery tasks and queue them from a web application! 🚀
`,
};
