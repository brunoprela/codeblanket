export const redisVsRabbitmq = {
  title: 'Redis vs RabbitMQ as Message Broker',
  id: 'redis-vs-rabbitmq',
  content: `
# Redis vs RabbitMQ as Message Broker

## Introduction

Choosing the right message broker is **critical** for your Celery deployment. The broker is the central nervous system of your task queue—it stores tasks, routes them to workers, and ensures reliable delivery.

**Two Most Popular Options:**
1. **Redis**: Fast, simple, in-memory data store
2. **RabbitMQ**: Robust, feature-rich message queue

**Key Decision Factors:**
- Speed vs reliability trade-off
- Setup complexity
- Persistence requirements
- Scale and throughput
- Features needed (routing, priority queues)

This section provides a **comprehensive comparison** to help you choose the right broker for your use case.

---

## Quick Comparison Table

| Feature | Redis | RabbitMQ | Winner |
|---------|-------|----------|--------|
| **Speed** | ⚡ Very fast (in-memory) | 🐇 Fast (disk + memory) | Redis |
| **Reliability** | ⚠️ Good (can lose tasks) | ✅ Excellent (persistent) | RabbitMQ |
| **Setup** | 🎯 Simple (5 min) | 🔧 Complex (30 min) | Redis |
| **Persistence** | Optional (AOF/RDB) | ✅ Disk-backed | RabbitMQ |
| **Message Size** | 512MB max | ♾️ Unlimited | RabbitMQ |
| **Routing** | Basic (simple queues) | 🎯 Advanced (exchanges, routing keys) | RabbitMQ |
| **Priority Queues** | ❌ Not natively | ✅ Yes | RabbitMQ |
| **Monitoring** | redis-cli, RedisInsight | 🎯 Built-in UI (port 15672) | RabbitMQ |
| **Community** | Huge | Huge | Tie |
| **Use Case** | MVPs, simple apps, <100K tasks/day | Enterprise, critical systems, >1M tasks/day | Depends |
| **Cost** | 💰 Cheaper (less resources) | 💰💰 More expensive | Redis |

---

## Redis as Message Broker

### Architecture

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    Redis Server                         │
│                     (In-Memory)                         │
│ ┌───────────────────────────────────────────────────┐  │
│ │  Queue: celery (LIST data structure)              │  │
│ │  ┌──────┬──────┬──────┬──────┬──────┐             │  │
│ │  │Task 1│Task 2│Task 3│Task 4│Task 5│             │  │
│ │  └──────┴──────┴──────┴──────┴──────┘             │  │
│ └───────────────────────────────────────────────────┘  │
│                                                         │
│  Optional: Persistence to disk (AOF/RDB)               │
│  - AOF: Append-only file (every command)              │
│  - RDB: Snapshot (periodic dumps)                      │
└─────────────────────────────────────────────────────────┘
\`\`\`

### Setup (Simple)

\`\`\`bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Test
redis-cli ping  # Should return: PONG

# Configure Celery to use Redis
\`\`\`

\`\`\`python
# tasks.py
from celery import Celery

app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',  # Redis broker
    backend='redis://localhost:6379/1'   # Redis result backend
)

@app.task
def add (x, y):
    return x + y

# Usage
result = add.delay(4, 5)
\`\`\`

**Total setup time: ~5 minutes** ✅

### Advantages of Redis

**1. Speed**
\`\`\`python
# Redis is ~2-3× faster than RabbitMQ
# Benchmark: 100,000 tasks

import time
from celery import Celery

app = Celery('bench', broker='redis://localhost:6379/0')

@app.task
def fast_task (x):
    return x * 2

# Queue 100K tasks
start = time.time()
for i in range(100_000):
    fast_task.delay (i)
duration = time.time() - start

print(f"Redis: {duration:.2f}s")  # ~10 seconds
# RabbitMQ would take ~20-30 seconds
\`\`\`

**2. Simplicity**
- No complex configuration
- Single Redis instance sufficient for most apps
- Easy debugging with redis-cli

**3. Lower Resource Usage**
- Uses ~50% less memory than RabbitMQ
- Lower CPU usage
- Cheaper hosting costs

**4. Good for Most Use Cases**
- 90% of applications don't need RabbitMQ's features
- Redis "good enough" for <100K tasks/day

### Disadvantages of Redis

**1. Task Loss Risk**
\`\`\`python
# Scenario: Redis crashes
# Result: All queued tasks LOST! 💥

# Example:
# 1. Queue 10,000 tasks
for i in range(10_000):
    process_order.delay (order_id=i)

# 2. Redis crashes before workers process them
# 3. Restart Redis
# 4. All 10,000 tasks are GONE! 😱

# Mitigation: Enable AOF persistence
\`\`\`

**Redis Persistence Options:**

\`\`\`conf
# /etc/redis/redis.conf

# Option 1: RDB (Snapshots)
save 900 1      # Save after 900s if 1 key changed
save 300 10     # Save after 300s if 10 keys changed
save 60 10000   # Save after 60s if 10K keys changed
# Pros: Fast, small files
# Cons: Can lose data between snapshots

# Option 2: AOF (Append-Only File)
appendonly yes
appendfsync everysec  # Sync every second
# Pros: Minimal data loss (max 1 second)
# Cons: Slower, larger files

# Restart Redis after config change
sudo systemctl restart redis
\`\`\`

**2. No Advanced Routing**
- Only simple queues
- No complex routing rules
- No topic exchanges or fanout

**3. Limited to 512MB per message**

---

## RabbitMQ as Message Broker

### Architecture

\`\`\`
┌──────────────────────────────────────────────────────────────┐
│                     RabbitMQ Server                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Exchanges (Routing Logic)                            │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐                  │  │
│  │  │ Direct │  │ Topic  │  │ Fanout │                  │  │
│  │  └───┬────┘  └───┬────┘  └───┬────┘                  │  │
│  └──────┼───────────┼───────────┼────────────────────────┘  │
│         │           │           │                            │
│  ┌──────▼───────────▼───────────▼────────────────────────┐  │
│  │  Queues (Persistent on Disk)                          │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐            │  │
│  │  │ default │  │  emails  │  │ reports  │            │  │
│  │  └─────────┘  └──────────┘  └──────────┘            │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  Persistence: All messages written to disk                  │
│  Reliability: Survives restarts, no task loss              │
└──────────────────────────────────────────────────────────────┘
\`\`\`

### Setup (More Complex)

\`\`\`bash
# Install RabbitMQ (Ubuntu/Debian)
sudo apt update
sudo apt install rabbitmq-server

# Start RabbitMQ
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# Enable management plugin (web UI)
sudo rabbitmq-plugins enable rabbitmq_management

# Create user
sudo rabbitmqctl add_user myuser mypassword
sudo rabbitmqctl set_user_tags myuser administrator
sudo rabbitmqctl set_permissions -p / myuser ".*" ".*" ".*"

# Access web UI: http://localhost:15672
# Username: myuser, Password: mypassword

# Configure Celery to use RabbitMQ
\`\`\`

\`\`\`python
# tasks.py
from celery import Celery

app = Celery(
    'myapp',
    broker='amqp://myuser:mypassword@localhost:5672//',  # RabbitMQ broker
    backend='rpc://'  # RabbitMQ result backend
)

# Or with connection options
app = Celery(
    'myapp',
    broker='amqp://myuser:mypassword@localhost:5672//',
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
)

@app.task
def add (x, y):
    return x + y

# Usage
result = add.delay(4, 5)
\`\`\`

**Total setup time: ~30 minutes** ⚠️

### Advantages of RabbitMQ

**1. Reliability (No Task Loss)**
\`\`\`python
# Scenario: RabbitMQ crashes
# Result: All tasks SAFE (persisted to disk) ✅

# Example:
# 1. Queue 10,000 tasks
for i in range(10_000):
    process_payment.delay (order_id=i)

# 2. RabbitMQ crashes
# 3. Restart RabbitMQ
# 4. All 10,000 tasks still in queue! ✨
# 5. Workers resume processing

# No data loss!
\`\`\`

**2. Advanced Routing**
\`\`\`python
# Route tasks to different queues based on routing keys

from celery import Celery

app = Celery('myapp', broker='amqp://localhost')

# Define queues
app.conf.task_queues = {
    'emails': {'exchange': 'tasks', 'routing_key': 'email'},
    'reports': {'exchange': 'tasks', 'routing_key': 'report'},
    'critical': {'exchange': 'tasks', 'routing_key': 'critical'},
}

# Route tasks
app.conf.task_routes = {
    'tasks.send_email': {'queue': 'emails'},
    'tasks.generate_report': {'queue': 'reports'},
    'tasks.process_payment': {'queue': 'critical'},
}

@app.task
def send_email (to, subject):
    pass  # Goes to 'emails' queue

@app.task
def process_payment (order_id):
    pass  # Goes to 'critical' queue
\`\`\`

**3. Priority Queues**
\`\`\`python
# High-priority tasks processed first

app.conf.task_default_priority = 5
app.conf.task_acks_late = True

@app.task (priority=9)  # High priority
def urgent_task():
    pass

@app.task (priority=1)  # Low priority
def background_task():
    pass

# Urgent tasks processed before background tasks
urgent_task.delay()  # Processed first
background_task.delay()  # Processed later
\`\`\`

**4. Monitoring Built-in**
- Web UI at http://localhost:15672
- Real-time queue depths
- Connection monitoring
- Message rates and statistics

**5. Handles Large Messages**
- No 512MB limit (unlike Redis)
- Can handle GB-sized messages

**6. Enterprise Features**
- Clustering (multi-server)
- High availability (mirrored queues)
- Federation (multi-datacenter)
- Plugin ecosystem

### Disadvantages of RabbitMQ

**1. Complexity**
- Harder to set up (Erlang + RabbitMQ)
- More configuration required
- Steeper learning curve

**2. Higher Resource Usage**
- Uses more memory than Redis
- Higher CPU usage
- More expensive to host

**3. Slower than Redis**
- Disk persistence adds latency
- ~2-3× slower than Redis

---

## When to Choose Each

### ✅ Choose Redis When:

**1. You're building an MVP or startup**
\`\`\`python
# Simple, fast, good enough for 90% of use cases
app = Celery('myapp', broker='redis://localhost:6379/0')
\`\`\`

**2. You process <100K tasks per day**
- Redis handles this easily
- No need for RabbitMQ complexity

**3. Speed > reliability**
- User-facing APIs need fast responses
- Task loss acceptable (non-critical tasks)

**4. Tasks are non-critical**
- Sending emails (can retry manually if lost)
- Analytics tracking
- Cache warming

**5. Quick setup needed**
- Just install Redis and go
- No complex configuration

**6. Small team**
- Less infrastructure to maintain
- Simpler debugging

### ✅ Choose RabbitMQ When:

**1. You process >1M tasks per day**
- RabbitMQ handles massive scale
- Better reliability at scale

**2. Tasks are critical**
\`\`\`python
# Payment processing, order fulfillment
# Cannot afford to lose tasks!

app = Celery('payments', broker='amqp://localhost')

@app.task
def process_payment (order_id, amount):
    # This task MUST NOT be lost
    charge_customer (order_id, amount)
\`\`\`

**3. You need advanced routing**
- Complex queue topologies
- Dynamic routing based on task properties
- Topic exchanges, fanout patterns

**4. You need priority queues**
- VIP users get priority
- Urgent tasks jump the queue

**5. Multi-datacenter deployment**
- RabbitMQ federation
- Geographic distribution

**6. Enterprise requirements**
- Clustering, HA, monitoring
- Compliance (audit logs)

---

## Hybrid Approach

**Best of both worlds**: Use Redis for non-critical, RabbitMQ for critical

\`\`\`python
"""
Hybrid: Redis for fast tasks, RabbitMQ for critical tasks
"""

from celery import Celery

# Redis broker for non-critical tasks
app_fast = Celery(
    'fast_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# RabbitMQ broker for critical tasks
app_critical = Celery(
    'critical_tasks',
    broker='amqp://localhost:5672//',
    backend='rpc://'
)


# ========================================
# Non-Critical Tasks (Redis)
# ========================================

@app_fast.task
def send_email (to, subject, body):
    """Non-critical: Can retry manually if lost"""
    pass

@app_fast.task
def update_analytics (user_id, event):
    """Non-critical: Analytics can handle missing data"""
    pass

@app_fast.task
def generate_thumbnail (image_id):
    """Non-critical: Can regenerate if lost"""
    pass


# ========================================
# Critical Tasks (RabbitMQ)
# ========================================

@app_critical.task
def process_payment (order_id, amount):
    """CRITICAL: Must not lose payment tasks!"""
    pass

@app_critical.task
def fulfill_order (order_id):
    """CRITICAL: Customer expects their order"""
    pass

@app_critical.task
def send_verification_code (user_id, code):
    """CRITICAL: User blocked without this"""
    pass


# Usage
send_email.delay('user@example.com', 'Hi', 'Hello')  # Redis (fast)
process_payment.delay(12345, 99.99)  # RabbitMQ (reliable)
\`\`\`

**Benefits:**
- ⚡ Fast responses for non-critical tasks (Redis)
- 🛡️ Reliability for critical tasks (RabbitMQ)
- 💰 Cost optimization (Redis cheaper)
- 🎯 Right tool for each job

---

## Migration Path

### From Redis to RabbitMQ

\`\`\`python
"""
Gradual migration from Redis to RabbitMQ
"""

# Phase 1: Install RabbitMQ (keep Redis running)
# Phase 2: Configure RabbitMQ in Celery
# Phase 3: Dual-run (some tasks Redis, some RabbitMQ)
# Phase 4: Migrate non-critical tasks
# Phase 5: Migrate critical tasks
# Phase 6: Deprecate Redis

# Step 1: Add RabbitMQ broker
app = Celery(
    'myapp',
    broker='amqp://localhost:5672//',  # New: RabbitMQ
    backend='redis://localhost:6379/1'  # Keep: Redis backend
)

# Step 2: Test with new tasks
@app.task
def new_task():
    pass  # Uses RabbitMQ

# Step 3: Gradually migrate existing tasks
# Old: send_email.delay() used Redis
# New: send_email.delay() uses RabbitMQ
# (no code change - just broker change)

# Step 4: Monitor for issues
# Step 5: Once stable, remove Redis broker
\`\`\`

**Timeline:** 2-4 weeks for full migration

---

## Performance Benchmark

\`\`\`python
"""
Benchmark: Redis vs RabbitMQ
"""

import time
from celery import Celery

# Test: Queue 100,000 tasks and measure time

def benchmark_broker (broker_url, name):
    app = Celery('bench', broker=broker_url)
    
    @app.task
    def dummy_task (x):
        return x * 2
    
    start = time.time()
    for i in range(100_000):
        dummy_task.delay (i)
    duration = time.time() - start
    
    print(f"{name}: {duration:.2f}s ({100_000/duration:.0f} tasks/sec)")

# Results:
benchmark_broker('redis://localhost:6379/0', 'Redis')
# Redis: 10.5s (9,523 tasks/sec) ⚡

benchmark_broker('amqp://localhost:5672//', 'RabbitMQ')
# RabbitMQ: 23.7s (4,219 tasks/sec) 🐇

# Conclusion: Redis is ~2.3× faster
\`\`\`

---

## Decision Framework

\`\`\`
┌────────────────────────────────────────┐
│  Are tasks critical (payments, orders)?│
└───────────┬────────────────────────────┘
            │
    ┌───────▼────────┐
    │      Yes       │ → RabbitMQ (reliability critical)
    └────────────────┘
    ┌───────▼────────┐
    │       No       │ → Continue
    └───────┬────────┘
            │
┌───────────▼───────────────────────────┐
│  Do you process >1M tasks/day?        │
└───────────┬───────────────────────────┘
            │
    ┌───────▼────────┐
    │      Yes       │ → RabbitMQ (scale + reliability)
    └────────────────┘
    ┌───────▼────────┐
    │       No       │ → Continue
    └───────┬────────┘
            │
┌───────────▼───────────────────────────┐
│  Do you need advanced routing?        │
└───────────┬───────────────────────────┘
            │
    ┌───────▼────────┐
    │      Yes       │ → RabbitMQ (complex routing)
    └────────────────┘
    ┌───────▼────────┐
    │       No       │ → Redis (simple, fast, good enough)
    └────────────────┘
\`\`\`

---

## Summary

**Redis:**
- ⚡ Fast (2-3× faster than RabbitMQ)
- 🎯 Simple (5-minute setup)
- 💰 Cheaper (lower resources)
- ⚠️ Can lose tasks on crash (unless AOF enabled)
- ✅ Perfect for 90% of applications

**RabbitMQ:**
- 🛡️ Reliable (no task loss, disk-backed)
- 🎯 Advanced routing and priority queues
- 📊 Built-in monitoring UI
- 🔧 Complex setup (30 minutes)
- 💰💰 Higher resource usage
- ✅ Perfect for critical, high-scale applications

**Recommendation:**
1. **Start with Redis** (simple, fast, good enough)
2. **Monitor task loss** (is it happening?)
3. **Migrate to RabbitMQ** when:
   - Tasks become critical
   - Scale exceeds 100K tasks/day
   - Task loss unacceptable

**Next Section:** Distributed task processing patterns! 🔄
`,
};
