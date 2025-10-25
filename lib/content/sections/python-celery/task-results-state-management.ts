export const taskResultsStateManagement = {
  title: 'Task Results & State Management',
  id: 'task-results-state-management',
  content: `
# Task Results & State Management

## Introduction

Every Celery task goes through a lifecycle with different states. Understanding task states and result management is crucial for building reliable distributed systems where you need to track task progress, retrieve results, and handle failures.

**Task States:**
- PENDING → STARTED → SUCCESS/FAILURE/RETRY

**Result Backend Options:**
- Redis (fast, in-memory)
- Database (persistent, SQL)
- RPC (temporary)
- None (fire-and-forget)

---

## Task States Lifecycle

\`\`\`
┌──────────┐
│ PENDING  │ ← Task queued but not started
└────┬─────┘
     │
     ▼
┌──────────┐
│ RECEIVED │ ← Worker received task (optional)
└────┬─────┘
     │
     ▼
┌──────────┐
│ STARTED  │ ← Task execution started
└────┬─────┘
     │
     ├──────────┐
     │          ▼
     │     ┌──────────┐
     │     │  RETRY   │ ← Task being retried
     │     └────┬─────┘
     │          │
     │          ▼
     │     (back to STARTED)
     │
     ▼
┌──────────┐       ┌──────────┐
│ SUCCESS  │       │ FAILURE  │
└──────────┘       └──────────┘
     │                  │
     ▼                  ▼
┌─────────────────────────────┐
│ Result stored in backend    │
└─────────────────────────────┘
\`\`\`

---

## Checking Task Results

\`\`\`python
"""
Retrieving Task Results
"""

from celery import Celery
from celery.result import AsyncResult

app = Celery('myapp', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

@app.task
def add(x, y):
    import time
    time.sleep(5)  # Simulate work
    return x + y

# Queue task
result = add.delay(4, 6)
print(f"Task ID: {result.id}")  # '550e8400-e29b-41d4-a716-446655440000'

# Method 1: Check if ready (non-blocking)
if result.ready():
    print("Task complete!")
    print(f"Result: {result.result}")
else:
    print("Task still running...")

# Method 2: Get result (blocking, waits)
output = result.get(timeout=10)  # Wait up to 10 seconds
print(f"Result: {output}")  # 10

# Method 3: Check status
print(f"Status: {result.status}")  # SUCCESS, PENDING, FAILURE, etc.

# Method 4: Check if successful
if result.successful():
    print(f"Task succeeded: {result.result}")
elif result.failed():
    print(f"Task failed: {result.traceback}")

# Method 5: Get info without waiting
print(f"State: {result.state}")
print(f"Info: {result.info}")  # Exception or result

# Method 6: Retrieve result by ID later
task_id = result.id
later_result = AsyncResult(task_id, app=app)
print(f"Retrieved result: {later_result.get()}")
\`\`\`

---

## Result Backend Configuration

### Redis Backend

\`\`\`python
from celery import Celery

app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'  # Separate DB for results
)

# Result configuration
app.conf.result_backend = 'redis://localhost:6379/1'
app.conf.result_expires = 3600  # Results expire after 1 hour
app.conf.result_compression = 'gzip'  # Compress results
app.conf.result_serializer = 'json'
app.conf.result_extended = True  # Store task args/kwargs
\`\`\`

### Database Backend

\`\`\`python
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='db+postgresql://user:pass@localhost/celery_results'
)

# Or SQLite for development
app.conf.result_backend = 'db+sqlite:///results.db'

# Database configuration
app.conf.database_table_names = {
    'task': 'celery_taskmeta',
    'group': 'celery_groupmeta',
}
\`\`\`

### No Result Backend (Fire-and-Forget)

\`\`\`python
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend=None  # No result storage
)

# Or disable per task
@app.task(ignore_result=True)
def fire_and_forget_task():
    """No result stored"""
    log_event("something happened")
\`\`\`

---

## Task State Tracking

\`\`\`python
"""
Tracking Task State and Progress
"""

from celery import Celery, current_task
from celery.result import AsyncResult

app = Celery('myapp', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

# Enable task tracking
app.conf.task_track_started = True  # Track STARTED state
app.conf.task_send_sent_event = True  # Send task-sent event

@app.task(bind=True)
def long_running_task(self, total_steps):
    """Task with progress updates"""
    for step in range(total_steps):
        # Do work
        time.sleep(1)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': step + 1,
                'total': total_steps,
                'percent': (step + 1) / total_steps * 100,
                'status': f'Processing step {step + 1} of {total_steps}'
            }
        )
    
    return {'status': 'complete', 'total': total_steps}

# Queue task
result = long_running_task.delay(10)

# Poll progress
import time
while not result.ready():
    if result.state == 'PROGRESS':
        info = result.info
        print(f"Progress: {info['percent']:.1f}% - {info['status']}")
    time.sleep(1)

# Get final result
print(f"Final result: {result.result}")
\`\`\`

---

## Result Methods

\`\`\`python
"""
AsyncResult Methods
"""

from celery.result import AsyncResult

# Get result object
result = AsyncResult(task_id, app=app)

# State checking
result.state        # Current state: PENDING, STARTED, SUCCESS, FAILURE, RETRY
result.status       # Alias for state
result.ready()      # True if task complete (SUCCESS or FAILURE)
result.successful() # True if SUCCESS
result.failed()     # True if FAILURE

# Getting results
result.get(timeout=10)              # Wait for result (blocking)
result.get(timeout=10, propagate=False)  # Don't raise exception on failure
result.result                       # Result value (None if not ready)
result.info                         # Result or exception info

# Error handling
result.traceback    # Exception traceback if failed

# Task metadata
result.id           # Task ID
result.task_id      # Alias for id
result.args         # Task arguments (if result_extended=True)
result.kwargs       # Task keyword arguments
result.task_name    # Task name
result.date_done    # Completion datetime

# Waiting
result.wait(timeout=10)  # Wait for completion (alias for get)

# Revoking
result.revoke(terminate=True)  # Cancel task

# Forgetting (remove from backend)
result.forget()
\`\`\`

---

## Custom States

\`\`\`python
"""
Custom Task States
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

@app.task(bind=True)
def multi_stage_task(self):
    """Task with custom states"""
    
    # Stage 1: Downloading
    self.update_state(
        state='DOWNLOADING',
        meta={'progress': 0, 'stage': 'download'}
    )
    download_data()
    
    # Stage 2: Processing
    self.update_state(
        state='PROCESSING',
        meta={'progress': 50, 'stage': 'process'}
    )
    process_data()
    
    # Stage 3: Uploading
    self.update_state(
        state='UPLOADING',
        meta={'progress': 75, 'stage': 'upload'}
    )
    upload_results()
    
    return {'status': 'complete', 'progress': 100}

# Check custom state
result = multi_stage_task.delay()
while not result.ready():
    state = result.state
    info = result.info
    
    if state in ['DOWNLOADING', 'PROCESSING', 'UPLOADING']:
        print(f"State: {state}, Progress: {info.get('progress', 0)}%")
    
    time.sleep(1)
\`\`\`

---

## Group Results

\`\`\`python
"""
Managing Group Results
"""

from celery import group

@app.task
def add(x, y):
    return x + y

# Create group (parallel execution)
job = group([
    add.s(2, 2),
    add.s(4, 4),
    add.s(8, 8),
    add.s(16, 16),
])

result = job.apply_async()

# Wait for all tasks
results = result.get(timeout=10)
print(f"Results: {results}")  # [4, 8, 16, 32]

# Check completion
if result.ready():
    print("All tasks complete!")

# Check if successful
if result.successful():
    print("All tasks succeeded!")

# Iterate results
for task_result in result.results:
    print(f"Task {task_result.id}: {task_result.result}")
\`\`\`

---

## Result Expiration

\`\`\`python
"""
Configuring Result Expiration
"""

# Global expiration
app.conf.result_expires = 3600  # 1 hour (seconds)

# Per-task expiration
@app.task(result_expires=300)  # 5 minutes
def short_lived_result():
    return "This result expires in 5 minutes"

# Never expire (not recommended!)
app.conf.result_expires = None

# Expire immediately after retrieval
@app.task(ignore_result=False, result_expires=0)
def one_time_result():
    """Result available once, then deleted"""
    return "Retrieved once"

result = one_time_result.delay()
value = result.get()  # Result retrieved
# result.get()  # Fails - result expired after first get()
\`\`\`

---

## Best Practices

\`\`\`python
"""
Result Management Best Practices
"""

# ✅ GOOD: Set expiration (prevent unbounded growth)
app.conf.result_expires = 3600  # 1 hour

# ✅ GOOD: Ignore results for fire-and-forget tasks
@app.task(ignore_result=True)
def log_event(event_type: str):
    """No result needed"""
    logger.info(f"Event: {event_type}")

# ✅ GOOD: Store large results externally
@app.task
def process_large_file(file_id):
    """Don't return 100MB result!"""
    result = process_file(file_id)  # 100MB
    
    # Upload to S3
    s3_url = upload_to_s3(result)
    
    # Return URL, not data
    return {'result_url': s3_url, 'size': len(result)}

# ✅ GOOD: Use result backend efficiently
@app.task
def frequent_task():
    """Runs 1000s of times - don't store results"""
    return None  # Or use ignore_result=True

# ❌ BAD: Return large data
@app.task
def bad_task():
    """Returns 100MB to result backend"""
    return [{'data': '...'} for _ in range(100000)]  # ❌ 100MB!

# ❌ BAD: Never expire results
app.conf.result_expires = None  # ❌ Redis fills up!

# ❌ BAD: Block waiting for result in API endpoint
@app.route('/process')
def process_endpoint():
    result = expensive_task.delay()
    return result.get(timeout=60)  # ❌ Blocks web worker!

# ✅ GOOD: Return task ID, poll separately
@app.route('/process')
def process_endpoint():
    result = expensive_task.delay()
    return {'task_id': result.id, 'status': 'processing'}

@app.route('/status/<task_id>')
def status_endpoint(task_id):
    result = AsyncResult(task_id, app=app)
    return {'state': result.state, 'result': result.result}
\`\`\`

---

## Monitoring Results

\`\`\`python
"""
Result Monitoring and Cleanup
"""

from celery import Celery
from celery.result import AsyncResult

app = Celery('myapp', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

# Monitor result backend size
@app.task
def monitor_result_backend():
    """Check result backend size"""
    import redis
    r = redis.Redis(host='localhost', port=6379, db=1)
    
    # Count keys
    keys = r.keys('celery-task-meta-*')
    count = len(keys)
    
    # Estimate size
    total_size = sum(r.memory_usage(key) for key in keys)
    
    if count > 100000:
        alert_ops(f"Result backend has {count} keys!")
    
    if total_size > 1_000_000_000:  # 1GB
        alert_ops(f"Result backend size: {total_size / 1e9:.2f} GB")

# Clean up old results
@app.task
def cleanup_old_results():
    """Remove results older than 24 hours"""
    import redis
    r = redis.Redis(host='localhost', port=6379, db=1)
    
    cutoff = time.time() - 86400  # 24 hours ago
    deleted = 0
    
    for key in r.scan_iter('celery-task-meta-*'):
        # Get task info
        task_id = key.decode().replace('celery-task-meta-', '')
        result = AsyncResult(task_id, app=app)
        
        # Check date_done
        if result.date_done and result.date_done.timestamp() < cutoff:
            result.forget()
            deleted += 1
    
    logger.info(f"Deleted {deleted} old results")

# Schedule cleanup
app.conf.beat_schedule = {
    'cleanup-results': {
        'task': 'tasks.cleanup_old_results',
        'schedule': crontab(hour=3, minute=0),  # 3 AM daily
    },
}
\`\`\`

---

## Production Configuration

\`\`\`python
"""
Production Result Backend Configuration
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# Result backend
app.conf.result_backend = 'redis://localhost:6379/1'

# Expiration (critical!)
app.conf.result_expires = 3600  # 1 hour

# Compression
app.conf.result_compression = 'gzip'

# Serialization
app.conf.result_serializer = 'json'
app.conf.accept_content = ['json']

# Extended results (store args/kwargs)
app.conf.result_extended = True

# Cache results in memory
app.conf.result_cache_max = 1000

# Backend options
app.conf.result_backend_transport_options = {
    'visibility_timeout': 3600,
    'retry_policy': {
        'timeout': 5.0
    }
}

# Per-task configuration
@app.task(
    ignore_result=False,     # Store result
    result_expires=1800,     # 30 minutes
    result_compression='gzip'
)
def important_task():
    return {"status": "complete"}
\`\`\`

---

## Summary

**Key Concepts:**
- **Task States**: PENDING → STARTED → SUCCESS/FAILURE/RETRY
- **Result Retrieval**: \`.get()\`, \`.ready()\`, \`.successful()\`
- **Result Backends**: Redis (fast), Database (persistent), None (fire-and-forget)
- **Progress Tracking**: \`self.update_state()\` for custom states

**Best Practices:**
- Set result expiration (prevent unbounded growth)
- Use \`ignore_result=True\` for fire-and-forget tasks
- Store large results externally (S3), return URLs
- Don't block web workers with \`.get()\`
- Clean up old results periodically
- Monitor result backend size

**Next Section:** Error handling, retries, and timeouts! 🔧
`,
};
