export const errorHandlingRetriesTimeouts = {
  title: 'Error Handling, Retries & Timeouts',
  id: 'error-handling-retries-timeouts',
  content: `
# Error Handling, Retries & Timeouts

## Introduction

In distributed systems, **failures are inevitable**. Network timeouts, API rate limits, temporary service outages—your Celery tasks will encounter errors. The difference between amateur and production-ready systems is **comprehensive error handling**.

**Key Challenges:**
- Network failures (timeouts, connection errors)
- External API failures (rate limiting, service outages)
- Resource exhaustion (memory, disk space)
- Permanent vs temporary errors (retry vs fail)
- Time limits (prevent infinite execution)

**This Section Covers:**
- Automatic retry strategies with exponential backoff
- Handling different error types appropriately
- Soft vs hard time limits
- Dead letter queues for permanent failures
- Production error monitoring and alerting

---

## The Problem: Naive Error Handling

\`\`\`python
"""
PROBLEMATIC: No error handling
"""

from celery import Celery
import requests

app = Celery('tasks', broker='redis://localhost:6379/0')


@app.task
def call_external_api (url: str):
    """
    What could go wrong?
    - Network timeout
    - Rate limiting (429)
    - Server error (500, 503)
    - Invalid response
    """
    response = requests.get (url)  # 💥 No timeout!
    return response.json()  # 💥 No error handling!


# Usage
result = call_external_api.delay('https://api.example.com/data')

# Problems:
# 1. No timeout → Task hangs forever if API is slow
# 2. No retry → Transient errors cause permanent failure
# 3. No error handling → Task crashes on 500 errors
# 4. No differentiation → Retries permanent errors (404)
\`\`\`

**What Happens:**
- API timeout → Task hangs forever 🕰️
- Rate limit (429) → Task fails (should retry later)
- Server error (503) → Task fails (should retry with backoff)
- Not found (404) → Task fails (shouldn't retry - permanent error)

---

## Solution 1: Automatic Retries with \`autoretry_for\`

\`\`\`python
"""
Production-ready: Automatic retries
"""

from celery import Celery
from requests.exceptions import RequestException, Timeout, ConnectionError
import requests

app = Celery('tasks', broker='redis://localhost:6379/0')


@app.task(
    bind=True,
    autoretry_for=(RequestException,),  # Auto-retry on these exceptions
    retry_kwargs={'max_retries': 5},     # Max 5 retry attempts
    retry_backoff=True,                  # Exponential backoff
    retry_backoff_max=600,               # Max 10 minutes between retries
    retry_jitter=True,                   # Add randomness to prevent thundering herd
)
def call_api_with_auto_retry (self, url: str):
    """
    Automatically retries on RequestException
    
    Retry schedule (exponential backoff):
    - Attempt 1: Immediate
    - Attempt 2: ~2 seconds later
    - Attempt 3: ~4 seconds later
    - Attempt 4: ~8 seconds later
    - Attempt 5: ~16 seconds later
    - Attempt 6: ~32 seconds later (or max 600s)
    """
    try:
        response = requests.get (url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx
        return response.json()
    except Timeout as exc:
        print(f"Timeout on attempt {self.request.retries + 1}")
        raise  # Auto-retry via autoretry_for
    except ConnectionError as exc:
        print(f"Connection error on attempt {self.request.retries + 1}")
        raise  # Auto-retry via autoretry_for
    except RequestException as exc:
        print(f"Request error on attempt {self.request.retries + 1}: {exc}")
        raise  # Auto-retry via autoretry_for


# Usage
result = call_api_with_auto_retry.delay('https://api.example.com/data')

# Task will retry automatically up to 5 times with exponential backoff
# Total attempts: 6 (initial + 5 retries)
# If all fail: Task marked as FAILURE
\`\`\`

**Benefits:**
- ✅ Automatic retries (no manual code)
- ✅ Exponential backoff (prevents overwhelming failing service)
- ✅ Jitter (prevents thundering herd problem)
- ✅ Max retries (prevents infinite loops)

---

## Solution 2: Manual Retry with \`self.retry()\`

For more control over retry logic:

\`\`\`python
"""
Manual retry control for different error types
"""

from celery import Celery
from requests.exceptions import HTTPError
import requests
import logging

app = Celery('tasks', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)


@app.task (bind=True, max_retries=5)
def call_api_with_manual_retry (self, url: str, data: dict):
    """
    Manual retry with different strategies for different errors
    
    - 429 Rate Limited → Respect Retry-After header
    - 503 Service Unavailable → Exponential backoff
    - 404 Not Found → Don't retry (permanent error)
    - Timeout → Short retry (30s)
    """
    try:
        response = requests.post (url, json=data, timeout=10)
        
        # Handle different HTTP status codes
        if response.status_code == 200:
            return response.json()
        
        elif response.status_code == 429:
            # Rate limited - respect Retry-After header
            retry_after = int (response.headers.get('Retry-After', 60))
            logger.warning (f"Rate limited. Retrying after {retry_after}s")
            raise self.retry (countdown=retry_after, exc=HTTPError (response=response))
        
        elif response.status_code == 503:
            # Temporary service unavailable - exponential backoff
            countdown = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s, 480s...
            countdown = min (countdown, 600)  # Cap at 10 minutes
            logger.warning (f"Service unavailable. Retrying in {countdown}s")
            raise self.retry (countdown=countdown, exc=HTTPError (response=response))
        
        elif response.status_code == 404:
            # Permanent error - don't retry
            logger.error (f"Resource not found: {url}")
            raise ValueError (f"Resource not found: {url}")
        
        elif response.status_code >= 500:
            # Server error - retry with backoff
            countdown = 60 * (2 ** self.request.retries)
            raise self.retry (countdown=countdown, exc=HTTPError (response=response))
        
        else:
            response.raise_for_status()
    
    except requests.Timeout as exc:
        # Network timeout - retry soon (transient issue)
        logger.warning (f"Timeout on attempt {self.request.retries + 1}")
        raise self.retry (exc=exc, countdown=30)
    
    except requests.ConnectionError as exc:
        # Connection error - retry with exponential backoff
        countdown = 60 * (2 ** self.request.retries)
        logger.warning (f"Connection error. Retrying in {countdown}s")
        raise self.retry (exc=exc, countdown=countdown)
    
    except ValueError:
        # Permanent error (404) - don't retry, just fail
        raise
    
    except Exception as exc:
        # Unknown error - retry but log
        logger.error (f"Unknown error: {exc}")
        if self.request.retries >= self.max_retries:
            # Max retries exceeded - send to dead letter queue
            send_to_dlq (url, data, str (exc))
            raise
        raise self.retry (exc=exc)


def send_to_dlq (url: str, data: dict, error: str):
    """Send failed task to dead letter queue for manual inspection"""
    # Store in database or separate queue
    logger.critical (f"Task permanently failed. URL: {url}, Error: {error}")
    # FailedTask.create (url=url, data=data, error=error)


# Usage
result = call_api_with_manual_retry.delay('https://api.example.com/orders', {'order_id': 123})
\`\`\`

**Retry Strategy Summary:**

| Error Type | Status Code | Strategy | Countdown |
|------------|-------------|----------|-----------|
| Rate Limited | 429 | Respect Retry-After | Header value (e.g., 60s) |
| Service Unavailable | 503 | Exponential backoff | 60s, 120s, 240s... |
| Not Found | 404 | **Don't retry** | N/A (permanent) |
| Server Error | 500-599 | Exponential backoff | 60s, 120s, 240s... |
| Timeout | N/A | Short retry | 30s |
| Connection Error | N/A | Exponential backoff | 60s, 120s, 240s... |

---

## Time Limits: Preventing Runaway Tasks

### The Problem: Tasks That Never End

\`\`\`python
"""
PROBLEMATIC: No time limits
"""

@app.task
def process_large_dataset (dataset_id: int):
    """
    What if this takes 10 hours? Or hangs forever?
    - Worker blocked for hours
    - Other tasks starved
    - No timeout
    """
    dataset = load_dataset (dataset_id)  # Could be huge!
    
    for item in dataset:  # Could be 1 billion items!
        process_item (item)
    
    return "Done"


# Problem: This task could run forever!
# - No time limit
# - Worker blocked indefinitely
# - No graceful shutdown
\`\`\`

### Solution: Soft and Hard Time Limits

\`\`\`python
"""
Production-ready: Time limits with graceful handling
"""

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
import logging

app = Celery('tasks', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    soft_time_limit=270,  # Soft limit: 4.5 minutes (raises exception)
    time_limit=300,       # Hard limit: 5 minutes (SIGKILL - forces termination)
)
def process_with_time_limit (self, dataset_id: int):
    """
    Soft limit (270s): Raises SoftTimeLimitExceeded exception
    - Task can catch it
    - Save progress
    - Cleanup resources
    - Graceful shutdown
    
    Hard limit (300s): Sends SIGKILL
    - 30s grace period after soft limit
    - Forces termination
    - No cleanup possible
    - Last resort
    """
    dataset = load_dataset (dataset_id)
    processed_items = []
    
    try:
        for i, item in enumerate (dataset):
            processed_items.append (process_item (item))
            
            # Periodic progress update
            if i % 1000 == 0:
                logger.info (f"Processed {i} items")
    
    except SoftTimeLimitExceeded:
        # Soft limit hit - save progress and retry
        logger.warning (f"Soft time limit exceeded. Processed {len (processed_items)} items")
        
        # Save checkpoint
        save_checkpoint (dataset_id, processed_items)
        
        # Optionally retry with remaining work
        remaining_ids = [item['id'] for item in dataset[len (processed_items):]]
        process_with_time_limit.delay (dataset_id, start_from=len (processed_items))
        
        raise  # Re-raise to mark task as failed (will be retried)
    
    return {
        'processed': len (processed_items),
        'status': 'complete'
    }


def save_checkpoint (dataset_id: int, processed_items: list):
    """Save progress for resumption"""
    # Save to Redis or database
    logger.info (f"Checkpoint saved: {len (processed_items)} items")


# Usage
result = process_with_time_limit.delay(12345)

# Timeline:
# T+0s: Task starts
# T+270s: Soft limit → SoftTimeLimitExceeded raised (graceful)
# T+300s: Hard limit → SIGKILL sent (forced termination)
# Grace period: 30 seconds for cleanup
\`\`\`

**Key Differences:**

| Limit Type | Time | Behavior | Can Catch? | Cleanup? |
|------------|------|----------|------------|----------|
| **Soft** | 270s | Raises exception | ✅ Yes | ✅ Yes (30s) |
| **Hard** | 300s | SIGKILL | ❌ No | ❌ No |

**Best Practice:** Set soft limit < hard limit (e.g., 270s soft, 300s hard = 30s grace period)

---

## Chunking Large Tasks

For tasks that process millions of records, **chunk them**:

\`\`\`python
"""
Production pattern: Chunking large tasks
"""

from celery import Celery, group

app = Celery('tasks', broker='redis://localhost:6379/0')


@app.task
def process_all_users():
    """
    Process 1 million users
    
    Problem: Would take 2 hours → exceeds time limit
    Solution: Split into 100 chunks of 10K users each
    """
    total_users = 1_000_000
    chunk_size = 10_000
    
    # Create chunks
    chunks = []
    for i in range(0, total_users, chunk_size):
        chunks.append((i, i + chunk_size))
    
    # Process chunks in parallel
    job = group (process_user_chunk.s (start, end) for start, end in chunks)
    result = job.apply_async()
    
    return {
        'total_chunks': len (chunks),
        'chunk_size': chunk_size,
        'group_id': result.id
    }


@app.task(
    bind=True,
    soft_time_limit=300,  # 5 minutes per chunk
    time_limit=330,
    max_retries=3
)
def process_user_chunk (self, start: int, end: int):
    """
    Process one chunk of users
    
    Each chunk:
    - Processes 10K users
    - Takes ~4 minutes (< 5 min limit)
    - Can fail independently
    - Retries only this chunk
    """
    checkpoint_key = f"checkpoint:users:{start}:{end}"
    
    # Check if already processed (idempotency)
    if redis_client.exists (checkpoint_key):
        return {'status': 'already_processed', 'range': (start, end)}
    
    processed = 0
    for user_id in range (start, end):
        try:
            process_user (user_id)
            processed += 1
        except Exception as e:
            logger.error (f"Failed to process user {user_id}: {e}")
    
    # Mark as complete
    redis_client.setex (checkpoint_key, 86400, 'done')  # 24h expiration
    
    return {
        'status': 'complete',
        'range': (start, end),
        'processed': processed
    }


def process_user (user_id: int):
    """Process single user"""
    pass


# Usage
result = process_all_users.delay()

# Benefits:
# - 100 chunks process in parallel (if 100 workers)
# - Each chunk < 5 minutes (within time limit)
# - Failure: Only failed chunk retries (not all 1M)
# - Checkpoints: Prevent reprocessing
# - Scalable: Add more workers = faster processing
\`\`\`

---

## Dead Letter Queue (DLQ)

Store permanently failed tasks for manual inspection:

\`\`\`python
"""
Production pattern: Dead Letter Queue
"""

from celery import Celery
from datetime import datetime
import json

app = Celery('tasks', broker='redis://localhost:6379/0')


class FailedTask:
    """Store failed tasks in database"""
    
    @staticmethod
    def create (task_name: str, args: tuple, kwargs: dict, error: str):
        """Save failed task to DLQ"""
        failed_task = {
            'task_name': task_name,
            'args': args,
            'kwargs': kwargs,
            'error': error,
            'failed_at': datetime.utcnow().isoformat(),
            'retries': kwargs.get('retries', 0)
        }
        
        # Save to database
        # db.session.add(FailedTaskModel(**failed_task))
        # db.session.commit()
        
        # Or save to file
        with open('/var/log/celery/dlq.jsonl', 'a') as f:
            f.write (json.dumps (failed_task) + '\\n')
        
        print(f"Task sent to DLQ: {task_name}")


@app.task (bind=True, max_retries=5)
def risky_task (self, user_id: int):
    """Task that might fail permanently"""
    try:
        # Attempt processing
        result = process_user_payment (user_id)
        return result
    
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            # Max retries exceeded - send to DLQ
            FailedTask.create(
                task_name=self.name,
                args=(user_id,),
                kwargs={'retries': self.request.retries},
                error=str (exc)
            )
            
            # Send alert to ops team
            send_alert(
                title="Task Failed After Max Retries",
                message=f"Task: {self.name}, User: {user_id}, Error: {exc}"
            )
            
            raise  # Mark as failed
        
        # Retry
        raise self.retry (exc=exc, countdown=60 * (2 ** self.request.retries))


def process_user_payment (user_id: int):
    """Simulate payment processing"""
    pass


def send_alert (title: str, message: str):
    """Send alert to ops team (PagerDuty, Slack, etc.)"""
    print(f"ALERT: {title} - {message}")


# View DLQ
def get_dlq_tasks():
    """Retrieve failed tasks for manual review"""
    with open('/var/log/celery/dlq.jsonl', 'r') as f:
        tasks = [json.loads (line) for line in f]
    return tasks


# Reprocess DLQ task
def retry_dlq_task (failed_task: dict):
    """Manually retry a failed task from DLQ"""
    task_name = failed_task['task_name']
    args = failed_task['args']
    kwargs = failed_task['kwargs']
    
    # Get task function
    task = app.tasks[task_name]
    
    # Retry
    result = task.delay(*args, **kwargs)
    return result.id
\`\`\`

---

## Idempotent Tasks

Make tasks **idempotent** (safe to execute multiple times):

\`\`\`python
"""
Production pattern: Idempotent tasks
"""

@app.task
def send_notification (user_id: int, notification_id: int):
    """
    Idempotent: Can be called multiple times safely
    
    Without idempotency:
    - Task retries → User gets 5 duplicate notifications 😱
    
    With idempotency:
    - Task retries → User gets 1 notification ✅
    """
    # Check if already sent
    cache_key = f"notification:sent:{notification_id}"
    
    if redis_client.exists (cache_key):
        return {'status': 'already_sent', 'notification_id': notification_id}
    
    # Send notification
    send_push_notification (user_id, notification_id)
    
    # Mark as sent (prevent duplicates)
    redis_client.setex (cache_key, 86400, 'sent')  # 24h expiration
    
    return {'status': 'sent', 'notification_id': notification_id}


@app.task
def process_payment (order_id: int):
    """
    Idempotent payment processing
    
    Critical: Payment must not be charged multiple times!
    """
    # Check if already processed
    payment = Payment.query.filter_by (order_id=order_id).first()
    
    if payment and payment.status == 'complete':
        return {'status': 'already_processed', 'payment_id': payment.id}
    
    # Process payment (with database unique constraint)
    try:
        charge = stripe.Charge.create(
            amount=get_order_amount (order_id),
            currency='usd',
            idempotency_key=f'order_{order_id}'  # Stripe idempotency
        )
        
        # Save with unique constraint on order_id
        payment = Payment(
            order_id=order_id,
            stripe_charge_id=charge.id,
            status='complete'
        )
        db.session.add (payment)
        db.session.commit()  # Unique constraint prevents duplicates
        
        return {'status': 'processed', 'payment_id': payment.id}
    
    except IntegrityError:
        # Duplicate (another worker processed it)
        db.session.rollback()
        return {'status': 'already_processed'}
\`\`\`

---

## Monitoring and Alerting

\`\`\`python
"""
Production monitoring for errors
"""

from celery.signals import task_failure, task_retry
import logging

logger = logging.getLogger(__name__)


@task_failure.connect
def task_failure_handler (sender=None, task_id=None, exception=None, args=None, kwargs=None, **kw):
    """
    Handle task failures
    
    Called when task fails after all retries
    """
    logger.error (f"Task {sender.name} [{task_id}] failed: {exception}")
    
    # Send to monitoring (Sentry, Datadog, etc.)
    sentry_sdk.capture_exception (exception)
    
    # Alert ops team for critical tasks
    if sender.name in ['process_payment', 'send_verification_email']:
        send_pagerduty_alert(
            title=f"Critical task failed: {sender.name}",
            details=f"Task ID: {task_id}, Error: {exception}"
        )


@task_retry.connect
def task_retry_handler (sender=None, task_id=None, reason=None, **kw):
    """
    Handle task retries
    
    Called when task is retried
    """
    logger.warning (f"Task {sender.name} [{task_id}] retry: {reason}")
    
    # Increment retry counter metric
    retry_counter.labels (task_name=sender.name).inc()


# Metrics
from prometheus_client import Counter

retry_counter = Counter('celery_task_retries_total', 'Task retries', ['task_name'])
failure_counter = Counter('celery_task_failures_total', 'Task failures', ['task_name'])
\`\`\`

---

## Summary

**Error Handling Strategies:**
1. **Automatic Retries**: Use \`autoretry_for\` for simple cases
2. **Manual Retries**: Use \`self.retry()\` for complex retry logic
3. **Exponential Backoff**: Prevent overwhelming failing services
4. **Respect Rate Limits**: Use Retry-After headers
5. **Don't Retry Permanent Errors**: 404, validation errors

**Time Limits:**
1. **Soft Limit**: Raises exception (graceful shutdown)
2. **Hard Limit**: SIGKILL (forced termination)
3. **Grace Period**: Soft < Hard (30-60s for cleanup)

**Best Practices:**
1. ✅ Implement dead letter queues
2. ✅ Make tasks idempotent
3. ✅ Chunk large tasks
4. ✅ Monitor failures and retries
5. ✅ Alert on critical failures

**Next Section:** Monitoring with Flower for real-time task visibility! 📊
`,
};
