export const backgroundTasks = {
  title: 'Background Tasks',
  id: 'background-tasks',
  content: `
# Background Tasks

## Introduction

APIs should respond fast. But some operations take time: sending emails, processing images, generating reports, updating analytics. If you execute these synchronously, users wait. Background tasks let you return a response immediately while work continues asynchronously.

**Why background tasks matter:**
- **User experience**: Instant API responses (< 200ms)
- **Scalability**: Handle thousands of concurrent requests
- **Reliability**: Retry failed tasks automatically
- **Resource management**: Process tasks when resources available
- **Separation of concerns**: API handles requests, workers handle heavy lifting

In production, background tasks solve:
- Email/SMS notifications after user actions
- Image/video processing after upload
- Report generation (PDF, Excel)
- Data exports and backups
- Third-party API calls (payments, analytics)
- Database maintenance tasks
- Webhook deliveries

In this section, you'll master:
- FastAPI built-in BackgroundTasks
- Celery for distributed task queues
- Redis as message broker
- Task monitoring and retries
- Production patterns

### Background Task Patterns

\`\`\`
1. Fire-and-forget: Return response immediately, task runs in background
2. Delayed execution: Schedule task for future time
3. Periodic tasks: Cron-like scheduled jobs
4. Task chains: One task triggers another
5. Distributed workers: Multiple servers processing tasks
\`\`\`

---

## FastAPI Background Tasks

### Built-in BackgroundTasks

\`\`\`python
"""
FastAPI Built-in Background Tasks
Simple, lightweight, same process
"""

from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def write_log (message: str):
    """
    Background task function
    Runs after response is sent
    """
    time.sleep(2)  # Simulate slow operation
    with open("log.txt", "a") as f:
        f.write (f"{message}\\n")

@app.post("/send-notification/")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    """
    API returns immediately,
    notification sent in background
    """
    # Add task to background queue
    background_tasks.add_task(
        write_log,
        f"Notification sent to {email}"
    )
    
    # Response sent immediately (before task completes)
    return {"message": "Notification will be sent"}

# Multiple background tasks
@app.post("/register/")
async def register_user(
    user: UserCreate,
    background_tasks: BackgroundTasks
):
    """
    Add multiple background tasks
    Execute in order after response
    """
    # Create user in database (fast)
    new_user = create_user (user)
    
    # Add background tasks
    background_tasks.add_task (send_welcome_email, user.email)
    background_tasks.add_task (create_default_settings, new_user.id)
    background_tasks.add_task (notify_admin, new_user.id)
    
    # Return immediately
    return {"user_id": new_user.id}

# Background task with parameters
def send_email_report(
    email: str,
    report_type: str,
    start_date: datetime,
    end_date: datetime
):
    """
    Background task with multiple parameters
    """
    # Generate report (slow)
    report_data = generate_report (report_type, start_date, end_date)
    
    # Send email
    send_email(
        to=email,
        subject=f"{report_type} Report",
        body=f"Report from {start_date} to {end_date}",
        attachments=[report_data]
    )

@app.post("/request-report/")
async def request_report(
    report_type: str,
    start_date: datetime,
    end_date: datetime,
    background_tasks: BackgroundTasks,
    current_user: User = Depends (get_current_user)
):
    """
    Generate report in background
    """
    background_tasks.add_task(
        send_email_report,
        email=current_user.email,
        report_type=report_type,
        start_date=start_date,
        end_date=end_date
    )
    
    return {"message": "Report generation started"}
\`\`\`

### When to Use BackgroundTasks

✅ **Good for:**
- Quick operations (< 10 seconds)
- Simple fire-and-forget tasks
- Logging, analytics, notifications
- No retry logic needed
- Single-server deployments

❌ **Not good for:**
- Long-running tasks (> 30 seconds)
- Tasks requiring retries
- Distributed systems (multiple servers)
- Task monitoring/tracking
- Scheduled/periodic tasks

---

## Celery Integration

### Celery Setup

\`\`\`python
"""
Celery: Distributed Task Queue
Production-grade background tasks
"""

from celery import Celery
from celery.result import AsyncResult
import os

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=REDIS_URL,  # Message broker
    backend=REDIS_URL  # Result backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minute timeout
    task_soft_time_limit=25 * 60,  # 25 minute soft timeout
    task_acks_late=True,  # Acknowledge after task completes
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
)

# Define tasks
@celery_app.task (bind=True, max_retries=3)
def send_email_task (self, recipient: str, subject: str, body: str):
    """
    Send email with retry logic
    
    self: Task instance (bind=True)
    max_retries: Retry up to 3 times
    """
    try:
        # Send email
        send_email (recipient, subject, body)
        
        return {"status": "sent", "recipient": recipient}
        
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(
            exc=exc,
            countdown=2 ** self.request.retries  # 2, 4, 8 seconds
        )

@celery_app.task
def process_uploaded_image (image_id: int):
    """
    Process image: resize, optimize, generate thumbnails
    """
    # Get image from database
    image = get_image (image_id)
    
    # Process (slow operations)
    resized = resize_image (image.path, width=1200)
    thumbnail = create_thumbnail (image.path, size=200)
    optimized = optimize_image (resized)
    
    # Update database
    update_image(
        image_id,
        processed_path=optimized,
        thumbnail_path=thumbnail,
        status="processed"
    )
    
    return {"image_id": image_id, "status": "processed"}

@celery_app.task
def generate_monthly_report (user_id: int, month: int, year: int):
    """
    Generate monthly analytics report
    Long-running task (5-10 minutes)
    """
    # Fetch data
    data = fetch_monthly_data (user_id, month, year)
    
    # Generate charts
    charts = generate_charts (data)
    
    # Create PDF
    pdf_path = create_pdf_report (data, charts)
    
    # Upload to S3
    s3_url = upload_to_s3(pdf_path)
    
    # Notify user
    send_email_task.delay(
        recipient=get_user_email (user_id),
        subject=f"Monthly Report - {month}/{year}",
        body=f"Your report is ready: {s3_url}"
    )
    
    return {"report_url": s3_url}
\`\`\`

### FastAPI + Celery Integration

\`\`\`python
"""
Integrate Celery with FastAPI
"""

from fastapi import FastAPI

app = FastAPI()

@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends (get_db)
):
    """
    Upload image and process in background
    """
    # Save file (fast)
    image_path = save_upload (file)
    
    # Create database record
    image = Image (path=image_path, status="pending")
    db.add (image)
    db.commit()
    
    # Trigger Celery task (returns immediately)
    task = process_uploaded_image.delay (image.id)
    
    return {
        "image_id": image.id,
        "task_id": task.id,
        "status": "processing"
    }

@app.get("/task-status/{task_id}")
async def get_task_status (task_id: str):
    """
    Check Celery task status
    """
    task_result = AsyncResult (task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": task_result.state,  # PENDING, STARTED, SUCCESS, FAILURE
        "result": task_result.result if task_result.ready() else None,
        "error": str (task_result.info) if task_result.failed() else None
    }

@app.post("/request-report/")
async def request_report(
    month: int,
    year: int,
    current_user: User = Depends (get_current_user)
):
    """
    Generate report asynchronously
    """
    # Trigger Celery task
    task = generate_monthly_report.delay(
        user_id=current_user.id,
        month=month,
        year=year
    )
    
    return {
        "task_id": task.id,
        "message": "Report generation started",
        "estimated_time": "5-10 minutes"
    }
\`\`\`

---

## Task Monitoring & Management

### Task Result Tracking

\`\`\`python
"""
Track task progress and results
"""

from celery import current_task

@celery_app.task (bind=True)
def long_running_task (self, data_size: int):
    """
    Task with progress updates
    """
    total = data_size
    
    for i in range (total):
        # Process item
        process_item (i)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i + 1,
                'total': total,
                'percent': int(((i + 1) / total) * 100)
            }
        )
        
        time.sleep(0.1)
    
    return {'status': 'completed', 'total': total}

# FastAPI endpoint to track progress
@app.get("/task-progress/{task_id}")
async def get_task_progress (task_id: str):
    """
    Get real-time task progress
    """
    task_result = AsyncResult (task_id, app=celery_app)
    
    if task_result.state == 'PROGRESS':
        return {
            "task_id": task_id,
            "status": "in_progress",
            "current": task_result.info.get('current', 0),
            "total": task_result.info.get('total', 1),
            "percent": task_result.info.get('percent', 0)
        }
    
    elif task_result.state == 'SUCCESS':
        return {
            "task_id": task_id,
            "status": "completed",
            "result": task_result.result
        }
    
    else:
        return {
            "task_id": task_id,
            "status": task_result.state.lower()
        }
\`\`\`

### Task Retries & Error Handling

\`\`\`python
"""
Retry failed tasks with backoff
"""

from celery.exceptions import MaxRetriesExceededError

@celery_app.task (bind=True, max_retries=5, default_retry_delay=60)
def call_external_api (self, url: str, payload: dict):
    """
    Call external API with retry on failure
    
    max_retries: 5 attempts
    default_retry_delay: 60 seconds between retries
    """
    try:
        response = requests.post (url, json=payload, timeout=10)
        response.raise_for_status()
        
        return response.json()
        
    except (requests.RequestException, requests.Timeout) as exc:
        # Retry with exponential backoff
        try:
            raise self.retry(
                exc=exc,
                countdown=60 * (2 ** self.request.retries)  # 60, 120, 240, 480, 960
            )
        except MaxRetriesExceededError:
            # All retries exhausted, log error
            logger.error(
                f"Failed to call {url} after {self.request.retries} retries",
                extra={"url": url, "error": str (exc)}
            )
            
            # Send alert
            send_alert_to_admin(
                f"External API call failed: {url}",
                error=str (exc)
            )
            
            raise

# Task with custom retry logic
@celery_app.task (bind=True)
def process_payment (self, order_id: int):
    """
    Process payment with idempotency
    """
    try:
        # Get order
        order = get_order (order_id)
        
        # Check if already processed (idempotency)
        if order.payment_status == "completed":
            return {"status": "already_completed", "order_id": order_id}
        
        # Process payment
        result = payment_gateway.charge(
            amount=order.total,
            currency="USD",
            source=order.payment_method_id,
            idempotency_key=f"order_{order_id}"
        )
        
        # Update order
        order.payment_status = "completed"
        order.payment_id = result.id
        save_order (order)
        
        return {"status": "success", "payment_id": result.id}
        
    except PaymentDeclinedError as exc:
        # Don't retry if card declined
        logger.warning (f"Payment declined for order {order_id}")
        update_order_status (order_id, "payment_failed")
        send_payment_failed_email (order_id)
        
        raise  # Don't retry
        
    except PaymentGatewayError as exc:
        # Retry on gateway errors
        if self.request.retries < self.max_retries:
            raise self.retry (exc=exc, countdown=30)
        else:
            # All retries failed, escalate
            send_alert_to_admin (f"Payment processing failed for order {order_id}")
            raise
\`\`\`

---

## Scheduled & Periodic Tasks

### Celery Beat for Scheduling

\`\`\`python
"""
Periodic tasks with Celery Beat
"""

from celery.schedules import crontab

# Configure periodic tasks
celery_app.conf.beat_schedule = {
    'send-daily-report': {
        'task': 'tasks.send_daily_report',
        'schedule': crontab (hour=8, minute=0),  # 8 AM every day
    },
    'cleanup-old-sessions': {
        'task': 'tasks.cleanup_sessions',
        'schedule': crontab (hour=2, minute=0),  # 2 AM every day
    },
    'process-pending-orders': {
        'task': 'tasks.process_pending_orders',
        'schedule': 300.0,  # Every 5 minutes
    },
    'generate-weekly-analytics': {
        'task': 'tasks.generate_weekly_analytics',
        'schedule': crontab (hour=9, minute=0, day_of_week=1),  # 9 AM every Monday
    },
}

# Periodic tasks
@celery_app.task
def send_daily_report():
    """
    Send daily summary to all users
    Runs at 8 AM every day
    """
    users = get_all_active_users()
    
    for user in users:
        report_data = generate_user_report (user.id)
        send_email_task.delay(
            recipient=user.email,
            subject="Daily Summary",
            body=render_report (report_data)
        )

@celery_app.task
def cleanup_sessions():
    """
    Delete expired sessions
    Runs at 2 AM every day
    """
    cutoff = datetime.utcnow() - timedelta (days=30)
    deleted_count = delete_sessions_before (cutoff)
    
    logger.info (f"Cleaned up {deleted_count} expired sessions")

@celery_app.task
def process_pending_orders():
    """
    Process orders stuck in pending state
    Runs every 5 minutes
    """
    pending_orders = get_pending_orders()
    
    for order in pending_orders:
        # Check if stuck for > 30 minutes
        if (datetime.utcnow() - order.created_at).seconds > 1800:
            # Cancel or retry
            process_stuck_order (order.id)
\`\`\`

---

## Production Patterns

### Task Architecture

\`\`\`python
"""
Production-ready task architecture
"""

# Task base class with common patterns
class ProductionTask (celery_app.Task):
    """
    Base task class with production patterns
    """
    autoretry_for = (
        Exception,  # Retry on any exception
    )
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True  # Exponential backoff
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True  # Add randomness to backoff
    
    def on_failure (self, exc, task_id, args, kwargs, einfo):
        """
        Called when task fails
        """
        logger.error(
            f"Task {self.name} failed",
            extra={
                "task_id": task_id,
                "exception": str (exc),
                "args": args,
                "kwargs": kwargs
            }
        )
        
        # Send alert for critical tasks
        if self.name in CRITICAL_TASKS:
            send_alert_to_slack(
                f"❌ Critical task failed: {self.name}",
                error=str (exc)
            )
    
    def on_success (self, retval, task_id, args, kwargs):
        """
        Called when task succeeds
        """
        logger.info(
            f"Task {self.name} succeeded",
            extra={"task_id": task_id, "result": retval}
        )

# Use production task class
@celery_app.task (base=ProductionTask)
def important_task (data: dict):
    """
    Automatically gets retry logic and monitoring
    """
    process_data (data)

# Task chaining
from celery import chain, group, chord

# Sequential tasks
workflow = chain(
    download_data.s (url),
    process_data.s(),
    upload_results.s()
)
result = workflow.apply_async()

# Parallel tasks
parallel_tasks = group(
    process_chunk.s (chunk)
    for chunk in data_chunks
)
results = parallel_tasks.apply_async()

# Parallel + callback
callback_workflow = chord(
    group (process_user_data.s (user_id) for user_id in user_ids),
    aggregate_results.s()  # Called after all parallel tasks complete
)
result = callback_workflow.apply_async()
\`\`\`

### Monitoring & Observability

\`\`\`python
"""
Monitor Celery tasks in production
"""

from celery.signals import task_prerun, task_postrun, task_failure
import time

# Track task duration
@task_prerun.connect
def task_prerun_handler (sender=None, task_id=None, task=None, **kwargs):
    """
    Called before task starts
    """
    task.start_time = time.time()

@task_postrun.connect
def task_postrun_handler (sender=None, task_id=None, task=None, retval=None, **kwargs):
    """
    Called after task completes
    """
    duration = time.time() - task.start_time
    
    # Log task duration
    logger.info(
        f"Task {task.name} completed",
        extra={
            "task_id": task_id,
            "duration_seconds": duration
        }
    )
    
    # Track metrics (Prometheus)
    TASK_DURATION.labels (task_name=task.name).observe (duration)

@task_failure.connect
def task_failure_handler (sender=None, task_id=None, exception=None, **kwargs):
    """
    Called when task fails
    """
    # Track failures (Prometheus)
    TASK_FAILURES.labels (task_name=sender.name).inc()

# Health check endpoint
@app.get("/health/celery")
async def celery_health():
    """
    Check if Celery workers are responsive
    """
    try:
        # Inspect Celery workers
        inspect = celery_app.control.inspect()
        active = inspect.active()
        stats = inspect.stats()
        
        if not stats:
            return {
                "status": "unhealthy",
                "error": "No workers available"
            }
        
        return {
            "status": "healthy",
            "workers": len (stats),
            "active_tasks": sum (len (tasks) for tasks in (active or {}).values())
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str (e)
        }
\`\`\`

---

## Summary

### Key Takeaways

✅ **BackgroundTasks**: Simple fire-and-forget for quick operations  
✅ **Celery**: Production-grade distributed task queue with retries  
✅ **Redis**: Message broker for task distribution  
✅ **Monitoring**: Track task status, progress, and failures  
✅ **Retries**: Exponential backoff for transient failures  
✅ **Scheduling**: Celery Beat for periodic tasks (cron-like)  
✅ **Task chains**: Sequential and parallel task workflows  
✅ **Observability**: Logging, metrics, health checks

### Best Practices

**1. Choose the right tool**:
- BackgroundTasks for simple, quick tasks (< 10s)
- Celery for production: retries, monitoring, distribution

**2. Idempotency**:
- Tasks should be safe to retry
- Use idempotency keys for external APIs
- Check if work already done before processing

**3. Timeouts**:
- Set task_time_limit to prevent hanging tasks
- Use soft_time_limit for graceful shutdown

**4. Monitoring**:
- Log task start/completion/failure
- Track task duration metrics
- Alert on critical task failures

**5. Error handling**:
- Retry transient errors (network, timeouts)
- Don't retry permanent errors (invalid input)
- Escalate after max retries

### When to Use What

| Use Case | Tool | Why |
|----------|------|-----|
| Send welcome email | BackgroundTasks | Quick, simple |
| Process uploaded image | Celery | Can take 1-5 minutes |
| Generate monthly report | Celery | Long-running (5-10 min) |
| Daily cleanup job | Celery Beat | Scheduled task |
| Payment processing | Celery | Needs retries |

### Next Steps

In the next section, we'll explore **WebSockets & Real-Time Communication**: implementing bi-directional, real-time features like chat, notifications, and live dashboards with FastAPI WebSockets.

**Production mindset**: Background tasks are critical for performance and user experience. Choose the right tool, implement retries, and monitor everything!
`,
};
