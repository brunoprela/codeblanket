export const backgroundTasksQuiz = [
  {
    id: 1,
    question:
      "Compare FastAPI's built-in BackgroundTasks with Celery for handling asynchronous operations. Design a system for an e-commerce platform that needs to: (1) send order confirmation emails immediately, (2) process credit card payments with retry logic, (3) generate daily sales reports at midnight, and (4) resize product images after upload. For each requirement, would you use BackgroundTasks or Celery? Justify your choices with code examples showing how you would implement each, including error handling and monitoring.",
    answer: `**BackgroundTasks vs Celery Analysis**:

**BackgroundTasks** (FastAPI built-in):
- Runs in same process as API server
- No external dependencies (Redis, workers)
- Simple fire-and-forget
- No retry logic
- No persistence (lost if server crashes)
- Good for: Quick tasks (< 10 seconds), simple operations, logging

**Celery** (Distributed task queue):
- Runs in separate worker processes
- Requires message broker (Redis/RabbitMQ)
- Retry logic with exponential backoff
- Task persistence and monitoring
- Can scale horizontally (multiple workers)
- Good for: Long-running tasks, retries, scheduled jobs, distributed systems

**E-Commerce System Design**:

**1. Order Confirmation Emails → BackgroundTasks**

Why: Fast operation (< 1 second), non-critical if fails, immediate execution needed.

\`\`\`python
from fastapi import BackgroundTasks

def send_confirmation_email(email: str, order_id: int):
    """Quick email send"""
    try:
        email_service.send(
            to=email,
            template="order_confirmation",
            context={"order_id": order_id}
        )
    except Exception as e:
        logger.error(f"Failed to send confirmation: {e}")
        # Non-critical, just log

@app.post("/orders")
async def create_order(
    order: OrderCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Create order and send confirmation"""
    # Save order (fast)
    new_order = save_order(order)
    
    # Send email in background
    background_tasks.add_task(
        send_confirmation_email,
        email=current_user.email,
        order_id=new_order.id
    )
    
    return {"order_id": new_order.id}
\`\`\`

**2. Credit Card Payments → Celery**

Why: Critical operation, needs retries, can take 5-30 seconds, must be reliable.

\`\`\`python
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task(bind=True, max_retries=5, default_retry_delay=30)
def process_payment_task(self, order_id: int):
    """
    Process payment with retry logic
    Critical: money involved!
    """
    try:
        order = get_order(order_id)
        
        # Idempotency check
        if order.payment_status == "completed":
            return {"status": "already_completed"}
        
        # Call payment gateway (can fail)
        result = stripe.Charge.create(
            amount=int(order.total * 100),
            currency="usd",
            source=order.payment_token,
            idempotency_key=f"order_{order_id}"  # Prevent double charging
        )
        
        # Update order
        order.payment_status = "completed"
        order.payment_id = result.id
        save_order(order)
        
        # Trigger fulfillment
        fulfill_order_task.delay(order_id)
        
        return {"status": "success", "payment_id": result.id}
        
    except stripe.CardError as e:
        # Card declined - don't retry
        logger.warning(f"Payment declined for order {order_id}: {e}")
        update_order_status(order_id, "payment_failed")
        send_payment_failed_email(order_id)
        raise  # Don't retry
        
    except (stripe.APIConnectionError, stripe.APIError) as e:
        # Gateway error - retry
        logger.error(f"Payment gateway error for order {order_id}: {e}")
        
        if self.request.retries < self.max_retries:
            # Exponential backoff: 30, 60, 120, 240, 480 seconds
            raise self.retry(
                exc=e,
                countdown=30 * (2 ** self.request.retries)
            )
        else:
            # All retries exhausted
            update_order_status(order_id, "payment_error")
            send_alert_to_admin(f"Payment failed after retries: {order_id}")
            raise

@app.post("/orders/{order_id}/pay")
async def process_payment(
    order_id: int,
    current_user: User = Depends(get_current_user)
):
    """Trigger payment processing"""
    # Trigger Celery task (returns immediately)
    task = process_payment_task.delay(order_id)
    
    return {
        "task_id": task.id,
        "status": "processing"
    }
\`\`\`

**3. Daily Sales Reports → Celery Beat**

Why: Scheduled task, runs at specific time, long-running (5-10 minutes).

\`\`\`python
from celery.schedules import crontab

# Configure periodic tasks
celery_app.conf.beat_schedule = {
    'generate-daily-sales-report': {
        'task': 'tasks.generate_daily_report',
        'schedule': crontab(hour=0, minute=0),  # Midnight UTC
    },
}

@celery_app.task
def generate_daily_report():
    """
    Generate daily sales report
    Runs at midnight every day
    """
    yesterday = date.today() - timedelta(days=1)
    
    # Fetch sales data (slow query)
    sales = fetch_daily_sales(yesterday)
    
    # Generate report (1-2 minutes)
    report_data = calculate_metrics(sales)
    charts = generate_charts(report_data)
    
    # Create PDF (30-60 seconds)
    pdf_path = create_pdf_report(report_data, charts)
    
    # Upload to S3
    s3_url = upload_to_s3(pdf_path, key=f"reports/daily_{yesterday}.pdf")
    
    # Notify admins
    send_email_to_admins(
        subject=f"Daily Sales Report - {yesterday}",
        body=f"Report ready: {s3_url}"
    )
    
    logger.info(f"Daily report generated for {yesterday}")
    
    return {"date": str(yesterday), "url": s3_url}
\`\`\`

**4. Product Image Resizing → Celery**

Why: CPU-intensive, can take 10-60 seconds, needs retries if upload fails.

\`\`\`python
@celery_app.task(bind=True, max_retries=3)
def process_product_image_task(self, product_id: int, image_path: str):
    """
    Resize and optimize product image
    Generate thumbnails
    """
    try:
        # Download original (if from S3)
        local_path = download_image(image_path)
        
        # Resize to standard sizes (CPU-intensive, 5-30 seconds)
        large = resize_image(local_path, width=1200)
        medium = resize_image(local_path, width=600)
        small = resize_image(local_path, width=300)
        thumbnail = resize_image(local_path, width=100)
        
        # Optimize (reduce file size)
        optimized_large = optimize_image(large, quality=85)
        optimized_medium = optimize_image(medium, quality=85)
        optimized_small = optimize_image(small, quality=85)
        
        # Upload to S3
        large_url = upload_to_s3(optimized_large, f"products/{product_id}/large.jpg")
        medium_url = upload_to_s3(optimized_medium, f"products/{product_id}/medium.jpg")
        small_url = upload_to_s3(optimized_small, f"products/{product_id}/small.jpg")
        thumbnail_url = upload_to_s3(thumbnail, f"products/{product_id}/thumb.jpg")
        
        # Update product
        update_product(product_id, {
            "image_large": large_url,
            "image_medium": medium_url,
            "image_small": small_url,
            "image_thumbnail": thumbnail_url,
            "image_status": "processed"
        })
        
        # Cleanup
        cleanup_temp_files([local_path, large, medium, small, thumbnail])
        
        return {"product_id": product_id, "status": "processed"}
        
    except Exception as exc:
        logger.error(f"Image processing failed for product {product_id}: {exc}")
        
        # Retry on failure
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))
        else:
            # Mark as failed
            update_product(product_id, {"image_status": "failed"})
            raise

@app.post("/products/{product_id}/upload-image")
async def upload_product_image(
    product_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload product image and process in background"""
    # Save original (fast)
    original_path = save_upload_to_s3(file, f"products/{product_id}/original.jpg")
    
    # Update product
    update_product(product_id, {
        "image_original": original_path,
        "image_status": "processing"
    })
    
    # Trigger processing
    task = process_product_image_task.delay(product_id, original_path)
    
    return {
        "product_id": product_id,
        "task_id": task.id,
        "status": "processing"
    }
\`\`\`

**Monitoring & Health Checks**:

\`\`\`python
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Check Celery task status"""
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    return {
        "task_id": task_id,
        "status": task.state,  # PENDING, STARTED, SUCCESS, FAILURE
        "result": task.result if task.ready() else None,
        "error": str(task.info) if task.failed() else None
    }

@app.get("/health/workers")
async def check_celery_workers():
    """Health check for Celery workers"""
    inspect = celery_app.control.inspect()
    stats = inspect.stats()
    active = inspect.active()
    
    if not stats:
        return {"status": "unhealthy", "error": "No workers available"}
    
    return {
        "status": "healthy",
        "workers": len(stats),
        "active_tasks": sum(len(tasks) for tasks in (active or {}).values())
    }
\`\`\`

**Summary Table**:

| Task | Tool | Reason | Execution Time |
|------|------|--------|---------------|
| Order confirmation email | BackgroundTasks | Fast, simple, non-critical | < 1 second |
| Payment processing | Celery | Critical, needs retries, idempotency | 5-30 seconds |
| Daily sales report | Celery Beat | Scheduled, long-running | 5-10 minutes |
| Image processing | Celery | CPU-intensive, needs retries | 10-60 seconds |`,
  },
  {
    id: 2,
    question:
      "Design a comprehensive retry strategy for background tasks that interact with external APIs (payment gateways, email services, SMS providers). Some failures are transient (network timeouts) and should be retried, while others are permanent (invalid API key, card declined) and shouldn't be retried. How would you categorize errors? What retry backoff strategy would you use and why? Implement a Celery task that processes payments with intelligent retry logic, including exponential backoff, max retries, dead letter queues for failed tasks, and alerting when tasks fail permanently. How would you test this retry logic?",
    answer: `**Intelligent Retry Strategy Design**:

**1. Error Categorization**:

\`\`\`python
"""
Categorize errors for retry decisions
"""

from typing import Type
from requests.exceptions import Timeout, ConnectionError

# Transient errors - SHOULD RETRY
TRANSIENT_ERRORS = (
    Timeout,  # Network timeout
    ConnectionError,  # Network connection failed
    # Payment gateway specific
    stripe.error.APIConnectionError,  # Can't reach Stripe
    stripe.error.RateLimitError,  # Rate limit (wait and retry)
    # HTTP status codes
    # 408: Request Timeout
    # 429: Too Many Requests
    # 500: Internal Server Error (gateway issue)
    # 502: Bad Gateway
    # 503: Service Unavailable
    # 504: Gateway Timeout
)

# Permanent errors - DON'T RETRY
PERMANENT_ERRORS = (
    # Payment specific
    stripe.error.CardError,  # Card declined, invalid, etc.
    stripe.error.InvalidRequestError,  # Bad parameters
    # Authentication
    stripe.error.AuthenticationError,  # Invalid API key
    # HTTP status codes
    # 400: Bad Request (our fault)
    # 401: Unauthorized (bad credentials)
    # 403: Forbidden (not allowed)
    # 404: Not Found
    # 422: Unprocessable Entity (validation error)
)

# Rate limit errors - RETRY with longer delay
RATE_LIMIT_ERRORS = (
    stripe.error.RateLimitError,
    # HTTP 429
)

def should_retry_error(exc: Exception) -> bool:
    """
    Determine if error should be retried
    """
    # Check if transient
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    
    # Check HTTP status codes
    if hasattr(exc, 'http_status'):
        status = exc.http_status
        # Retry on 5xx (server errors)
        if 500 <= status < 600:
            return True
        # Retry on 408 (timeout) and 429 (rate limit)
        if status in [408, 429]:
            return True
    
    # Don't retry permanent errors
    return False
\`\`\`

**2. Backoff Strategies**:

\`\`\`python
"""
Different backoff strategies
"""

def linear_backoff(retry_count: int, base_delay: int = 60) -> int:
    """Linear: 60, 120, 180, 240, 300"""
    return base_delay * (retry_count + 1)

def exponential_backoff(retry_count: int, base_delay: int = 10) -> int:
    """Exponential: 10, 20, 40, 80, 160"""
    return base_delay * (2 ** retry_count)

def fibonacci_backoff(retry_count: int, base_delay: int = 10) -> int:
    """Fibonacci: 10, 10, 20, 30, 50, 80, 130"""
    if retry_count == 0:
        return base_delay
    
    fib = [1, 1]
    for i in range(retry_count):
        fib.append(fib[-1] + fib[-2])
    
    return base_delay * fib[-1]

# Best for most cases: Exponential with jitter
import random

def exponential_backoff_with_jitter(retry_count: int, base_delay: int = 10, max_delay: int = 600) -> int:
    """
    Exponential backoff with jitter
    
    Prevents thundering herd problem:
    - Multiple tasks failing at same time
    - All retry at same time
    - Overwhelm service again
    
    Jitter adds randomness: 10-20, 15-40, 25-80, 40-160
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    jitter = random.uniform(0, delay * 0.3)  # ±30% randomness
    return int(delay + jitter)
\`\`\`

**3. Production-Ready Payment Task**:

\`\`\`python
"""
Payment processing with intelligent retry
"""

from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError
import logging

logger = logging.getLogger(__name__)

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

# Configure Celery
celery_app.conf.update(
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    task_track_started=True,  # Track when task starts
)

class PaymentTask(Task):
    """
    Base task for payment operations
    Includes retry logic and error handling
    """
    autoretry_for = TRANSIENT_ERRORS
    retry_kwargs = {'max_retries': 5}
    retry_backoff = True  # Use exponential backoff
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True  # Add randomness
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails permanently"""
        order_id = kwargs.get('order_id') or (args[0] if args else None)
        
        logger.error(
            f"Payment task failed permanently for order {order_id}",
            extra={
                "task_id": task_id,
                "order_id": order_id,
                "exception": str(exc),
                "retries": self.request.retries
            }
        )
        
        # Send alert
        send_alert_to_slack(
            channel="#payments-alerts",
            text=f"🚨 Payment failed permanently for order {order_id}",
            fields={
                "Order ID": order_id,
                "Error": str(exc),
                "Retries": self.request.retries
            }
        )
        
        # Update order status
        if order_id:
            update_order_status(order_id, "payment_failed", error=str(exc))
            send_payment_failed_email(order_id)

@celery_app.task(bind=True, base=PaymentTask)
def process_payment(self, order_id: int):
    """
    Process payment with intelligent retry
    """
    try:
        # Get order
        order = get_order(order_id)
        
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        # Idempotency check
        if order.payment_status == "completed":
            logger.info(f"Order {order_id} already paid")
            return {"status": "already_completed", "order_id": order_id}
        
        # Process payment
        logger.info(f"Processing payment for order {order_id}, attempt {self.request.retries + 1}")
        
        result = stripe.Charge.create(
            amount=int(order.total * 100),
            currency="usd",
            source=order.payment_token,
            idempotency_key=f"order_{order_id}",  # Prevent double charging
            metadata={"order_id": order_id},
            timeout=10  # 10 second timeout
        )
        
        # Success! Update order
        order.payment_status = "completed"
        order.payment_id = result.id
        save_order(order)
        
        logger.info(f"Payment successful for order {order_id}, payment_id: {result.id}")
        
        # Trigger fulfillment
        fulfill_order.delay(order_id)
        
        return {
            "status": "success",
            "order_id": order_id,
            "payment_id": result.id
        }
        
    except stripe.error.CardError as e:
        # PERMANENT: Card declined, expired, invalid
        logger.warning(f"Card declined for order {order_id}: {e.user_message}")
        
        # Update order
        update_order_status(
            order_id,
            "payment_declined",
            error=e.user_message
        )
        
        # Notify customer
        send_payment_declined_email(order_id, reason=e.user_message)
        
        # DON'T RETRY
        raise
        
    except stripe.error.InvalidRequestError as e:
        # PERMANENT: Bad parameters (our bug)
        logger.error(f"Invalid payment request for order {order_id}: {e}")
        
        # Alert developers (this is a bug)
        send_alert_to_pagerduty(
            summary=f"Invalid payment request for order {order_id}",
            details=str(e),
            severity="error"
        )
        
        # DON'T RETRY
        raise
        
    except stripe.error.RateLimitError as e:
        # RATE LIMITED: Retry with longer delay
        logger.warning(f"Rate limited for order {order_id}, retrying...")
        
        # Custom retry with longer delay
        if self.request.retries < 10:  # More retries for rate limits
            raise self.retry(
                exc=e,
                countdown=min(120 * (2 ** self.request.retries), 3600)  # Up to 1 hour
            )
        else:
            raise MaxRetriesExceededError(f"Rate limit persists after {self.request.retries} retries")
        
    except (stripe.error.APIConnectionError, stripe.error.APIError, Timeout) as e:
        # TRANSIENT: Network or gateway issue, RETRY
        logger.warning(f"Transient error for order {order_id}: {e}, retry {self.request.retries + 1}/5")
        
        # Check max retries
        if self.request.retries >= 5:
            logger.error(f"Max retries exceeded for order {order_id}")
            raise MaxRetriesExceededError(f"Payment failed after {self.request.retries} retries")
        
        # Retry with exponential backoff + jitter
        countdown = exponential_backoff_with_jitter(
            self.request.retries,
            base_delay=30,  # Start at 30 seconds
            max_delay=600   # Max 10 minutes
        )
        
        logger.info(f"Retrying order {order_id} in {countdown} seconds")
        
        raise self.retry(exc=e, countdown=countdown)
        
    except Exception as e:
        # UNKNOWN: Log and retry with caution
        logger.error(f"Unexpected error for order {order_id}: {e}")
        
        # Alert team
        send_alert_to_pagerduty(
            summary=f"Unexpected payment error for order {order_id}",
            details=str(e),
            severity="critical"
        )
        
        # Retry once with long delay
        if self.request.retries < 1:
            raise self.retry(exc=e, countdown=300)  # 5 minutes
        else:
            raise
\`\`\`

**4. Dead Letter Queue**:

\`\`\`python
"""
Handle permanently failed tasks
"""

# Dead letter queue
DEAD_LETTER_QUEUE = "failed_payments"

@celery_app.task
def handle_dead_letter_payment(order_id: int, error: str):
    """
    Handle payment that failed permanently
    Manual investigation required
    """
    logger.critical(f"Payment {order_id} moved to dead letter queue: {error}")
    
    # Store in database
    failed_payment = FailedPayment(
        order_id=order_id,
        error=error,
        created_at=datetime.utcnow(),
        status="pending_investigation"
    )
    db.add(failed_payment)
    db.commit()
    
    # Alert finance team
    send_email_to_finance_team(
        subject=f"Payment Failed - Manual Review Required",
        body=f"Order {order_id} failed: {error}\\n\\nRequires manual investigation."
    )
    
    # Create ticket in support system
    create_support_ticket(
        title=f"Payment Failed: Order {order_id}",
        description=error,
        priority="high",
        assignee="finance-team"
    )

# Override on_failure to use dead letter queue
class PaymentTaskWithDLQ(PaymentTask):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Send to dead letter queue on permanent failure"""
        super().on_failure(exc, task_id, args, kwargs, einfo)
        
        order_id = kwargs.get('order_id') or (args[0] if args else None)
        
        # Send to dead letter queue
        handle_dead_letter_payment.delay(order_id, str(exc))
\`\`\`

**5. Testing Retry Logic**:

\`\`\`python
"""
Test retry behavior
"""

import pytest
from unittest.mock import patch, Mock

def test_payment_success_no_retry():
    """Successful payment should not retry"""
    with patch('stripe.Charge.create') as mock_charge:
        mock_charge.return_value = Mock(id="ch_123")
        
        result = process_payment(order_id=1)
        
        assert result["status"] == "success"
        assert mock_charge.call_count == 1  # Called once, no retries

def test_payment_transient_error_retries():
    """Transient error should retry with backoff"""
    with patch('stripe.Charge.create') as mock_charge:
        # Fail twice, succeed on third
        mock_charge.side_effect = [
            stripe.error.APIConnectionError("Network error"),
            stripe.error.APIConnectionError("Network error"),
            Mock(id="ch_123")  # Success
        ]
        
        result = process_payment(order_id=1)
        
        assert result["status"] == "success"
        assert mock_charge.call_count == 3  # Retried twice

def test_payment_permanent_error_no_retry():
    """Permanent error (card declined) should not retry"""
    with patch('stripe.Charge.create') as mock_charge:
        mock_charge.side_effect = stripe.error.CardError(
            message="Card declined",
            param=None,
            code="card_declined"
        )
        
        with pytest.raises(stripe.error.CardError):
            process_payment(order_id=1)
        
        assert mock_charge.call_count == 1  # No retries

def test_payment_max_retries_exceeded():
    """Should fail after max retries"""
    with patch('stripe.Charge.create') as mock_charge:
        # Always fail
        mock_charge.side_effect = stripe.error.APIConnectionError("Network error")
        
        with pytest.raises(MaxRetriesExceededError):
            process_payment(order_id=1)
        
        assert mock_charge.call_count == 6  # Initial + 5 retries

def test_payment_idempotency():
    """Should not process payment twice"""
    # Order already paid
    order = Mock(payment_status="completed")
    
    with patch('get_order', return_value=order):
        result = process_payment(order_id=1)
        
        assert result["status"] == "already_completed"

def test_exponential_backoff_timing():
    """Verify exponential backoff delays"""
    delays = [exponential_backoff_with_jitter(i, base_delay=10, max_delay=600) for i in range(5)]
    
    # Each delay should be roughly 2x previous (with jitter)
    assert 10 <= delays[0] <= 30  # ~10-20
    assert 15 <= delays[1] <= 50  # ~20-40
    assert 30 <= delays[2] <= 100  # ~40-80
    assert 70 <= delays[3] <= 200  # ~80-160
    assert 140 <= delays[4] <= 400  # ~160-320
\`\`\`

**Best Practices Summary**:

1. **Categorize errors**: Transient (retry) vs permanent (don't retry)
2. **Exponential backoff with jitter**: Prevents thundering herd
3. **Max retries**: Fail eventually, don't retry forever
4. **Idempotency**: Safe to retry (use idempotency keys)
5. **Alerting**: Notify team when tasks fail permanently
6. **Dead letter queue**: Handle failed tasks manually
7. **Comprehensive logging**: Track every retry attempt
8. **Testing**: Test success, transient failures, permanent failures, max retries`,
  },
  {
    id: 3,
    question:
      'Design a monitoring and observability strategy for a production Celery deployment processing thousands of tasks per minute. What metrics would you track? How would you detect and alert on task failures, slow tasks, worker failures, and queue backlogs? Implement a comprehensive monitoring solution using Prometheus and Grafana, including health check endpoints, custom metrics, and alerting rules. How would you debug a situation where tasks are being accepted but not processed (stuck in queue)? What are the common causes and solutions?',
    answer: `**Production Celery Monitoring Strategy**:

**1. Key Metrics to Track**:

\`\`\`python
"""
Prometheus metrics for Celery
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Task counters
TASK_SUBMITTED = Counter(
    'celery_tasks_submitted_total',
    'Total tasks submitted',
    ['task_name']
)

TASK_STARTED = Counter(
    'celery_tasks_started_total',
    'Total tasks started',
    ['task_name']
)

TASK_SUCCESS = Counter(
    'celery_tasks_success_total',
    'Total tasks completed successfully',
    ['task_name']
)

TASK_FAILURE = Counter(
    'celery_tasks_failure_total',
    'Total tasks failed',
    ['task_name', 'exception_type']
)

TASK_RETRY = Counter(
    'celery_tasks_retry_total',
    'Total task retries',
    ['task_name']
)

# Task duration
TASK_DURATION = Histogram(
    'celery_task_duration_seconds',
    'Task execution duration',
    ['task_name'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]  # 100ms to 10min
)

# Queue metrics
QUEUE_LENGTH = Gauge(
    'celery_queue_length',
    'Number of tasks in queue',
    ['queue_name']
)

QUEUE_ACTIVE_TASKS = Gauge(
    'celery_queue_active_tasks',
    'Number of tasks currently being processed',
    ['queue_name']
)

# Worker metrics
WORKER_STATUS = Gauge(
    'celery_worker_status',
    'Worker status (1=up, 0=down)',
    ['worker_name']
)

WORKER_POOL_SIZE = Gauge(
    'celery_worker_pool_size',
    'Worker pool size (number of concurrent workers)',
    ['worker_name']
)

# Application metrics
CELERY_INFO = Info(
    'celery_info',
    'Celery version and configuration'
)
\`\`\`

**2. Instrumentation with Celery Signals**:

\`\`\`python
"""
Hook into Celery signals to track metrics
"""

from celery.signals import (
    task_prerun, task_postrun, task_failure, task_retry,
    worker_ready, worker_shutdown, worker_heartbeat
)

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Called before task starts"""
    TASK_STARTED.labels(task_name=sender.name).inc()
    
    # Store start time on task instance
    task.start_time = time.time()
    
    logger.info(
        f"Task started: {sender.name}",
        extra={"task_id": task_id, "task_name": sender.name}
    )

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, retval=None, **kwargs):
    """Called after task completes successfully"""
    TASK_SUCCESS.labels(task_name=sender.name).inc()
    
    # Track duration
    duration = time.time() - getattr(task, 'start_time', time.time())
    TASK_DURATION.labels(task_name=sender.name).observe(duration)
    
    logger.info(
        f"Task completed: {sender.name}",
        extra={
            "task_id": task_id,
            "task_name": sender.name,
            "duration_seconds": duration,
            "result": retval
        }
    )

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Called when task fails"""
    TASK_FAILURE.labels(
        task_name=sender.name,
        exception_type=type(exception).__name__
    ).inc()
    
    logger.error(
        f"Task failed: {sender.name}",
        extra={
            "task_id": task_id,
            "task_name": sender.name,
            "exception": str(exception)
        }
    )

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, **kwargs):
    """Called when task is retried"""
    TASK_RETRY.labels(task_name=sender.name).inc()
    
    logger.warning(
        f"Task retrying: {sender.name}",
        extra={
            "task_id": task_id,
            "task_name": sender.name,
            "reason": str(reason)
        }
    )

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker starts"""
    WORKER_STATUS.labels(worker_name=sender.hostname).set(1)
    
    logger.info(f"Worker ready: {sender.hostname}")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when worker shuts down"""
    WORKER_STATUS.labels(worker_name=sender.hostname).set(0)
    
    logger.info(f"Worker shutdown: {sender.hostname}")
\`\`\`

**3. Health Check Endpoints**:

\`\`\`python
"""
FastAPI health check endpoints
"""

from fastapi import FastAPI
from celery import Celery
from datetime import datetime, timedelta

app = FastAPI()
celery_app = Celery('tasks')

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/health/celery")
async def celery_health():
    """
    Comprehensive Celery health check
    """
    try:
        # Check if we can reach Celery
        inspect = celery_app.control.inspect()
        
        # Get worker stats
        stats = inspect.stats()
        if not stats:
            return {
                "status": "unhealthy",
                "error": "No workers available",
                "workers": 0
            }
        
        # Get active tasks
        active = inspect.active()
        active_count = sum(len(tasks) for tasks in (active or {}).values())
        
        # Get reserved tasks (in prefetch)
        reserved = inspect.reserved()
        reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())
        
        # Get registered tasks
        registered = inspect.registered()
        
        # Check queue lengths
        queue_stats = {}
        for queue in ['default', 'high_priority', 'low_priority']:
            try:
                length = celery_app.control.inspect().active_queues()
                queue_stats[queue] = length
            except:
                pass
        
        return {
            "status": "healthy",
            "workers": {
                "count": len(stats),
                "names": list(stats.keys())
            },
            "tasks": {
                "active": active_count,
                "reserved": reserved_count,
                "registered": len(next(iter(registered.values()), []))
            },
            "queues": queue_stats
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/health/celery/workers")
async def celery_workers():
    """
    Detailed worker information
    """
    inspect = celery_app.control.inspect()
    
    return {
        "stats": inspect.stats(),
        "active": inspect.active(),
        "reserved": inspect.reserved(),
        "scheduled": inspect.scheduled(),
        "registered": inspect.registered()
    }

@app.get("/health/celery/queues")
async def celery_queues():
    """
    Queue information
    """
    # Get queue lengths from Redis
    from redis import Redis
    redis_client = Redis.from_url(REDIS_URL)
    
    queues = ['default', 'high_priority', 'low_priority']
    queue_info = {}
    
    for queue in queues:
        length = redis_client.llen(f'celery_queue_{queue}')
        queue_info[queue] = {
            "length": length,
            "status": "healthy" if length < 1000 else "backlog"
        }
    
    return queue_info
\`\`\`

**4. Grafana Dashboard**:

\`\`\`yaml
# Prometheus alerting rules
# /etc/prometheus/celery_alerts.yml

groups:
  - name: celery_alerts
    interval: 30s
    rules:
      # Task failure rate too high
      - alert: HighTaskFailureRate
        expr: |
          rate(celery_tasks_failure_total[5m]) /
          rate(celery_tasks_started_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High task failure rate"
          description: "{{ $value | humanizePercentage }} of tasks failing"
      
      # Queue backlog growing
      - alert: QueueBacklog
        expr: celery_queue_length > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Queue backlog detected"
          description: "Queue {{ $labels.queue_name }} has {{ $value }} tasks"
      
      # No workers available
      - alert: NoWorkersAvailable
        expr: sum(celery_worker_status) == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "No Celery workers available"
          description: "All Celery workers are down"
      
      # Slow tasks
      - alert: SlowTaskExecution
        expr: |
          histogram_quantile(0.95, 
            rate(celery_task_duration_seconds_bucket[5m])
          ) > 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Tasks executing slowly"
          description: "95th percentile task duration is {{ $value }}s"
      
      # Worker crashed
      - alert: WorkerCrashed
        expr: changes(celery_worker_status[5m]) > 0 and celery_worker_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Celery worker crashed"
          description: "Worker {{ $labels.worker_name }} crashed"
\`\`\`

**5. Debugging Stuck Queue**:

**Symptoms**: Tasks submitted but not processed, queue length growing.

**Common Causes & Solutions**:

\`\`\`python
"""
Diagnostic script for stuck tasks
"""

import redis
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

def diagnose_stuck_queue():
    """
    Diagnose why tasks are stuck
    """
    print("=== Celery Diagnostic Report ===\\n")
    
    # 1. Check if workers are running
    print("1. Checking workers...")
    inspect = celery_app.control.inspect()
    stats = inspect.stats()
    
    if not stats:
        print("❌ NO WORKERS RUNNING!")
        print("Solution: Start workers with: celery -A tasks worker --loglevel=info")
        return
    else:
        print(f"✅ {len(stats)} worker(s) running")
        for worker_name in stats.keys():
            print(f"   - {worker_name}")
    
    # 2. Check if workers are accepting tasks
    print("\\n2. Checking if workers accepting tasks...")
    active_queues = inspect.active_queues()
    
    if not active_queues:
        print("❌ Workers not consuming from any queue!")
        print("Solution: Check worker configuration")
    else:
        for worker, queues in active_queues.items():
            print(f"   {worker}: {[q['name'] for q in queues]}")
    
    # 3. Check Redis connection
    print("\\n3. Checking Redis connection...")
    try:
        redis_client = redis.from_url('redis://localhost:6379/0')
        redis_client.ping()
        print("✅ Redis connection OK")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return
    
    # 4. Check queue lengths
    print("\\n4. Checking queue lengths...")
    queues = ['celery', 'default', 'high_priority', 'low_priority']
    
    for queue in queues:
        try:
            length = redis_client.llen(queue)
            if length > 0:
                print(f"   {queue}: {length} tasks")
        except:
            pass
    
    # 5. Check for tasks in unacked state
    print("\\n5. Checking unacknowledged tasks...")
    reserved = inspect.reserved()
    
    if reserved:
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        print(f"   {total_reserved} task(s) reserved but not completed")
        print("   Possible causes:")
        print("   - Workers stuck on long-running tasks")
        print("   - Worker processes killed mid-task")
        print("   Solution: Restart workers or increase timeout")
    
    # 6. Check worker pool settings
    print("\\n6. Checking worker configuration...")
    for worker_name, worker_stats in stats.items():
        pool = worker_stats.get('pool', {})
        print(f"   {worker_name}:")
        print(f"      Pool: {pool.get('implementation')}")
        print(f"      Max concurrency: {pool.get('max-concurrency')}")
        print(f"      Prefetch multiplier: {worker_stats.get('prefetch_count')}")
    
    # 7. Check for exceptions in worker logs
    print("\\n7. Check worker logs for exceptions:")
    print("   tail -f /var/log/celery/worker.log")
    
    print("\\n=== Diagnostic Complete ===")

# Run diagnostic
if __name__ == "__main__":
    diagnose_stuck_queue()
\`\`\`

**Common Issues & Fixes**:

1. **No workers running**:
   \`\`\`bash
   # Start workers
   celery - A tasks worker--loglevel = info--concurrency = 4
   \`\`\`

2. **Workers not consuming from queue**:
   \`\`\`python
   # Check worker is listening to correct queue
   celery - A tasks worker--loglevel = info--queues =default, high_priority
   \`\`\`

3. **Workers stuck on long tasks**:
   \`\`\`python
   # Increase concurrency
celery - A tasks worker--concurrency = 8
   
   # Or use different queues for long / short tasks
    \`\`\`

4. **Memory leak in workers**:
   \`\`\`python
   # Restart workers after N tasks
celery - A tasks worker--max - tasks - per - child=1000
    \`\`\`

5. **Redis connection pool exhausted**:
   \`\`\`python
   # Increase Redis connections
celery_app.conf.broker_pool_limit = 10
    \`\`\`

**Monitoring Checklist**:

✅ Track: Task success/failure rate, duration, queue length, worker status  
✅ Alert: High failure rate, queue backlog, no workers, slow tasks  
✅ Dashboard: Real-time Grafana with task throughput and latency graphs  
✅ Health checks: Expose \`/health/celery\` endpoint  
✅ Logging: Structured logs with task_id, task_name, duration  
✅ Diagnostic tools: Scripts to identify stuck queues  
✅ Regular reviews: Analyze slow tasks weekly`,
  },
].map(({ id, ...q }, idx) => ({
  id: `fastapi-background-tasks-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
