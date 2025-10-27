export const taskQueueFundamentals = {
  title: 'Task Queue Fundamentals',
  id: 'task-queue-fundamentals',
  content: `
# Task Queue Fundamentals

## Introduction

**Task queues** are the backbone of scalable applications, enabling asynchronous processing of time-consuming operations. Instead of making users wait for slow operations (sending emails, generating reports, processing images), you queue these tasks and process them in the background.

**Why Task Queues Matter:**
- **User Experience**: Instant responses vs waiting 30 seconds for email sending
- **Scalability**: Process millions of tasks across distributed workers
- **Reliability**: Retry failed tasks automatically
- **Resource Management**: Smooth out traffic spikes

### The Problem: Synchronous Processing

\`\`\`python
"""
PROBLEMATIC: Synchronous email sending
"""

from flask import Flask, request, jsonify
import smtplib
from email.mime.text import MIMEText
import time

app = Flask(__name__)

@app.route('/register', methods=['POST'])
def register_user():
    email = request.json['email']
    username = request.json['username']
    
    # Step 1: Save user to database (fast ~10ms)
    user_id = save_to_database (email, username)
    
    # Step 2: Send welcome email (SLOW ~3-5 seconds!)
    send_welcome_email (email, username)  # BLOCKS HERE ⏰
    
    # Step 3: Generate user report (SLOW ~10 seconds!)
    generate_welcome_report (user_id)  # BLOCKS HERE TOO ⏰
    
    # User waits 13-15 seconds before seeing response! 😱
    return jsonify({'message': 'User registered'}), 201


def send_welcome_email (email: str, username: str):
    """Simulates slow email sending"""
    time.sleep(3)  # Network latency, SMTP connection
    msg = MIMEText (f"Welcome {username}!")
    msg['Subject'] = 'Welcome!'
    msg['To'] = email
    # ... SMTP sending logic ...
    print(f"Email sent to {email}")


def generate_welcome_report (user_id: int):
    """Simulates expensive report generation"""
    time.sleep(10)  # Database queries, PDF generation
    print(f"Report generated for user {user_id}")


def save_to_database (email: str, username: str) -> int:
    """Fast database write"""
    return 12345


# Testing
if __name__ == '__main__':
    # Simulate user registration
    import requests
    
    start = time.time()
    response = requests.post('http://localhost:5000/register', 
                           json={'email': 'user@example.com', 'username': 'alice'})
    duration = time.time() - start
    
    print(f"Response: {response.json()}")
    print(f"Duration: {duration:.2f}s")  # ~13-15 seconds! 🐌
\`\`\`

**Problems:**1. **Poor UX**: User waits 13+ seconds for simple registration
2. **Resource Waste**: Web workers blocked during I/O operations
3. **No Retry**: If email fails, user registration rolls back
4. **Scalability**: 100 concurrent registrations = 100 blocked threads
5. **Timeouts**: Browsers timeout after 30-60 seconds

---

## The Solution: Task Queues

**Architecture Pattern:**

\`\`\`
┌─────────┐       ┌─────────────┐       ┌─────────┐       ┌─────────┐
│  User   │──────▶│  Web Server │──────▶│  Queue  │──────▶│ Worker  │
└─────────┘ HTTP  └─────────────┘ Task  └─────────┘ Task  └─────────┘
   (1)              (2)                    (3)               (4)
Sends request    Queues task          Stores task        Processes task
                 Returns immediately   (Redis/RabbitMQ)   asynchronously
\`\`\`

### How It Works:

1. **User makes request** → "Register user with email alice@example.com"
2. **Web server:**
   - Saves user to database (fast ~10ms)
   - **Queues** email task to Redis/RabbitMQ
   - Returns success response immediately (total ~50ms)
3. **Task stored in queue** → Waiting for available worker
4. **Worker picks up task** → Processes email in background
5. **User gets instant response** → No waiting for slow operations!

---

## Synchronous vs Asynchronous Processing

### Synchronous (Blocking):

\`\`\`python
"""
Synchronous: Everything happens sequentially
"""

def process_order_sync (order_id: int):
    # These run one after another (user waits for all)
    validate_payment (order_id)           # 2 seconds
    update_inventory (order_id)           # 1 second
    send_confirmation_email (order_id)    # 3 seconds
    notify_warehouse (order_id)           # 2 seconds
    generate_invoice (order_id)           # 5 seconds
    # Total: 13 seconds of user waiting! 🐌
    
    return "Order processed"


# User experience:
import time
start = time.time()
result = process_order_sync(12345)
print(f"Done in {time.time() - start:.1f}s")  # 13.0s
\`\`\`

### Asynchronous (Non-Blocking with Task Queue):

\`\`\`python
"""
Asynchronous: Critical work now, rest queued for later
"""

from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')


@app.task
def send_confirmation_email (order_id: int):
    """Background task"""
    time.sleep(3)
    print(f"Email sent for order {order_id}")


@app.task
def notify_warehouse (order_id: int):
    """Background task"""
    time.sleep(2)
    print(f"Warehouse notified for order {order_id}")


@app.task
def generate_invoice (order_id: int):
    """Background task"""
    time.sleep(5)
    print(f"Invoice generated for order {order_id}")


def process_order_async (order_id: int):
    # Do critical work NOW (user waits)
    validate_payment (order_id)    # 2 seconds - MUST wait
    update_inventory (order_id)    # 1 second - MUST wait
    
    # Queue non-critical work for LATER (instant)
    send_confirmation_email.delay (order_id)   # Queued ~1ms
    notify_warehouse.delay (order_id)          # Queued ~1ms
    generate_invoice.delay (order_id)          # Queued ~1ms
    
    # User gets response after 3 seconds instead of 13!
    return "Order processed"


# User experience:
start = time.time()
result = process_order_async(12345)
print(f"Done in {time.time() - start:.1f}s")  # 3.0s ✨ (10s faster!)

# Meanwhile, workers process tasks in background:
# T+0s: Worker picks up email task (runs for 3s)
# T+3s: Worker picks up warehouse task (runs for 2s)
# T+5s: Worker picks up invoice task (runs for 5s)
# T+10s: All tasks complete (but user already got response!)
\`\`\`

**Benefits:**
- **10× faster response**: 3s vs 13s
- **Better UX**: User doesn't wait for background tasks
- **Scalability**: Add more workers to process tasks faster
- **Reliability**: Tasks retried automatically if they fail

---

## Message Broker Role

The **message broker** is the middleman between web servers (producers) and workers (consumers).

### How Message Brokers Work:

\`\`\`
┌──────────────┐                ┌──────────────────┐                ┌──────────┐
│ Web Server 1 │───push task───▶│                  │◀───pull task───│ Worker 1 │
└──────────────┘                │  Message Broker  │                └──────────┘
┌──────────────┐                │  (Redis/RabbitMQ)│                ┌──────────┐
│ Web Server 2 │───push task───▶│                  │◀───pull task───│ Worker 2 │
└──────────────┘                │  - Stores tasks  │                └──────────┘
┌──────────────┐                │  - Queues tasks  │                ┌──────────┐
│ Web Server 3 │───push task───▶│  - Routes tasks  │◀───pull task───│ Worker 3 │
└──────────────┘                └──────────────────┘                └──────────┘
     (Fast)                      (Persistent storage)                  (Slow)
    Return                        Reliable queue                    Process tasks
  immediately                     Survives restarts                 in background
\`\`\`

### Key Responsibilities:

**1. Task Storage:**
- Tasks stored in memory (Redis) or disk (RabbitMQ)
- Persistent: Survives broker restarts
- Ordered: FIFO (first-in, first-out)

**2. Task Routing:**
- Route tasks to specific queues
- Priority queues for urgent tasks
- Fanout to multiple workers

**3. Reliability:**
- Acknowledgments: Worker confirms task completion
- Requeuing: Failed tasks go back to queue
- Dead letter queues: Store permanently failed tasks

---

## Producer-Consumer Pattern

The task queue implements the **producer-consumer pattern**:

\`\`\`python
"""
Producer-Consumer Pattern with Celery
"""

from celery import Celery
import time

app = Celery('myapp', broker='redis://localhost:6379/0')


# ========================================
# PRODUCER (Web Server / API)
# ========================================

def producer_web_server():
    """
    Producer: Creates tasks and adds them to queue
    
    This runs in your web application (Flask, Django, FastAPI)
    """
    print("Producer: Received user request")
    
    # Queue tasks (producers don't process, just queue)
    result1 = process_video.delay (video_id=123)
    result2 = send_email.delay (email='user@example.com')
    result3 = generate_report.delay (user_id=456)
    
    print(f"Producer: Queued 3 tasks")
    print(f"Producer: Task IDs: {result1.id}, {result2.id}, {result3.id}")
    
    # Return immediately (don't wait for tasks)
    return "Tasks queued successfully"


# ========================================
# CONSUMER (Celery Worker)
# ========================================

@app.task (name='process_video')
def process_video (video_id: int):
    """
    Consumer: Processes tasks from queue
    
    This runs in background workers
    """
    print(f"Consumer Worker: Processing video {video_id}")
    time.sleep(10)  # Simulate video processing
    print(f"Consumer Worker: Video {video_id} complete")
    return f"Video {video_id} processed"


@app.task (name='send_email')
def send_email (email: str):
    """Consumer task"""
    print(f"Consumer Worker: Sending email to {email}")
    time.sleep(2)
    print(f"Consumer Worker: Email sent to {email}")
    return f"Email sent to {email}"


@app.task (name='generate_report')
def generate_report (user_id: int):
    """Consumer task"""
    print(f"Consumer Worker: Generating report for user {user_id}")
    time.sleep(5)
    print(f"Consumer Worker: Report complete for user {user_id}")
    return f"Report generated for user {user_id}"


# ========================================
# EXECUTION FLOW
# ========================================

if __name__ == '__main__':
    print("=== Producer-Consumer Pattern Demo ===\\n")
    
    # Producer: Queue tasks
    start = time.time()
    result = producer_web_server()
    producer_time = time.time() - start
    
    print(f"\\nProducer completed in {producer_time:.3f}s")
    print("(Producer doesn't wait - returns immediately!)\\n")
    
    print("Consumers (workers) processing tasks in background...")
    print("(Run 'celery -A myapp worker' in another terminal)")
\`\`\`

### Multiple Consumers:

\`\`\`
Queue: [Task1] [Task2] [Task3] [Task4] [Task5] [Task6]
          │       │       │       │       │       │
          ▼       ▼       ▼       ▼       ▼       ▼
       Worker1  Worker2  Worker3  Worker1  Worker2  Worker3
       (Task1)  (Task2)  (Task3)  (Task4)  (Task5)  (Task6)
         3s       5s       2s       3s       5s       2s
         
Timeline:
T+0s: All 3 workers start processing (parallel!)
T+2s: Worker3 done, picks up Task4 (if exists)
T+3s: Worker1 done, picks up Task5 (if exists)
T+5s: Worker2 done, picks up Task6 (if exists)

Total time: 5s (vs 20s if sequential!)
\`\`\`

---

## Task Queue Benefits

### 1. **Improved User Experience**

\`\`\`python
# WITHOUT task queue:
# User: Click "Register" → Wait 15 seconds → See success
# Duration: 15 seconds 🐌

# WITH task queue:
# User: Click "Register" → See success immediately → Receive email later
# Duration: 0.05 seconds ✨
\`\`\`

### 2. **Scalability**

\`\`\`
Scenario: 1000 concurrent user registrations

WITHOUT task queue:
- 1000 web workers blocked for 15s each
- Need 1000 concurrent connections
- System overwhelmed 🔥

WITH task queue:
- 10 web workers handle requests (50ms each)
- Queue 1000 tasks to Redis (instant)
- 50 background workers process tasks
- System scales horizontally ✨
\`\`\`

### 3. **Reliability with Retries**

\`\`\`python
@app.task (bind=True, max_retries=3)
def send_email (self, email: str):
    try:
        # Attempt to send email
        smtp_send (email)
    except SMTPException as exc:
        # Retry with exponential backoff
        raise self.retry (exc=exc, countdown=60 * (2 ** self.request.retries))

# Automatic retry schedule:
# Attempt 1: Immediate
# Attempt 2: After 60s (2^0 * 60)
# Attempt 3: After 120s (2^1 * 60)
# Attempt 4: After 240s (2^2 * 60)
# After 3 retries: Task marked as failed
\`\`\`

### 4. **Resource Management**

\`\`\`python
"""
Smooth out traffic spikes with queues
"""

# E-commerce site: 10,000 orders in 1 minute (Black Friday)

# WITHOUT task queue:
# - All 10,000 orders processed immediately
# - Database overloaded
# - API timeouts
# - System crashes 💥

# WITH task queue:
# - 10,000 tasks queued in Redis (instant)
# - 100 workers process 100 tasks/min
# - Takes 100 minutes to complete
# - System stable, predictable load ✨
\`\`\`

---

## Common Use Cases

### 1. **Email Sending**
\`\`\`python
@app.task
def send_welcome_email (user_id: int):
    """Send email asynchronously"""
    user = User.query.get (user_id)
    send_email (user.email, "Welcome!", template='welcome.html')
\`\`\`

### 2. **Report Generation**
\`\`\`python
@app.task
def generate_monthly_report (company_id: int):
    """Generate PDF report (slow)"""
    data = fetch_month_data (company_id)
    pdf = generate_pdf (data)
    upload_to_s3(pdf, f'reports/{company_id}/monthly.pdf')
    notify_user (company_id, pdf_url)
\`\`\`

### 3. **Data Processing**
\`\`\`python
@app.task
def process_uploaded_image (image_id: int):
    """Resize, compress, generate thumbnails"""
    image = Image.open (get_image_path (image_id))
    
    # Generate multiple sizes
    thumbnail = image.resize((150, 150))
    medium = image.resize((600, 600))
    
    # Upload to CDN
    upload_to_cdn (thumbnail, f'thumbnails/{image_id}.jpg')
    upload_to_cdn (medium, f'medium/{image_id}.jpg')
\`\`\`

### 4. **Scheduled Jobs**
\`\`\`python
from celery.schedules import crontab

@app.on_after_configure.connect
def setup_periodic_tasks (sender, **kwargs):
    # Run every day at midnight
    sender.add_periodic_task(
        crontab (hour=0, minute=0),
        backup_database.s(),
    )

@app.task
def backup_database():
    """Daily database backup"""
    dump = pg_dump()
    upload_to_s3(dump, f'backups/{today}.sql')
\`\`\`

### 5. **External API Calls**
\`\`\`python
@app.task (bind=True, max_retries=5)
def fetch_stock_price (self, symbol: str):
    """Call external API with retries"""
    try:
        response = requests.get (f'https://api.stocks.com/{symbol}')
        price = response.json()['price']
        cache.set (f'price:{symbol}', price, timeout=60)
    except requests.RequestException as exc:
        raise self.retry (exc=exc, countdown=30)
\`\`\`

---

## When to Use Task Queues

### ✅ **Use Task Queues For:**1. **Operations that take >1 second**
   - Email sending, SMS, push notifications
   - PDF/report generation
   - Image/video processing
   
2. **External API calls**
   - Third-party webhooks
   - Payment processing callbacks
   - Social media posting
   
3. **Scheduled/periodic jobs**
   - Daily backups
   - Cleanup old data
   - Send weekly digests
   
4. **Batch processing**
   - Process 10,000 uploaded CSVs
   - Send 1M marketing emails
   - Generate thumbnails for 100K images
   
5. **Non-critical operations**
   - Analytics tracking
   - Logging to third-party services
   - Updating recommendation models

### ❌ **Don't Use Task Queues For:**1. **Operations needed for response**
   - User authentication (must verify immediately)
   - Payment authorization (must know result)
   - Search results (user expects immediate results)
   
2. **Very fast operations (<100ms)**
   - Simple database queries
   - Cache reads
   - Basic calculations
   - Overhead of queuing > operation time
   
3. **Operations requiring transaction atomicity**
   - Must succeed/fail with database transaction
   - Can't be eventually consistent

---

## Task Queue Architecture

\`\`\`python
"""
Complete Task Queue Architecture
"""

# ============================================
# Component 1: Message Broker (Redis/RabbitMQ)
# ============================================
# - Stores tasks
# - Manages queues
# - Handles routing

# ============================================
# Component 2: Web Application (Producer)
# ============================================
from flask import Flask, request
from celery import Celery

app = Flask(__name__)
celery_app = Celery('myapp', broker='redis://localhost:6379/0')

@app.route('/process', methods=['POST'])
def process_request():
    data = request.json
    
    # Queue task
    result = celery_task.delay (data)
    
    # Return task ID (user can check status later)
    return {'task_id': result.id}, 202

# ============================================
# Component 3: Celery Tasks (Task Definitions)
# ============================================
@celery_app.task (name='celery_task')
def celery_task (data):
    """Process data (runs in worker)"""
    # Heavy processing...
    return result

# ============================================
# Component 4: Celery Workers (Consumers)
# ============================================
# Run in separate processes:
# celery -A myapp worker --loglevel=info --concurrency=4

# ============================================
# Component 5: Result Backend (Optional)
# ============================================
# Stores task results for later retrieval
# Configuration:
celery_app.conf.result_backend = 'redis://localhost:6379/1'

# Check task status:
result = celery_task.delay (data)
print(result.status)  # PENDING, STARTED, SUCCESS, FAILURE
print(result.result)  # Return value (if complete)
\`\`\`

---

## Real-World Example: E-Commerce Order Processing

\`\`\`python
"""
E-Commerce Order Processing with Task Queue
"""

from flask import Flask, request, jsonify
from celery import Celery, group
import stripe
import time

app = Flask(__name__)
celery_app = Celery('ecommerce', broker='redis://localhost:6379/0')


# ========================================
# Web Endpoint (Producer)
# ========================================

@app.route('/orders', methods=['POST'])
def create_order():
    """
    Create order endpoint
    
    User experience: Immediate response (< 100ms)
    Background: All slow operations queued
    """
    data = request.json
    user_id = data['user_id']
    items = data['items']
    payment_method = data['payment_method']
    
    # Step 1: Validate (FAST - must happen now)
    if not validate_items (items):
        return {'error': 'Invalid items'}, 400
    
    # Step 2: Create order record (FAST - must happen now)
    order = create_order_in_db (user_id, items)
    order_id = order.id
    
    # Step 3: Charge payment (CRITICAL - must wait for result)
    # But still fast (Stripe API ~500ms)
    payment_result = stripe.Charge.create(
        amount=calculate_total (items),
        currency='usd',
        source=payment_method
    )
    
    if not payment_result.success:
        cancel_order (order_id)
        return {'error': 'Payment failed'}, 402
    
    # Step 4: Queue ALL slow, non-critical operations
    # These don't block user response!
    
    job = group([
        send_confirmation_email.s (order_id),
        notify_warehouse.s (order_id),
        update_inventory.s (items),
        generate_invoice.s (order_id),
        send_sms_notification.s (user_id, order_id),
        update_analytics.s (order_id),
        trigger_shipping_label.s (order_id)
    ])
    
    result = job.apply_async()
    
    # Step 5: Return immediately!
    return {
        'order_id': order_id,
        'status': 'processing',
        'message': 'Order placed successfully'
    }, 201
    # Total time: ~600ms (user happy! ✨)


# ========================================
# Background Tasks (Consumers)
# ========================================

@celery_app.task
def send_confirmation_email (order_id: int):
    """Send order confirmation (3s)"""
    time.sleep(3)
    print(f"✉️  Email sent for order {order_id}")


@celery_app.task
def notify_warehouse (order_id: int):
    """Notify warehouse to ship (2s)"""
    time.sleep(2)
    print(f"📦 Warehouse notified for order {order_id}")


@celery_app.task
def update_inventory (items: list):
    """Update stock quantities (1s)"""
    time.sleep(1)
    print(f"📊 Inventory updated for {len (items)} items")


@celery_app.task
def generate_invoice (order_id: int):
    """Generate PDF invoice (5s)"""
    time.sleep(5)
    print(f"🧾 Invoice generated for order {order_id}")


@celery_app.task
def send_sms_notification (user_id: int, order_id: int):
    """Send SMS confirmation (2s)"""
    time.sleep(2)
    print(f"📱 SMS sent to user {user_id} for order {order_id}")


@celery_app.task
def update_analytics (order_id: int):
    """Update analytics (1s)"""
    time.sleep(1)
    print(f"📈 Analytics updated for order {order_id}")


@celery_app.task
def trigger_shipping_label (order_id: int):
    """Generate shipping label (4s)"""
    time.sleep(4)
    print(f"🏷️  Shipping label generated for order {order_id}")


# Helper functions
def validate_items (items): return True
def create_order_in_db (user_id, items): 
    class Order: 
        id = 12345
    return Order()
def calculate_total (items): return 9999
def cancel_order (order_id): pass
\`\`\`

**Timeline:**

\`\`\`
Without Task Queue:
User clicks "Place Order" → Wait 18 seconds → See confirmation
Total: 18 seconds 🐌

With Task Queue:
User clicks "Place Order" → Wait 0.6 seconds → See confirmation
                                              ↓
                            Background workers process tasks
Total: 0.6 seconds ✨ (30× faster!)

Background processing (user doesn't wait):
T+0s: 7 tasks queued
T+1s: Inventory updated
T+2s: SMS sent, warehouse notified
T+3s: Email sent
T+4s: Shipping label done
T+5s: Invoice done
All tasks complete by T+5s (but user already gone!)
\`\`\`

---

## Summary

**Key Concepts:**
- Task queues enable asynchronous processing
- Message broker stores and routes tasks
- Producer-consumer pattern separates task creation from execution
- Workers process tasks in background (user doesn't wait)

**Benefits:**
- **Fast responses**: < 100ms instead of 10+ seconds
- **Scalability**: Add workers to handle more load
- **Reliability**: Automatic retries for failed tasks
- **Resource management**: Smooth out traffic spikes

**When to Use:**
- Operations taking > 1 second
- External API calls
- Scheduled jobs
- Batch processing
- Non-critical operations

**Next Section:** We'll dive into Celery architecture and set up your first Celery project! 🚀
`,
};
