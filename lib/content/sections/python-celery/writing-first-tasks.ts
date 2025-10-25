export const writingFirstTasks = {
  title: 'Writing Your First Tasks',
  id: 'writing-first-tasks',
  content: `
# Writing Your First Tasks

## Introduction

**Tasks** are the core building blocks of Celery. A task is a Python function decorated with \`@app.task\` that can be executed asynchronously by workers. This section covers how to define, call, and manage tasks effectively.

**What You'll Learn:**
- Defining tasks with \`@app.task\` decorator
- Calling tasks with \`.delay()\` and \`.apply_async()\`
- Task arguments and return values
- Task naming and discovery
- Best practices for task design

---

## Defining Your First Task

\`\`\`python
"""
Basic Task Definition
"""

from celery import Celery

# Initialize Celery app
app = Celery(
    'myapp',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Define a simple task
@app.task
def add (x: int, y: int) -> int:
    """Add two numbers"""
    return x + y

# Define a task that takes longer
@app.task
def send_email (email: str, subject: str, body: str) -> str:
    """Send an email (simulated)"""
    import time
    import smtplib
    
    print(f"Sending email to {email}...")
    time.sleep(2)  # Simulate email sending
    
    # In production: actual SMTP sending
    # msg = MIMEText (body)
    # msg['Subject'] = subject
    # msg['To'] = email
    # smtp = smtplib.SMTP('localhost')
    # smtp.send_message (msg)
    # smtp.quit()
    
    return f"Email sent to {email}"

# Define a task with more complex logic
@app.task
def process_user_data (user_id: int) -> dict:
    """Process user data"""
    import time
    
    # Step 1: Fetch user from database
    print(f"Fetching user {user_id}...")
    time.sleep(0.5)
    
    # Step 2: Process data
    print(f"Processing data for user {user_id}...")
    time.sleep(1)
    
    # Step 3: Update analytics
    print(f"Updating analytics for user {user_id}...")
    time.sleep(0.5)
    
    return {
        'user_id': user_id,
        'status': 'processed',
        'timestamp': time.time()
    }
\`\`\`

**Key Points:**
- Use \`@app.task\` decorator to mark function as a task
- Tasks are regular Python functions (can take args, return values)
- Add type hints for clarity (not required but recommended)
- Include docstrings to document task purpose

---

## Task Function Syntax

\`\`\`python
"""
Task Function Patterns
"""

from celery import Celery
from typing import List, Dict, Optional
import logging

app = Celery('myapp', broker='redis://localhost:6379/0')

# Pattern 1: Simple function (no arguments)
@app.task
def cleanup_temp_files():
    """No arguments, no return value"""
    import os
    import glob
    
    temp_files = glob.glob('/tmp/myapp_*')
    for file in temp_files:
        os.remove (file)
    
    print(f"Deleted {len (temp_files)} temp files")

# Pattern 2: Positional arguments
@app.task
def multiply (x: int, y: int) -> int:
    """Positional arguments"""
    return x * y

# Pattern 3: Keyword arguments
@app.task
def create_user (username: str, email: str, age: Optional[int] = None) -> dict:
    """Keyword arguments with defaults"""
    user = {
        'username': username,
        'email': email,
        'age': age or 18,
        'created_at': time.time()
    }
    # Save to database...
    return user

# Pattern 4: Variable arguments
@app.task
def sum_numbers(*args: int) -> int:
    """Variable positional arguments"""
    return sum (args)

# Pattern 5: Complex return types
@app.task
def fetch_user_stats (user_id: int) -> Dict[str, any]:
    """Complex return type (dict)"""
    return {
        'user_id': user_id,
        'posts': 42,
        'followers': 1337,
        'following': 256,
        'likes': 9999
    }

# Pattern 6: No return value (side effects only)
@app.task
def log_event (event_type: str, data: dict) -> None:
    """Side effect task (logging)"""
    logger = logging.getLogger(__name__)
    logger.info (f"Event: {event_type}", extra=data)
    # No return value needed

# Pattern 7: Lists/iterables
@app.task
def process_batch (items: List[dict]) -> List[dict]:
    """Process list of items"""
    results = []
    for item in items:
        result = process_item (item)
        results.append (result)
    return results
\`\`\`

---

## Calling Tasks: .delay() Method

The **simplest way** to queue a task is using \`.delay()\`:

\`\`\`python
"""
Using .delay() to Queue Tasks
"""

from tasks import add, send_email, process_user_data

# ========================================
# .delay() - Simple Task Queuing
# ========================================

# Example 1: Simple task
result = add.delay(4, 6)

print(f"Task queued!")
print(f"Task ID: {result.id}")
print(f"Task state: {result.state}")  # PENDING

# Wait for result (blocking)
output = result.get (timeout=10)
print(f"Result: {output}")  # 10

# Example 2: Task with multiple arguments
result = send_email.delay(
    email='user@example.com',
    subject='Welcome!',
    body='Thanks for signing up!'
)

print(f"Email task queued: {result.id}")

# Example 3: Task with keyword arguments
result = process_user_data.delay (user_id=12345)

# Check if task completed (non-blocking)
import time

for i in range(10):
    if result.ready():
        print(f"Task complete! Result: {result.result}")
        break
    else:
        print(f"Waiting... ({i+1}s)")
        time.sleep(1)

# ========================================
# .delay() is Shorthand for .apply_async()
# ========================================

# These are equivalent:
result1 = add.delay(4, 6)
result2 = add.apply_async (args=(4, 6))

# .delay() is cleaner and more Pythonic
# Use .delay() for simple cases
# Use .apply_async() when you need advanced options
\`\`\`

**\`.delay()\` Pros:**
- ✅ Simple, clean syntax
- ✅ Pythonic (feels like regular function call)
- ✅ Good for 90% of use cases

**\`.delay()\` Cons:**
- ❌ Can't set advanced options (countdown, expires, priority)
- ❌ Can't set execution options (queue, routing key)

---

## Calling Tasks: .apply_async() Method

For **advanced control**, use \`.apply_async()\`:

\`\`\`python
"""
Using .apply_async() for Advanced Task Queuing
"""

from tasks import add, send_email, process_user_data
from celery import group, chain, chord
from datetime import datetime, timedelta

# ========================================
# Basic Usage
# ========================================

# Equivalent to .delay()
result = add.apply_async (args=(4, 6))

# ========================================
# Advanced Option 1: Countdown (Delay Execution)
# ========================================

# Execute after 60 seconds
result = send_email.apply_async(
    kwargs={
        'email': 'user@example.com',
        'subject': 'Reminder',
        'body': 'Don\\'t forget!'
    },
    countdown=60  # Wait 60 seconds before execution
)

print(f"Email will be sent in 60 seconds. Task ID: {result.id}")

# ========================================
# Advanced Option 2: ETA (Execute at Specific Time)
# ========================================

# Execute at specific datetime
eta_time = datetime.utcnow() + timedelta (hours=2)

result = process_user_data.apply_async(
    kwargs={'user_id': 12345},
    eta=eta_time  # Execute in 2 hours
)

print(f"Task scheduled for {eta_time}")

# ========================================
# Advanced Option 3: Expiration
# ========================================

# Task expires if not executed within 1 hour
result = process_user_data.apply_async(
    kwargs={'user_id': 12345},
    expires=3600  # Expire after 3600 seconds (1 hour)
)

# Or use datetime
expires_at = datetime.utcnow() + timedelta (hours=1)
result = process_user_data.apply_async(
    kwargs={'user_id': 12345},
    expires=expires_at
)

# ========================================
# Advanced Option 4: Priority
# ========================================

# Set task priority (0-9, higher = more priority)
# Requires broker support (RabbitMQ supports, Redis limited)

result = send_email.apply_async(
    kwargs={'email': 'vip@example.com', 'subject': 'Urgent', 'body': 'Important!'},
    priority=9  # Highest priority
)

result = send_email.apply_async(
    kwargs={'email': 'regular@example.com', 'subject': 'Newsletter', 'body': 'Hi!'},
    priority=3  # Low priority
)

# ========================================
# Advanced Option 5: Queue Routing
# ========================================

# Send task to specific queue
result = process_user_data.apply_async(
    kwargs={'user_id': 12345},
    queue='high-priority'  # Send to 'high-priority' queue
)

# ========================================
# Advanced Option 6: Retry Options
# ========================================

result = send_email.apply_async(
    kwargs={'email': 'user@example.com', 'subject': 'Hi', 'body': 'Hello'},
    retry=True,
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 60,  # Retry after 0s, 60s, 120s
        'interval_max': 600,
    }
)

# ========================================
# Advanced Option 7: Task ID (Custom)
# ========================================

# Provide custom task ID (useful for idempotency)
result = add.apply_async(
    args=(4, 6),
    task_id='my-custom-task-id-12345'
)

print(f"Task ID: {result.id}")  # 'my-custom-task-id-12345'

# ========================================
# Advanced Option 8: Serializer
# ========================================

# Use specific serializer for this task
result = process_user_data.apply_async(
    kwargs={'user_id': 12345},
    serializer='json'  # or 'pickle', 'msgpack', 'yaml'
)

# ========================================
# Combined Options
# ========================================

# Use multiple options together
result = send_email.apply_async(
    kwargs={
        'email': 'user@example.com',
        'subject': 'Welcome',
        'body': 'Thanks for signing up!'
    },
    countdown=300,          # 5 minutes delay
    expires=3600,           # Expire after 1 hour
    priority=7,             # High priority
    queue='emails',         # Send to 'emails' queue
    retry=True,             # Enable retry
    task_id=f'email-{user_id}'  # Custom ID
)

print(f"Complex task queued: {result.id}")
\`\`\`

**\`.apply_async()\` Options Summary:**

| Option | Type | Description |
|--------|------|-------------|
| \`args\` | tuple | Positional arguments |
| \`kwargs\` | dict | Keyword arguments |
| \`countdown\` | int | Delay in seconds |
| \`eta\` | datetime | Execute at specific time |
| \`expires\` | int/datetime | Task expiration |
| \`priority\` | int (0-9) | Task priority |
| \`queue\` | str | Target queue name |
| \`retry\` | bool | Enable retry |
| \`retry_policy\` | dict | Retry configuration |
| \`task_id\` | str | Custom task ID |
| \`serializer\` | str | Serialization format |

---

## Task Arguments and Return Values

\`\`\`python
"""
Task Arguments and Return Values
"""

from celery import Celery
from typing import List, Dict, Tuple
import json

app = Celery('myapp', broker='redis://localhost:6379/0')

# ========================================
# Simple Types (Serializable)
# ========================================

@app.task
def simple_types(
    integer: int,
    floating: float,
    string: str,
    boolean: bool,
    none_val: None
) -> dict:
    """All simple types are serializable"""
    return {
        'integer': integer,
        'floating': floating,
        'string': string,
        'boolean': boolean,
        'none_val': none_val
    }

# Call task
result = simple_types.delay(42, 3.14, 'hello', True, None)
print(result.get())  # {'integer': 42, 'floating': 3.14, ...}

# ========================================
# Collections (Lists, Dicts, Tuples)
# ========================================

@app.task
def process_collections(
    items: List[int],
    mapping: Dict[str, any],
    coordinates: Tuple[float, float]
) -> dict:
    """Lists, dicts, tuples are serializable"""
    total = sum (items)
    keys = list (mapping.keys())
    distance = (coordinates[0] ** 2 + coordinates[1] ** 2) ** 0.5
    
    return {
        'total': total,
        'keys': keys,
        'distance': distance
    }

# Call task
result = process_collections.delay(
    items=[1, 2, 3, 4, 5],
    mapping={'name': 'Alice', 'age': 30},
    coordinates=(3.0, 4.0)
)
print(result.get())  # {'total': 15, 'keys': ['name', 'age'], 'distance': 5.0}

# ========================================
# Complex Objects (NOT Serializable!)
# ========================================

import datetime
import decimal

@app.task
def problematic_types(
    timestamp: datetime.datetime,  # ❌ Not JSON serializable!
    price: decimal.Decimal         # ❌ Not JSON serializable!
) -> dict:
    """These types cause errors with JSON serializer"""
    return {
        'timestamp': timestamp,
        'price': price
    }

# ❌ This will fail with JSON serializer!
# result = problematic_types.delay (datetime.datetime.now(), decimal.Decimal('19.99'))
# Error: Object of type datetime is not JSON serializable

# ✅ Solution: Convert to serializable types
@app.task
def fixed_types(
    timestamp: str,  # Pass as ISO string
    price: float     # Pass as float
) -> dict:
    """Convert to serializable types"""
    # Convert back in task
    dt = datetime.datetime.fromisoformat (timestamp)
    decimal_price = decimal.Decimal (str (price))
    
    return {
        'timestamp': timestamp,
        'price': price
    }

# ✅ This works!
result = fixed_types.delay(
    timestamp=datetime.datetime.now().isoformat(),
    price=19.99
)

# ========================================
# Large Data (Not Recommended!)
# ========================================

@app.task
def process_large_data (data: List[dict]):  # ❌ Bad practice!
    """Passing large data as argument"""
    # Process 100,000 records...
    return len (data)

# ❌ Don't do this! (passes 100MB through queue)
# large_dataset = [{...}, {...}, ...]  # 100,000 items
# result = process_large_data.delay (large_dataset)

# ✅ Better: Pass reference (ID, URL, S3 key)
@app.task
def process_large_data_better (data_url: str):
    """Pass reference to data"""
    import requests
    
    # Download data in worker
    response = requests.get (data_url)
    data = response.json()
    
    # Process data
    return len (data)

# ✅ This is better!
result = process_large_data_better.delay(
    data_url='https://s3.amazonaws.com/mybucket/dataset.json'
)

# ========================================
# Return Value Guidelines
# ========================================

@app.task
def good_return_values() -> dict:
    """Return serializable data structures"""
    return {
        'status': 'success',
        'count': 42,
        'items': ['a', 'b', 'c'],
        'metadata': {
            'processed_at': datetime.datetime.now().isoformat(),
            'duration': 1.5
        }
    }

@app.task
def return_nothing():
    """Side-effect task (no return needed)"""
    # Do something (log, send email, etc.)
    print("Task executed")
    # No return statement

@app.task
def return_large_result():  # ❌ Bad practice!
    """Don't return large data"""
    # ❌ Returning 100 MB result
    return [{'data': '...'} for _ in range(100000)]

@app.task
def return_large_result_better() -> str:
    """Return reference to large data"""
    # Generate large result
    result_data = [{'data': '...'} for _ in range(100000)]
    
    # Upload to S3
    import boto3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='mybucket',
        Key='results/task-12345.json',
        Body=json.dumps (result_data)
    )
    
    # ✅ Return URL instead
    return 'https://s3.amazonaws.com/mybucket/results/task-12345.json'
\`\`\`

**Best Practices for Arguments:**
- ✅ Pass IDs, URLs, S3 keys (not large objects)
- ✅ Use serializable types (int, float, str, bool, list, dict)
- ✅ Convert datetime to ISO string
- ✅ Keep arguments small (<1KB)
- ❌ Don't pass large data (>100KB)
- ❌ Don't pass complex objects (datetime, Decimal, custom classes)

---

## Task Naming and Identification

\`\`\`python
"""
Task Naming and Discovery
"""

from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

# ========================================
# Default Task Naming
# ========================================

@app.task
def send_email (email: str):
    """Default name: 'tasks.send_email' (module.function)"""
    pass

print(send_email.name)  # 'tasks.send_email'

# ========================================
# Custom Task Names
# ========================================

@app.task (name='my_custom_email_task')
def send_email_custom (email: str):
    """Custom task name"""
    pass

print(send_email_custom.name)  # 'my_custom_email_task'

# ========================================
# Namespaced Task Names
# ========================================

@app.task (name='emails.send')
def send_email_namespaced (email: str):
    """Namespaced task name (organizational)"""
    pass

@app.task (name='emails.send_bulk')
def send_bulk_email (emails: List[str]):
    pass

@app.task (name='users.create')
def create_user (username: str):
    pass

@app.task (name='users.delete')
def delete_user (user_id: int):
    pass

# Benefits:
# - Organized by domain (emails.*, users.*)
# - Easy to route tasks by namespace
# - Clear task purpose

# ========================================
# Task Discovery
# ========================================

# Celery automatically discovers tasks in imported modules

# Option 1: Import explicitly
from tasks import send_email, create_user

# Option 2: Auto-discover from modules
app = Celery('myapp', broker='redis://localhost:6379/0')
app.autodiscover_tasks(['myapp.tasks', 'myapp.emails'])

# Option 3: Auto-discover from installed apps (Django)
app.autodiscover_tasks()

# ========================================
# Listing All Tasks
# ========================================

# Get list of registered tasks
print(app.tasks.keys())
# ['celery.backend_cleanup', 'celery.chain', 'tasks.send_email', ...]

# Check if task exists
if 'tasks.send_email' in app.tasks:
    print("Task registered!")

# ========================================
# Task Identification (Task ID)
# ========================================

# Every task instance gets unique ID
result = send_email.delay('user@example.com')
print(f"Task ID: {result.id}")  # '550e8400-e29b-41d4-a716-446655440000'

# Custom task ID (idempotency)
result = send_email.apply_async(
    kwargs={'email': 'user@example.com'},
    task_id='email-user-12345'
)
print(f"Task ID: {result.id}")  # 'email-user-12345'

# Use custom ID to prevent duplicates
def send_welcome_email (user_id: int):
    """Send welcome email (idempotent)"""
    task_id = f'welcome-email-{user_id}'
    
    # Check if already queued
    from celery.result import AsyncResult
    result = AsyncResult (task_id, app=app)
    
    if result.state == 'PENDING':
        # Not queued yet, queue it
        send_email.apply_async(
            kwargs={'email': get_user_email (user_id)},
            task_id=task_id
        )
    else:
        print(f"Welcome email already sent/queued for user {user_id}")
\`\`\`

---

## Task Best Practices

\`\`\`python
"""
Task Design Best Practices
"""

from celery import Celery
from typing import Optional
import logging

app = Celery('myapp', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)

# ========================================
# Best Practice 1: Make Tasks Idempotent
# ========================================

@app.task
def send_welcome_email_idempotent (user_id: int):
    """
    Idempotent: Can be called multiple times safely
    
    Result is same whether called once or 10 times
    """
    # Check if already sent
    if email_already_sent (user_id, 'welcome'):
        logger.info (f"Welcome email already sent to user {user_id}")
        return
    
    # Send email
    send_email (get_user_email (user_id), 'Welcome!', 'Thanks for joining!')
    
    # Mark as sent
    mark_email_sent (user_id, 'welcome')

# ========================================
# Best Practice 2: Keep Tasks Small and Focused
# ========================================

# ❌ Bad: One giant task
@app.task
def process_order_bad (order_id: int):
    """Giant task doing everything"""
    validate_order (order_id)
    charge_payment (order_id)
    update_inventory (order_id)
    send_confirmation_email (order_id)
    notify_warehouse (order_id)
    generate_invoice (order_id)
    update_analytics (order_id)
    # 7 responsibilities in one task!

# ✅ Good: Break into smaller tasks
@app.task
def validate_order (order_id: int):
    """Single responsibility: Validate"""
    # Just validation logic
    pass

@app.task
def charge_payment (order_id: int):
    """Single responsibility: Payment"""
    # Just payment logic
    pass

@app.task
def send_confirmation_email (order_id: int):
    """Single responsibility: Email"""
    # Just email logic
    pass

# Chain them together
from celery import chain

def process_order_good (order_id: int):
    """Orchestrate smaller tasks"""
    workflow = chain(
        validate_order.s (order_id),
        charge_payment.s(),
        send_confirmation_email.s()
    )
    workflow.apply_async()

# ========================================
# Best Practice 3: Handle Failures Gracefully
# ========================================

@app.task (bind=True, max_retries=3)
def send_email_with_retry (self, email: str, subject: str, body: str):
    """Task with proper error handling"""
    try:
        # Attempt to send
        send_email_via_smtp (email, subject, body)
        logger.info (f"Email sent successfully to {email}")
        
    except SMTPException as exc:
        # Log error
        logger.error (f"SMTP error sending to {email}: {exc}")
        
        # Retry with exponential backoff
        raise self.retry(
            exc=exc,
            countdown=60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        )
        
    except Exception as exc:
        # Unexpected error - don't retry, log and alert
        logger.exception (f"Unexpected error sending email to {email}")
        alert_ops_team (f"Email task failed: {exc}")
        raise

# ========================================
# Best Practice 4: Add Logging
# ========================================

@app.task
def process_user_data_with_logging (user_id: int):
    """Task with comprehensive logging"""
    logger.info (f"Starting processing for user {user_id}")
    
    try:
        # Step 1
        logger.debug (f"Fetching user {user_id} from database")
        user = fetch_user (user_id)
        
        # Step 2
        logger.debug (f"Processing data for user {user_id}")
        result = process_data (user)
        
        # Step 3
        logger.debug (f"Saving results for user {user_id}")
        save_results (result)
        
        logger.info (f"Successfully processed user {user_id}")
        return {'status': 'success', 'user_id': user_id}
        
    except Exception as exc:
        logger.exception (f"Failed to process user {user_id}: {exc}")
        raise

# ========================================
# Best Practice 5: Set Time Limits
# ========================================

@app.task(
    time_limit=300,      # Hard limit: 5 minutes
    soft_time_limit=240  # Soft limit: 4 minutes (raises exception)
)
def long_running_task_with_limits():
    """Task with time limits"""
    from celery.exceptions import SoftTimeLimitExceeded
    
    try:
        # Long-running operation
        result = expensive_computation()
        return result
        
    except SoftTimeLimitExceeded:
        # Clean up before hard limit
        logger.warning("Task approaching time limit, cleaning up...")
        cleanup()
        raise

# ========================================
# Best Practice 6: Use Type Hints
# ========================================

@app.task
def well_typed_task(
    user_id: int,
    email: str,
    preferences: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    Task with full type hints
    
    Args:
        user_id: User database ID
        email: User email address
        preferences: Optional user preferences
    
    Returns:
        Dictionary with processing results
    """
    # Implementation
    return {'status': 'success'}

# ========================================
# Best Practice 7: Don't Block Workers
# ========================================

# ❌ Bad: Blocking operation in task
@app.task
def bad_blocking_task (url: str):
    """Blocks worker for 10 seconds"""
    import time
    time.sleep(10)  # ❌ Wastes worker resources

# ✅ Good: Use appropriate timeouts
@app.task
def good_non_blocking_task (url: str):
    """Proper timeout handling"""
    import requests
    
    try:
        response = requests.get (url, timeout=5)  # ✅ 5-second timeout
        return response.json()
    except requests.Timeout:
        logger.warning (f"Timeout fetching {url}")
        raise

# ========================================
# Best Practice 8: Validate Inputs
# ========================================

@app.task
def task_with_validation (user_id: int, email: str):
    """Validate inputs before processing"""
    # Validate user_id
    if not isinstance (user_id, int) or user_id <= 0:
        raise ValueError (f"Invalid user_id: {user_id}")
    
    # Validate email
    import re
    if not re.match (r'^[^@]+@[^@]+\\.[^@]+$', email):
        raise ValueError (f"Invalid email: {email}")
    
    # Process (inputs validated)
    send_email (email, 'Hello', f'User {user_id}')
\`\`\`

---

## Complete Example: Real-World Task Design

\`\`\`python
"""
Complete Real-World Example: Image Processing Service
"""

from celery import Celery
from typing import Dict, Optional
import logging
import boto3
from PIL import Image
import tempfile
import requests

app = Celery('image_service', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)

@app.task(
    name='images.process',
    bind=True,
    max_retries=3,
    time_limit=300,
    soft_time_limit=270
)
def process_image(
    self,
    image_url: str,
    user_id: int,
    sizes: Optional[Dict[str, tuple]] = None
) -> Dict[str, any]:
    """
    Process uploaded image: Download, resize, upload to S3
    
    Args:
        image_url: URL of uploaded image
        user_id: User who uploaded image
        sizes: Dict of size names to dimensions, e.g. {'thumb': (150, 150)}
    
    Returns:
        Dict with S3 URLs for each size
    
    Example:
        result = process_image.delay(
            image_url='https://temp.com/upload.jpg',
            user_id=12345,
            sizes={'thumb': (150, 150), 'medium': (600, 600)}
        )
    """
    from celery.exceptions import SoftTimeLimitExceeded
    
    # Default sizes
    if sizes is None:
        sizes = {
            'thumbnail': (150, 150),
            'medium': (600, 600),
            'large': (1200, 1200)
        }
    
    logger.info (f"Processing image for user {user_id}: {image_url}")
    
    try:
        # Step 1: Download image
        logger.debug("Downloading image...")
        response = requests.get (image_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Step 2: Load image
        with tempfile.NamedTemporaryFile (delete=False, suffix='.jpg') as tmp:
            for chunk in response.iter_content (chunk_size=8192):
                tmp.write (chunk)
            tmp.flush()
            
            logger.debug("Loading image...")
            img = Image.open (tmp.name)
            
            # Step 3: Generate sizes
            results = {}
            s3 = boto3.client('s3')
            
            for size_name, dimensions in sizes.items():
                logger.debug (f"Generating {size_name} ({dimensions[0]}x{dimensions[1]})")
                
                # Resize
                img_resized = img.copy()
                img_resized.thumbnail (dimensions, Image.LANCZOS)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile (delete=False, suffix='.jpg') as size_tmp:
                    img_resized.save (size_tmp.name, 'JPEG', quality=85)
                    
                    # Upload to S3
                    s3_key = f'images/{user_id}/{size_name}/{tmp.name.split("/")[-1]}'
                    s3.upload_file(
                        size_tmp.name,
                        'my-bucket',
                        s3_key,
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )
                    
                    s3_url = f'https://s3.amazonaws.com/my-bucket/{s3_key}'
                    results[size_name] = s3_url
                    
                    logger.debug (f"{size_name} uploaded to {s3_url}")
        
        logger.info (f"Successfully processed image for user {user_id}")
        
        return {
            'status': 'success',
            'user_id': user_id,
            'urls': results,
            'original_url': image_url
        }
        
    except SoftTimeLimitExceeded:
        logger.warning (f"Task approaching time limit for user {user_id}")
        # Clean up
        raise
        
    except requests.RequestException as exc:
        # Network error - retry
        logger.error (f"Network error downloading image: {exc}")
        raise self.retry (exc=exc, countdown=60 * (2 ** self.request.retries))
        
    except Exception as exc:
        # Unexpected error - log and fail
        logger.exception (f"Failed to process image for user {user_id}: {exc}")
        raise

# Queue task
result = process_image.delay(
    image_url='https://uploads.com/image.jpg',
    user_id=12345,
    sizes={'thumb': (150, 150), 'medium': (600, 600)}
)

print(f"Task queued: {result.id}")
\`\`\`

---

## Summary

**Key Concepts:**
- Define tasks with \`@app.task\` decorator
- Call tasks with \`.delay()\` (simple) or \`.apply_async()\` (advanced)
- Tasks are regular Python functions with args and return values
- Use serializable types for arguments (int, str, list, dict)
- Name tasks for organization (\`emails.send\`, \`users.create\`)

**Best Practices:**
1. Make tasks idempotent
2. Keep tasks small and focused
3. Handle failures gracefully
4. Add comprehensive logging
5. Set time limits
6. Use type hints
7. Don't block workers
8. Validate inputs

**Next Section:** We'll configure task routing, queues, and advanced Celery settings! 🚀
`,
};
