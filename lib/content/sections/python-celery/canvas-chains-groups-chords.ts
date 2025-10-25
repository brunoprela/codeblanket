export const canvasChainsGroupsChords = {
  title: 'Canvas: Chains, Groups, Chords',
  id: 'canvas-chains-groups-chords',
  content: `
# Canvas: Chains, Groups, Chords

## Introduction

**Canvas** is Celery's powerful workflow API for composing complex task workflows. Instead of manually orchestrating tasks, Canvas provides three elegant primitives:

1. **chain**: Sequential execution (A → B → C)
2. **group**: Parallel execution ([A, B, C])
3. **chord**: Parallel + callback (group → callback)

**Why Canvas?**
- **Declarative**: Express complex workflows simply
- **Composable**: Combine primitives for advanced patterns
- **Reliable**: Built-in error handling and retries
- **Scalable**: Automatic parallelization

---

## Chains: Sequential Execution

**chain**: Execute tasks one after another, passing results forward.

\`\`\`python
"""
Chains: Sequential task execution
"""

from celery import Celery, chain

app = Celery('chains', broker='redis://localhost:6379/0')


@app.task
def download_image(url: str) -> str:
    """Step 1: Download image"""
    print(f"Downloading {url}")
    return f"/tmp/image_{url.split('/')[-1]}"


@app.task
def resize_image(image_path: str) -> str:
    """Step 2: Resize image (depends on download)"""
    print(f"Resizing {image_path}")
    return image_path.replace('.jpg', '_resized.jpg')


@app.task
def upload_to_cdn(image_path: str) -> str:
    """Step 3: Upload to CDN (depends on resize)"""
    print(f"Uploading {image_path}")
    return f"https://cdn.example.com/{image_path.split('/')[-1]}"


# ========================================
# METHOD 1: chain() function
# ========================================

workflow = chain(
    download_image.s('https://example.com/photo.jpg'),
    resize_image.s(),  # Receives result from download_image
    upload_to_cdn.s()  # Receives result from resize_image
)

result = workflow.apply_async()
cdn_url = result.get()  # Wait for entire chain to complete

print(f"Final CDN URL: {cdn_url}")


# ========================================
# METHOD 2: Pipe operator (more elegant)
# ========================================

workflow = (
    download_image.s('https://example.com/photo.jpg') |
    resize_image.s() |
    upload_to_cdn.s()
)

result = workflow.apply_async()


# ========================================
# Real-World Example: E-Commerce Order
# ========================================

@app.task
def validate_payment(order_id: int) -> int:
    """Step 1: Validate payment"""
    print(f"Validating payment for order {order_id}")
    return order_id


@app.task
def reserve_inventory(order_id: int) -> int:
    """Step 2: Reserve inventory"""
    print(f"Reserving inventory for order {order_id}")
    return order_id


@app.task
def ship_order(order_id: int) -> dict:
    """Step 3: Ship order"""
    print(f"Shipping order {order_id}")
    return {'order_id': order_id, 'tracking': 'TRACK123'}


@app.task
def send_confirmation(result: dict) -> str:
    """Step 4: Send confirmation email"""
    print(f"Sending confirmation for order {result['order_id']}")
    return "Email sent"


# Process order workflow
order_workflow = (
    validate_payment.s(12345) |
    reserve_inventory.s() |
    ship_order.s() |
    send_confirmation.s()
)

result = order_workflow.apply_async()
print(result.get())  # "Email sent"
\`\`\`

**Timeline:**
\`\`\`
T+0s:  validate_payment starts
T+2s:  validate_payment complete → reserve_inventory starts
T+4s:  reserve_inventory complete → ship_order starts
T+9s:  ship_order complete → send_confirmation starts
T+10s: send_confirmation complete → Done!
\`\`\`

---

## Groups: Parallel Execution

**group**: Execute tasks in parallel across multiple workers.

\`\`\`python
"""
Groups: Parallel task execution
"""

from celery import Celery, group

app = Celery('groups', broker='redis://localhost:6379/0')


@app.task
def process_user(user_id: int) -> dict:
    """Process single user (takes 3 seconds)"""
    time.sleep(3)
    return {'user_id': user_id, 'status': 'processed'}


# ========================================
# Sequential Processing (SLOW)
# ========================================

def process_users_sequential(user_ids):
    """
    Process 100 users sequentially
    
    Time: 100 users × 3 seconds = 300 seconds (5 minutes)
    """
    results = []
    for user_id in user_ids:
        result = process_user.delay(user_id).get()
        results.append(result)
    return results


# ========================================
# Parallel Processing with group (FAST)
# ========================================

def process_users_parallel(user_ids):
    """
    Process 100 users in parallel
    
    Time: 3 seconds (if you have 100 workers)
    or 30 seconds (if you have 10 workers)
    
    100× faster! ⚡
    """
    job = group(process_user.s(user_id) for user_id in user_ids)
    result = job.apply_async()
    return result.get()  # List of all results


# Usage
user_ids = range(1, 101)  # 100 users

# Sequential: 300 seconds 🐌
# results = process_users_sequential(user_ids)

# Parallel: 30 seconds ⚡ (with 10 workers)
results = process_users_parallel(user_ids)


# ========================================
# Real-World Example: Thumbnail Generation
# ========================================

@app.task
def generate_thumbnail(image_id: int, size: tuple) -> str:
    """Generate thumbnail of specific size"""
    print(f"Generating {size} thumbnail for image {image_id}")
    return f"thumbnail_{image_id}_{size[0]}x{size[1]}.jpg"


def generate_all_thumbnails(image_id: int):
    """Generate multiple thumbnail sizes in parallel"""
    sizes = [(150, 150), (300, 300), (600, 600), (1200, 1200)]
    
    job = group(
        generate_thumbnail.s(image_id, size) for size in sizes
    )
    
    result = job.apply_async()
    thumbnails = result.get()  # All 4 thumbnails
    
    print(f"Generated thumbnails: {thumbnails}")
    return thumbnails


# Usage
generate_all_thumbnails(12345)
# Generates 4 sizes in parallel (4× faster than sequential)
\`\`\`

---

## Chords: Parallel + Callback

**chord**: Execute tasks in parallel (header), then run callback with all results.

\`\`\`python
"""
Chords: Parallel processing with aggregation callback
"""

from celery import Celery, chord

app = Celery('chords', broker='redis://localhost:6379/0')


@app.task
def fetch_data_from_api(api_id: int) -> dict:
    """Fetch data from external API"""
    print(f"Fetching from API {api_id}")
    return {'api_id': api_id, 'value': api_id * 100}


@app.task
def aggregate_results(results: List[dict]) -> dict:
    """
    Callback: Aggregate results from all APIs
    
    Called after ALL header tasks complete
    """
    total_apis = len(results)
    total_value = sum(r['value'] for r in results)
    
    print(f"Aggregated {total_apis} APIs: total value = {total_value}")
    
    return {
        'total_apis': total_apis,
        'total_value': total_value,
        'average': total_value / total_apis if total_apis else 0
    }


# ========================================
# Chord: Header (parallel) + Callback
# ========================================

callback = aggregate_results.s()
header = [fetch_data_from_api.s(i) for i in range(1, 11)]  # 10 APIs

result = chord(header)(callback)
summary = result.get()

print(summary)
# {'total_apis': 10, 'total_value': 5500, 'average': 550.0}


# ========================================
# Real-World Example: Daily Report Generation
# ========================================

@app.task
def fetch_user_stats(user_id: int) -> dict:
    """Fetch statistics for single user"""
    return {
        'user_id': user_id,
        'orders': user_id * 10,
        'revenue': user_id * 100
    }


@app.task
def generate_report(user_stats: List[dict]) -> dict:
    """Generate summary report from all user stats"""
    total_users = len(user_stats)
    total_orders = sum(s['orders'] for s in user_stats)
    total_revenue = sum(s['revenue'] for s in user_stats)
    
    report = {
        'date': datetime.now().isoformat(),
        'total_users': total_users,
        'total_orders': total_orders,
        'total_revenue': total_revenue,
        'avg_revenue_per_user': total_revenue / total_users if total_users else 0
    }
    
    # Save report to database
    save_report(report)
    
    # Send to stakeholders
    send_email('ceo@company.com', 'Daily Report', report)
    
    return report


def generate_daily_report(user_ids):
    """
    Generate daily report for all users
    
    1. Fetch stats for all users in parallel (header)
    2. Aggregate and generate report (callback)
    """
    callback = generate_report.s()
    header = [fetch_user_stats.s(user_id) for user_id in user_ids]
    
    result = chord(header)(callback)
    report = result.get()
    
    print(f"Report generated: {report}")
    return report


# Usage
user_ids = range(1, 10001)  # 10,000 users
generate_daily_report(user_ids)
# Processes 10K users in parallel, then generates report
\`\`\`

---

## Complex Workflows: Combining Primitives

\`\`\`python
"""
Advanced: Combining chains, groups, and chords
"""

from celery import Celery, chain, group, chord

app = Celery('advanced', broker='redis://localhost:6379/0')


# ========================================
# Pattern 1: Chain of Groups
# ========================================

@app.task
def download_file(url: str) -> str:
    return f"file_{url}"

@app.task
def process_file(file_path: str) -> dict:
    return {'file': file_path, 'processed': True}

@app.task
def aggregate_files(results: List[dict]) -> str:
    return f"Processed {len(results)} files"

# Download files in parallel, then process in parallel
workflow = chain(
    group(download_file.s(url) for url in ['url1', 'url2', 'url3']),
    group(process_file.s() for _ in range(3))
)


# ========================================
# Pattern 2: Chord with Chain Callback
# ========================================

# Parallel processing → chain of post-processing steps
workflow = chord(
    [fetch_data_from_api.s(i) for i in range(10)],
    chain(
        aggregate_results.s(),
        save_to_database.s(),
        send_notification.s()
    )
)


# ========================================
# Pattern 3: Nested Chords
# ========================================

# Process in parallel, aggregate, process aggregates in parallel
inner_chord = chord(
    [process_chunk.s(i) for i in range(100)],
    aggregate_chunk_results.s()
)

outer_chord = chord(
    [inner_chord for _ in range(10)],
    final_aggregation.s()
)


# ========================================
# Real-World: E-Commerce Analytics Pipeline
# ========================================

@app.task
def fetch_orders(date: str) -> List[dict]:
    """Fetch orders for date"""
    return []  # List of orders

@app.task
def process_order_batch(orders: List[dict]) -> dict:
    """Process batch of orders"""
    return {'revenue': sum(o['total'] for o in orders)}

@app.task
def fetch_users(date: str) -> List[dict]:
    """Fetch user activity for date"""
    return []  # List of users

@app.task
def process_user_batch(users: List[dict]) -> dict:
    """Process batch of users"""
    return {'active_users': len(users)}

@app.task
def generate_analytics(results: List[dict]) -> dict:
    """Final analytics report"""
    return {
        'revenue': sum(r.get('revenue', 0) for r in results),
        'active_users': sum(r.get('active_users', 0) for r in results)
    }

# Complex pipeline: Fetch + process in parallel, then aggregate
analytics_pipeline = chord(
    [
        chain(fetch_orders.s('2025-01-01'), process_order_batch.s()),
        chain(fetch_users.s('2025-01-01'), process_user_batch.s())
    ],
    generate_analytics.s()
)

result = analytics_pipeline.apply_async()
analytics = result.get()
\`\`\`

---

## Error Handling in Canvas

\`\`\`python
"""
Error handling in chains and chords
"""

from celery import Celery, chain, chord
from celery.exceptions import Ignore

app = Celery('errors', broker='redis://localhost:6379/0')


# ========================================
# Chain Error Handling
# ========================================

@app.task(bind=True, max_retries=3)
def risky_task(self, x):
    """Task that might fail"""
    try:
        if x < 0:
            raise ValueError("Negative number!")
        return x * 2
    except ValueError as exc:
        raise self.retry(exc=exc, countdown=10)

@app.task
def safe_task(x):
    """This won't run if risky_task fails"""
    return x + 10

# Chain: If risky_task fails, safe_task never runs
workflow = chain(risky_task.s(-5), safe_task.s())
result = workflow.apply_async()
# Result: FAILURE (risky_task failed)


# ========================================
# Chord Error Handling
# ========================================

@app.task(bind=True, max_retries=3)
def process_item(self, item_id: int) -> dict:
    """Process item (might fail)"""
    try:
        if item_id % 10 == 0:  # Simulate failure
            raise Exception("Item processing failed")
        return {'item_id': item_id, 'status': 'success'}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30)

@app.task
def aggregate_items(results: List[dict]) -> dict:
    """
    Callback: Handle mixed success/failure results
    
    Filter out failed tasks, aggregate successful ones
    """
    successful = [r for r in results if r and r.get('status') == 'success']
    failed = len(results) - len(successful)
    
    return {
        'total': len(results),
        'successful': len(successful),
        'failed': failed
    }

# Chord with error handling
callback = aggregate_items.s()
header = [process_item.s(i) for i in range(100)]

result = chord(header)(callback)
summary = result.get()
# {'total': 100, 'successful': 90, 'failed': 10}
\`\`\`

---

## Performance Optimization

\`\`\`python
"""
Optimizing Canvas performance
"""

# ========================================
# Tip 1: Use signatures (.s()) not .delay()
# ========================================

# WRONG: Executes immediately (not part of workflow)
workflow = chain(
    task1.delay(5),  # ❌ Executes now!
    task2.s()
)

# CORRECT: Creates signature (deferred execution)
workflow = chain(
    task1.s(5),  # ✅ Deferred
    task2.s()
)


# ========================================
# Tip 2: Batch tasks in chunks
# ========================================

# SLOW: 10,000 individual tasks
job = group(process.s(i) for i in range(10_000))

# FAST: 100 batch tasks
chunks = [range(i, i+100) for i in range(0, 10_000, 100)]
job = group(process_batch.s(chunk) for chunk in chunks)


# ========================================
# Tip 3: Use immutable signatures for reuse
# ========================================

from celery import signature

# Mutable (default) - args can change
sig = task.s(5)

# Immutable - args frozen
sig = task.si(5)  # or task.s(5).set(immutable=True)

# Useful in groups/chords where args shouldn't propagate
\`\`\`

---

## Summary

**Canvas Primitives:**

| Primitive | Execution | Use Case | Example |
|-----------|-----------|----------|---------|
| **chain** | Sequential (A→B→C) | Dependent tasks | Download → Process → Upload |
| **group** | Parallel ([A,B,C]) | Independent tasks | Process 100 users simultaneously |
| **chord** | Parallel + callback | Aggregate results | Fetch APIs → Generate report |

**Benefits:**
- **Declarative**: Express workflows elegantly
- **Parallel**: Automatic parallelization
- **Composable**: Combine primitives
- **Reliable**: Built-in retries

**Best Practices:**
- Use .s() for signatures (not .delay())
- Batch tasks into chunks
- Handle errors in callbacks
- Use immutable signatures when needed

**Next Section:** Production deployment! 🚀
`,
};
