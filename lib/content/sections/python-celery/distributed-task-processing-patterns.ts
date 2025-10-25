export const distributedTaskProcessingPatterns = {
  title: 'Distributed Task Processing Patterns',
  id: 'distributed-task-processing-patterns',
  content: `
# Distributed Task Processing Patterns

## Introduction

Distributed task processing is about **breaking down large problems** into smaller tasks that run **in parallel** across multiple workers. This section covers essential patterns that enable you to process millions of records efficiently.

**Key Patterns:**
- **Map-Reduce**: Parallel processing + aggregation
- **Fan-Out/Fan-In**: Distribute work, combine results
- **Task Chunking**: Break large tasks into manageable pieces
- **Distributed Locking**: Ensure only one worker processes a resource
- **Idempotency**: Safe to execute multiple times

---

## Map-Reduce Pattern

**Concept**: Process data in parallel (map), then aggregate results (reduce).

\`\`\`python
"""
Map-Reduce: Process 1 million records in parallel
"""

from celery import Celery, group
from typing import List

app = Celery('mapreduce', broker='redis://localhost:6379/0')


# ========================================
# MAP PHASE: Process chunks in parallel
# ========================================

@app.task
def map_process_chunk(records: List[int]) -> List[int]:
    """
    Map: Process chunk of records
    
    Input: [1, 2, 3, ..., 10000]
    Output: [2, 4, 6, ..., 20000] (doubled)
    """
    return [record * 2 for record in records]


# ========================================
# REDUCE PHASE: Aggregate results
# ========================================

@app.task
def reduce_sum_results(results: List[List[int]]) -> int:
    """
    Reduce: Aggregate all results
    
    Input: [[2,4,6], [8,10,12], ...] (list of lists)
    Output: 100000050000 (sum of all)
    """
    total = 0
    for chunk_result in results:
        total += sum(chunk_result)
    return total


# ========================================
# ORCHESTRATION: Map-Reduce workflow
# ========================================

def mapreduce_workflow():
    """
    Process 1 million records using map-reduce
    
    Timeline:
    - Split: 1M records → 100 chunks of 10K
    - Map: 100 workers process in parallel (~1 minute)
    - Reduce: Sum all results (~1 second)
    - Total: ~61 seconds
    
    vs Sequential: ~16 hours
    Speedup: 940× faster! 🚀
    """
    # Prepare data
    total_records = 1_000_000
    chunk_size = 10_000
    
    # Split into chunks
    chunks = []
    for i in range(0, total_records, chunk_size):
        chunk = list(range(i, i + chunk_size))
        chunks.append(chunk)
    
    print(f"Split {total_records:,} records into {len(chunks)} chunks")
    
    # MAP PHASE: Process chunks in parallel
    map_job = group(map_process_chunk.s(chunk) for chunk in chunks)
    map_result = map_job.apply_async()
    
    # Wait for all map tasks to complete
    map_results = map_result.get()  # List of results from each chunk
    
    print(f"Map phase complete: {len(map_results)} chunks processed")
    
    # REDUCE PHASE: Aggregate results
    reduce_result = reduce_sum_results.delay(map_results)
    final_sum = reduce_result.get()
    
    print(f"Reduce phase complete: Sum = {final_sum:,}")
    
    return final_sum


# Usage
if __name__ == '__main__':
    result = mapreduce_workflow()
    print(f"Final result: {result}")
\`\`\`

**Benefits:**
- ✅ **Parallel processing**: 100 workers process simultaneously
- ✅ **Scalable**: Add more workers = faster processing
- ✅ **Fault-tolerant**: Failed chunks retry independently
- ✅ **Efficient**: 940× faster than sequential

---

## Fan-Out/Fan-In Pattern

**Concept**: Distribute work to multiple workers (fan-out), then combine results (fan-in).

\`\`\`python
"""
Fan-Out/Fan-In: Process orders in parallel
"""

from celery import Celery, chord
import time

app = Celery('fanout', broker='redis://localhost:6379/0')


# ========================================
# FAN-OUT: Parallel processing
# ========================================

@app.task
def process_order(order_id: int) -> dict:
    """Process single order (takes 5 seconds)"""
    time.sleep(5)  # Simulate processing
    return {
        'order_id': order_id,
        'status': 'processed',
        'total': order_id * 100
    }


# ========================================
# FAN-IN: Combine results
# ========================================

@app.task
def combine_orders(order_results: List[dict]) -> dict:
    """Combine all order results"""
    total_orders = len(order_results)
    total_revenue = sum(order['total'] for order in order_results)
    
    return {
        'total_orders': total_orders,
        'total_revenue': total_revenue,
        'average_order': total_revenue / total_orders if total_orders else 0
    }


# ========================================
# ORCHESTRATION: Fan-Out/Fan-In workflow
# ========================================

def process_daily_orders():
    """
    Process 100 orders using fan-out/fan-in
    
    Without parallelism:
    - 100 orders × 5 seconds = 500 seconds (8.3 minutes)
    
    With parallelism (10 workers):
    - 10 batches × 5 seconds = 50 seconds
    - 10× faster! ⚡
    """
    order_ids = range(1, 101)  # 100 orders
    
    # Chord: Fan-out (parallel) + Fan-in (callback)
    callback = combine_orders.s()
    header = [process_order.s(order_id) for order_id in order_ids]
    
    result = chord(header)(callback)
    summary = result.get()
    
    print(f"Processed {summary['total_orders']} orders")
    print(f"Total revenue: \${summary['total_revenue']:,}")
    print(f"Average order: \${summary['average_order']:.2f}")

return summary


# Usage
if __name__ == '__main__':
    process_daily_orders()
\`\`\`

---

## Task Chunking Pattern

**Problem**: Task processes 10M records but times out after 5 minutes.

**Solution**: Split into 1000 chunks of 10K records each.

\`\`\`python
"""
Task Chunking: Process 10M records without timeout
"""

from celery import Celery, group
import redis

app = Celery('chunking', broker='redis://localhost:6379/0')
redis_client = redis.Redis()


@app.task(soft_time_limit=300, time_limit=330)
def process_chunk(start_id: int, end_id: int) -> dict:
    """
    Process chunk of records
    
    Each chunk:
    - Processes 10K records
    - Takes ~4 minutes (< 5 minute limit)
    - Can fail independently
    """
    checkpoint_key = f"checkpoint:processed:{start_id}:{end_id}"
    
    # Check if already processed (idempotency)
    if redis_client.exists(checkpoint_key):
        return {'status': 'already_processed', 'range': (start_id, end_id)}
    
    # Process records
    processed = 0
    for record_id in range(start_id, end_id):
        try:
            process_record(record_id)
            processed += 1
        except Exception as e:
            logger.error(f"Failed record {record_id}: {e}")
    
    # Mark chunk as complete
    redis_client.setex(checkpoint_key, 86400, 'done')
    
    return {
        'status': 'complete',
        'range': (start_id, end_id),
        'processed': processed
    }


def process_all_records():
    """
    Process 10 million records
    
    Strategy:
    - Split into 1000 chunks of 10K each
    - Each chunk < 5 minutes (no timeout)
    - Process chunks in parallel
    - Failed chunks retry independently
    """
    total_records = 10_000_000
    chunk_size = 10_000
    
    # Create chunks
    chunks = []
    for i in range(0, total_records, chunk_size):
        chunks.append((i, i + chunk_size))
    
    print(f"Processing {total_records:,} records in {len(chunks)} chunks")
    
    # Process in parallel
    job = group(process_chunk.s(start, end) for start, end in chunks)
    result = job.apply_async()
    
    # Monitor progress
    completed = 0
    while not result.ready():
        time.sleep(10)
        completed = sum(1 for r in result.results if r.ready())
        progress = (completed / len(chunks)) * 100
        print(f"Progress: {progress:.1f}% ({completed}/{len(chunks)} chunks)")
    
    # Final results
    results = result.get()
    total_processed = sum(r['processed'] for r in results if r['status'] == 'complete')
    
    print(f"Complete! Processed {total_processed:,} records")
    
    return results


def process_record(record_id: int):
    """Process single record"""
    pass  # Your processing logic
\`\`\`

---

## Distributed Locking Pattern

**Problem**: Multiple workers try to process the same resource (race condition).

**Solution**: Use distributed lock to ensure only one worker processes it.

\`\`\`python
"""
Distributed Locking: Prevent race conditions
"""

from celery import Celery
from celery.exceptions import Ignore
import redis
import logging

app = Celery('locking', broker='redis://localhost:6379/0')
redis_client = redis.Redis()
logger = logging.getLogger(__name__)


@app.task(bind=True)
def process_user_balance(self, user_id: int):
    """
    Process user balance (must be exclusive)
    
    Scenario without locking:
    - Worker 1 reads balance: $100
    - Worker 2 reads balance: $100
    - Worker 1 adds $50: $150
    - Worker 2 adds $30: $130 ❌ (should be $180!)
    
    With locking:
    - Worker 1 acquires lock
    - Worker 2 waits
    - Worker 1 processes: $100 + $50 = $150
    - Worker 1 releases lock
    - Worker 2 acquires lock
    - Worker 2 processes: $150 + $30 = $180 ✅
    """
    lock_key = f"lock:user:{user_id}"
    lock = redis_client.lock(lock_key, timeout=300, blocking_timeout=10)
    
    # Try to acquire lock
    if not lock.acquire(blocking=False):
        logger.warning(f"User {user_id} already being processed by another worker")
        raise Ignore()  # Another worker has it
    
    try:
        # Critical section (only one worker executes this)
        balance = get_user_balance(user_id)
        logger.info(f"Processing user {user_id}, balance: ${balance}")
        
        # Update balance
        new_balance = balance + 50
        set_user_balance(user_id, new_balance)
        
        logger.info(f"User {user_id} processed, new balance: ${new_balance}")
        
        return {'user_id': user_id, 'balance': new_balance}
    
    finally:
        # Always release lock
        lock.release()


def get_user_balance(user_id: int) -> float:
    """Get user balance from database"""
    return 100.0  # Example


def set_user_balance(user_id: int, balance: float):
    """Update user balance in database"""
    pass  # Example


# Usage
process_user_balance.delay(123)
\`\`\`

---

## Idempotency Pattern

**Problem**: Task executed multiple times (due to retries) causes duplicates.

**Solution**: Make task idempotent (same result whether called 1× or 100×).

\`\`\`python
"""
Idempotency: Safe to execute multiple times
"""

from celery import Celery
import redis

app = Celery('idempotent', broker='redis://localhost:6379/0')
redis_client = redis.Redis()


@app.task
def send_notification(user_id: int, notification_id: int, message: str):
    """
    Idempotent notification sending
    
    Without idempotency:
    - Task retries 3 times
    - User gets 3 duplicate notifications ❌
    
    With idempotency:
    - Task retries 3 times
    - User gets 1 notification ✅
    """
    cache_key = f"notification:sent:{notification_id}"
    
    # Check if already sent
    if redis_client.exists(cache_key):
        return {'status': 'already_sent', 'notification_id': notification_id}
    
    # Send notification
    send_push(user_id, message)
    
    # Mark as sent (prevents duplicates on retry)
    redis_client.setex(cache_key, 86400, 'sent')  # 24h expiration
    
    return {'status': 'sent', 'notification_id': notification_id}


@app.task
def process_payment(order_id: int, amount: float):
    """
    Idempotent payment processing
    
    CRITICAL: Must not charge customer multiple times!
    """
    # Check if already processed
    payment = Payment.query.filter_by(order_id=order_id).first()
    
    if payment and payment.status == 'complete':
        return {'status': 'already_processed', 'payment_id': payment.id}
    
    # Process payment with Stripe idempotency key
    try:
        charge = stripe.Charge.create(
            amount=int(amount * 100),
            currency='usd',
            idempotency_key=f'order_{order_id}'  # Stripe prevents duplicates
        )
        
        # Save to database (with unique constraint on order_id)
        payment = Payment(
            order_id=order_id,
            stripe_charge_id=charge.id,
            amount=amount,
            status='complete'
        )
        db.session.add(payment)
        db.session.commit()
        
        return {'status': 'processed', 'payment_id': payment.id}
    
    except IntegrityError:
        # Duplicate (another worker processed it)
        db.session.rollback()
        return {'status': 'already_processed'}


def send_push(user_id: int, message: str):
    """Send push notification"""
    pass  # Implementation
\`\`\`

---

## Summary

**Map-Reduce:**
- Process data in parallel (map)
- Aggregate results (reduce)
- Use for massive datasets (millions of records)

**Fan-Out/Fan-In:**
- Distribute work to N workers (fan-out)
- Combine results (fan-in)
- Use Celery chord for implementation

**Task Chunking:**
- Break large tasks into small chunks
- Each chunk < time limit
- Process chunks in parallel
- Failed chunks retry independently

**Distributed Locking:**
- Use Redis locks for exclusive access
- Prevents race conditions
- Critical for balance updates, inventory

**Idempotency:**
- Make tasks safe to execute multiple times
- Check if already processed
- Use idempotency keys (Stripe)
- Database unique constraints

**Next Section:** Canvas workflows (chains, groups, chords)! 🎨
`,
};
