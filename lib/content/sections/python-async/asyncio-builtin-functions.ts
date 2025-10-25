export const asyncioBuiltinFunctions = {
  title: 'asyncio Built-in Functions',
  id: 'asyncio-builtin-functions',
  content: `
# asyncio Built-in Functions

## Introduction

The **asyncio** module provides a rich set of built-in functions for common async patterns. Mastering these functions allows you to write efficient, readable async code without reinventing the wheel.

### Why Built-in Functions Matter

\`\`\`python
"""
Without Built-in Functions vs With
"""

import asyncio

# ❌ Without: Manual coordination
async def fetch_all_manual (urls):
    results = []
    tasks = []
    for url in urls:
        task = asyncio.create_task (fetch (url))
        tasks.append (task)
    
    for task in tasks:
        result = await task
        results.append (result)
    
    return results

# ✅ With: Built-in gather()
async def fetch_all_builtin (urls):
    return await asyncio.gather(*[fetch (url) for url in urls])

# Much cleaner and handles errors properly!
\`\`\`

By the end of this section, you'll master:
- \`asyncio.gather()\` for concurrent execution
- \`asyncio.wait()\` for fine-grained control
- \`asyncio.sleep()\` for non-blocking delays
- \`asyncio.to_thread()\` for running blocking code
- \`asyncio.Queue\` for producer-consumer patterns
- Semaphores and locks for synchronization
- Timeout utilities for safety
- Event and condition primitives

---

## asyncio.run()

The main entry point for async programs.

\`\`\`python
"""
asyncio.run(): Running Async Programs
"""

import asyncio

async def main():
    print("Hello Async World!")
    await asyncio.sleep(1)
    return "Done"

# Run the async program
result = asyncio.run (main())
print(f"Result: {result}")

# What asyncio.run() does:
# 1. Creates a new event loop
# 2. Runs main() until complete
# 3. Closes the event loop
# 4. Returns the result

# Equivalent manual code:
loop = asyncio.new_event_loop()
try:
    result = loop.run_until_complete (main())
finally:
    loop.close()

# Always use asyncio.run() in Python 3.7+
# It\'s simpler and handles cleanup properly
\`\`\`

---

## asyncio.gather()

Run multiple coroutines concurrently and collect results.

\`\`\`python
"""
asyncio.gather(): Concurrent Execution
"""

import asyncio
import time

async def fetch_data (source, delay):
    await asyncio.sleep (delay)
    return f"Data from {source}"

async def main():
    start = time.time()
    
    # Run all concurrently
    results = await asyncio.gather(
        fetch_data("API-1", 2.0),
        fetch_data("API-2", 1.0),
        fetch_data("API-3", 1.5),
    )
    
    elapsed = time.time() - start
    print(f"Results: {results}")
    print(f"Time: {elapsed:.2f}s")  # ~2.0s (max delay)

asyncio.run (main())

# Output:
# Results: ['Data from API-1', 'Data from API-2', 'Data from API-3']
# Time: 2.00s

# Key: All run together, wait for slowest
\`\`\`

### gather() with Error Handling

\`\`\`python
"""
gather() Error Handling Strategies
"""

import asyncio

async def task_success (n):
    await asyncio.sleep(0.5)
    return f"Success {n}"

async def task_failure (n):
    await asyncio.sleep(0.5)
    raise ValueError (f"Failed {n}")

async def test_gather_errors():
    # Strategy 1: Stop on first exception (default)
    print("Strategy 1: Stop on first exception")
    try:
        results = await asyncio.gather(
            task_success(1),
            task_failure(2),  # This will fail
            task_success(3),
        )
    except ValueError as e:
        print(f"Caught: {e}")
    
    # Strategy 2: Collect all results/exceptions
    print("\nStrategy 2: Collect all")
    results = await asyncio.gather(
        task_success(1),
        task_failure(2),
        task_success(3),
        return_exceptions=True  # Don't raise, collect exceptions
    )
    
    for i, result in enumerate (results):
        if isinstance (result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

asyncio.run (test_gather_errors())

# Output:
# Strategy 1: Stop on first exception
# Caught: Failed 2
#
# Strategy 2: Collect all
# Task 0 succeeded: Success 1
# Task 1 failed: Failed 2
# Task 2 succeeded: Success 3
\`\`\`

---

## asyncio.wait()

More control than gather(), returns (done, pending) sets.

\`\`\`python
"""
asyncio.wait(): Fine-Grained Control
"""

import asyncio
import time

async def worker (n, delay):
    await asyncio.sleep (delay)
    return f"Worker {n} done"

async def test_wait():
    # Create tasks
    tasks = {
        asyncio.create_task (worker(1, 1.0)),
        asyncio.create_task (worker(2, 2.0)),
        asyncio.create_task (worker(3, 3.0)),
    }
    
    # Wait for all to complete
    done, pending = await asyncio.wait (tasks)
    
    print(f"Completed: {len (done)}")
    print(f"Pending: {len (pending)}")
    
    for task in done:
        print(f"Result: {task.result()}")

asyncio.run (test_wait())

# Output:
# Completed: 3
# Pending: 0
# Result: Worker 1 done
# Result: Worker 2 done
# Result: Worker 3 done
\`\`\`

### wait() with Different Strategies

\`\`\`python
"""
wait() Return Conditions
"""

import asyncio
import time

async def worker (n):
    await asyncio.sleep (n)
    return f"Worker {n}"

async def test_wait_strategies():
    # Strategy 1: Wait for first completion
    print("Wait for FIRST_COMPLETED:")
    tasks = {
        asyncio.create_task (worker(3)),
        asyncio.create_task (worker(1)),
        asyncio.create_task (worker(2)),
    }
    
    start = time.time()
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.1f}s")  # ~1.0s
    print(f"Done: {len (done)}, Pending: {len (pending)}")
    
    # Cancel remaining
    for task in pending:
        task.cancel()
    
    # Strategy 2: Wait with timeout
    print("\nWait with timeout:")
    tasks = {
        asyncio.create_task (worker(3)),
        asyncio.create_task (worker(1)),
        asyncio.create_task (worker(2)),
    }
    
    start = time.time()
    done, pending = await asyncio.wait(
        tasks,
        timeout=1.5  # Wait at most 1.5 seconds
    )
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.1f}s")  # ~1.5s
    print(f"Done: {len (done)}, Pending: {len (pending)}")
    
    # Cancel remaining
    for task in pending:
        task.cancel()

asyncio.run (test_wait_strategies())

# Output:
# Wait for FIRST_COMPLETED:
# Time: 1.0s
# Done: 1, Pending: 2
#
# Wait with timeout:
# Time: 1.5s
# Done: 2, Pending: 1
\`\`\`

---

## asyncio.sleep()

Non-blocking sleep (essential for async code).

\`\`\`python
"""
asyncio.sleep() vs time.sleep()
"""

import asyncio
import time

async def blocking_sleep():
    """BAD: Blocks event loop"""
    print("Starting blocking sleep")
    time.sleep(2)  # ❌ Blocks entire event loop!
    print("Finished blocking sleep")

async def non_blocking_sleep():
    """GOOD: Non-blocking"""
    print("Starting non-blocking sleep")
    await asyncio.sleep(2)  # ✅ Other tasks can run
    print("Finished non-blocking sleep")

async def other_work():
    """This can run while sleeping"""
    for i in range(4):
        print(f"  Doing work {i}")
        await asyncio.sleep(0.5)

async def test_sleeps():
    # Test 1: Blocking sleep (bad)
    print("Test 1: Blocking sleep")
    start = time.time()
    await asyncio.gather(
        blocking_sleep(),
        other_work(),
    )
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s\n")  # ~4.0s (sequential!)
    
    # Test 2: Non-blocking sleep (good)
    print("Test 2: Non-blocking sleep")
    start = time.time()
    await asyncio.gather(
        non_blocking_sleep(),
        other_work(),
    )
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")  # ~2.0s (concurrent!)

asyncio.run (test_sleeps())

# Always use await asyncio.sleep() in async code!
# Never use time.sleep() (blocks event loop)
\`\`\`

---

## asyncio.to_thread()

Run blocking I/O in a thread (Python 3.9+).

\`\`\`python
"""
asyncio.to_thread(): Running Blocking Code
"""

import asyncio
import time
import requests  # Blocking HTTP library

def blocking_http_request (url):
    """Blocking function (not async)"""
    response = requests.get (url)
    return response.json()

async def fetch_data_correctly (url):
    """Run blocking code in thread"""
    # This runs in a thread pool, not blocking event loop
    data = await asyncio.to_thread (blocking_http_request, url)
    return data

async def main():
    # Fetch multiple URLs concurrently
    # Even though requests is blocking, to_thread makes it work
    urls = [
        "https://api.example.com/users/1",
        "https://api.example.com/users/2",
        "https://api.example.com/users/3",
    ]
    
    results = await asyncio.gather(*[
        fetch_data_correctly (url) for url in urls
    ])
    
    return results

# asyncio.run (main())

# Use cases for to_thread():
# - Blocking I/O (file operations, subprocess)
# - Libraries without async support (requests, PIL)
# - CPU-bound work (if can't use multiprocessing)

# Note: Still limited by thread pool size (~32 threads)
# For true parallelism, use multiprocessing
\`\`\`

---

## asyncio.Queue

Producer-consumer pattern for async code.

\`\`\`python
"""
asyncio.Queue: Producer-Consumer Pattern
"""

import asyncio
import random

async def producer (queue, producer_id):
    """Produce items and put in queue"""
    for i in range(5):
        item = f"Item-{producer_id}-{i}"
        await asyncio.sleep (random.uniform(0.1, 0.5))
        await queue.put (item)
        print(f"Producer {producer_id}: Produced {item}")
    
    print(f"Producer {producer_id}: Done")

async def consumer (queue, consumer_id):
    """Consume items from queue"""
    while True:
        try:
            # Wait up to 2 seconds for item
            item = await asyncio.wait_for (queue.get(), timeout=2.0)
            print(f"  Consumer {consumer_id}: Processing {item}")
            await asyncio.sleep (random.uniform(0.1, 0.3))
            queue.task_done()
        except asyncio.TimeoutError:
            print(f"  Consumer {consumer_id}: Timeout, exiting")
            break

async def main():
    # Create queue
    queue = asyncio.Queue (maxsize=10)
    
    # Start producers and consumers
    producers = [
        asyncio.create_task (producer (queue, i))
        for i in range(2)
    ]
    
    consumers = [
        asyncio.create_task (consumer (queue, i))
        for i in range(3)
    ]
    
    # Wait for producers to finish
    await asyncio.gather(*producers)
    
    # Wait for queue to be empty
    await queue.join()
    
    # Cancel consumers (they'll timeout and exit)
    for c in consumers:
        c.cancel()

asyncio.run (main())

# Output: Producers create items, consumers process them concurrently
# Multiple consumers can process items in parallel
# Queue handles synchronization automatically
\`\`\`

---

## Semaphores

Limit concurrent access to resources.

\`\`\`python
"""
asyncio.Semaphore: Rate Limiting
"""

import asyncio
import time

async def access_resource (semaphore, n):
    """Access resource with semaphore"""
    async with semaphore:
        print(f"Task {n}: Acquired semaphore")
        await asyncio.sleep(1)  # Simulate work
        print(f"Task {n}: Released semaphore")

async def test_semaphore():
    # Allow only 3 concurrent accesses
    semaphore = asyncio.Semaphore(3)
    
    start = time.time()
    
    # Create 10 tasks, but only 3 can run at once
    await asyncio.gather(*[
        access_resource (semaphore, i)
        for i in range(10)
    ])
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")  # ~4s (10 tasks / 3 concurrent / 1s each)

asyncio.run (test_semaphore())

# Output shows only 3 tasks running at a time
# Useful for:
# - Rate limiting API requests
# - Limiting database connections
# - Controlling resource usage
\`\`\`

### BoundedSemaphore

\`\`\`python
"""
BoundedSemaphore: Prevent Over-Release
"""

import asyncio

async def test_bounded():
    # BoundedSemaphore prevents releasing more than acquired
    sem = asyncio.BoundedSemaphore(2)
    
    async with sem:
        print("Acquired once")
    
    try:
        sem.release()  # ❌ Error: Can't release more than acquired
    except ValueError as e:
        print(f"Error: {e}")
    
    # Regular Semaphore allows over-release (usually wrong)
    regular_sem = asyncio.Semaphore(2)
    async with regular_sem:
        pass
    regular_sem.release()  # ✓ Allowed (but probably wrong)
    
    print("BoundedSemaphore prevents bugs from over-release")

asyncio.run (test_bounded())

# Use BoundedSemaphore by default (safer)
\`\`\`

---

## Locks

Mutual exclusion for critical sections.

\`\`\`python
"""
asyncio.Lock: Mutual Exclusion
"""

import asyncio

class Counter:
    """Thread-safe counter with lock"""
    
    def __init__(self):
        self.value = 0
        self.lock = asyncio.Lock()
    
    async def increment (self):
        """Increment counter (critical section)"""
        async with self.lock:
            # Only one coroutine can be here at a time
            current = self.value
            await asyncio.sleep(0.01)  # Simulate work
            self.value = current + 1

async def test_lock():
    counter = Counter()
    
    # 100 concurrent increments
    await asyncio.gather(*[
        counter.increment()
        for _ in range(100)
    ])
    
    print(f"Final count: {counter.value}")  # 100 (correct!)

asyncio.run (test_lock())

# Without lock, would get race conditions
# Lock ensures only one coroutine in critical section
\`\`\`

---

## Events

Signal between coroutines.

\`\`\`python
"""
asyncio.Event: Signaling
"""

import asyncio

async def waiter (event, n):
    """Wait for event to be set"""
    print(f"Waiter {n}: Waiting for event")
    await event.wait()
    print(f"Waiter {n}: Event received!")

async def setter (event):
    """Set event after delay"""
    print("Setter: Waiting 2 seconds...")
    await asyncio.sleep(2)
    print("Setter: Setting event")
    event.set()

async def test_event():
    event = asyncio.Event()
    
    # Multiple waiters
    waiters = [
        asyncio.create_task (waiter (event, i))
        for i in range(3)
    ]
    
    # One setter
    setter_task = asyncio.create_task (setter (event))
    
    # Wait for all
    await asyncio.gather(*waiters, setter_task)

asyncio.run (test_event())

# Output:
# All waiters wait
# Setter sets event after 2s
# All waiters wake up at once

# Use cases:
# - Startup synchronization
# - Shutdown signals
# - Barrier pattern
\`\`\`

---

## Summary

### Key Built-in Functions

| Function | Purpose | Use When |
|----------|---------|----------|
| \`asyncio.run()\` | Run async program | Main entry point |
| \`asyncio.gather()\` | Run multiple concurrently | Want all results |
| \`asyncio.wait()\` | Fine-grained control | Need done/pending sets |
| \`asyncio.sleep()\` | Non-blocking delay | Need to pause |
| \`asyncio.to_thread()\` | Run blocking code | Using sync libraries |
| \`asyncio.Queue\` | Producer-consumer | Processing pipeline |
| \`Semaphore\` | Limit concurrency | Rate limiting |
| \`Lock\` | Mutual exclusion | Protect critical sections |
| \`Event\` | Signaling | Coordination |

### Best Practices

1. **Use gather() for simple cases**
   - Want all results
   - Order matters
   - Simple error handling

2. **Use wait() for complex cases**
   - Need to cancel pending
   - Timeout with partial results
   - Race conditions

3. **Always use asyncio.sleep()**
   - Never time.sleep() in async code

4. **Use to_thread() for blocking code**
   - When async version not available
   - Limited by thread pool size

5. **Use Semaphore for rate limiting**
   - API request limits
   - Resource pools
   - Concurrent access control

### Next Steps

Now that you know asyncio built-ins, we'll explore:
- Async HTTP with aiohttp
- Async database operations
- Real-world async patterns
- Production deployment

**Remember**: asyncio provides primitives for most async patterns. Learn these well before building custom solutions.
`,
};
