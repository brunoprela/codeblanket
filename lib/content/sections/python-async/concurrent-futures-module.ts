export const concurrentFuturesModule = {
  title: 'concurrent.futures Module',
  id: 'concurrent-futures-module',
  content: `
# concurrent.futures Module

## Introduction

The **concurrent.futures** module provides a high-level interface for asynchronously executing callables using threads or processes. It's the easiest way to add concurrency to Python code without managing threads/processes directly.

### Why concurrent.futures?

\`\`\`python
"""
Simple Concurrent Execution
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

# Sequential: 5 seconds
results = [task(i) for i in range(5)]

# Concurrent: 1 second
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(task, range(5)))

# Same simple interface for processes!
with ProcessPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(task, range(5)))
\`\`\`

By the end of this section, you'll master:
- ThreadPoolExecutor and ProcessPoolExecutor
- submit() vs map() methods
- Future objects and callbacks
- Exception handling
- Timeout and cancellation
- Production patterns

---

## ThreadPoolExecutor

### Basic Usage

\`\`\`python
"""
ThreadPoolExecutor Basics
"""

from concurrent.futures import ThreadPoolExecutor
import time

def download_file(url):
    """Simulate file download"""
    print(f"Downloading {url}...")
    time.sleep(2)
    return f"Content from {url}"

urls = [
    'https://example.com/file1',
    'https://example.com/file2',
    'https://example.com/file3',
]

# Create thread pool
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit tasks
    futures = [executor.submit(download_file, url) for url in urls]
    
    # Get results
    for future in futures:
        result = future.result()
        print(result)

# All 3 downloads complete in ~2 seconds (vs 6 seconds sequential)
\`\`\`

### map() Method

\`\`\`python
"""
ThreadPoolExecutor.map() for Batch Processing
"""

from concurrent.futures import ThreadPoolExecutor
import time

def process_item(item):
    """Process single item"""
    time.sleep(1)
    return item * 2

items = list(range(10))

# Using map() - simpler for batch operations
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_item, items))
    print(f"Results: {results}")

# map() returns results in order
# Blocks until all complete
# 10 items with 5 workers = 2 seconds
\`\`\`

### submit() Method

\`\`\`python
"""
ThreadPoolExecutor.submit() for Fine Control
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def fetch_data(source, delay):
    """Fetch data from source"""
    time.sleep(delay)
    return f"Data from {source}"

with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit tasks with different arguments
    future1 = executor.submit(fetch_data, "API-1", 1)
    future2 = executor.submit(fetch_data, "API-2", 2)
    future3 = executor.submit(fetch_data, "API-3", 0.5)
    
    # Process as they complete (not in order)
    futures = [future1, future2, future3]
    for future in as_completed(futures):
        result = future.result()
        print(result)

# Output order: API-3, API-1, API-2
# (as_completed returns futures as they finish)
\`\`\`

---

## ProcessPoolExecutor

### Basic Usage

\`\`\`python
"""
ProcessPoolExecutor for CPU-Bound Work
"""

from concurrent.futures import ProcessPoolExecutor
import time

def cpu_intensive_task(n):
    """CPU-intensive calculation"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

if __name__ == '__main__':
    data = [10_000_000] * 8
    
    # Using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_intensive_task, data))
    
    print(f"Completed {len(results)} tasks")

# 4× speedup on 4-core CPU
\`\`\`

### Choosing Worker Count

\`\`\`python
"""
Optimal Worker Count
"""

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# For CPU-bound work (ProcessPoolExecutor)
cpu_count = os.cpu_count()
print(f"CPU cores: {cpu_count}")

# Use all cores
with ProcessPoolExecutor(max_workers=cpu_count) as executor:
    pass

# Or leave 1 core free for system
with ProcessPoolExecutor(max_workers=cpu_count - 1) as executor:
    pass

# For I/O-bound work (ThreadPoolExecutor)
# Rule of thumb: 2-5× number of cores
io_workers = cpu_count * 2

with ThreadPoolExecutor(max_workers=io_workers) as executor:
    pass

# For high-concurrency I/O: even more
with ThreadPoolExecutor(max_workers=100) as executor:
    pass
\`\`\`

---

## Future Objects

### Working with Futures

\`\`\`python
"""
Future Object Methods
"""

from concurrent.futures import ThreadPoolExecutor
import time

def slow_task(n):
    time.sleep(n)
    return n * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit task
    future = executor.submit(slow_task, 3)
    
    # Check if done
    print(f"Done: {future.done()}")  # False
    
    # Check if running
    print(f"Running: {future.running()}")  # True
    
    # Wait for completion
    time.sleep(1)
    print(f"Done: {future.done()}")  # False (still running)
    
    # Get result (blocks until done)
    result = future.result()
    print(f"Result: {result}")  # 6
    
    print(f"Done: {future.done()}")  # True
\`\`\`

### Timeouts

\`\`\`python
"""
Future Timeouts
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

def slow_operation():
    time.sleep(10)
    return "completed"

with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(slow_operation)
    
    try:
        # Wait maximum 2 seconds
        result = future.result(timeout=2.0)
    except TimeoutError:
        print("Operation timed out")
        # Task still running! Can cancel it
        future.cancel()
\`\`\`

### Callbacks

\`\`\`python
"""
Future Callbacks
"""

from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

def callback(future):
    """Called when future completes"""
    result = future.result()
    print(f"Task completed with result: {result}")

with ThreadPoolExecutor(max_workers=2) as executor:
    # Submit task with callback
    future1 = executor.submit(task, 5)
    future1.add_done_callback(callback)
    
    future2 = executor.submit(task, 10)
    future2.add_done_callback(callback)

# Callbacks execute as tasks complete
# Output:
# Task completed with result: 25
# Task completed with result: 100
\`\`\`

---

## Error Handling

### Handling Exceptions

\`\`\`python
"""
Exception Handling in Futures
"""

from concurrent.futures import ThreadPoolExecutor
import time

def failing_task(n):
    time.sleep(1)
    if n == 2:
        raise ValueError(f"Task {n} failed")
    return n * 2

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(failing_task, i) for i in range(4)]
    
    for i, future in enumerate(futures):
        try:
            result = future.result()
            print(f"Task {i}: {result}")
        except ValueError as e:
            print(f"Task {i} failed: {e}")

# Output:
# Task 0: 0
# Task 1: 2
# Task 2 failed: Task 2 failed
# Task 3: 6
\`\`\`

### Using as_completed()

\`\`\`python
"""
Process Results as They Complete
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def task(n):
    """Task with random duration"""
    duration = random.uniform(0.5, 2.0)
    time.sleep(duration)
    return n, duration

with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit tasks
    futures = {executor.submit(task, i): i for i in range(10)}
    
    # Process as they complete
    for future in as_completed(futures):
        task_id = futures[future]
        try:
            result, duration = future.result()
            print(f"Task {task_id} completed in {duration:.2f}s")
        except Exception as e:
            print(f"Task {task_id} failed: {e}")

# Results appear in completion order, not submission order
\`\`\`

---

## Production Patterns

### Batch Processing

\`\`\`python
"""
Production Pattern: Batch Processing
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any

class BatchProcessor:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        chunk_size: int = 100
    ):
        """Process items in batches"""
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process in chunks
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                
                # Submit chunk
                futures = [
                    executor.submit(process_func, item)
                    for item in chunk
                ]
                
                # Collect results
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        errors.append(str(e))
        
        return {
            'results': results,
            'errors': errors,
            'success_rate': len(results) / len(items)
        }

# Usage
processor = BatchProcessor(max_workers=10)

def process_item(item):
    # Process logic
    return item * 2

items = list(range(1000))
output = processor.process_batch(items, process_item)
print(f"Processed {len(output['results'])} items")
\`\`\`

### Rate Limiting

\`\`\`python
"""
Rate-Limited Execution
"""

from concurrent.futures import ThreadPoolExecutor
import time
from threading import Semaphore

class RateLimitedExecutor:
    def __init__(self, max_workers: int, rate_limit: int):
        """
        max_workers: Number of concurrent workers
        rate_limit: Maximum operations per second
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = Semaphore(rate_limit)
        self.rate_limit = rate_limit
    
    def submit(self, func, *args, **kwargs):
        """Submit task with rate limiting"""
        def wrapper():
            with self.semaphore:
                try:
                    return func(*args, **kwargs)
                finally:
                    # Release after minimum interval
                    time.sleep(1.0 / self.rate_limit)
        
        return self.executor.submit(wrapper)
    
    def shutdown(self):
        self.executor.shutdown()

# Usage: Limit to 10 API calls per second
executor = RateLimitedExecutor(max_workers=5, rate_limit=10)

def api_call(endpoint):
    # Make API call
    pass

futures = [executor.submit(api_call, f"endpoint-{i}") for i in range(100)]
results = [f.result() for f in futures]

executor.shutdown()
\`\`\`

### Progress Tracking

\`\`\`python
"""
Track Progress of Concurrent Tasks
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_item(item):
    """Process item"""
    time.sleep(0.5)
    return item * 2

items = list(range(100))

with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all tasks
    futures = {executor.submit(process_item, item): item for item in items}
    
    # Track progress
    completed = 0
    total = len(futures)
    
    for future in as_completed(futures):
        completed += 1
        percentage = (completed / total) * 100
        
        try:
            result = future.result()
            print(f"Progress: {percentage:.1f}% ({completed}/{total})")
        except Exception as e:
            print(f"Task failed: {e}")

print("All tasks complete!")
\`\`\`

---

## Integrating with Asyncio

### Running in Executor

\`\`\`python
"""
Mix concurrent.futures with asyncio
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

def blocking_io(n):
    """Blocking I/O operation"""
    print(f"Starting blocking operation {n}")
    time.sleep(2)
    return f"Result {n}"

async def main():
    loop = asyncio.get_running_loop()
    
    # Create executor
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Run blocking operation in executor
        tasks = [
            loop.run_in_executor(executor, blocking_io, i)
            for i in range(5)
        ]
        
        # Await all
        results = await asyncio.gather(*tasks)
        print(f"Results: {results}")

asyncio.run(main())

# Allows mixing async code with blocking libraries
\`\`\`

---

## Summary

### Key Concepts

1. **ThreadPoolExecutor**: Thread pool for I/O-bound work
2. **ProcessPoolExecutor**: Process pool for CPU-bound work
3. **submit()**: Submit single task, returns Future
4. **map()**: Batch process, returns results in order
5. **as_completed()**: Process results as they finish
6. **Future**: Represents pending result, supports timeout

### Best Practices

- Use context manager (\`with\`) for automatic cleanup
- Choose appropriate worker count (CPU count for processes)
- Handle exceptions in futures
- Use as_completed() for progressive processing
- Set timeouts for long-running tasks

### Common Patterns

- **Batch processing**: Process large datasets in parallel
- **Rate limiting**: Control request rate
- **Progress tracking**: Monitor completion percentage
- **Retry logic**: Resubmit failed tasks

### Next Steps

Now that you master concurrent.futures, we'll explore:
- Race conditions and synchronization
- Debugging concurrent applications
- Advanced production patterns

**Remember**: concurrent.futures is the simplest way to add concurrency to Python!
\`,
};
