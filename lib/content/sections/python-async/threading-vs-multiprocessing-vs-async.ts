export const threadingVsMultiprocessingVsAsync = {
  title: 'Threading vs Multiprocessing vs Async',
  id: 'threading-vs-multiprocessing-vs-async',
  content: `
# Threading vs Multiprocessing vs Async

## Introduction

Python offers three major concurrency models: **Threading** (concurrent execution in one process), **Multiprocessing** (parallel execution across processes), and **Async** (cooperative multitasking for I/O). **Choosing the wrong model can make your code 10-100× slower.**

### The Fundamental Question

\`\`\`python
"""
When to Use Each Concurrency Model
"""

# ❌ Wrong: Threading for CPU-bound work (GIL kills performance)
import threading
def cpu_task():
    return sum (i**2 for i in range(10_000_000))

threads = [threading.Thread (target=cpu_task) for _ in range(4)]
# Result: 4× slower than single-threaded! (GIL contention)

# ✅ Right: Multiprocessing for CPU-bound work
from multiprocessing import Pool
with Pool(4) as pool:
    results = pool.map (cpu_task, range(4))
# Result: 4× faster! (true parallelism)

# ❌ Wrong: Multiprocessing for I/O-bound work (high overhead)
with Pool(100) as pool:
    results = pool.map (fetch_url, urls)  # 100 processes = high overhead

# ✅ Right: Async for I/O-bound work
async def fetch_all():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url (session, url) for url in urls]
        return await asyncio.gather(*tasks)
# Result: 100 concurrent requests, minimal overhead
\`\`\`

By the end of this section, you'll master:
- The Global Interpreter Lock (GIL) and its impact
- Threading: When and how to use it
- Multiprocessing: True parallelism for CPU work
- Async: High-concurrency I/O
- Performance benchmarks and trade-offs
- Choosing the right model for your use case

---

## The Global Interpreter Lock (GIL)

### What is the GIL?

\`\`\`python
"""
The GIL Explained
"""

# The GIL is a mutex that protects access to Python objects
# Only ONE thread can execute Python bytecode at a time

import threading
import time

def cpu_bound_work():
    """Pure CPU work"""
    total = 0
    for i in range(10_000_000):
        total += i ** 2
    return total

# Single-threaded
start = time.time()
result = cpu_bound_work()
single_time = time.time() - start
print(f"Single thread: {single_time:.2f}s")

# Multi-threaded (2 threads)
start = time.time()
threads = [
    threading.Thread (target=cpu_bound_work),
    threading.Thread (target=cpu_bound_work)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
multi_time = time.time() - start
print(f"Two threads: {multi_time:.2f}s")

# Result: Two threads SLOWER than single thread!
# Single thread: 1.2s
# Two threads: 1.5s (due to GIL contention)
\`\`\`

### When the GIL Releases

\`\`\`python
"""
GIL Releases During I/O Operations
"""

import threading
import time
import requests

def io_bound_work (url):
    """I/O-bound work (network request)"""
    response = requests.get (url)
    return len (response.content)

urls = ['https://example.com'] * 10

# Single-threaded I/O
start = time.time()
results = [io_bound_work (url) for url in urls]
single_time = time.time() - start
print(f"Single thread: {single_time:.2f}s")

# Multi-threaded I/O
start = time.time()
threads = [threading.Thread (target=io_bound_work, args=(url,)) for url in urls]
for t in threads:
    t.start()
for t in threads:
    t.join()
multi_time = time.time() - start
print(f"Ten threads: {multi_time:.2f}s")

# Result: Multi-threaded MUCH faster for I/O!
# Single thread: 5.0s (sequential)
# Ten threads: 0.6s (concurrent, GIL released during I/O)

# GIL releases during:
# - I/O operations (network, file, database)
# - time.sleep()
# - Some C extensions (NumPy, etc.)
\`\`\`

---

## Threading

### Basic Threading

\`\`\`python
"""
Threading Basics
"""

import threading
import time

def worker (name, duration):
    """Worker thread"""
    print(f"{name} starting")
    time.sleep (duration)  # GIL released during sleep
    print(f"{name} finished")

# Create threads
threads = [
    threading.Thread (target=worker, args=("Thread-1", 2)),
    threading.Thread (target=worker, args=("Thread-2", 1)),
    threading.Thread (target=worker, args=("Thread-3", 3)),
]

# Start threads
for t in threads:
    t.start()

# Wait for completion
for t in threads:
    t.join()

print("All threads complete")

# Output:
# Thread-1 starting
# Thread-2 starting
# Thread-3 starting
# Thread-2 finished (after 1s)
# Thread-1 finished (after 2s)
# Thread-3 finished (after 3s)
# All threads complete
\`\`\`

### Thread Pool

\`\`\`python
"""
ThreadPoolExecutor for Managing Thread Pool
"""

from concurrent.futures import ThreadPoolExecutor
import time

def task (n):
    """Task that takes time"""
    time.sleep(1)
    return n * n

# Create thread pool
with ThreadPoolExecutor (max_workers=5) as executor:
    # Submit tasks
    futures = [executor.submit (task, i) for i in range(10)]
    
    # Get results
    results = [future.result() for future in futures]
    print(f"Results: {results}")

# 10 tasks with 5 workers = ~2 seconds (vs 10 seconds sequential)
\`\`\`

### When to Use Threading

\`\`\`python
"""
Threading Use Cases
"""

# ✅ Good for threading:
# 1. I/O-bound operations
def download_files (urls):
    with ThreadPoolExecutor (max_workers=10) as executor:
        return list (executor.map (download, urls))

# 2. Blocking API calls
def fetch_user_data (user_ids):
    with ThreadPoolExecutor (max_workers=20) as executor:
        return list (executor.map (api_client.get_user, user_ids))

# 3. Database queries (with connection pooling)
def execute_queries (queries):
    with ThreadPoolExecutor (max_workers=10) as executor:
        return list (executor.map (db.execute, queries))

# ❌ Bad for threading:
# 1. CPU-bound work (use multiprocessing)
def compute_intensive (data):
    # Complex calculations (GIL prevents parallelism)
    pass

# 2. Very high concurrency (1000+ concurrent, use async)
def handle_10000_connections():
    # Threading overhead too high
    pass
\`\`\`

---

## Multiprocessing

### Basic Multiprocessing

\`\`\`python
"""
Multiprocessing Basics
"""

from multiprocessing import Process
import os

def worker (name):
    """Worker process"""
    print(f"{name} running in process {os.getpid()}")

if __name__ == '__main__':
    # Create processes
    processes = [
        Process (target=worker, args=("Process-1",)),
        Process (target=worker, args=("Process-2",)),
        Process (target=worker, args=("Process-3",)),
    ]
    
    # Start processes
    for p in processes:
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()
    
    print("All processes complete")

# Output:
# Process-1 running in process 12345
# Process-2 running in process 12346
# Process-3 running in process 12347
# All processes complete
\`\`\`

### Process Pool

\`\`\`python
"""
ProcessPoolExecutor for CPU-Bound Work
"""

from concurrent.futures import ProcessPoolExecutor
import time

def cpu_intensive_task (n):
    """CPU-intensive calculation"""
    total = 0
    for i in range (n):
        total += i ** 2
    return total

if __name__ == '__main__':
    # Create process pool
    with ProcessPoolExecutor (max_workers=4) as executor:
        # Map work across processes
        results = list (executor.map(
            cpu_intensive_task,
            [10_000_000] * 8
        ))
    
    print(f"Results: {len (results)} computed")

# With 4 cores: 8 tasks complete in ~2× single-task time
# vs 8× time with sequential execution
\`\`\`

### Multiprocessing Pool

\`\`\`python
"""
multiprocessing.Pool for Parallel Processing
"""

from multiprocessing import Pool
import numpy as np

def process_chunk (data):
    """Process data chunk"""
    # CPU-intensive processing
    result = np.sum (data ** 2)
    return result

if __name__ == '__main__':
    # Generate large dataset
    data = np.random.rand(10_000_000)
    
    # Split into chunks
    chunks = np.array_split (data, 8)
    
    # Process in parallel
    with Pool (processes=4) as pool:
        results = pool.map (process_chunk, chunks)
    
    total = sum (results)
    print(f"Total: {total}")

# 4× faster than sequential processing (on 4-core CPU)
\`\`\`

### When to Use Multiprocessing

\`\`\`python
"""
Multiprocessing Use Cases
"""

# ✅ Good for multiprocessing:
# 1. CPU-bound computations
def parallel_computation (data_chunks):
    with Pool (processes=4) as pool:
        return pool.map (compute, data_chunks)

# 2. Data processing pipelines
def process_large_dataset (dataset):
    with Pool (processes=8) as pool:
        return pool.map (process_record, dataset)

# 3. Image/video processing
def process_images (image_files):
    with Pool (processes=4) as pool:
        return pool.map (transform_image, image_files)

# ❌ Bad for multiprocessing:
# 1. I/O-bound work (overhead too high)
def fetch_urls (urls):
    # Process creation overhead > I/O time
    pass

# 2. Shared state (requires expensive IPC)
def update_shared_counter():
    # Inter-process communication is slow
    pass

# 3. Small tasks (overhead dominates)
def add_numbers (a, b):
    # Process creation > actual work
    pass
\`\`\`

---

## Async (asyncio)

### Basic Async

\`\`\`python
"""
Async Basics
"""

import asyncio

async def worker (name, duration):
    """Async worker"""
    print(f"{name} starting")
    await asyncio.sleep (duration)
    print(f"{name} finished")
    return name

async def main():
    # Create tasks
    tasks = [
        worker("Task-1", 2),
        worker("Task-2", 1),
        worker("Task-3", 3),
    ]
    
    # Run concurrently
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")

asyncio.run (main())

# Output:
# Task-1 starting
# Task-2 starting
# Task-3 starting
# Task-2 finished (after 1s)
# Task-1 finished (after 2s)
# Task-3 finished (after 3s)
# Results: ['Task-1', 'Task-2', 'Task-3']
\`\`\`

### Async HTTP Example

\`\`\`python
"""
Async HTTP: High Concurrency with Low Overhead
"""

import asyncio
import aiohttp

async def fetch_url (session, url):
    """Fetch single URL"""
    async with session.get (url) as response:
        return await response.text()

async def fetch_all_urls (urls):
    """Fetch all URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url (session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = ['https://example.com'] * 100
results = asyncio.run (fetch_all_urls (urls))

# 100 concurrent requests in ~1 second
# vs 100+ seconds sequential
# Minimal memory overhead (single thread)
\`\`\`

### When to Use Async

\`\`\`python
"""
Async Use Cases
"""

# ✅ Good for async:
# 1. High-concurrency I/O (1000+ concurrent)
async def handle_websocket_connections (connections):
    # Handle 10,000+ concurrent WebSocket connections
    await asyncio.gather(*[handle_conn (c) for c in connections])

# 2. API servers
async def api_handler (request):
    # FastAPI, aiohttp server for high-throughput APIs
    data = await fetch_from_db (request.user_id)
    return response (data)

# 3. Web scraping at scale
async def scrape_websites (urls):
    # Scrape 10,000 URLs concurrently
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[scrape (session, url) for url in urls])

# ❌ Bad for async:
# 1. CPU-bound work (no parallelism)
async def compute_intensive():
    # Blocks event loop, prevents other tasks from running
    return sum (i**2 for i in range(10_000_000))

# 2. Blocking libraries
async def use_blocking_lib():
    # requests library blocks event loop
    response = requests.get (url)  # ❌ Bad!
    # Use aiohttp instead
\`\`\`

---

## Performance Comparison

### Benchmark: I/O-Bound Work

\`\`\`python
"""
I/O-Bound Performance Comparison
"""

import time
import asyncio
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def fetch_sync (url):
    """Synchronous fetch"""
    return requests.get (url).text

async def fetch_async (session, url):
    """Async fetch"""
    async with session.get (url) as response:
        return await response.text()

urls = ['https://httpbin.org/delay/1'] * 10

# 1. Sequential
start = time.time()
results = [fetch_sync (url) for url in urls]
print(f"Sequential: {time.time() - start:.2f}s")
# Result: ~10 seconds

# 2. Threading
start = time.time()
with ThreadPoolExecutor (max_workers=10) as executor:
    results = list (executor.map (fetch_sync, urls))
print(f"Threading: {time.time() - start:.2f}s")
# Result: ~1 second

# 3. Multiprocessing
start = time.time()
with ProcessPoolExecutor (max_workers=10) as executor:
    results = list (executor.map (fetch_sync, urls))
print(f"Multiprocessing: {time.time() - start:.2f}s")
# Result: ~1.5 seconds (slower due to overhead)

# 4. Async
async def fetch_all():
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*[fetch_async (session, url) for url in urls])

start = time.time()
results = asyncio.run (fetch_all())
print(f"Async: {time.time() - start:.2f}s")
# Result: ~1 second (fastest, lowest overhead)

# Winner: Async (fastest, lowest memory)
\`\`\`

### Benchmark: CPU-Bound Work

\`\`\`python
"""
CPU-Bound Performance Comparison
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_task():
    """CPU-intensive task"""
    return sum (i**2 for i in range(10_000_000))

tasks = [cpu_task] * 4

# 1. Sequential
start = time.time()
results = [task() for task in tasks]
print(f"Sequential: {time.time() - start:.2f}s")
# Result: ~4.8 seconds

# 2. Threading
start = time.time()
with ThreadPoolExecutor (max_workers=4) as executor:
    results = list (executor.map (lambda f: f(), tasks))
print(f"Threading: {time.time() - start:.2f}s")
# Result: ~5.2 seconds (SLOWER due to GIL contention!)

# 3. Multiprocessing
start = time.time()
with ProcessPoolExecutor (max_workers=4) as executor:
    results = list (executor.map (lambda f: f(), tasks))
print(f"Multiprocessing: {time.time() - start:.2f}s")
# Result: ~1.2 seconds (4× speedup!)

# 4. Async
async def async_tasks():
    loop = asyncio.get_event_loop()
    return await asyncio.gather(*[loop.run_in_executor(None, task) for task in tasks])

start = time.time()
results = asyncio.run (async_tasks())
print(f"Async (with executor): {time.time() - start:.2f}s")
# Result: ~5.0 seconds (uses thread pool, has GIL)

# Winner: Multiprocessing (only true parallelism)
\`\`\`

---

## Decision Matrix

### Choosing the Right Model

\`\`\`python
"""
Decision Matrix for Concurrency Model
"""

def choose_concurrency_model (workload_type, concurrency_level, task_duration):
    """
    Choose best concurrency model
    
    Parameters:
    - workload_type: 'cpu' or 'io'
    - concurrency_level: number of concurrent operations
    - task_duration: 'short' (<100ms) or 'long' (>100ms)
    """
    
    if workload_type == 'cpu':
        if concurrency_level <= cpu_count():
            return "Multiprocessing (ProcessPoolExecutor)"
        else:
            return "Multiprocessing with chunking"
    
    elif workload_type == 'io':
        if concurrency_level < 50:
            return "Threading (ThreadPoolExecutor)"
        elif concurrency_level < 1000:
            return "Async (asyncio + aiohttp)"
        else:
            return "Async (definitely asyncio)"
    
    return "Sequential (simplest)"

# Examples:
print(choose_concurrency_model('cpu', 4, 'long'))
# → Multiprocessing (ProcessPoolExecutor)

print(choose_concurrency_model('io', 100, 'long'))
# → Async (asyncio + aiohttp)

print(choose_concurrency_model('io', 10, 'short'))
# → Threading (ThreadPoolExecutor)
\`\`\`

---

## Summary

### Key Concepts

1. **GIL**: Python\'s Global Interpreter Lock prevents true parallel CPU execution in threads
2. **Threading**: Good for I/O-bound work (GIL releases during I/O), bad for CPU-bound
3. **Multiprocessing**: True parallelism for CPU-bound work, high overhead
4. **Async**: Best for high-concurrency I/O (1000+), single-threaded

### Decision Guide

| Workload | Concurrency | Best Choice | Why |
|----------|-------------|-------------|-----|
| CPU-bound | Any | Multiprocessing | GIL prevents threading parallelism |
| I/O-bound | < 50 | Threading | Simple, effective, low overhead |
| I/O-bound | 50-1000 | Async | Lower overhead than threading |
| I/O-bound | 1000+ | Async | Only option for high concurrency |

### Performance Rules

- **Threading**: 1-10× speedup for I/O (limited by GIL)
- **Multiprocessing**: N× speedup for CPU (N = cores)
- **Async**: 100-1000× speedup for I/O (scales to 10K+ concurrent)

### Next Steps

Now that you understand concurrency models, we'll explore:
- concurrent.futures in depth
- Race conditions and synchronization
- Advanced async patterns

**Remember**: Choose based on workload type (CPU vs I/O) and concurrency level!
`,
};
