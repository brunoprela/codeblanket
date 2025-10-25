export const concurrencyFundamentals = {
  title: 'Concurrency Fundamentals',
  id: 'concurrency-fundamentals',
  content: `
# Concurrency Fundamentals

## Introduction

**Concurrency** is the ability of a program to make progress on multiple tasks seemingly at the same time. In Python, understanding concurrency is critical for building high-performance applications that don't waste time waiting for I/O operations.

### Why Concurrency Matters

Consider a web scraper that fetches 100 web pages. With synchronous code, you fetch them one at a time:

\`\`\`python
# Synchronous approach - SLOW
import requests

urls = ['https://api.example.com/data/' + str (i) for i in range(100)]
responses = []

for url in urls:
    response = requests.get (url)  # Waits for each request to complete
    responses.append (response.json())

# Takes 100 × 0.5s = 50 seconds if each request takes 0.5s
\`\`\`

With asynchronous code, you can start all 100 requests and wait for them concurrently:

\`\`\`python
# Asynchronous approach - FAST
import asyncio
import aiohttp

async def fetch_all (urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get (url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# Takes ~0.5s - all requests happen concurrently!
\`\`\`

**Result**: 100× speed improvement for this I/O-bound task!

By the end of this section, you'll understand:
- The difference between concurrency and parallelism
- Why async matters for I/O-bound operations
- When to use async vs threads vs multiprocessing
- CPU-bound vs I/O-bound task characteristics
- How async fits into the Python ecosystem
- Common use cases for asynchronous programming

---

## Concurrency vs Parallelism

These terms are often confused, but they have distinct meanings:

### Concurrency

**Dealing with multiple tasks at once** (task switching)

- Single-core CPU can do concurrency
- Tasks make progress by interleaving
- While one task waits (I/O), another runs
- **Key**: Tasks don't run simultaneously, they just make progress together

\`\`\`
Task A: [====== WAIT ======] [====]
Task B:        [====] [== WAIT ==]
Time:   0---5---10---15---20---25
\`\`\`

### Parallelism

**Doing multiple tasks simultaneously** (truly at the same time)

- Requires multiple CPU cores
- Tasks run at exactly the same moment
- Used for CPU-bound operations
- **Key**: Tasks execute simultaneously on different cores

\`\`\`
Core 1: [====================]  Task A
Core 2: [====================]  Task B
Time:   0---5---10---15---20
\`\`\`

### Python Example

\`\`\`python
"""
Demonstrating Concurrency vs Parallelism
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O-bound task: Benefits from CONCURRENCY
def io_bound_task (n):
    """Simulates API call or file I/O"""
    print(f"Task {n}: Starting I/O operation")
    time.sleep(1)  # Simulates waiting for I/O
    print(f"Task {n}: I/O complete")
    return n * 2

# CPU-bound task: Benefits from PARALLELISM  
def cpu_bound_task (n):
    """Simulates heavy computation"""
    print(f"Task {n}: Starting computation")
    total = 0
    for i in range(10_000_000):  # Pure computation
        total += i
    print(f"Task {n}: Computation complete")
    return total

# Test 1: Sequential (SLOW for both)
print("\\n=== Sequential Execution ===")
start = time.time()
results = [io_bound_task (i) for i in range(3)]
print(f"Time: {time.time() - start:.2f}s")  # ~3 seconds

# Test 2: Threading for I/O (FAST - concurrency)
print("\\n=== Threading for I/O ===")
start = time.time()
with ThreadPoolExecutor (max_workers=3) as executor:
    results = list (executor.map (io_bound_task, range(3)))
print(f"Time: {time.time() - start:.2f}s")  # ~1 second (all wait together)

# Test 3: Multiprocessing for CPU (FAST - parallelism)
print("\\n=== Multiprocessing for CPU ===")
start = time.time()
with ProcessPoolExecutor (max_workers=3) as executor:
    results = list (executor.map (cpu_bound_task, range(3)))
print(f"Time: {time.time() - start:.2f}s")  # ~1/3 sequential time

# Test 4: Async for I/O (FASTEST - true async concurrency)
async def async_io_task (n):
    """Async version of I/O task"""
    print(f"Task {n}: Starting async I/O")
    await asyncio.sleep(1)  # Non-blocking wait
    print(f"Task {n}: Async I/O complete")
    return n * 2

async def run_async():
    results = await asyncio.gather(
        async_io_task(0),
        async_io_task(1), 
        async_io_task(2)
    )
    return results

print("\\n=== Async for I/O ===")
start = time.time()
results = asyncio.run (run_async())
print(f"Time: {time.time() - start:.2f}s")  # ~1 second
\`\`\`

**Key Insight**: Use concurrency (async/threading) for I/O-bound tasks, parallelism (multiprocessing) for CPU-bound tasks.

---

## CPU-Bound vs I/O-Bound Tasks

Understanding whether your task is CPU-bound or I/O-bound is critical for choosing the right concurrency model.

### I/O-Bound Tasks

**Characteristics**:
- Spend most time waiting for external operations
- CPU is mostly idle
- Examples: Network requests, database queries, file operations
- Limited by I/O speed, not CPU speed

**Indicators**:
- High wait time
- Low CPU usage (often <10%)
- Latency-sensitive
- Benefit from concurrency (async/threading)

\`\`\`python
"""
I/O-Bound Example: Web API Requests
"""

import asyncio
import aiohttp
import time

async def fetch_user (session, user_id):
    """Fetch user data from API (I/O-bound)"""
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"
    
    async with session.get (url) as response:
        # 90% of time spent here waiting for network response
        # CPU is idle during this wait
        data = await response.json()
        return data

async def main():
    """Fetch 10 users concurrently"""
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_user (session, i) for i in range(1, 11)]
        users = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"Fetched {len (users)} users in {elapsed:.2f}s")
    print(f"CPU was idle {(elapsed * 0.95):.2f}s waiting for network")

# asyncio.run (main())
# Output: ~0.5s for all 10 requests (concurrent)
# vs ~5s if done sequentially
\`\`\`

### CPU-Bound Tasks

**Characteristics**:
- Spend most time in computation
- CPU is maxed out
- Examples: Data processing, encryption, image manipulation, calculations
- Limited by CPU speed

**Indicators**:
- High CPU usage (often 100%)
- Little waiting
- Throughput-sensitive
- Benefit from parallelism (multiprocessing)

\`\`\`python
"""
CPU-Bound Example: Image Processing
"""

from concurrent.futures import ProcessPoolExecutor
import time

def process_image (image_id):
    """Simulate CPU-intensive image processing"""
    # Pure computation - CPU at 100% during this
    total = 0
    for i in range(10_000_000):
        total += i ** 2
    
    return f"Processed image {image_id}"

def main():
    """Process images using multiple CPU cores"""
    images = list (range(8))  # 8 images to process
    
    # Sequential processing
    start = time.time()
    results = [process_image (i) for i in images]
    sequential_time = time.time() - start
    print(f"Sequential: {sequential_time:.2f}s")
    
    # Parallel processing (multiprocessing)
    start = time.time()
    with ProcessPoolExecutor (max_workers=4) as executor:
        results = list (executor.map (process_image, images))
    parallel_time = time.time() - start
    print(f"Parallel (4 cores): {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")

# main()
# Output:
# Sequential: 8.2s
# Parallel (4 cores): 2.1s
# Speedup: 3.9x (near-perfect scaling for 4 cores)
\`\`\`

### Comparison Matrix

| Aspect | I/O-Bound | CPU-Bound |
|--------|-----------|-----------|
| **Bottleneck** | External operations | CPU cycles |
| **CPU Usage** | Low (5-20%) | High (90-100%) |
| **Wait Time** | High | Low |
| **Best Approach** | Async/Threading | Multiprocessing |
| **Scaling** | Can handle 1000s concurrent | Limited by CPU cores |
| **Examples** | API calls, DB queries | Math, encryption, ML |

\`\`\`python
"""
Decision Framework: Choose the Right Approach
"""

import psutil
import time

def profile_task (task_func, *args):
    """Profile a task to determine if CPU or I/O bound"""
    cpu_before = psutil.cpu_percent (interval=0.1)
    
    start = time.time()
    task_func(*args)
    elapsed = time.time() - start
    
    cpu_during = psutil.cpu_percent (interval=0.1)
    
    if cpu_during > 80:
        print(f"✓ CPU-Bound: {cpu_during}% CPU usage")
        print("  → Use multiprocessing")
    else:
        print(f"✓ I/O-Bound: {cpu_during}% CPU usage, {elapsed:.2f}s elapsed")
        print("  → Use async or threading")

# Example usage
def io_task():
    time.sleep(1)  # Simulate I/O

def cpu_task():
    sum (i ** 2 for i in range(10_000_000))

profile_task (io_task)
# Output: ✓ I/O-Bound: 5% CPU usage, 1.00s elapsed → Use async

profile_task (cpu_task)  
# Output: ✓ CPU-Bound: 100% CPU usage → Use multiprocessing
\`\`\`

---

## Why Async Matters for I/O

Asynchronous programming is a game-changer for I/O-bound applications. Let\'s understand why.

### The Problem with Blocking I/O

\`\`\`python
"""
Blocking I/O: The CPU Waits Idle
"""

import requests
import time

def fetch_sync (url):
    """Blocking HTTP request"""
    response = requests.get (url)
    return response.text

def main():
    urls = [
        'https://httpbin.org/delay/1',  # Each takes 1 second
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/1',
    ]
    
    start = time.time()
    results = []
    
    for url in urls:
        print(f"Fetching {url}...")
        result = fetch_sync (url)  # BLOCKS HERE - CPU idle
        results.append (result)
        print(f"  Received {len (result)} bytes")
    
    elapsed = time.time() - start
    print(f"\\nTotal time: {elapsed:.2f}s")  # ~3 seconds
    print(f"CPU idle time: ~{elapsed * 0.95:.2f}s (wasted!)")

# Timeline:
# 0s: Start request 1 → CPU waits...
# 1s: Receive request 1 → Start request 2 → CPU waits...
# 2s: Receive request 2 → Start request 3 → CPU waits...
# 3s: Receive request 3 → Done
\`\`\`

### The Async Solution

\`\`\`python
"""
Async I/O: Start All Requests, Wait Together
"""

import asyncio
import aiohttp
import time

async def fetch_async (session, url):
    """Non-blocking HTTP request"""
    async with session.get (url) as response:
        return await response.text()

async def main():
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/1',
    ]
    
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Start ALL requests at once
        tasks = [fetch_async (session, url) for url in urls]
        
        print("Starting all requests concurrently...")
        results = await asyncio.gather(*tasks)
        
        print(f"Received {len (results)} responses")
    
    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s")  # ~1 second!
    print(f"Speedup: 3× faster")

# Timeline:
# 0s: Start ALL 3 requests → CPU waits...
# 1s: Receive ALL 3 responses → Done
#
# All requests happened concurrently during that 1 second!
\`\`\`

### Real-World Impact

\`\`\`python
"""
Real-World Example: Microservices Architecture
"""

import asyncio
import aiohttp
import time

async def get_user (session, user_id):
    """Fetch user from user service"""
    async with session.get (f'http://user-service/users/{user_id}') as r:
        return await r.json()

async def get_orders (session, user_id):
    """Fetch orders from order service"""
    async with session.get (f'http://order-service/orders?user={user_id}') as r:
        return await r.json()

async def get_recommendations (session, user_id):
    """Fetch recommendations from ML service"""
    async with session.get (f'http://ml-service/recommend/{user_id}') as r:
        return await r.json()

async def build_dashboard (user_id):
    """Build user dashboard by calling multiple services"""
    async with aiohttp.ClientSession() as session:
        # Without async: 150ms + 200ms + 300ms = 650ms total
        # With async: max(150ms, 200ms, 300ms) = 300ms total
        
        user, orders, recommendations = await asyncio.gather(
            get_user (session, user_id),          # 150ms
            get_orders (session, user_id),         # 200ms
            get_recommendations (session, user_id) # 300ms
        )
        
        return {
            'user': user,
            'orders': orders,
            'recommendations': recommendations
        }

# Result: 650ms → 300ms (2.2× faster)
# At scale: Serve 100 users/second instead of 45 users/second
\`\`\`

---

## Blocking vs Non-Blocking Operations

Understanding the difference between blocking and non-blocking operations is fundamental to async programming.

### Blocking Operations

**Definition**: Operations that prevent the program from doing anything else while they complete.

\`\`\`python
"""
Blocking Operation Example
"""

import time

def blocking_operation():
    """This blocks the entire program"""
    print("Starting operation...")
    time.sleep(2)  # BLOCKS for 2 seconds
    print("Operation complete")
    return "result"

def main():
    print("Before blocking call")
    result = blocking_operation()  # Program frozen for 2 seconds
    print(f"After blocking call: {result}")
    print("Can continue now")

# Timeline:
# 0s: "Before blocking call"
# 0s: "Starting operation..."
# (2 seconds of nothing - program frozen)
# 2s: "Operation complete"
# 2s: "After blocking call: result"
# 2s: "Can continue now"
\`\`\`

### Non-Blocking Operations

**Definition**: Operations that allow the program to continue doing other work while waiting.

\`\`\`python
"""
Non-Blocking Operation Example
"""

import asyncio

async def non_blocking_operation():
    """This doesn't block the event loop"""
    print("Starting operation...")
    await asyncio.sleep(2)  # NON-BLOCKING wait
    print("Operation complete")
    return "result"

async def do_other_work():
    """This can run while waiting"""
    for i in range(5):
        print(f"  Doing other work... {i}")
        await asyncio.sleep(0.5)

async def main():
    print("Before non-blocking call")
    
    # Start operation but don't wait yet
    task = asyncio.create_task (non_blocking_operation())
    
    # Do other work while operation runs
    await do_other_work()
    
    # Now wait for result
    result = await task
    print(f"After non-blocking call: {result}")

# Timeline:
# 0.0s: "Before non-blocking call"
# 0.0s: "Starting operation..."
# 0.0s: "  Doing other work... 0"
# 0.5s: "  Doing other work... 1"
# 1.0s: "  Doing other work... 2"
# 1.5s: "  Doing other work... 3"
# 2.0s: "Operation complete"
# 2.0s: "  Doing other work... 4"
# 2.5s: "After non-blocking call: result"
\`\`\`

### Identifying Blocking Code

\`\`\`python
"""
Common Blocking Operations in Python
"""

import time
import requests

# ❌ BLOCKING - Avoid in async functions
def blocking_examples():
    time.sleep(1)                    # ❌ Blocks event loop
    requests.get('https://api.com')  # ❌ Blocks event loop
    with open('file.txt') as f:      # ❌ Blocks event loop
        data = f.read()
    
    # Any synchronous I/O blocks

# ✅ NON-BLOCKING - Use in async functions
async def non_blocking_examples():
    await asyncio.sleep(1)           # ✅ Non-blocking wait
    
    async with aiohttp.ClientSession() as session:
        await session.get('https://api.com')  # ✅ Non-blocking
    
    async with aiofiles.open('file.txt') as f:
        data = await f.read()        # ✅ Non-blocking
\`\`\`

---

## When to Use Async

Not every application needs async. Use it when it provides clear benefits.

### ✅ Good Use Cases for Async

1. **Web Servers / APIs**
\`\`\`python
# Handle many concurrent requests efficiently
# Each request spends most time on I/O (database, external APIs)

from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user (user_id: int):
    # While waiting for database, server can handle other requests
    user = await db.fetch_user (user_id)
    return user

# Can handle 10,000+ concurrent connections on single process
\`\`\`

2. **Web Scraping**
\`\`\`python
# Fetch thousands of pages concurrently
async def scrape_websites (urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page (session, url) for url in urls]
        pages = await asyncio.gather(*tasks)
    return pages

# 100× faster than sequential scraping
\`\`\`

3. **Microservices Communication**
\`\`\`python
# Make multiple service calls in parallel
async def get_user_data (user_id):
    profile, orders, reviews = await asyncio.gather(
        profile_service.get (user_id),
        order_service.get (user_id),
        review_service.get (user_id)
    )
    return combine (profile, orders, reviews)
\`\`\`

4. **Real-Time Applications**
\`\`\`python
# WebSocket server handling many concurrent connections
async def websocket_handler (websocket):
    async for message in websocket:
        # Handle message while managing thousands of other connections
        await process_message (message)
\`\`\`

5. **Database Operations**
\`\`\`python
# Execute multiple queries concurrently
async def get_dashboard_data():
    users, orders, revenue = await asyncio.gather(
        db.query("SELECT COUNT(*) FROM users"),
        db.query("SELECT COUNT(*) FROM orders"),
        db.query("SELECT SUM(amount) FROM orders")
    )
    return {'users': users, 'orders': orders, 'revenue': revenue}
\`\`\`

### ❌ Bad Use Cases for Async

1. **CPU-Intensive Operations**
\`\`\`python
# ❌ Async doesn't help with pure computation
async def process_data (data):
    result = 0
    for i in range(10_000_000):
        result += i ** 2  # CPU-bound loop
    return result

# This will still block the event loop!
# Use multiprocessing instead
\`\`\`

2. **Simple Scripts**
\`\`\`python
# ❌ Unnecessary complexity for simple scripts
async def read_and_print():
    async with aiofiles.open('file.txt') as f:
        content = await f.read()
    print(content)

# Just use regular file operations for simple scripts
with open('file.txt') as f:
    print(f.read())
\`\`\`

3. **Blocking Library Interactions**
\`\`\`python
# ❌ If library doesn't support async, you don't benefit
import some_blocking_library

async def use_blocking_lib():
    # This BLOCKS despite being in async function!
    result = some_blocking_library.do_something()
    return result

# Need async-compatible libraries to benefit
\`\`\`

### Decision Tree

\`\`\`
Is your application I/O-bound?
├─ No (CPU-bound) → Use multiprocessing
│
└─ Yes (I/O-bound)
    │
    ├─ Simple script, few I/O operations?
    │  └─ No → Use synchronous code (simpler)
    │
    ├─ Need to handle thousands of concurrent operations?
    │  └─ Yes → Use async
    │
    ├─ Have async-compatible libraries available?
    │  └─ No → Use threading as fallback
    │
    └─ Otherwise → Async is a great choice!
\`\`\`

---

## Common Async Use Cases

Let\'s explore real-world scenarios where async shines.

### Use Case 1: Building a Fast Web API

\`\`\`python
"""
FastAPI: Async Web Framework
Handle thousands of concurrent requests
"""

from fastapi import FastAPI
import asyncio

app = FastAPI()

async def query_database (user_id: int):
    """Simulate database query"""
    await asyncio.sleep(0.1)  # DB query takes 100ms
    return {'id': user_id, 'name': f'User {user_id}'}

async def fetch_external_api (user_id: int):
    """Simulate external API call"""
    await asyncio.sleep(0.2)  # API call takes 200ms
    return {'recommendations': ['item1', 'item2']}

@app.get("/users/{user_id}/dashboard")
async def get_dashboard (user_id: int):
    """
    Build user dashboard from multiple sources
    Sequential: 100ms + 200ms = 300ms
    Async: max(100ms, 200ms) = 200ms
    """
    user, recommendations = await asyncio.gather(
        query_database (user_id),
        fetch_external_api (user_id)
    )
    
    return {
        'user': user,
        'recommendations': recommendations
    }

# Can handle 5,000 requests/second per worker
# vs 500 requests/second with synchronous code
\`\`\`

### Use Case 2: Concurrent Database Operations

\`\`\`python
"""
Async Database Queries with asyncpg
"""

import asyncpg
import asyncio

async def init_db():
    """Initialize async database connection"""
    return await asyncpg.connect(
        host='localhost',
        database='mydb',
        user='user',
        password='password'
    )

async def get_user_stats (conn, user_id):
    """Get user statistics with multiple queries"""
    
    # Execute queries concurrently
    user, orders, reviews, activity = await asyncio.gather(
        conn.fetchrow('SELECT * FROM users WHERE id = $1', user_id),
        conn.fetch('SELECT * FROM orders WHERE user_id = $1', user_id),
        conn.fetch('SELECT * FROM reviews WHERE user_id = $1', user_id),
        conn.fetch('SELECT * FROM activity WHERE user_id = $1', user_id)
    )
    
    return {
        'user': dict (user),
        'total_orders': len (orders),
        'total_reviews': len (reviews),
        'activities': [dict (a) for a in activity]
    }

async def main():
    conn = await init_db()
    
    # Get stats for multiple users concurrently
    user_ids = range(1, 101)
    stats = await asyncio.gather(
        *[get_user_stats (conn, uid) for uid in user_ids]
    )
    
    await conn.close()
    return stats

# Process 100 users in ~1 second vs ~30 seconds sequentially
\`\`\`

### Use Case 3: Web Scraping at Scale

\`\`\`python
"""
Async Web Scraper
Scrape thousands of pages efficiently
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def fetch_page (session, url):
    """Fetch a single page"""
    try:
        async with session.get (url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def parse_page (html):
    """Parse HTML (CPU-bound, but fast)"""
    soup = BeautifulSoup (html, 'html.parser')
    return {
        'title': soup.find('title').text if soup.find('title') else '',
        'links': [a['href'] for a in soup.find_all('a', href=True)]
    }

async def scrape_website (start_url, max_pages=100):
    """Scrape website starting from URL"""
    visited = set()
    to_visit = {start_url}
    results = []
    
    async with aiohttp.ClientSession() as session:
        while to_visit and len (visited) < max_pages:
            # Process up to 10 pages concurrently
            batch = [to_visit.pop() for _ in range (min(10, len (to_visit)))]
            
            # Fetch batch concurrently
            pages = await asyncio.gather(
                *[fetch_page (session, url) for url in batch]
            )
            
            for url, html in zip (batch, pages):
                if html:
                    visited.add (url)
                    data = await parse_page (html)
                    results.append({'url': url, 'data': data})
                    
                    # Add new URLs to visit
                    for link in data['links']:
                        if link not in visited:
                            to_visit.add (link)
    
    return results

# Scrape 1000 pages in 2 minutes vs 30 minutes sequentially
\`\`\`

### Use Case 4: Chat Application (WebSockets)

\`\`\`python
"""
Real-Time Chat with WebSockets
Handle thousands of concurrent connections
"""

import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

class ConnectionManager:
    """Manage active WebSocket connections"""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect (self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect (self, user_id: str):
        self.active_connections.pop (user_id, None)
    
    async def broadcast (self, message: dict):
        """Send message to all connected users"""
        # Send to all clients concurrently
        await asyncio.gather(
            *[conn.send_json (message) 
              for conn in self.active_connections.values()],
            return_exceptions=True
        )

manager = ConnectionManager()

@app.websocket("/chat/{user_id}")
async def chat_endpoint (websocket: WebSocket, user_id: str):
    """Handle chat connection"""
    await manager.connect (user_id, websocket)
    
    try:
        while True:
            # Wait for message from this user
            message = await websocket.receive_json()
            
            # Broadcast to all users (non-blocking)
            await manager.broadcast({
                'user': user_id,
                'message': message['text']
            })
    
    except WebSocketDisconnect:
        manager.disconnect (user_id)
        await manager.broadcast({
            'system': f'{user_id} left the chat'
        })

# Can handle 10,000+ concurrent connections on single server
\`\`\`

---

## Summary

### Key Takeaways

1. **Concurrency ≠ Parallelism**
   - Concurrency: Task switching (one core)
   - Parallelism: Simultaneous execution (multiple cores)

2. **I/O-Bound → Async/Threading**
   - Tasks spend time waiting for external operations
   - CPU mostly idle
   - Async provides massive speedups

3. **CPU-Bound → Multiprocessing**
   - Tasks are pure computation
   - CPU at 100%
   - Need multiple cores for speedup

4. **Async Shines For**:
   - Web servers/APIs (thousands of concurrent requests)
   - Web scraping (hundreds of concurrent fetches)
   - Database operations (multiple queries in parallel)
   - Microservices (parallel service calls)
   - Real-time apps (WebSockets, chat, streaming)

5. **Avoid Async For**:
   - CPU-intensive operations
   - Simple scripts with few I/O operations
   - When libraries don't support async

### Performance Impact

| Scenario | Sequential | Async | Speedup |
|----------|-----------|-------|---------|
| 100 API calls (500ms each) | 50s | 0.5s | 100× |
| Web scraping (1000 pages) | 30min | 2min | 15× |
| Microservices (3 calls) | 650ms | 300ms | 2.2× |
| Database queries (4 queries) | 400ms | 100ms | 4× |

### Next Steps

Now that you understand concurrency fundamentals, we'll dive into:
- Event loop internals (how async actually works)
- Coroutines and async/await syntax
- Tasks and futures
- Building real async applications

**Remember**: Async is a tool, not a requirement. Use it when it provides clear benefits for I/O-bound operations.
`,
};
