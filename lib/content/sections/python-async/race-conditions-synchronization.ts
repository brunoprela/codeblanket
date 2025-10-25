export const raceConditionsSynchronization = {
  title: 'Race Conditions & Synchronization',
  id: 'race-conditions-synchronization',
  content: `
# Race Conditions & Synchronization

## Introduction

**Race conditions** occur when multiple threads/tasks access shared state concurrently, leading to unpredictable behavior. **Synchronization primitives** (locks, semaphores, events) prevent race conditions by coordinating access.

### The Classic Race Condition

\`\`\`python
"""
Race Condition Example
"""

import threading

counter = 0

def increment():
    global counter
    for _ in range(100_000):
        counter += 1  # NOT atomic!

# Sequential: predictable
counter = 0
increment()
increment()
print(f"Sequential: {counter}")  # 200,000

# Concurrent: race condition!
counter = 0
threads = [threading.Thread (target=increment) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"Concurrent: {counter}")  # ~150,000 (unpredictable!)

# Why? counter += 1 is actually:
# 1. Read counter
# 2. Add 1
# 3. Write counter
# Threads interleave these steps, losing updates
\`\`\`

By the end of this section, you'll master:
- Identifying race conditions
- Threading locks and synchronization
- Async locks and synchronization
- Deadlock prevention
- Lock-free programming patterns

---

## Threading Locks

### Basic Lock Usage

\`\`\`python
"""
Fix Race Condition with Lock
"""

import threading

counter = 0
lock = threading.Lock()

def increment_safe():
    global counter
    for _ in range(100_000):
        with lock:
            counter += 1  # Protected by lock

# Now it's safe!
counter = 0
threads = [threading.Thread (target=increment_safe) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"With lock: {counter}")  # Always 200,000
\`\`\`

### RLock (Reentrant Lock)

\`\`\`python
"""
RLock: Lock that can be acquired multiple times by same thread
"""

import threading

lock = threading.RLock()

def recursive_function (n):
    """Acquire lock recursively"""
    with lock:
        print(f"Acquired lock at level {n}")
        if n > 0:
            recursive_function (n - 1)

# Works! Same thread can acquire RLock multiple times
thread = threading.Thread (target=recursive_function, args=(3,))
thread.start()
thread.join()

# Output:
# Acquired lock at level 3
# Acquired lock at level 2
# Acquired lock at level 1
# Acquired lock at level 0

# Regular Lock would deadlock here!
\`\`\`

---

## Threading Semaphore

### Basic Semaphore

\`\`\`python
"""
Semaphore: Allow N threads to access resource
"""

import threading
import time

# Only 3 threads can access simultaneously
semaphore = threading.Semaphore(3)

def access_resource (n):
    """Access limited resource"""
    print(f"Thread {n} waiting...")
    with semaphore:
        print(f"Thread {n} accessing resource")
        time.sleep(2)  # Use resource
    print(f"Thread {n} released resource")

# Create 10 threads, but only 3 run concurrently
threads = [threading.Thread (target=access_resource, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Useful for: Database connection pools, API rate limiting, Resource pooling
\`\`\`

---

## Threading Event

### Basic Event Usage

\`\`\`python
"""
Event: Signal between threads
"""

import threading
import time

event = threading.Event()

def worker():
    """Wait for signal"""
    print("Worker waiting for signal...")
    event.wait()  # Block until set()
    print("Worker received signal, starting work")
    time.sleep(2)
    print("Worker done")

def coordinator():
    """Send signal after delay"""
    print("Coordinator sleeping...")
    time.sleep(3)
    print("Coordinator sending signal")
    event.set()  # Wake up waiting threads

# Start threads
worker_thread = threading.Thread (target=worker)
coord_thread = threading.Thread (target=coordinator)

worker_thread.start()
coord_thread.start()

worker_thread.join()
coord_thread.join()

# Output:
# Worker waiting for signal...
# Coordinator sleeping...
# Coordinator sending signal
# Worker received signal, starting work
# Worker done
\`\`\`

---

## Async Locks

### asyncio.Lock

\`\`\`python
"""
Async Lock for Coroutines
"""

import asyncio

counter = 0
lock = asyncio.Lock()

async def increment_async():
    global counter
    for _ in range(100_000):
        async with lock:
            counter += 1

async def main():
    global counter
    counter = 0
    
    # Run 10 concurrent tasks
    await asyncio.gather(*[increment_async() for _ in range(10)])
    
    print(f"Counter: {counter}")  # Always 1,000,000

asyncio.run (main())
\`\`\`

### asyncio.Semaphore

\`\`\`python
"""
Async Semaphore: Rate Limiting
"""

import asyncio
import aiohttp

# Limit to 10 concurrent requests
semaphore = asyncio.Semaphore(10)

async def fetch_url (session, url):
    """Fetch URL with rate limiting"""
    async with semaphore:
        async with session.get (url) as response:
            return await response.text()

async def fetch_all (urls):
    """Fetch all URLs with concurrency limit"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url (session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Even with 1000 URLs, only 10 concurrent requests
urls = ['https://example.com'] * 1000
results = asyncio.run (fetch_all (urls))
\`\`\`

### asyncio.Event

\`\`\`python
"""
Async Event: Coordinate Coroutines
"""

import asyncio

async def worker (event, name):
    """Wait for event"""
    print(f"{name} waiting...")
    await event.wait()
    print(f"{name} starting work")
    await asyncio.sleep(1)
    print(f"{name} done")

async def coordinator (event):
    """Trigger event"""
    await asyncio.sleep(2)
    print("Coordinator triggering event")
    event.set()

async def main():
    event = asyncio.Event()
    
    await asyncio.gather(
        worker (event, "Worker-1"),
        worker (event, "Worker-2"),
        worker (event, "Worker-3"),
        coordinator (event)
    )

asyncio.run (main())
\`\`\`

---

## Deadlock

### Deadlock Example

\`\`\`python
"""
Deadlock: Two threads waiting for each other
"""

import threading
import time

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    """Acquire lock1, then lock2"""
    with lock1:
        print("Thread 1 acquired lock1")
        time.sleep(0.1)
        with lock2:  # Waits forever if thread2 has lock2
            print("Thread 1 acquired lock2")

def thread2():
    """Acquire lock2, then lock1"""
    with lock2:
        print("Thread 2 acquired lock2")
        time.sleep(0.1)
        with lock1:  # Waits forever if thread1 has lock1
            print("Thread 2 acquired lock1")

# Deadlock! Both threads waiting for each other
t1 = threading.Thread (target=thread1)
t2 = threading.Thread (target=thread2)
t1.start()
t2.start()
# Program hangs forever
\`\`\`

### Preventing Deadlock

\`\`\`python
"""
Prevent Deadlock: Always acquire locks in same order
"""

import threading
import time

lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1_safe():
    """Always acquire lock1, then lock2"""
    with lock1:
        print("Thread 1 acquired lock1")
        time.sleep(0.1)
        with lock2:
            print("Thread 1 acquired lock2")

def thread2_safe():
    """Also acquire lock1, then lock2 (same order!)"""
    with lock1:
        print("Thread 2 acquired lock1")
        time.sleep(0.1)
        with lock2:
            print("Thread 2 acquired lock2")

# No deadlock - same lock order
t1 = threading.Thread (target=thread1_safe)
t2 = threading.Thread (target=thread2_safe)
t1.start()
t2.start()
t1.join()
t2.join()
print("Completed without deadlock")
\`\`\`

---

## Lock-Free Patterns

### Using Queue

\`\`\`python
"""
Lock-Free: Use Queue for thread-safe communication
"""

import threading
import queue
import time

def producer (q):
    """Produce items"""
    for i in range(10):
        item = f"item-{i}"
        q.put (item)
        print(f"Produced {item}")
        time.sleep(0.1)

def consumer (q):
    """Consume items"""
    while True:
        try:
            item = q.get (timeout=1)
            print(f"Consumed {item}")
            q.task_done()
        except queue.Empty:
            break

# Thread-safe queue (no explicit locks needed!)
q = queue.Queue()

prod = threading.Thread (target=producer, args=(q,))
cons1 = threading.Thread (target=consumer, args=(q,))
cons2 = threading.Thread (target=consumer, args=(q,))

prod.start()
cons1.start()
cons2.start()

prod.join()
cons1.join()
cons2.join()
\`\`\`

### Async Queue

\`\`\`python
"""
Async Queue: Lock-Free async communication
"""

import asyncio

async def producer (q):
    """Produce items"""
    for i in range(10):
        item = f"item-{i}"
        await q.put (item)
        print(f"Produced {item}")
        await asyncio.sleep(0.1)

async def consumer (q, name):
    """Consume items"""
    while True:
        try:
            item = await asyncio.wait_for (q.get(), timeout=1)
            print(f"{name} consumed {item}")
        except asyncio.TimeoutError:
            break

async def main():
    q = asyncio.Queue()
    
    await asyncio.gather(
        producer (q),
        consumer (q, "Consumer-1"),
        consumer (q, "Consumer-2"),
    )

asyncio.run (main())
\`\`\`

---

## Production Patterns

### Thread-Safe Counter

\`\`\`python
"""
Thread-Safe Counter Class
"""

import threading

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment (self, amount=1):
        """Increment counter"""
        with self._lock:
            self._value += amount
    
    def decrement (self, amount=1):
        """Decrement counter"""
        with self._lock:
            self._value -= amount
    
    @property
    def value (self):
        """Get current value"""
        with self._lock:
            return self._value

# Usage
counter = ThreadSafeCounter()

def worker():
    for _ in range(1000):
        counter.increment()

threads = [threading.Thread (target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Final count: {counter.value}")  # Always 10,000
\`\`\`

### Async Resource Pool

\`\`\`python
"""
Async Resource Pool with Semaphore
"""

import asyncio

class ResourcePool:
    def __init__(self, size: int):
        self.semaphore = asyncio.Semaphore (size)
        self.resources = [f"Resource-{i}" for i in range (size)]
        self.available = set (self.resources)
        self.lock = asyncio.Lock()
    
    async def acquire (self):
        """Acquire resource from pool"""
        await self.semaphore.acquire()
        async with self.lock:
            resource = self.available.pop()
        return resource
    
    async def release (self, resource):
        """Release resource back to pool"""
        async with self.lock:
            self.available.add (resource)
        self.semaphore.release()

# Usage
async def use_resource (pool):
    resource = await pool.acquire()
    try:
        print(f"Using {resource}")
        await asyncio.sleep(1)
    finally:
        await pool.release (resource)

async def main():
    pool = ResourcePool (size=3)
    
    # 10 tasks, but only 3 resources available
    await asyncio.gather(*[use_resource (pool) for _ in range(10)])

asyncio.run (main())
\`\`\`

---

## Summary

### Key Concepts

1. **Race Condition**: Multiple threads accessing shared state concurrently
2. **Lock**: Mutual exclusion (only 1 thread at a time)
3. **Semaphore**: Allow N threads to access simultaneously
4. **Event**: Signal/coordinate between threads
5. **Deadlock**: Threads waiting for each other (preventable with lock ordering)

### Threading Synchronization

- \`threading.Lock\`: Basic mutual exclusion
- \`threading.RLock\`: Reentrant lock (same thread can acquire multiple times)
- \`threading.Semaphore\`: Limit concurrent access to N
- \`threading.Event\`: Signal between threads

### Async Synchronization

- \`asyncio.Lock\`: Async mutual exclusion
- \`asyncio.Semaphore\`: Async concurrency limiting
- \`asyncio.Event\`: Async coordination
- \`asyncio.Queue\`: Lock-free async communication

### Best Practices

- Use locks only when necessary (performance cost)
- Keep critical sections small
- Always acquire locks in same order (prevent deadlock)
- Prefer lock-free patterns (Queue) when possible
- Use context managers (\`with lock\`) for safety

**Remember**: Synchronization is essential for shared state in concurrent programs!
`,
};
