export const asyncDesignPatterns = {
  title: 'Async Design Patterns',
  id: 'async-design-patterns',
  content: `
# Async Design Patterns

## Introduction

Async design patterns are reusable solutions to common concurrency problems. **Master these patterns to build scalable, maintainable async applications.**

### Pattern Overview

\`\`\`python
"""
Common Async Patterns
"""

# 1. Producer-Consumer: Decouple production and consumption
# 2. Worker Pool: Process tasks with fixed workers
# 3. Rate Limiter: Control request rate
# 4. Circuit Breaker: Prevent cascading failures
# 5. Retry with Backoff: Handle transient failures
# 6. Timeout Wrapper: Add timeouts to operations
# 7. Pub/Sub: Event-driven architecture
# 8. Pipeline: Multi-stage processing
\`\`\`

By the end of this section, you'll master:
- Producer-Consumer pattern
- Worker pool pattern
- Rate limiting strategies
- Circuit breaker implementation
- Retry patterns
- Pipeline processing
- Real-world pattern combinations

---

## Producer-Consumer Pattern

### Basic Implementation

\`\`\`python
"""
Producer-Consumer with asyncio.Queue
"""

import asyncio
import random

async def producer (queue, n_items):
    """Produce items"""
    for i in range (n_items):
        item = f"item-{i}"
        await queue.put (item)
        print(f"Produced {item}")
        await asyncio.sleep (random.uniform(0.1, 0.5))

    # Signal completion
    await queue.put(None)

async def consumer (queue, name):
    """Consume items"""
    while True:
        item = await queue.get()

        if item is None:
            # End signal
            await queue.put(None)  # Pass to other consumers
            break

        print(f"{name} consuming {item}")
        await asyncio.sleep (random.uniform(0.2, 0.6))
        queue.task_done()

async def main():
    queue = asyncio.Queue (maxsize=10)  # Bounded queue

    # Start producer and consumers
    await asyncio.gather(
        producer (queue, 20),
        consumer (queue, "Consumer-1"),
        consumer (queue, "Consumer-2"),
        consumer (queue, "Consumer-3"),
    )

asyncio.run (main())
\`\`\`

---

## Worker Pool Pattern

### Fixed Worker Pool

\`\`\`python
"""
Worker Pool for Task Processing
"""

import asyncio

class WorkerPool:
    def __init__(self, n_workers):
        self.queue = asyncio.Queue()
        self.workers = []
        self.n_workers = n_workers

    async def worker (self, worker_id):
        """Worker that processes tasks"""
        while True:
            task = await self.queue.get()

            if task is None:
                break

            try:
                func, args, kwargs = task
                result = await func(*args, **kwargs)
                print(f"Worker-{worker_id} completed task")
            except Exception as e:
                print(f"Worker-{worker_id} error: {e}")
            finally:
                self.queue.task_done()

    async def start (self):
        """Start all workers"""
        self.workers = [
            asyncio.create_task (self.worker (i))
            for i in range (self.n_workers)
        ]

    async def submit (self, func, *args, **kwargs):
        """Submit task to pool"""
        await self.queue.put((func, args, kwargs))

    async def shutdown (self):
        """Shutdown pool"""
        # Send stop signals
        for _ in range (self.n_workers):
            await self.queue.put(None)

        # Wait for workers
        await asyncio.gather(*self.workers)

# Usage
async def task (n):
    await asyncio.sleep(1)
    return n * 2

async def main():
    pool = WorkerPool (n_workers=5)
    await pool.start()

    # Submit tasks
    for i in range(20):
        await pool.submit (task, i)

    # Wait for completion
    await pool.queue.join()
    await pool.shutdown()

asyncio.run (main())
\`\`\`

---

## Rate Limiter Pattern

### Token Bucket Rate Limiter

\`\`\`python
"""
Token Bucket Rate Limiter
"""

import asyncio
import time

class RateLimiter:
    def __init__(self, rate: int, per: float = 1.0):
        """
        rate: Number of operations
        per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()

    async def acquire (self):
        """Acquire permission to proceed"""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current

            # Add tokens based on time passed
            self.allowance += time_passed * (self.rate / self.per)

            if self.allowance > self.rate:
                self.allowance = self.rate

            if self.allowance < 1.0:
                # Not enough tokens, wait
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep (sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

# Usage
async def api_call (limiter, n):
    await limiter.acquire()
    print(f"Request {n} at {time.time():.2f}")
    # Make actual API call

async def main():
    limiter = RateLimiter (rate=10, per=1.0)  # 10 requests/second

    await asyncio.gather(*[api_call (limiter, i) for i in range(50)])

asyncio.run (main())
\`\`\`

---

## Circuit Breaker Pattern

### Production Circuit Breaker

\`\`\`python
"""
Circuit Breaker Pattern
"""

import asyncio
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = asyncio.Lock()

    async def call (self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        async with self.lock:
            if self.state == CircuitState.OPEN:
                # Check if should try again
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    print("Circuit HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            # Success
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    print("Circuit CLOSED")
                self.failure_count = 0

            return result

        except self.expected_exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    print(f"Circuit OPEN (failures: {self.failure_count})")

            raise

# Usage
breaker = CircuitBreaker (failure_threshold=3)

async def unreliable_api():
    import random
    if random.random() < 0.7:
        raise ConnectionError("API unavailable")
    return "success"

async def main():
    for i in range(20):
        try:
            result = await breaker.call (unreliable_api)
            print(f"Request {i}: {result}")
        except Exception as e:
            print(f"Request {i}: {e}")

        await asyncio.sleep(1)

asyncio.run (main())
\`\`\`

---

## Retry Pattern

### Exponential Backoff Retry

\`\`\`python
"""
Retry with Exponential Backoff
"""

import asyncio
import random
from typing import Callable, TypeVar

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    *args,
    **kwargs
) -> T:
    """Retry function with exponential backoff"""
    delay = initial_delay

    for attempt in range (max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            # Add jitter to prevent thundering herd
            if jitter:
                actual_delay = delay * (0.5 + random.random())
            else:
                actual_delay = delay

            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {actual_delay:.2f}s...")

            await asyncio.sleep (actual_delay)
            delay *= backoff_factor

# Usage
async def flaky_operation():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Temporary failure")
    return "success"

async def main():
    result = await retry_with_backoff(
        flaky_operation,
        max_attempts=5,
        initial_delay=1.0,
        backoff_factor=2.0
    )
    print(f"Final result: {result}")

asyncio.run (main())
\`\`\`

---

## Pipeline Pattern

### Multi-Stage Pipeline

\`\`\`python
"""
Multi-Stage Processing Pipeline
"""

import asyncio

class Pipeline:
    def __init__(self):
        self.stages = []

    def add_stage (self, func):
        """Add processing stage"""
        self.stages.append (func)
        return self

    async def process (self, items):
        """Process items through pipeline"""
        queue_in = asyncio.Queue()
        queue_out = asyncio.Queue()

        # Load initial items
        for item in items:
            await queue_in.put (item)
        await queue_in.put(None)  # End marker

        # Process through stages
        for stage_func in self.stages:
            await self._process_stage (stage_func, queue_in, queue_out)
            queue_in = queue_out
            queue_out = asyncio.Queue()

        # Collect results
        results = []
        while True:
            item = await queue_in.get()
            if item is None:
                break
            results.append (item)

        return results

    async def _process_stage (self, func, queue_in, queue_out):
        """Process single stage"""
        while True:
            item = await queue_in.get()

            if item is None:
                await queue_out.put(None)
                break

            try:
                result = await func (item)
                await queue_out.put (result)
            except Exception as e:
                print(f"Stage error: {e}")

# Usage
async def stage1(item):
    """Parse"""
    await asyncio.sleep(0.1)
    return {'parsed': item}

async def stage2(item):
    """Validate"""
    await asyncio.sleep(0.1)
    item['valid'] = True
    return item

async def stage3(item):
    """Transform"""
    await asyncio.sleep(0.1)
    item['transformed'] = True
    return item

async def main():
    pipeline = Pipeline()
    pipeline.add_stage (stage1)
    pipeline.add_stage (stage2)
    pipeline.add_stage (stage3)

    items = list (range(10))
    results = await pipeline.process (items)
    print(f"Results: {len (results)} items processed")

asyncio.run (main())
\`\`\`

---

## Summary

### Key Patterns

1. **Producer-Consumer**: Decouple production/consumption with Queue
2. **Worker Pool**: Fixed workers processing tasks
3. **Rate Limiter**: Token bucket for API rate limiting
4. **Circuit Breaker**: Fail fast when service down
5. **Retry**: Exponential backoff for transient failures
6. **Pipeline**: Multi-stage processing

### When to Use Each

- **Producer-Consumer**: Different speeds (fast producer, slow consumer)
- **Worker Pool**: Limited concurrency, task processing
- **Rate Limiter**: API rate limits, resource protection
- **Circuit Breaker**: External dependencies, prevent cascading failures
- **Retry**: Network operations, transient failures
- **Pipeline**: ETL, multi-stage data processing

### Combining Patterns

- Worker Pool + Rate Limiter: Process tasks with rate limiting
- Circuit Breaker + Retry: Retry with circuit breaker protection
- Pipeline + Worker Pool: Parallel processing at each stage

**Remember**: Choose patterns based on your concurrency requirements!
`,
};
