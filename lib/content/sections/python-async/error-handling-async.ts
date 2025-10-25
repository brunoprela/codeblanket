export const errorHandlingAsync = {
  title: 'Error Handling in Async Code',
  id: 'error-handling-async',
  content: `
# Error Handling in Async Code

## Introduction

Error handling in async code is more complex than synchronous code. Exceptions can occur in multiple concurrent tasks, during task cancellation, or in background tasks. **Proper error handling prevents silent failures and ensures application reliability.**

### Why Async Error Handling is Different

\`\`\`python
"""
Sync vs Async Error Handling
"""

import asyncio

# Synchronous: Simple try/except
def sync_function():
    try:
        result = risky_operation()
        return result
    except ValueError as e:
        print(f"Error: {e}")
        return None

# Async: More complex scenarios
async def async_function():
    # Scenario 1: Single coroutine
    try:
        result = await risky_operation()
    except ValueError as e:
        print(f"Error: {e}")
    
    # Scenario 2: Multiple concurrent tasks
    results = await asyncio.gather(
        task1(), task2(), task3(),
        return_exceptions=True  # Critical!
    )
    # results may contain exceptions!
    
    # Scenario 3: Task cancellation
    task = asyncio.create_task (long_operation())
    try:
        result = await asyncio.wait_for (task, timeout=5.0)
    except asyncio.TimeoutError:
        task.cancel()  # Must handle cancellation
\`\`\`

By the end of this section, you'll master:
- Exception handling in coroutines
- Handling errors in concurrent tasks
- Task cancellation and cleanup
- Timeout handling
- Background task errors
- Production error patterns
- Debugging async errors

---

## Basic Exception Handling

### Try/Except in Coroutines

\`\`\`python
"""
Basic Async Exception Handling
"""

import asyncio

async def fetch_data (url):
    """Fetch data with error handling"""
    try:
        # Simulated network request
        await asyncio.sleep(0.1)
        
        if 'invalid' in url:
            raise ValueError (f"Invalid URL: {url}")
        
        return f"Data from {url}"
    
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None
    
    finally:
        # Always runs (cleanup)
        print(f"✅ Cleanup for {url}")

async def main():
    result = await fetch_data("invalid-url")
    print(f"Result: {result}")

asyncio.run (main())

# Output:
# ❌ Error: Invalid URL: invalid-url
# ✅ Cleanup for invalid-url
# Result: None
\`\`\`

### Propagating Exceptions

\`\`\`python
"""
Exception Propagation in Async Code
"""

import asyncio

async def low_level_operation():
    """Low-level operation that may fail"""
    await asyncio.sleep(0.1)
    raise ConnectionError("Database connection failed")

async def mid_level_operation():
    """Mid-level: Catches and re-raises"""
    try:
        return await low_level_operation()
    except ConnectionError as e:
        print(f"Mid-level caught: {e}")
        raise  # Re-raise to higher level

async def high_level_operation():
    """High-level: Final error handler"""
    try:
        return await mid_level_operation()
    except ConnectionError as e:
        print(f"High-level handling: {e}")
        return "fallback-value"

async def main():
    result = await high_level_operation()
    print(f"Final result: {result}")

asyncio.run (main())

# Output:
# Mid-level caught: Database connection failed
# High-level handling: Database connection failed
# Final result: fallback-value
\`\`\`

---

## Handling Errors in Concurrent Tasks

### asyncio.gather() with return_exceptions

\`\`\`python
"""
Handle Errors in Multiple Concurrent Tasks
"""

import asyncio

async def task (n):
    """Task that may fail"""
    await asyncio.sleep(0.1)
    if n == 2:
        raise ValueError (f"Task {n} failed")
    return f"Task {n} succeeded"

async def without_return_exceptions():
    """❌ Bad: First exception stops everything"""
    try:
        results = await asyncio.gather(
            task(1), task(2), task(3)
        )
        print(f"Results: {results}")
    except ValueError as e:
        print(f"Error: {e}")
        # Only caught exception for task 2
        # Tasks 1 and 3 results lost!

async def with_return_exceptions():
    """✅ Good: All tasks complete, exceptions returned"""
    results = await asyncio.gather(
        task(1), task(2), task(3),
        return_exceptions=True  # Critical!
    )
    
    for i, result in enumerate (results, 1):
        if isinstance (result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

async def main():
    print("Without return_exceptions:")
    await without_return_exceptions()
    
    print("\\nWith return_exceptions:")
    await with_return_exceptions()

asyncio.run (main())

# Output:
# Without return_exceptions:
# Error: Task 2 failed
#
# With return_exceptions:
# Task 1 succeeded: Task 1 succeeded
# Task 2 failed: Task 2 failed
# Task 3 succeeded: Task 3 succeeded
\`\`\`

### Processing Results with Error Handling

\`\`\`python
"""
Robust Pattern for Handling Multiple Task Results
"""

import asyncio
from typing import List, Union

async def fetch_url (url: str) -> str:
    """Fetch URL (may fail)"""
    await asyncio.sleep(0.1)
    if 'error' in url:
        raise ConnectionError (f"Failed to fetch {url}")
    return f"Content from {url}"

async def fetch_all_urls (urls: List[str]):
    """Fetch all URLs, handle errors individually"""
    # Execute all tasks concurrently
    results = await asyncio.gather(
        *[fetch_url (url) for url in urls],
        return_exceptions=True
    )
    
    # Process results
    successes = []
    failures = []
    
    for url, result in zip (urls, results):
        if isinstance (result, Exception):
            failures.append({'url': url, 'error': str (result)})
        else:
            successes.append({'url': url, 'content': result})
    
    # Report
    print(f"✅ Successes: {len (successes)}")
    print(f"❌ Failures: {len (failures)}")
    
    return successes, failures

async def main():
    urls = [
        'https://example.com',
        'https://error.com',
        'https://another.com',
    ]
    
    successes, failures = await fetch_all_urls (urls)
    
    for failure in failures:
        print(f"Failed: {failure['url']} - {failure['error']}")

asyncio.run (main())
\`\`\`

---

## Task Cancellation

### Handling CancelledError

\`\`\`python
"""
Proper Task Cancellation Handling
"""

import asyncio

async def cancellable_task():
    """Task that handles cancellation"""
    try:
        print("Task started")
        await asyncio.sleep(10)  # Long operation
        print("Task completed")
        return "success"
    
    except asyncio.CancelledError:
        print("Task cancelled, cleaning up...")
        # Cleanup code here
        raise  # Re-raise CancelledError (important!)
    
    finally:
        print("Task cleanup")

async def main():
    # Create task
    task = asyncio.create_task (cancellable_task())
    
    # Let it run briefly
    await asyncio.sleep(0.5)
    
    # Cancel it
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Main caught cancellation")

asyncio.run (main())

# Output:
# Task started
# Task cancelled, cleaning up...
# Task cleanup
# Main caught cancellation
\`\`\`

### Graceful Shutdown

\`\`\`python
"""
Graceful Shutdown of Multiple Tasks
"""

import asyncio
import signal

class Application:
    def __init__(self):
        self.tasks = []
        self.shutdown_event = asyncio.Event()
    
    async def worker (self, name: str):
        """Worker task"""
        try:
            while not self.shutdown_event.is_set():
                print(f"{name} working...")
                await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            print(f"{name} cancelled, saving state...")
            await asyncio.sleep(0.1)  # Simulate state save
            raise
    
    async def start (self):
        """Start application"""
        # Create worker tasks
        self.tasks = [
            asyncio.create_task (self.worker (f"Worker-{i}"))
            for i in range(3)
        ]
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Cancel all tasks
        print("Shutting down...")
        for task in self.tasks:
            task.cancel()
        
        # Wait for all to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        print("Shutdown complete")
    
    def shutdown (self):
        """Trigger shutdown"""
        self.shutdown_event.set()

async def main():
    app = Application()
    
    # Setup signal handler
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGINT,
        app.shutdown
    )
    
    await app.start()

# asyncio.run (main())
# Press Ctrl+C to trigger graceful shutdown
\`\`\`

---

## Timeout Handling

### asyncio.wait_for()

\`\`\`python
"""
Timeout Handling with wait_for
"""

import asyncio

async def slow_operation():
    """Operation that takes too long"""
    await asyncio.sleep(10)
    return "completed"

async def with_timeout():
    """Execute with timeout"""
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=2.0
        )
        print(f"Result: {result}")
    
    except asyncio.TimeoutError:
        print("Operation timed out")

asyncio.run (with_timeout())

# Output: Operation timed out
\`\`\`

### Timeout with Cleanup

\`\`\`python
"""
Timeout with Resource Cleanup
"""

import asyncio

async def operation_with_resources():
    """Operation that needs cleanup"""
    try:
        print("Acquiring resources...")
        resource = "database-connection"
        
        print("Processing...")
        await asyncio.sleep(10)  # Long operation
        
        return "success"
    
    except asyncio.CancelledError:
        print("Operation cancelled, releasing resources...")
        # Cleanup code
        raise
    
    finally:
        print("Cleanup completed")

async def main():
    try:
        result = await asyncio.wait_for(
            operation_with_resources(),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        print("Timed out")

asyncio.run (main())

# Output:
# Acquiring resources...
# Processing...
# Operation cancelled, releasing resources...
# Cleanup completed
# Timed out
\`\`\`

---

## Background Task Errors

### Background Task Pattern

\`\`\`python
"""
Handle Errors in Background Tasks
"""

import asyncio

class BackgroundTaskManager:
    def __init__(self):
        self.tasks = set()
    
    def create_task (self, coro):
        """Create background task with error handling"""
        task = asyncio.create_task (coro)
        self.tasks.add (task)
        task.add_done_callback (self.tasks.discard)
        task.add_done_callback (self._handle_task_result)
        return task
    
    def _handle_task_result (self, task):
        """Handle task completion/errors"""
        try:
            # Retrieve exception if any
            exc = task.exception()
            if exc:
                print(f"❌ Background task failed: {exc}")
        except asyncio.CancelledError:
            print(f"⚠️  Background task cancelled")
    
    async def shutdown (self):
        """Shutdown all background tasks"""
        if self.tasks:
            print(f"Shutting down {len (self.tasks)} tasks...")
            for task in self.tasks:
                task.cancel()
            
            await asyncio.gather(*self.tasks, return_exceptions=True)

async def background_worker (n):
    """Background worker that may fail"""
    await asyncio.sleep(1)
    if n == 2:
        raise ValueError (f"Worker {n} failed")
    print(f"Worker {n} completed")

async def main():
    manager = BackgroundTaskManager()
    
    # Create background tasks
    for i in range(5):
        manager.create_task (background_worker (i))
    
    # Let them run
    await asyncio.sleep(2)
    
    # Shutdown
    await manager.shutdown()

asyncio.run (main())
\`\`\`

---

## Production Error Patterns

### Retry with Exponential Backoff

\`\`\`python
"""
Retry Pattern for Transient Failures
"""

import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    coro_func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args, **kwargs
) -> T:
    """Retry coroutine with exponential backoff"""
    delay = initial_delay
    
    for attempt in range (max_retries):
        try:
            return await coro_func(*args, **kwargs)
        
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt, raise
                raise
            
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {delay}s...")
            await asyncio.sleep (delay)
            delay *= backoff_factor

async def flaky_operation():
    """Operation that fails randomly"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("Connection failed")
    return "success"

async def main():
    try:
        result = await retry_with_backoff(
            flaky_operation,
            max_retries=5,
            initial_delay=1.0,
            backoff_factor=2.0
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"All retries failed: {e}")

asyncio.run (main())
\`\`\`

### Circuit Breaker Pattern

\`\`\`python
"""
Circuit Breaker for Failing Services
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
        timeout: float = 60.0,
        recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    async def call (self, coro_func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                print("🟡 Circuit HALF_OPEN, testing...")
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = await coro_func(*args, **kwargs)
            
            # Success
            if self.state == CircuitState.HALF_OPEN:
                print("✅ Circuit CLOSED, recovered")
                self.state = CircuitState.CLOSED
            
            self.failure_count = 0
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                print(f"🔴 Circuit OPEN (failures: {self.failure_count})")
                self.state = CircuitState.OPEN
            
            raise

async def unreliable_service():
    """Service that fails"""
    import random
    if random.random() < 0.8:
        raise ConnectionError("Service unavailable")
    return "success"

async def main():
    breaker = CircuitBreaker (failure_threshold=3)
    
    for i in range(10):
        try:
            result = await breaker.call (unreliable_service)
            print(f"Request {i}: {result}")
        except Exception as e:
            print(f"Request {i} failed: {e}")
        
        await asyncio.sleep(1)

asyncio.run (main())
\`\`\`

---

## Debugging Async Errors

### asyncio Debug Mode

\`\`\`python
"""
Enable asyncio Debug Mode
"""

import asyncio

async def problematic_task():
    """Task with issues"""
    await asyncio.sleep(1)
    # Forgot to await!
    asyncio.sleep(1)  # Warning in debug mode

async def main():
    await problematic_task()

# Enable debug mode
asyncio.run (main(), debug=True)

# Or set environment variable:
# PYTHONASYNCIODEBUG=1 python script.py
\`\`\`

### Logging Async Exceptions

\`\`\`python
"""
Comprehensive Async Error Logging
"""

import asyncio
import logging
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_exceptions (func):
    """Decorator to log exceptions in async functions"""
    @functools.wraps (func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception (f"Exception in {func.__name__}: {e}")
            raise
    return wrapper

@log_exceptions
async def risky_operation():
    """Operation that may fail"""
    await asyncio.sleep(0.1)
    raise ValueError("Something went wrong")

async def main():
    try:
        await risky_operation()
    except ValueError:
        pass  # Already logged

asyncio.run (main())
\`\`\`

---

## Summary

### Key Concepts

1. **return_exceptions=True**: Critical for gather() to handle all errors
2. **CancelledError**: Must handle and re-raise for proper cleanup
3. **wait_for()**: Timeout with automatic cancellation
4. **Background tasks**: Need explicit error handling (add_done_callback)
5. **Retry patterns**: Exponential backoff for transient failures
6. **Circuit breaker**: Prevent cascading failures

### Best Practices

- Always use \`return_exceptions=True\` with gather()
- Handle CancelledError and re-raise
- Use wait_for() for timeouts
- Log all async exceptions
- Implement graceful shutdown
- Use circuit breakers for external services

### Common Pitfalls

- ❌ Forgetting \`return_exceptions=True\` (loses results)
- ❌ Swallowing CancelledError (breaks cancellation)
- ❌ Ignoring background task errors (silent failures)
- ❌ No timeout on external calls (hangs forever)

### Next Steps

Now that you master async error handling, you'll learn:
- Advanced async patterns
- Testing async code
- Production deployment

**Remember**: Proper error handling is critical for reliable async applications!
`,
};
