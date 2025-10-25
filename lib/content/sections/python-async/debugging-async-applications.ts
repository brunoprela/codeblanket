export const debuggingAsyncApplications = {
  title: 'Debugging Async Applications',
  id: 'debugging-async-applications',
  content: `
# Debugging Async Applications

## Introduction

Debugging async code is challenging: tasks run concurrently, exceptions can be silenced, and traditional debugging tools don't show the full picture. **Master async debugging to build reliable production systems.**

### Common Async Bugs

\`\`\`python
"""
Common Async Bug: Forgot await
"""

import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def process():
    # ❌ Bug: Forgot await
    result = fetch_data()  # Returns coroutine, not result!
    print(result)  # <coroutine object fetch_data>
    
    # ✅ Correct
    result = await fetch_data()
    print(result)  # "data"

# asyncio debug mode catches this!
asyncio.run (process(), debug=True)
# Warning: coroutine 'fetch_data' was never awaited
\`\`\`

By the end of this section, you'll master:
- asyncio debug mode
- Logging async operations
- Tracing task execution
- Finding silent failures
- Performance profiling
- Production debugging patterns

---

## asyncio Debug Mode

### Enabling Debug Mode

\`\`\`python
"""
Enable asyncio Debug Mode
"""

import asyncio
import warnings

# Method 1: Environment variable
# PYTHONASYNCIODEBUG=1 python script.py

# Method 2: In code
asyncio.run (main(), debug=True)

# Method 3: Get event loop
loop = asyncio.get_event_loop()
loop.set_debug(True)

# Debug mode catches:
# - Coroutines not awaited
# - Slow callbacks (>100ms)
# - Tasks destroyed with pending exceptions
\`\`\`

### Debug Mode Examples

\`\`\`python
"""
What Debug Mode Catches
"""

import asyncio

async def forgot_await():
    """Bug: Forgot to await"""
    result = asyncio.sleep(1)  # ❌ Should be: await asyncio.sleep(1)
    return result

async def slow_callback():
    """Bug: Slow callback blocks event loop"""
    import time
    time.sleep(2)  # ❌ Should be: await asyncio.sleep(2)
    return "done"

async def pending_exception():
    """Bug: Task with exception never awaited"""
    async def failing_task():
        raise ValueError("Task failed")
    
    task = asyncio.create_task (failing_task())
    # Forgot to await task!

async def main():
    await forgot_await()
    await slow_callback()
    await pending_exception()

# Run with debug mode
asyncio.run (main(), debug=True)

# Output warnings:
# Warning: coroutine 'sleep' was never awaited
# Executing <Handle ...> took 2.001 seconds
# Task exception was never retrieved: ValueError
\`\`\`

---

## Logging Async Operations

### Basic Async Logging

\`\`\`python
"""
Comprehensive Async Logging
"""

import asyncio
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(task_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

def log_async (func):
    """Decorator to log async function execution"""
    @wraps (func)
    async def wrapper(*args, **kwargs):
        task_id = id (asyncio.current_task())
        logger.info (f"{func.__name__} started", extra={'task_id': task_id})
        
        try:
            result = await func(*args, **kwargs)
            logger.info (f"{func.__name__} completed", extra={'task_id': task_id})
            return result
        except Exception as e:
            logger.error(
                f"{func.__name__} failed: {e}",
                extra={'task_id': task_id},
                exc_info=True
            )
            raise
    
    return wrapper

@log_async
async def fetch_data (url):
    """Fetch data with logging"""
    await asyncio.sleep(1)
    if 'error' in url:
        raise ValueError("Bad URL")
    return f"Data from {url}"

async def main():
    await asyncio.gather(
        fetch_data("https://example.com"),
        fetch_data("https://error.com"),
        return_exceptions=True
    )

asyncio.run (main())
\`\`\`

### Structured Logging

\`\`\`python
"""
Structured Logging for Async Operations
"""

import asyncio
import json
import logging

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger (name)
    
    def log (self, level, message, **context):
        """Log with structured context"""
        task = asyncio.current_task()
        context.update({
            'task_id': id (task) if task else None,
            'task_name': task.get_name() if task else None,
        })
        
        log_entry = {
            'message': message,
            'context': context
        }
        
        self.logger.log (level, json.dumps (log_entry))

logger = StructuredLogger(__name__)

async def process_item (item_id):
    logger.log (logging.INFO, "Processing item", item_id=item_id)
    await asyncio.sleep(1)
    logger.log (logging.INFO, "Item processed", item_id=item_id, status="success")

# Logs are easily parseable for analysis
\`\`\`

---

## Task Tracing

### Tracking Active Tasks

\`\`\`python
"""
Track All Active Tasks
"""

import asyncio

async def monitor_tasks():
    """Monitor all running tasks"""
    while True:
        tasks = asyncio.all_tasks()
        print(f"Active tasks: {len (tasks)}")
        
        for task in tasks:
            print(f"  - {task.get_name()}: {task}")
        
        await asyncio.sleep(5)

async def worker (name, duration):
    """Worker task"""
    await asyncio.sleep (duration)
    return f"{name} done"

async def main():
    # Start monitor
    monitor = asyncio.create_task (monitor_tasks(), name="Monitor")
    
    # Start workers
    workers = [
        asyncio.create_task (worker (f"Worker-{i}", i), name=f"Worker-{i}")
        for i in range(5)
    ]
    
    await asyncio.gather(*workers)
    monitor.cancel()

asyncio.run (main())
\`\`\`

### Task Stack Traces

\`\`\`python
"""
Get Stack Traces for Stuck Tasks
"""

import asyncio
import traceback

async def debug_stuck_tasks():
    """Debug tasks that might be stuck"""
    while True:
        await asyncio.sleep(10)
        
        tasks = asyncio.all_tasks()
        for task in tasks:
            if not task.done():
                print(f"Task: {task.get_name()}")
                # Get stack trace
                stack = task.get_stack()
                for frame in stack:
                    traceback.print_stack (frame)

# Helps identify where tasks are stuck
\`\`\`

---

## Finding Silent Failures

### Background Task Errors

\`\`\`python
"""
Catch Silent Background Task Failures
"""

import asyncio

def create_task_with_error_handling (coro, name=None):
    """Create task with automatic error logging"""
    task = asyncio.create_task (coro, name=name)
    
    def handle_result (task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Expected
        except Exception as e:
            print(f"❌ Task {task.get_name()} failed: {e}")
    
    task.add_done_callback (handle_result)
    return task

async def failing_background_task():
    """Task that fails silently without error handling"""
    await asyncio.sleep(1)
    raise ValueError("Background task failed")

async def main():
    # ❌ Bad: Error is silent
    # task = asyncio.create_task (failing_background_task())
    
    # ✅ Good: Error is logged
    task = create_task_with_error_handling(
        failing_background_task(),
        name="BackgroundTask"
    )
    
    await asyncio.sleep(2)

asyncio.run (main())
\`\`\`

---

## Performance Profiling

### Measuring Task Duration

\`\`\`python
"""
Profile Async Task Performance
"""

import asyncio
import time
from functools import wraps

def profile_async (func):
    """Profile async function execution time"""
    @wraps (func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - start
            print(f"{func.__name__} took {duration:.3f}s")
    
    return wrapper

@profile_async
async def slow_operation():
    await asyncio.sleep(2)
    return "done"

@profile_async
async def fast_operation():
    await asyncio.sleep(0.1)
    return "done"

async def main():
    await slow_operation()
    await fast_operation()

asyncio.run (main())

# Output:
# slow_operation took 2.001s
# fast_operation took 0.101s
\`\`\`

### Detecting Slow Callbacks

\`\`\`python
"""
Detect Callbacks That Block Event Loop
"""

import asyncio
import time

async def slow_callback_detection():
    """Detect slow synchronous callbacks"""
    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = 0.1  # Warn if >100ms
    
    async def good_task():
        await asyncio.sleep(1)  # Non-blocking
    
    async def bad_task():
        time.sleep(1)  # ❌ Blocks event loop!
    
    await good_task()
    await bad_task()  # Will trigger warning in debug mode

asyncio.run (slow_callback_detection(), debug=True)
\`\`\`

---

## Production Debugging Patterns

### Health Check Endpoint

\`\`\`python
"""
Health Check with Task Status
"""

import asyncio
from datetime import datetime

class AsyncApplication:
    def __init__(self):
        self.tasks = {}
        self.start_time = datetime.now()
    
    def register_task (self, name, task):
        """Register task for monitoring"""
        self.tasks[name] = {
            'task': task,
            'started': datetime.now(),
            'status': 'running'
        }
    
    async def health_check (self):
        """Return application health status"""
        return {
            'status': 'healthy',
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'tasks': {
                name: {
                    'status': info['status'],
                    'running_time': (datetime.now() - info['started']).total_seconds(),
                    'done': info['task'].done()
                }
                for name, info in self.tasks.items()
            }
        }

# Usage in FastAPI
# @app.get("/health")
# async def health():
#     return await app.health_check()
\`\`\`

### Distributed Tracing

\`\`\`python
"""
Trace Request Across Async Operations
"""

import asyncio
import uuid
from contextvars import ContextVar

# Context variable for request tracing
request_id: ContextVar[str] = ContextVar('request_id', default=None)

async def fetch_user (user_id):
    """Fetch user with tracing"""
    trace_id = request_id.get()
    print(f"[{trace_id}] Fetching user {user_id}")
    await asyncio.sleep(0.1)
    return {'id': user_id, 'name': 'Alice'}

async def fetch_orders (user_id):
    """Fetch orders with tracing"""
    trace_id = request_id.get()
    print(f"[{trace_id}] Fetching orders for user {user_id}")
    await asyncio.sleep(0.1)
    return [{'id': 1}, {'id': 2}]

async def handle_request (user_id):
    """Handle request with tracing"""
    # Set request ID for this request
    trace_id = str (uuid.uuid4())
    request_id.set (trace_id)
    
    print(f"[{trace_id}] Request started")
    
    user, orders = await asyncio.gather(
        fetch_user (user_id),
        fetch_orders (user_id)
    )
    
    print(f"[{trace_id}] Request completed")
    return {'user': user, 'orders': orders}

# All async operations share same request_id
\`\`\`

---

## Summary

### Key Debugging Techniques

1. **Debug Mode**: \`asyncio.run (main(), debug=True)\` catches common mistakes
2. **Logging**: Log task start/end with task IDs
3. **Task Monitoring**: Track active tasks with \`asyncio.all_tasks()\`
4. **Error Handling**: Use done callbacks for background tasks
5. **Profiling**: Measure task duration to find bottlenecks

### Common Bugs

- Forgot \`await\` (coroutine never executes)
- Blocking calls (\`time.sleep\` instead of \`asyncio.sleep\`)
- Silent background task failures
- Slow callbacks blocking event loop

### Production Patterns

- Health check endpoints with task status
- Structured logging with request IDs
- Distributed tracing with context variables
- Automatic error logging for background tasks

### Best Practices

- Always run with debug mode in development
- Log all async operations with context
- Monitor active tasks in production
- Profile performance regularly
- Handle all background task errors

**Remember**: Debug mode is your friend—use it in development to catch async bugs early!
`,
};
