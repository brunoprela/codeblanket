export const tasksAndFutures = {
  title: 'Tasks and Futures',
  id: 'tasks-and-futures',
  content: `
# Tasks and Futures

## Introduction

While coroutines are the building blocks of async code, **Tasks** are how you actually run them concurrently. A Task wraps a coroutine and schedules it to run on the event loop, allowing you to start multiple operations without immediately waiting for them.

**Futures** are lower-level objects that represent a value that will be available in the future. Understanding both is essential for advanced async programming.

### The Power of Tasks

\`\`\`python
"""
Without Tasks: Sequential Execution
"""
import asyncio
import time

async def fetch_data (n):
    await asyncio.sleep(1)
    return f"Data {n}"

async def without_tasks():
    start = time.time()
    
    # Waiting for each one sequentially
    result1 = await fetch_data(1)  # Wait 1s
    result2 = await fetch_data(2)  # Wait another 1s
    result3 = await fetch_data(3)  # Wait another 1s
    
    print(f"Time: {time.time() - start:.2f}s")  # ~3.0s
    return [result1, result2, result3]

"""
With Tasks: Concurrent Execution
"""
async def with_tasks():
    start = time.time()
    
    # Create tasks (starts execution immediately!)
    task1 = asyncio.create_task (fetch_data(1))
    task2 = asyncio.create_task (fetch_data(2))
    task3 = asyncio.create_task (fetch_data(3))
    
    # All three are running now
    # Wait for completion
    results = await asyncio.gather (task1, task2, task3)
    
    print(f"Time: {time.time() - start:.2f}s")  # ~1.0s
    return results

asyncio.run (without_tasks())  # ~3.0s
asyncio.run (with_tasks())  # ~1.0s (3× faster!)
\`\`\`

By the end of this section, you'll understand:
- Creating and managing tasks
- Task cancellation and timeout handling
- Task groups (Python 3.11+)
- Gathering and waiting for multiple tasks
- Futures and their relationship to tasks
- Best practices for concurrent task execution
- Common pitfalls and debugging techniques

---

## Creating Tasks

Tasks are created with \`asyncio.create_task()\` and begin executing immediately on the event loop.

### Basic Task Creation

\`\`\`python
"""
Creating and Awaiting Tasks
"""
import asyncio

async def my_coroutine (n):
    print(f"Coroutine {n} starting")
    await asyncio.sleep(1)
    print(f"Coroutine {n} finishing")
    return n * 2

async def main():
    # Create a task (starts running immediately!)
    task = asyncio.create_task (my_coroutine(5))
    
    print("Task created, doing other work...")
    await asyncio.sleep(0.5)
    print("Other work done")
    
    # Now wait for the task to complete
    result = await task
    print(f"Result: {result}")

asyncio.run (main())

# Output:
# Coroutine 5 starting
# Task created, doing other work...
# Other work done
# Coroutine 5 finishing
# Result: 10

# Key insight: Task starts running as soon as created, not when awaited!
\`\`\`

### Multiple Tasks

\`\`\`python
"""
Creating Multiple Tasks
"""
import asyncio
import time

async def fetch_user (user_id):
    """Simulate fetching user data"""
    await asyncio.sleep(0.5)
    return {'id': user_id, 'name': f'User {user_id}'}

async def main():
    start = time.time()
    
    # Create multiple tasks
    tasks = [
        asyncio.create_task (fetch_user (i))
        for i in range(10)
    ]
    
    # All 10 tasks are now running concurrently!
    print(f"Created {len (tasks)} tasks")
    
    # Wait for all to complete
    users = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"Fetched {len (users)} users in {elapsed:.2f}s")

asyncio.run (main())

# Output: Fetched 10 users in 0.50s
# (vs 5.0s if done sequentially!)
\`\`\`

### Task Naming (Python 3.8+)

\`\`\`python
"""
Naming Tasks for Debugging
"""
import asyncio

async def background_task():
    while True:
        await asyncio.sleep(1)
        print("Background task running...")

async def main():
    # Create task with descriptive name
    task = asyncio.create_task(
        background_task(),
        name="background-worker-1"
    )
    
    # Get task name
    print(f"Task name: {task.get_name()}")
    
    # Change task name
    task.set_name("worker-1-renamed")
    
    await asyncio.sleep(3)
    task.cancel()

# Useful for debugging with many concurrent tasks!
\`\`\`

---

## Task States and Lifecycle

Tasks progress through several states during their lifecycle.

### Task States

\`\`\`python
"""
Understanding Task States
"""
import asyncio

async def slow_operation():
    await asyncio.sleep(2)
    return "Done"

async def main():
    # Create task
    task = asyncio.create_task (slow_operation())
    
    # Check if done
    print(f"Done: {task.done()}")  # False (just started)
    
    # Check if cancelled
    print(f"Cancelled: {task.cancelled()}")  # False
    
    # Wait a bit
    await asyncio.sleep(0.1)
    print(f"Done after 0.1s: {task.done()}")  # Still False
    
    # Wait for completion
    result = await task
    
    print(f"Done after await: {task.done()}")  # True
    print(f"Result: {task.result()}")  # "Done"
    print(f"Exception: {task.exception()}")  # None (no error)

asyncio.run (main())
\`\`\`

### Getting Task Results

\`\`\`python
"""
Multiple Ways to Get Task Results
"""
import asyncio

async def compute (n):
    await asyncio.sleep(0.5)
    return n ** 2

async def main():
    task = asyncio.create_task (compute(5))
    
    # Method 1: Await the task
    result = await task
    print(f"Method 1: {result}")  # 25
    
    # Method 2: Get result after task is done
    task2 = asyncio.create_task (compute(10))
    await task2  # Wait for completion
    result = task2.result()  # Get result without awaiting again
    print(f"Method 2: {result}")  # 100
    
    # ❌ Don't do this: Get result before task is done
    task3 = asyncio.create_task (compute(15))
    try:
        result = task3.result()  # Task not done yet!
    except asyncio.InvalidStateError as e:
        print(f"Error: {e}")  # InvalidStateError: Result is not set

asyncio.run (main())
\`\`\`

### Task Exceptions

\`\`\`python
"""
Handling Task Exceptions
"""
import asyncio

async def failing_task():
    await asyncio.sleep(0.5)
    raise ValueError("Something went wrong!")

async def main():
    task = asyncio.create_task (failing_task())
    
    try:
        await task
    except ValueError as e:
        print(f"Caught exception: {e}")
    
    # After catching, can check exception on task object
    print(f"Task exception: {task.exception()}")
    
    # Task is done but with exception
    print(f"Task done: {task.done()}")  # True
    print(f"Task cancelled: {task.cancelled()}")  # False

asyncio.run (main())

# Important: If you don't await a task that raises an exception,
# Python will show a warning when the task is garbage collected:
# "Task exception was never retrieved"
\`\`\`

---

## Task Cancellation

Cancelling tasks is essential for handling timeouts and shutting down operations.

### Basic Cancellation

\`\`\`python
"""
Cancelling a Task
"""
import asyncio

async def long_running_task():
    try:
        print("Task starting...")
        await asyncio.sleep(10)  # Long operation
        print("Task completed")
        return "result"
    except asyncio.CancelledError:
        print("Task was cancelled!")
        raise  # Must re-raise!

async def main():
    task = asyncio.create_task (long_running_task())
    
    # Let it run for 1 second
    await asyncio.sleep(1)
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Confirmed: Task cancelled")

asyncio.run (main())

# Output:
# Task starting...
# Task was cancelled!
# Confirmed: Task cancelled
\`\`\`

### Graceful Cancellation with Cleanup

\`\`\`python
"""
Cancellation with Resource Cleanup
"""
import asyncio

class DatabaseConnection:
    def __init__(self):
        self.closed = False
    
    async def close (self):
        print("Closing database connection...")
        await asyncio.sleep(0.1)
        self.closed = True
        print("Connection closed")

async def database_task (conn):
    try:
        print("Starting database operations...")
        await asyncio.sleep(10)
        return "Success"
    except asyncio.CancelledError:
        print("Database task cancelled, cleaning up...")
        await conn.close()
        raise  # Important: Re-raise CancelledError!

async def main():
    conn = DatabaseConnection()
    task = asyncio.create_task (database_task (conn))
    
    await asyncio.sleep(0.5)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Task fully cancelled and cleaned up")
    
    print(f"Connection closed: {conn.closed}")

asyncio.run (main())

# Output:
# Starting database operations...
# Database task cancelled, cleaning up...
# Closing database connection...
# Connection closed
# Task fully cancelled and cleaned up
# Connection closed: True
\`\`\`

### Cancelling Multiple Tasks

\`\`\`python
"""
Cancelling a Group of Tasks
"""
import asyncio

async def worker (n):
    try:
        while True:
            print(f"Worker {n} working...")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print(f"Worker {n} cancelled")
        raise

async def main():
    # Create multiple workers
    tasks = [
        asyncio.create_task (worker (i))
        for i in range(5)
    ]
    
    # Let them work for 3 seconds
    await asyncio.sleep(3)
    
    # Cancel all tasks
    print("Cancelling all workers...")
    for task in tasks:
        task.cancel()
    
    # Wait for all to finish cancelling
    await asyncio.gather(*tasks, return_exceptions=True)
    
    print("All workers cancelled")

asyncio.run (main())
\`\`\`

---

## Timeouts

Timeouts are implemented using task cancellation.

### Simple Timeout

\`\`\`python
"""
Timeout with asyncio.wait_for()
"""
import asyncio

async def slow_operation():
    print("Operation starting...")
    await asyncio.sleep(5)
    print("Operation complete")
    return "result"

async def main():
    try:
        # Timeout after 2 seconds
        result = await asyncio.wait_for (slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run (main())

# Output:
# Operation starting...
# Operation timed out!  (after 2 seconds)
\`\`\`

### Timeout Context Manager (Python 3.11+)

\`\`\`python
"""
Timeout with Context Manager
"""
import asyncio

async def fetch_data():
    async with asyncio.timeout(2.0):  # Python 3.11+
        # This block has 2 second timeout
        await asyncio.sleep(5)
        return "data"

async def main():
    try:
        result = await fetch_data()
    except asyncio.TimeoutError:
        print("Timed out!")

# For Python <3.11, use asyncio.wait_for()
\`\`\`

### Timeout with Partial Results

\`\`\`python
"""
Get Results from Non-Timeout Tasks
"""
import asyncio

async def fetch_with_delay (n, delay):
    await asyncio.sleep (delay)
    return f"Result {n}"

async def main():
    tasks = [
        asyncio.create_task (fetch_with_delay(1, 0.5)),
        asyncio.create_task (fetch_with_delay(2, 1.5)),
        asyncio.create_task (fetch_with_delay(3, 2.5)),
    ]
    
    # Wait up to 2 seconds
    done, pending = await asyncio.wait(
        tasks,
        timeout=2.0
    )
    
    # Get results from completed tasks
    results = [task.result() for task in done]
    print(f"Completed: {results}")
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
    
    print(f"Cancelled: {len (pending)} tasks")

asyncio.run (main())

# Output:
# Completed: ['Result 1', 'Result 2']
# Cancelled: 1 tasks
\`\`\`

---

## Gathering and Waiting

Multiple strategies for waiting on multiple tasks.

### asyncio.gather()

\`\`\`python
"""
Gather: Wait for All Tasks
"""
import asyncio

async def fetch_data (source):
    await asyncio.sleep (source * 0.5)
    return f"Data from {source}"

async def main():
    # Wait for all tasks to complete
    results = await asyncio.gather(
        fetch_data(1),
        fetch_data(2),
        fetch_data(3),
    )
    
    print(f"Results: {results}")
    # Results: ['Data from 1', 'Data from 2', 'Data from 3']
    
    # With error handling
    async def might_fail (n):
        if n == 2:
            raise ValueError("Failed!")
        return n * 10
    
    results = await asyncio.gather(
        might_fail(1),
        might_fail(2),
        might_fail(3),
        return_exceptions=True  # Don't stop on first error
    )
    
    for i, result in enumerate (results):
        if isinstance (result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

asyncio.run (main())

# Output:
# Results: ['Data from 1', 'Data from 2', 'Data from 3']
# Task 0 succeeded: 10
# Task 1 failed: Failed!
# Task 2 succeeded: 30
\`\`\`

### asyncio.wait()

\`\`\`python
"""
Wait: More Control Than Gather
"""
import asyncio

async def worker (n):
    await asyncio.sleep (n)
    return n ** 2

async def main():
    tasks = [
        asyncio.create_task (worker(1)),
        asyncio.create_task (worker(2)),
        asyncio.create_task (worker(3)),
    ]
    
    # Wait for all to complete
    done, pending = await asyncio.wait (tasks)
    
    results = [task.result() for task in done]
    print(f"All done: {sorted (results)}")
    
    # Wait for first to complete
    tasks = [
        asyncio.create_task (worker(1)),
        asyncio.create_task (worker(2)),
        asyncio.create_task (worker(3)),
    ]
    
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    
    print(f"First done: {list (done)[0].result()}")
    
    # Cancel remaining
    for task in pending:
        task.cancel()

asyncio.run (main())
\`\`\`

### as_completed()

\`\`\`python
"""
Process Results as They Complete
"""
import asyncio
import time

async def fetch_with_delay (n, delay):
    await asyncio.sleep (delay)
    return f"Result {n}"

async def main():
    tasks = [
        fetch_with_delay(1, 2.0),
        fetch_with_delay(2, 0.5),
        fetch_with_delay(3, 1.0),
    ]
    
    # Process results in completion order (not submission order!)
    for coro in asyncio.as_completed (tasks):
        result = await coro
        print(f"{time.time():.1f}: {result}")

asyncio.run (main())

# Output:
# 0.5: Result 2  (completed first)
# 1.0: Result 3  (completed second)
# 2.0: Result 1  (completed last)
\`\`\`

---

## Task Groups (Python 3.11+)

Task groups provide structured concurrency with automatic cancellation.

### Basic Task Groups

\`\`\`python
"""
Task Groups for Structured Concurrency
"""
import asyncio

async def worker (n):
    await asyncio.sleep (n)
    return n * 10

async def main():
    async with asyncio.TaskGroup() as tg:
        # All tasks added to group
        task1 = tg.create_task (worker(1))
        task2 = tg.create_task (worker(2))
        task3 = tg.create_task (worker(3))
    
    # When exiting, all tasks automatically awaited
    print(f"Results: {task1.result()}, {task2.result()}, {task3.result()}")

# Python 3.11+ required
# asyncio.run (main())
\`\`\`

### Task Groups with Error Handling

\`\`\`python
"""
Task Group Cancellation on Error
"""
import asyncio

async def worker (n):
    await asyncio.sleep (n)
    if n == 2:
        raise ValueError("Worker 2 failed!")
    return n * 10

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task (worker(1))
            task2 = tg.create_task (worker(2))  # Will fail
            task3 = tg.create_task (worker(3))
    except* ValueError as eg:  # Exception group
        print(f"Caught errors: {eg.exceptions}")
    
    # When task2 fails, task1 and task3 are automatically cancelled
    # This is "structured concurrency"

# Key benefit: No orphaned tasks!
\`\`\`

---

## Futures

Futures are lower-level than tasks, representing values that will be available later.

### Creating Futures

\`\`\`python
"""
Working with Futures
"""
import asyncio

async def main():
    loop = asyncio.get_running_loop()
    
    # Create a future
    future = loop.create_future()
    
    # Schedule result to be set later
    def set_result():
        future.set_result("Hello from the future!")
    
    loop.call_later(2.0, set_result)
    
    print("Waiting for future...")
    result = await future
    print(f"Got: {result}")

asyncio.run (main())
\`\`\`

### Bridging Callbacks to Async

\`\`\`python
"""
Converting Callback-Based Code to Async
"""
import asyncio

class LegacyAPI:
    """Old callback-based API"""
    def fetch_data (self, callback):
        """Simulate async operation with callback"""
        loop = asyncio.get_event_loop()
        loop.call_later(1.0, callback, "legacy data")

async def fetch_data_async (api):
    """Wrap callback API for async use"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    def callback (data):
        future.set_result (data)
    
    api.fetch_data (callback)
    return await future

async def main():
    api = LegacyAPI()
    result = await fetch_data_async (api)
    print(f"Result: {result}")

asyncio.run (main())
\`\`\`

---

## Best Practices

### DO: Create Tasks Early

\`\`\`python
# ✅ Good: Start tasks early
async def main():
    # Create all tasks immediately (they start running)
    task1 = asyncio.create_task (operation1())
    task2 = asyncio.create_task (operation2())
    
    # Do other work...
    
    # Then wait for results
    result1, result2 = await asyncio.gather (task1, task2)
\`\`\`

### DON'T: Await Immediately

\`\`\`python
# ❌ Bad: Await immediately (sequential)
async def main():
    result1 = await operation1()  # Wait here
    result2 = await operation2()  # Then wait here
    
    # Operations run sequentially, not concurrently!
\`\`\`

### DO: Name Your Tasks

\`\`\`python
# ✅ Good: Named tasks for debugging
task = asyncio.create_task(
    fetch_data (user_id),
    name=f"fetch-user-{user_id}"
)
\`\`\`

### DO: Handle Cancellation

\`\`\`python
# ✅ Good: Proper cancellation handling
async def my_task():
    try:
        await long_operation()
    except asyncio.CancelledError:
        await cleanup()
        raise  # Must re-raise!
\`\`\`

---

## Summary

### Key Takeaways

1. **Tasks vs Coroutines**
   - Coroutine: Function definition (async def)
   - Task: Wrapped coroutine running on event loop
   - Create tasks with \`asyncio.create_task()\`

2. **Task Execution**
   - Tasks start immediately when created
   - Use \`await\` to wait for completion
   - Multiple tasks run concurrently

3. **Cancellation**
   - Cancel with \`task.cancel()\`
   - Handle \`CancelledError\` for cleanup
   - Always re-raise \`CancelledError\`

4. **Gathering Results**
   - \`asyncio.gather()\`: Wait for all, get results in order
   - \`asyncio.wait()\`: More control, returns (done, pending)
   - \`asyncio.as_completed()\`: Process in completion order

5. **Timeouts**
   - \`asyncio.wait_for()\`: Timeout for single operation
   - \`asyncio.timeout()\`: Context manager (Python 3.11+)
   - Timeout triggers cancellation

6. **Task Groups** (Python 3.11+)
   - Structured concurrency
   - Automatic cancellation on error
   - Prevents orphaned tasks

7. **Futures**
   - Lower-level than tasks
   - Bridge callback-based code to async
   - Usually use tasks instead

### Next Steps

Now that you master tasks and futures, we'll explore:
- Async context managers for resource management
- More asyncio built-in functions
- Real-world patterns for concurrent operations
- Error handling strategies at scale

**Remember**: Tasks are the key to concurrency in async Python. Master task creation, cancellation, and gathering for effective async programming.
`,
};
