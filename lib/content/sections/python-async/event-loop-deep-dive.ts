export const eventLoopDeepDive = {
  title: 'Event Loop Deep Dive',
  id: 'event-loop-deep-dive',
  content: `
# Event Loop Deep Dive

## Introduction

The **event loop** is the heart of asyncio—it's the engine that makes asynchronous programming work. Understanding how the event loop operates is crucial for writing efficient async code and debugging performance issues.

### What Problem Does the Event Loop Solve?

Consider this scenario: You have 100 tasks, each needs to make a network request (which takes 1 second). How can one thread handle all 100 tasks?

**Answer**: The event loop continuously checks which tasks are ready to run and switches between them:

\`\`\`
Time  Task 1          Task 2          Task 3
0s    Start request → Start request → Start request →
      (waiting...)    (waiting...)    (waiting...)
1s    ← Response      ← Response      ← Response
      Process data    Process data    Process data
\`\`\`

The event loop doesn't wait idly—it manages hundreds of tasks, switching between them whenever they're ready.

By the end of this section, you'll understand:
- How the event loop orchestrates async tasks
- Event loop internal architecture
- Running and managing the event loop
- Event loop policies and customization
- Task scheduling mechanisms
- Callbacks and futures
- Performance characteristics and optimization

---

## What is an Event Loop?

The event loop is a **programming construct that waits for and dispatches events or messages in a program**. It continuously checks for work to do and executes it.

### Core Concept

\`\`\`python
"""
Simplified Event Loop Concept
(This is what asyncio does internally)
"""

def simple_event_loop():
    """Simplified event loop implementation"""
    tasks = []  # List of pending tasks
    
    while tasks:
        # Check each task
        for task in tasks[:]:  # Copy list to allow modification
            if task.is_ready():
                # Task is ready to run
                try:
                    task.step()  # Execute next step
                except StopIteration:
                    # Task completed
                    tasks.remove (task)
                except Exception as e:
                    # Task failed
                    task.set_exception (e)
                    tasks.remove (task)
            else:
                # Task waiting for I/O, check later
                pass
        
        # Brief pause to avoid busy-waiting
        time.sleep(0.001)  # 1ms

# This is approximately what asyncio.run() does!
\`\`\`

### Real Event Loop

\`\`\`python
"""
Actual asyncio Event Loop Usage
"""

import asyncio

async def task1():
    print("Task 1: Starting")
    await asyncio.sleep(1)  # "I'm waiting, event loop can run other tasks"
    print("Task 1: Done")
    return "Result 1"

async def task2():
    print("Task 2: Starting")
    await asyncio.sleep(0.5)
    print("Task 2: Done")
    return "Result 2"

async def main():
    """Main coroutine that creates tasks"""
    # Create tasks (registers them with event loop)
    t1 = asyncio.create_task (task1())
    t2 = asyncio.create_task (task2())
    
    # Event loop will manage both tasks
    results = await asyncio.gather (t1, t2)
    print(f"Results: {results}")

# Start the event loop and run main
asyncio.run (main())

# Output:
# Task 1: Starting
# Task 2: Starting
# Task 2: Done (after 0.5s)
# Task 1: Done (after 1s)
# Results: ['Result 1', 'Result 2']
\`\`\`

**Key Insight**: The event loop runs in a single thread, but manages multiple tasks by switching between them at await points.

---

## Event Loop Architecture

The asyncio event loop has several components working together:

\`\`\`
┌─────────────────────────────────────┐
│       Application Code              │
│   (async functions/coroutines)      │
└──────────────┬──────────────────────┘
               │ await
               ↓
┌─────────────────────────────────────┐
│         Event Loop                  │
│  ┌──────────────────────────────┐   │
│  │   Ready Queue                │   │
│  │   (tasks ready to run)       │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │   I/O Selector               │   │
│  │   (monitors file descriptors) │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │   Scheduled Callbacks        │   │
│  │   (timers, delayed calls)    │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│      Operating System               │
│  (network I/O, file I/O, etc.)      │
└─────────────────────────────────────┘
\`\`\`

### Event Loop Cycle

\`\`\`python
"""
Event Loop Execution Cycle
"""

def event_loop_cycle():
    """
    One iteration of the event loop (simplified)
    """
    
    # Phase 1: Run all ready tasks
    while ready_queue:
        task = ready_queue.pop(0)
        task.step()  # Run until next await
    
    # Phase 2: Check for completed I/O
    ready_ios = io_selector.select (timeout=0)
    for io in ready_ios:
        task = io.task
        ready_queue.append (task)
    
    # Phase 3: Run scheduled callbacks
    now = time.time()
    while scheduled_callbacks and scheduled_callbacks[0].when <= now:
        callback = scheduled_callbacks.pop(0)
        callback.run()
    
    # Phase 4: Check for idle work
    if not ready_queue and not scheduled_callbacks:
        # Wait for I/O with timeout
        ready_ios = io_selector.select (timeout=next_timer_time)

# This cycle repeats until no more work
\`\`\`

### Practical Example: Seeing the Event Loop Work

\`\`\`python
"""
Visualizing Event Loop Operations
"""

import asyncio
import time

class TracingEventLoop:
    """Wrapper to trace event loop operations"""
    
    def __init__(self, loop):
        self.loop = loop
        self.call_count = 0
    
    async def trace_task (self, name, delay):
        """Task that prints each step"""
        self.call_count += 1
        print(f"[{time.time():.2f}] {name}: Starting (call #{self.call_count})")
        
        await asyncio.sleep (delay)
        
        self.call_count += 1
        print(f"[{time.time():.2f}] {name}: Resumed (call #{self.call_count})")
        
        return f"{name} result"

async def main():
    """Run multiple tasks and see event loop switching"""
    tracer = TracingEventLoop (asyncio.get_event_loop())
    
    start = time.time()
    
    # Create three tasks with different delays
    results = await asyncio.gather(
        tracer.trace_task("Task-A", 0.3),
        tracer.trace_task("Task-B", 0.1),
        tracer.trace_task("Task-C", 0.2),
    )
    
    elapsed = time.time() - start
    
    print(f"\\nCompleted in {elapsed:.2f}s")
    print(f"Total event loop calls: {tracer.call_count}")
    print(f"Results: {results}")

asyncio.run (main())

# Output:
# [0.00] Task-A: Starting (call #1)
# [0.00] Task-B: Starting (call #2)
# [0.00] Task-C: Starting (call #3)
# [0.10] Task-B: Resumed (call #4)
# [0.20] Task-C: Resumed (call #5)
# [0.30] Task-A: Resumed (call #6)
# 
# Completed in 0.30s
# Total event loop calls: 6
# Results: ['Task-A result', 'Task-B result', 'Task-C result']
\`\`\`

**Observation**: The event loop made 6 calls total (3 starts + 3 resumes), completing all tasks in 0.3s (the maximum delay) instead of 0.6s (sum of delays).

---

## Running the Event Loop

There are several ways to run the event loop, depending on your use case.

### Method 1: asyncio.run() (Recommended for Python 3.7+)

\`\`\`python
"""
asyncio.run(): The Modern Way
"""

import asyncio

async def main():
    """Main entry point"""
    print("Starting application")
    await asyncio.sleep(1)
    print("Application complete")
    return "Done"

# Run the event loop (creates, runs, and closes it)
result = asyncio.run (main())
print(f"Result: {result}")

# asyncio.run() does three things:
# 1. Creates a new event loop
# 2. Runs the coroutine until complete
# 3. Closes the event loop and cleanup

# Equivalent to:
# loop = asyncio.new_event_loop()
# try:
#     result = loop.run_until_complete (main())
# finally:
#     loop.close()
\`\`\`

### Method 2: Manual Event Loop Management

\`\`\`python
"""
Manual Event Loop Control
Use when you need fine-grained control
"""

import asyncio

async def task():
    await asyncio.sleep(1)
    return "Task complete"

# Get or create event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop (loop)

try:
    # Run a single coroutine
    result = loop.run_until_complete (task())
    print(f"Result: {result}")
    
    # Run multiple coroutines
    results = loop.run_until_complete (asyncio.gather(
        task(),
        task(),
        task()
    ))
    print(f"Results: {results}")
    
finally:
    # Always close the loop
    loop.close()

# When to use manual control:
# - Integration with existing frameworks
# - Need to reuse the same loop
# - Custom event loop configuration
# - Testing and debugging
\`\`\`

### Method 3: Getting the Running Event Loop

\`\`\`python
"""
Access Currently Running Event Loop
"""

import asyncio

async def inner_task():
    """Task that needs the current event loop"""
    loop = asyncio.get_running_loop()  # Python 3.7+
    
    print(f"Loop: {loop}")
    print(f"Loop is running: {loop.is_running()}")
    
    # Schedule a callback
    loop.call_later(1.0, lambda: print("Delayed callback!"))
    
    await asyncio.sleep(1.5)

async def main():
    await inner_task()

asyncio.run (main())

# Output:
# Loop: <_UnixSelectorEventLoop running=True ...>
# Loop is running: True
# Delayed callback!  (after 1 second)
\`\`\`

### Method 4: Running Forever (Server Pattern)

\`\`\`python
"""
Long-Running Event Loop
Common for servers and daemons
"""

import asyncio
import signal

async def background_task():
    """Task that runs continuously"""
    counter = 0
    while True:
        counter += 1
        print(f"Background task running... {counter}")
        await asyncio.sleep(2)

async def main():
    """Setup and start background tasks"""
    # Start background task
    task = asyncio.create_task (background_task())
    
    # Simulate other work
    await asyncio.sleep(10)
    
    # Cancel background task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Background task cancelled")

# For servers that run forever:
# loop = asyncio.new_event_loop()
# loop.run_forever()  # Runs until loop.stop() called

asyncio.run (main())
\`\`\`

### Method 5: Event Loop in Jupyter/IPython

\`\`\`python
"""
Running Async in Jupyter Notebooks
"""

# Jupyter has a running event loop already!
# Can't use asyncio.run() (would create nested loop)

# Option 1: Use await directly (Jupyter 7.0+)
result = await some_coroutine()

# Option 2: Use nest_asyncio (older Jupyter)
import nest_asyncio
nest_asyncio.apply()

asyncio.run (main())

# Option 3: Use IPython magic
%load_ext asyncio
%%asyncio
async def notebook_task():
    await asyncio.sleep(1)
    return "Done"

result = await notebook_task()
\`\`\`

---

## Event Loop Policies

Event loop policies control how event loops are created and managed across different contexts (threads, processes).

### Default Policy

\`\`\`python
"""
Understanding Event Loop Policies
"""

import asyncio
import threading

# Get current event loop policy
policy = asyncio.get_event_loop_policy()
print(f"Default policy: {policy}")  # Usually WindowsSelectorEventLoopPolicy or UnixDefaultEventLoopPolicy

# Create a new event loop using the policy
loop = policy.new_event_loop()
print(f"Created loop: {loop}")

# Each thread can have its own event loop
def thread_function():
    """Function running in separate thread"""
    # This thread needs its own event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop (loop)
    
    async def thread_task():
        print(f"Running in thread: {threading.current_thread().name}")
        await asyncio.sleep(1)
        return "Thread result"
    
    result = loop.run_until_complete (thread_task())
    loop.close()
    return result

# Start thread
thread = threading.Thread (target=thread_function)
thread.start()
thread.join()

# Main thread has its own event loop
async def main_task():
    print(f"Running in main thread")
    await asyncio.sleep(1)

asyncio.run (main_task())
\`\`\`

### Custom Event Loop Policy

\`\`\`python
"""
Creating Custom Event Loop Policy
For specialized requirements
"""

import asyncio
import uvloop  # High-performance event loop implementation

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy (uvloop.EventLoopPolicy())

# Now all event loops use uvloop
async def fast_task():
    """This runs on uvloop (typically 2-4× faster)"""
    await asyncio.sleep(0.001)
    return "Fast!"

result = asyncio.run (fast_task())

# Use case: Production applications needing maximum performance
# uvloop is 2-4× faster than default asyncio for I/O operations
\`\`\`

### Windows-Specific: ProactorEventLoop

\`\`\`python
"""
Windows Event Loop Selection
"""

import asyncio
import sys

if sys.platform == 'win32':
    # Python 3.8+: ProactorEventLoop is default on Windows
    # Supports subprocesses and pipes (SelectorEventLoop doesn't)
    
    asyncio.set_event_loop_policy (asyncio.WindowsProactorEventLoopPolicy())
    
    # Or for compatibility with older networking code:
    asyncio.set_event_loop_policy (asyncio.WindowsSelectorEventLoopPolicy())

async def windows_task():
    """Task using Windows-specific features"""
    # ProactorEventLoop required for subprocess support on Windows
    process = await asyncio.create_subprocess_exec(
        'python', '--version',
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    print(f"Python version: {stdout.decode()}")

if sys.platform == 'win32':
    asyncio.run (windows_task())
\`\`\`

---

## Task Scheduling

The event loop provides several ways to schedule work.

### Immediate Scheduling: call_soon()

\`\`\`python
"""
Schedule Callback to Run on Next Loop Iteration
"""

import asyncio

def callback (n):
    """Simple callback function"""
    print(f"Callback executed with n={n}")

async def main():
    loop = asyncio.get_running_loop()
    
    # Schedule callbacks
    loop.call_soon (callback, 1)
    loop.call_soon (callback, 2)
    loop.call_soon (callback, 3)
    
    print("Callbacks scheduled")
    
    # Give event loop a chance to run callbacks
    await asyncio.sleep(0)
    
    print("After callbacks")

asyncio.run (main())

# Output:
# Callbacks scheduled
# Callback executed with n=1
# Callback executed with n=2
# Callback executed with n=3
# After callbacks

# Use case: Break up long-running synchronous work
def process_items (items):
    """Process items without blocking too long"""
    loop = asyncio.get_running_loop()
    
    def process_batch (batch):
        for item in batch:
            # Process item
            pass
        
        # If more items, schedule next batch
        if items:
            next_batch = items[:100]
            items[:] = items[100:]
            loop.call_soon (process_batch, next_batch)
    
    process_batch (items[:100])
    items[:] = items[100:]
\`\`\`

### Delayed Scheduling: call_later()

\`\`\`python
"""
Schedule Callback After Delay
"""

import asyncio
import time

def delayed_callback (start_time):
    """Callback executed after delay"""
    elapsed = time.time() - start_time
    print(f"Callback executed after {elapsed:.2f}s")

async def main():
    loop = asyncio.get_running_loop()
    start = time.time()
    
    # Schedule callbacks at different times
    loop.call_later(1.0, delayed_callback, start)  # After 1 second
    loop.call_later(2.0, delayed_callback, start)  # After 2 seconds
    loop.call_later(0.5, delayed_callback, start)  # After 0.5 seconds
    
    print("Callbacks scheduled")
    
    # Wait for callbacks to execute
    await asyncio.sleep(2.5)

asyncio.run (main())

# Output:
# Callbacks scheduled
# Callback executed after 0.50s
# Callback executed after 1.00s
# Callback executed after 2.00s

# Use case: Implementing timeouts
class TimeoutManager:
    def __init__(self, loop):
        self.loop = loop
        self.handle = None
    
    def set_timeout (self, callback, delay):
        """Set a timeout"""
        self.cancel_timeout()
        self.handle = self.loop.call_later (delay, callback)
    
    def cancel_timeout (self):
        """Cancel pending timeout"""
        if self.handle:
            self.handle.cancel()
            self.handle = None
\`\`\`

### Absolute Time Scheduling: call_at()

\`\`\`python
"""
Schedule Callback at Absolute Time
"""

import asyncio
import time

async def main():
    loop = asyncio.get_running_loop()
    
    # Get loop time (monotonic clock)
    now = loop.time()
    print(f"Current loop time: {now:.2f}")
    
    # Schedule at absolute time
    target = now + 2.0
    
    def callback():
        actual = loop.time()
        print(f"Callback at {actual:.2f} (target was {target:.2f})")
    
    loop.call_at (target, callback)
    
    # Wait for callback
    await asyncio.sleep(2.5)

asyncio.run (main())

# Use case: Scheduling at specific times
async def schedule_at_specific_time (hour, minute, callback):
    """Schedule callback at specific time of day"""
    loop = asyncio.get_running_loop()
    
    from datetime import datetime, timedelta
    
    now = datetime.now()
    target = now.replace (hour=hour, minute=minute, second=0, microsecond=0)
    
    if target <= now:
        # Time has passed today, schedule for tomorrow
        target += timedelta (days=1)
    
    delay = (target - now).total_seconds()
    loop.call_later (delay, callback)
\`\`\`

### Priority Queue: call_soon_threadsafe()

\`\`\`python
"""
Schedule from Another Thread
Thread-safe scheduling
"""

import asyncio
import threading
import time

def blocking_work (loop, message):
    """Work done in separate thread"""
    time.sleep(2)  # Simulate blocking I/O
    
    # Schedule callback in event loop (thread-safe)
    loop.call_soon_threadsafe (lambda: print(f"From thread: {message}"))

async def main():
    loop = asyncio.get_running_loop()
    
    # Start blocking work in thread
    thread = threading.Thread(
        target=blocking_work,
        args=(loop, "Work complete!")
    )
    thread.start()
    
    print("Thread started, doing async work...")
    
    # Do async work while thread runs
    for i in range(5):
        await asyncio.sleep(0.5)
        print(f"Async work {i}")
    
    thread.join()

asyncio.run (main())

# Output:
# Thread started, doing async work...
# Async work 0
# Async work 1
# Async work 2
# Async work 3
# From thread: Work complete!
# Async work 4

# Use case: Integrating threading with asyncio
class AsyncThreadPool:
    """Run blocking functions in thread pool, return to async"""
    
    def __init__(self, loop, max_workers=4):
        self.loop = loop
        self.executor = ThreadPoolExecutor (max_workers=max_workers)
    
    async def run (self, func, *args):
        """Run blocking function in thread, return result to async"""
        future = self.loop.run_in_executor (self.executor, func, *args)
        return await future
\`\`\`

---

## Callbacks and Futures

Understanding callbacks and futures helps you work with lower-level asyncio APIs.

### Callbacks

\`\`\`python
"""
Working with Callbacks
"""

import asyncio

def operation_complete (future):
    """Callback when operation completes"""
    try:
        result = future.result()
        print(f"Operation succeeded: {result}")
    except Exception as e:
        print(f"Operation failed: {e}")

async def async_operation():
    """Some async operation"""
    await asyncio.sleep(1)
    return "Success!"

async def main():
    # Create task
    task = asyncio.create_task (async_operation())
    
    # Add callback
    task.add_done_callback (operation_complete)
    
    # Wait for completion
    await task

asyncio.run (main())

# Output:
# Operation succeeded: Success!
\`\`\`

### Futures

\`\`\`python
"""
Working with Futures
Bridge between callbacks and coroutines
"""

import asyncio

async def main():
    loop = asyncio.get_running_loop()
    
    # Create a future
    future = loop.create_future()
    
    def set_result():
        """Callback that sets future result"""
        future.set_result("Result from callback!")
    
    # Schedule result to be set after 1 second
    loop.call_later(1.0, set_result)
    
    print("Waiting for future...")
    result = await future
    print(f"Got result: {result}")

asyncio.run (main())

# Use case: Wrapping callback-based APIs
class CallbackAPI:
    """Legacy callback-based API"""
    
    def fetch_data (self, callback):
        """Fetch data, call callback when done"""
        # Simulate async operation
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, callback, "Data!")

async def use_callback_api():
    """Wrap callback API for async use"""
    api = CallbackAPI()
    loop = asyncio.get_running_loop()
    
    # Create future to await
    future = loop.create_future()
    
    # Callback sets future result
    def callback (data):
        future.set_result (data)
    
    # Start operation
    api.fetch_data (callback)
    
    # Await future
    result = await future
    return result

# asyncio.run (use_callback_api())
\`\`\`

---

## Event Loop Performance

Understanding event loop performance characteristics helps you write efficient code.

### Performance Characteristics

\`\`\`python
"""
Event Loop Performance Benchmarks
"""

import asyncio
import time

async def measure_task_creation_overhead():
    """Measure overhead of creating tasks"""
    iterations = 10000
    
    async def empty_task():
        pass
    
    # Measure task creation
    start = time.time()
    tasks = [asyncio.create_task (empty_task()) for _ in range (iterations)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    per_task = (elapsed / iterations) * 1_000_000  # microseconds
    print(f"Task creation overhead: {per_task:.2f}μs per task")
    print(f"Can handle {int(1_000_000 / per_task)} tasks per second")

async def measure_context_switching():
    """Measure context switching overhead"""
    iterations = 1000
    
    async def switch_task():
        for _ in range(100):
            await asyncio.sleep(0)  # Yield control
    
    start = time.time()
    await asyncio.gather(*[switch_task() for _ in range (iterations)])
    elapsed = time.time() - start
    
    switches = iterations * 100
    per_switch = (elapsed / switches) * 1_000_000  # microseconds
    print(f"Context switch: {per_switch:.2f}μs")

async def measure_io_multiplexing():
    """Measure I/O multiplexing efficiency"""
    num_connections = 1000
    
    async def connection():
        await asyncio.sleep(0.1)
    
    start = time.time()
    await asyncio.gather(*[connection() for _ in range (num_connections)])
    elapsed = time.time() - start
    
    print(f"{num_connections} concurrent I/O operations: {elapsed:.2f}s")
    print(f"Throughput: {num_connections / elapsed:.0f} ops/second")

async def main():
    print("=== Event Loop Performance ===\\n")
    await measure_task_creation_overhead()
    print()
    await measure_context_switching()
    print()
    await measure_io_multiplexing()

asyncio.run (main())

# Typical output:
# Task creation overhead: 15μs per task
# Can handle 66,666 tasks per second
# 
# Context switch: 2μs
# 
# 1000 concurrent I/O operations: 0.10s
# Throughput: 10,000 ops/second
\`\`\`

---

## Summary

### Key Takeaways

1. **Event Loop is the Engine**
   - Manages all async tasks in a single thread
   - Switches between tasks at await points
   - Monitors I/O and schedules callbacks

2. **Running the Event Loop**
   - \`asyncio.run()\`: Simple, recommended for Python 3.7+
   - Manual loop control for advanced use cases
   - Thread-specific loops for multi-threading

3. **Task Scheduling**
   - \`call_soon()\`: Next loop iteration
   - \`call_later()\`: After delay
   - \`call_at()\`: At absolute time
   - \`call_soon_threadsafe()\`: From other threads

4. **Performance**
   - Very low overhead (~15μs per task)
   - Can handle 10,000+ concurrent I/O operations
   - Single-threaded but highly efficient

5. **Best Practices**
   - Use \`asyncio.run()\` for simple cases
   - Don't block the event loop with CPU-intensive work
   - Use futures to wrap callback-based APIs
   - Profile with event loop debug mode

### Next Steps

Now that you understand the event loop, we'll explore:
- Coroutines and async/await syntax
- Creating and managing tasks
- Practical async patterns
- Building production async applications

**Remember**: The event loop is powerful but single-threaded. Keep operations non-blocking to maintain responsiveness.
`,
};
