export const coroutinesAsyncAwait = {
  title: 'Coroutines & async/await',
  id: 'coroutines-async-await',
  content: `
# Coroutines & async/await

## Introduction

**Coroutines** are the fundamental building blocks of async programming in Python. They're special functions that can pause execution and resume later, allowing the event loop to run other tasks while waiting.

The \`async def\` and \`await\` keywords introduced in Python 3.5 make writing asynchronous code almost as simple as writing synchronous code, but with dramatically better performance for I/O-bound operations.

### The Power of Coroutines

\`\`\`python
"""
Synchronous vs Asynchronous Comparison
"""

import time
import asyncio

# Synchronous: Takes 3 seconds
def sync_fetch():
    time.sleep(1)  # Simulate I/O
    return "data"

def sync_main():
    start = time.time()
    results = [sync_fetch() for _ in range(3)]
    print(f"Sync: {time.time() - start:.2f}s")  # ~3.00s

# Asynchronous: Takes 1 second!
async def async_fetch():
    await asyncio.sleep(1)  # Simulate async I/O
    return "data"

async def async_main():
    start = time.time()
    results = await asyncio.gather(*[async_fetch() for _ in range(3)])
    print(f"Async: {time.time() - start:.2f}s")  # ~1.00s

sync_main()
asyncio.run (async_main())

# 3× speedup with just async/await!
\`\`\`

By the end of this section, you'll understand:
- What coroutines are and how they work
- The \`async def\` and \`await\` syntax
- Coroutine objects and their lifecycle
- Async generators and comprehensions
- Best practices for writing async code
- Common pitfalls and how to avoid them

---

## What Are Coroutines?

A **coroutine** is a function that can suspend its execution before reaching return, and can indirectly pass control to another coroutine for some time.

### Traditional Functions vs Coroutines

\`\`\`python
"""
Regular Function: Runs Start to Finish
"""

def regular_function():
    print("Start")
    result = expensive_operation()  # Blocks here
    print("End")
    return result

# Once called, runs until return or exception
# Cannot pause in the middle

"""
Coroutine: Can Pause and Resume
"""

async def coroutine_function():
    print("Start")
    result = await async_operation()  # Pauses here, returns control
    print("End")  # Resumes here when operation completes
    return result

# Can pause at 'await' points
# Event loop can run other coroutines while this waits
\`\`\`

### Coroutine Lifecycle

\`\`\`python
"""
Understanding Coroutine States
"""

import asyncio
import inspect

async def example_coroutine():
    print("Coroutine executing")
    await asyncio.sleep(1)
    return "result"

# Create coroutine object (doesn't execute yet)
coro = example_coroutine()
print(f"Type: {type (coro)}")  # <class 'coroutine'>
print(f"State: {inspect.getcoroutinestate (coro)}")  # CORO_CREATED

# Coroutine object is waiting to be executed
# Need to either:
# 1. await it from another coroutine
# 2. pass it to asyncio.run()
# 3. create a Task from it

# Execute it
result = asyncio.run (coro)
print(f"Result: {result}")

# States during lifecycle:
# CORO_CREATED: Just created, not started
# CORO_RUNNING: Currently executing
# CORO_SUSPENDED: Paused at await
# CORO_CLOSED: Finished or cancelled
\`\`\`

---

## The async def Keyword

The \`async def\` keyword defines a coroutine function. When called, it returns a coroutine object.

### Basic Syntax

\`\`\`python
"""
Defining Coroutines with async def
"""

# Define a coroutine
async def greet (name):
    """A simple coroutine"""
    return f"Hello, {name}!"

# Calling it returns a coroutine object
coro = greet("Alice")
print(coro)  # <coroutine object greet at 0x...>

# To actually execute it:
result = asyncio.run (coro)
print(result)  # "Hello, Alice!"

# Warning: Forgetting to await a coroutine
async def bad_example():
    result = greet("Bob")  # ❌ Forgot await!
    print(result)  # Prints: <coroutine object greet at 0x...>
    # Python warning: RuntimeWarning: coroutine 'greet' was never awaited

async def good_example():
    result = await greet("Bob")  # ✅ Correct!
    print(result)  # Prints: "Hello, Bob!"

asyncio.run (good_example())
\`\`\`

### Coroutine Function Characteristics

\`\`\`python
"""
Properties of Async Functions
"""

async def async_function():
    """This is a coroutine function"""
    return 42

# Check if it's a coroutine function
import asyncio
print(asyncio.iscoroutinefunction (async_function))  # True

# Regular function
def regular_function():
    return 42

print(asyncio.iscoroutinefunction (regular_function))  # False

# Coroutine functions can:
# 1. Use 'await' expressions
# 2. Use 'async with' (async context managers)
# 3. Use 'async for' (async iteration)
# 4. Return values (like regular functions)
# 5. Raise exceptions (like regular functions)

async def full_featured_coroutine():
    """Demonstrates all async features"""

    # await expression
    await asyncio.sleep(0.1)

    # async with
    async with async_context_manager() as resource:
        data = resource.read()

    # async for
    async for item in async_generator():
        process (item)

    # Regular Python code
    result = compute_something()

    return result
\`\`\`

### Converting Regular Functions to Async

\`\`\`python
"""
Migration Pattern: Sync to Async
"""

# Original synchronous code
import requests

def fetch_user (user_id):
    response = requests.get (f"https://api.example.com/users/{user_id}")
    return response.json()

def get_users (user_ids):
    return [fetch_user (uid) for uid in user_ids]

# Time: N × request_time (sequential)

# Converted to async
import aiohttp

async def fetch_user (user_id):
    async with aiohttp.ClientSession() as session:
        async with session.get (f"https://api.example.com/users/{user_id}") as response:
            return await response.json()

async def get_users (user_ids):
    tasks = [fetch_user (uid) for uid in user_ids]
    return await asyncio.gather(*tasks)

# Time: max (request_times) (concurrent!)

# Pattern: Add 'async' to def, add 'await' to I/O operations
\`\`\`

---

## The await Keyword

The \`await\` keyword can only be used inside async functions. It suspends the coroutine until the awaited operation completes.

### What Can You Await?

\`\`\`python
"""
Awaitable Objects in Python
"""

import asyncio

# 1. Coroutines
async def my_coroutine():
    return "coroutine result"

async def main():
    result = await my_coroutine()  # ✅ Can await coroutines
    print(result)

# 2. Tasks
async def main():
    task = asyncio.create_task (my_coroutine())
    result = await task  # ✅ Can await tasks
    print(result)

# 3. Futures
async def main():
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # Set result later
    loop.call_later(1.0, lambda: future.set_result("future result"))

    result = await future  # ✅ Can await futures
    print(result)

# Cannot await regular functions!
def regular_function():
    return "regular result"

async def bad_example():
    result = await regular_function()  # ❌ TypeError!
    # TypeError: object 'str' can't be used in 'await' expression

asyncio.run (main())
\`\`\`

### await Behavior

\`\`\`python
"""
How await Works
"""

import asyncio
import time

async def step1():
    print(f"{time.time():.2f}: Step 1 starting")
    await asyncio.sleep(1)
    print(f"{time.time():.2f}: Step 1 complete")
    return "result1"

async def step2():
    print(f"{time.time():.2f}: Step 2 starting")
    await asyncio.sleep(0.5)
    print(f"{time.time():.2f}: Step 2 complete")
    return "result2"

async def sequential():
    """Wait for each step to complete"""
    start = time.time()

    result1 = await step1()  # Waits 1s
    result2 = await step2()  # Then waits 0.5s

    elapsed = time.time() - start
    print(f"Sequential: {elapsed:.2f}s")  # ~1.5s
    return result1, result2

async def concurrent():
    """Start both, then wait for both"""
    start = time.time()

    # Start both without waiting
    task1 = asyncio.create_task (step1())
    task2 = asyncio.create_task (step2())

    # Now wait for both
    result1 = await task1  # Waits ~1s (both run concurrently)
    result2 = await task2  # Already done!

    elapsed = time.time() - start
    print(f"Concurrent: {elapsed:.2f}s")  # ~1.0s
    return result1, result2

asyncio.run (sequential())
asyncio.run (concurrent())

# Output:
# Sequential: step1 then step2 = 1.5s
# Concurrent: step1 and step2 together = 1.0s
\`\`\`

### await vs return await

\`\`\`python
"""
Subtle Difference: await vs return await
"""

import asyncio

async def operation():
    await asyncio.sleep(1)
    return "done"

# Pattern 1: await then return
async def explicit_await():
    result = await operation()  # Wait, store result
    return result  # Return result

# Pattern 2: return await
async def return_await():
    return await operation()  # Wait and return directly

# Pattern 3: just return (no await!)
async def just_return():
    return operation()  # ❌ Returns coroutine object, not result!

# Which to use?
async def test():
    # These two are functionally equivalent
    r1 = await explicit_await()  # "done"
    r2 = await return_await()    # "done"

    # This is wrong!
    r3 = await just_return()  # TypeError or warning

    print(f"r1={r1}, r2={r2}")

# When to use each:
# - 'await then return': When you need the result for logic
async def with_logic():
    result = await operation()
    if result == "done":
        print("Success!")
    return result

# - 'return await': When just passing through
async def passthrough():
    return await operation()

# - 'just return': NEVER for coroutines (wrong!)

asyncio.run (test())
\`\`\`

---

## Async Generator Functions

Generators can also be async, allowing you to yield values asynchronously.

### Basic Async Generators

\`\`\`python
"""
Async Generators with async def + yield
"""

import asyncio

async def async_range (count):
    """Async version of range()"""
    for i in range (count):
        await asyncio.sleep(0.1)  # Simulate async operation
        yield i

async def main():
    print("Iterating async generator:")
    async for value in async_range(5):
        print(f"  Got: {value}")

asyncio.run (main())

# Output:
# Iterating async generator:
#   Got: 0  (after 0.1s)
#   Got: 1  (after 0.2s)
#   Got: 2  (after 0.3s)
#   Got: 3  (after 0.4s)
#   Got: 4  (after 0.5s)
\`\`\`

### Async Generators for Streaming Data

\`\`\`python
"""
Real-World Async Generator: Data Streaming
"""

import asyncio
import aiohttp

async def fetch_pages (urls):
    """Stream webpage contents as they're fetched"""
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.get (url) as response:
                    content = await response.text()
                    yield (url, content)
            except Exception as e:
                yield (url, f"Error: {e}")

async def main():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
    ]

    async for url, content in fetch_pages (urls):
        print(f"Fetched {url}: {len (content)} bytes")

# Fetched https://httpbin.org/delay/1: ... (after 1s)
# Fetched https://httpbin.org/delay/2: ... (after 3s total)
# Fetched https://httpbin.org/delay/3: ... (after 6s total)

# Process results as they arrive, don't wait for all!
\`\`\`

### Async Generator Expressions

\`\`\`python
"""
Async Generator Expressions (Python 3.6+)
"""

import asyncio

async def async_squares (n):
    """Generate squares asynchronously"""
    for i in range (n):
        await asyncio.sleep(0.1)
        yield i ** 2

async def main():
    # Async generator expression
    gen = (x async for x in async_squares(5) if x > 5)

    async for value in gen:
        print(value)

asyncio.run (main())

# Output: 9, 16 (values > 5)

# Use case: Filter async streams
async def filter_large_files (file_paths):
    """Yield only files larger than 1MB"""
    return (
        path
        async for path in async_scan_files (file_paths)
        if (await async_get_size (path)) > 1_000_000
    )
\`\`\`

---

## Async Comprehensions

List/dict/set comprehensions can be async when iterating over async iterables.

### Async List Comprehensions

\`\`\`python
"""
Async Comprehensions (Python 3.6+)
"""

import asyncio

async def async_range (count):
    for i in range (count):
        await asyncio.sleep(0.1)
        yield i

async def main():
    # Async list comprehension
    result = [x async for x in async_range(5)]
    print(f"Result: {result}")  # [0, 1, 2, 3, 4]

    # With condition
    evens = [x async for x in async_range(10) if x % 2 == 0]
    print(f"Evens: {evens}")  # [0, 2, 4, 6, 8]

    # Async dict comprehension
    squares = {x: x**2 async for x in async_range(5)}
    print(f"Squares: {squares}")  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

    # Async set comprehension
    unique = {x % 3 async for x in async_range(10)}
    print(f"Unique: {unique}")  # {0, 1, 2}

asyncio.run (main())
\`\`\`

### await in Comprehensions

\`\`\`python
"""
Using await Inside Comprehensions
"""

import asyncio
import aiohttp

async def fetch_size (url):
    """Fetch URL and return content size"""
    async with aiohttp.ClientSession() as session:
        async with session.get (url) as response:
            content = await response.text()
            return len (content)

async def main():
    urls = [
        "https://httpbin.org/delay/0",
        "https://httpbin.org/delay/0",
        "https://httpbin.org/delay/0",
    ]

    # ❌ Wrong: This runs sequentially!
    sizes_sequential = [await fetch_size (url) for url in urls]
    # Each await blocks before starting next

    # ✅ Correct: Create tasks, then gather
    sizes_concurrent = await asyncio.gather(*[fetch_size (url) for url in urls])
    # All fetches start immediately, wait for all to complete

    print(f"Sequential: {sizes_sequential}")
    print(f"Concurrent: {sizes_concurrent}")

# Key insight: 'await' in comprehension is still sequential
# For concurrency, use asyncio.gather() or create_task()
\`\`\`

### Practical Example: Batch Processing

\`\`\`python
"""
Async Comprehension for Batch Processing
"""

import asyncio
import aiohttp

async def process_user (session, user_id):
    """Fetch and process user data"""
    async with session.get (f"https://api.example.com/users/{user_id}") as response:
        user = await response.json()
        return {
            'id': user_id,
            'name': user.get('name'),
            'email': user.get('email'),
        }

async def main():
    user_ids = range(1, 101)  # 100 users

    async with aiohttp.ClientSession() as session:
        # Process all users concurrently
        tasks = [process_user (session, uid) for uid in user_ids]
        users = await asyncio.gather(*tasks)

    # Filter using list comprehension (sync, already have data)
    verified_users = [u for u in users if u.get('verified')]

    print(f"Processed {len (users)} users")
    print(f"Verified: {len (verified_users)}")

# 100 users fetched concurrently in ~1 second
# vs 50-100 seconds sequentially
\`\`\`

---

## Common Patterns and Best Practices

### Pattern 1: Concurrent Operations

\`\`\`python
"""
Running Multiple Coroutines Concurrently
"""

import asyncio

async def fetch_data (source):
    await asyncio.sleep(1)
    return f"data from {source}"

async def main():
    # ❌ Sequential (slow)
    result1 = await fetch_data("source1")  # Wait 1s
    result2 = await fetch_data("source2")  # Wait another 1s
    result3 = await fetch_data("source3")  # Wait another 1s
    # Total: 3 seconds

    # ✅ Concurrent (fast)
    results = await asyncio.gather(
        fetch_data("source1"),
        fetch_data("source2"),
        fetch_data("source3"),
    )
    # Total: 1 second (all run together)
\`\`\`

### Pattern 2: Error Handling

\`\`\`python
"""
Error Handling in Async Code
"""

import asyncio

async def risky_operation (will_fail=False):
    await asyncio.sleep(0.5)
    if will_fail:
        raise ValueError("Operation failed!")
    return "success"

async def main():
    # Try/except works normally
    try:
        result = await risky_operation (will_fail=True)
    except ValueError as e:
        print(f"Caught: {e}")

    # With gather, use return_exceptions=True
    results = await asyncio.gather(
        risky_operation (will_fail=False),
        risky_operation (will_fail=True),
        risky_operation (will_fail=False),
        return_exceptions=True,  # Don't stop on first error
    )

    for i, result in enumerate (results):
        if isinstance (result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

# Output:
# Caught: Operation failed!
# Task 0 succeeded: success
# Task 1 failed: Operation failed!
# Task 2 succeeded: success
\`\`\`

### Pattern 3: Timeouts

\`\`\`python
"""
Adding Timeouts to Async Operations
"""

import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "finally done"

async def main():
    try:
        # Wait at most 2 seconds
        result = await asyncio.wait_for (slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out!")

    # Multiple operations with overall timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                slow_operation(),
                slow_operation(),
            ),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Batch timed out!")

# Use case: API requests with timeout
async def fetch_with_timeout (url, timeout=5.0):
    async with aiohttp.ClientSession() as session:
        try:
            async with asyncio.timeout (timeout):  # Python 3.11+
                async with session.get (url) as response:
                    return await response.json()
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
\`\`\`

---

## Common Pitfalls

### Pitfall 1: Forgetting await

\`\`\`python
"""
Most Common Mistake: Forgetting await
"""

import asyncio

async def get_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    # ❌ Forgot await - gets coroutine object, not result!
    result = get_data()
    print(f"Result: {result}")  # <coroutine object get_data at 0x...>

    # ✅ Correct
    result = await get_data()
    print(f"Result: {result}")  # "data"

# Python 3.11+ warns: RuntimeWarning: coroutine 'get_data' was never awaited
\`\`\`

### Pitfall 2: Blocking Operations in Async Code

\`\`\`python
"""
Don't Block the Event Loop!
"""

import asyncio
import time

async def bad_example():
    # ❌ This blocks the entire event loop for 2 seconds!
    time.sleep(2)
    return "done"

async def good_example():
    # ✅ This yields control during the 2 seconds
    await asyncio.sleep(2)
    return "done"

async def test():
    start = time.time()

    # Start three concurrent operations
    await asyncio.gather(
        bad_example(),
        bad_example(),
        bad_example(),
    )

    print(f"Bad: {time.time() - start:.2f}s")  # ~6.0s (sequential!)

asyncio.run (test())

# Even though using asyncio.gather, time.sleep() blocks!
# Result: 6 seconds (2+2+2) instead of 2 seconds
\`\`\`

### Pitfall 3: Mixing Sync and Async

\`\`\`python
"""
Cannot Call Async from Sync Directly
"""

import asyncio

async def async_function():
    return "async result"

def sync_function():
    # ❌ This doesn't work!
    result = async_function()  # Just returns coroutine object

    # ❌ This doesn't work either!
    result = await async_function()  # SyntaxError: await outside async function

    # ✅ Correct way: Create event loop
    result = asyncio.run (async_function())  # Works, but creates new loop

    # ✅ Or use run_in_executor to call sync from async context
    # (Reverse: call sync from async)

# Solution: Keep sync and async code separate
# Or use run_in_executor for sync code that needs to be called from async
\`\`\`

---

## Summary

### Key Takeaways

1. **Coroutines are Pausable Functions**
   - Defined with \`async def\`
   - Return coroutine objects when called
   - Must be awaited to execute

2. **await Suspends Execution**
   - Only works inside async functions
   - Returns control to event loop
   - Resumes when operation completes

3. **Async Generators and Comprehensions**
   - Use \`async for\` to iterate async generators
   - Async comprehensions for concise code
   - Great for streaming data

4. **Best Practices**
   - Use \`asyncio.gather()\` for concurrency
   - Add timeouts with \`wait_for()\`
   - Handle errors with try/except and \`return_exceptions=True\`
   - Never block with sync I/O in async code

5. **Common Pitfalls**
   - Forgetting \`await\` (returns coroutine, not result)
   - Blocking operations (time.sleep, requests.get)
   - Mixing sync and async incorrectly

### async/await Cheat Sheet

\`\`\`python
# Define coroutine
async def my_coroutine():
    return "result"

# Call coroutine (await required!)
result = await my_coroutine()

# Run multiple concurrently
results = await asyncio.gather (coro1(), coro2(), coro3())

# With timeout
result = await asyncio.wait_for (my_coroutine(), timeout=5.0)

# Async generator
async def gen():
    yield 1
    yield 2

# Async iteration
async for value in gen():
    print(value)

# Async comprehension
values = [x async for x in gen()]
\`\`\`

### Next Steps

Now that you master coroutines and async/await, we'll explore:
- Tasks and task management
- Async context managers
- Building complete async applications
- Performance optimization techniques

**Remember**: async/await makes async code look like sync code, but the execution model is fundamentally different. Master the mental model of "pause and resume" for effective async programming.
`,
};
