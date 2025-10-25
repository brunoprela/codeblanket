/**
 * Async/Await & Asynchronous Programming Section
 */

export const asyncawaitSection = {
  id: 'async-await',
  title: 'Async/Await & Asynchronous Programming',
  content: `**What is Asynchronous Programming?**
Async programming allows your code to handle multiple tasks concurrently without blocking, making it ideal for I/O-bound operations like network requests, file I/O, and database queries.

**Key Concepts:**
- **Coroutine:** An async function defined with \`async def\`
- **Await:** Pauses execution until the awaited task completes
- **Event Loop:** Manages and schedules async tasks
- **Concurrency:** Multiple tasks make progress, but not necessarily in parallel

**Basic Async Function:**
\`\`\`python
import asyncio

async def fetch_data (url):
    """Async function (coroutine)"""
    print(f"Fetching {url}...")
    await asyncio.sleep(1)  # Simulate I/O operation
    return f"Data from {url}"

# Run async function
result = asyncio.run (fetch_data("https://api.example.com"))
print(result)
\`\`\`

**Running Multiple Tasks Concurrently:**
\`\`\`python
async def fetch_all_data():
    """Run multiple async tasks concurrently"""
    # All three requests run concurrently (not sequentially!)
    results = await asyncio.gather(
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com")
    )
    return results

# Total time: ~1 second (not 3 seconds!)
results = asyncio.run (fetch_all_data())
\`\`\`

**Why Async Matters:**
\`\`\`python
import time

# Synchronous version (3 seconds total)
def sync_fetch_all():
    results = []
    for url in ["url1", "url2", "url3"]:
        time.sleep(1)  # Blocking operation
        results.append (f"Data from {url}")
    return results

# Async version (1 second total - concurrent!)
async def async_fetch_all():
    tasks = [fetch_data (url) for url in ["url1", "url2", "url3"]]
    return await asyncio.gather(*tasks)
\`\`\`

**Common Async Patterns:**

**1. Creating Tasks:**
\`\`\`python
async def main():
    # Create tasks (start them immediately)
    task1 = asyncio.create_task (fetch_data("url1"))
    task2 = asyncio.create_task (fetch_data("url2"))
    
    # Do other work while tasks run
    print("Tasks are running in background...")
    
    # Wait for results
    result1 = await task1
    result2 = await task2
\`\`\`

**2. Timeout Handling:**
\`\`\`python
async def fetch_with_timeout (url, timeout=5):
    try:
        return await asyncio.wait_for(
            fetch_data (url), 
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return f"Request to {url} timed out"
\`\`\`

**3. Async Context Managers:**
\`\`\`python
class AsyncDatabaseConnection:
    async def __aenter__(self):
        print("Opening database connection...")
        await asyncio.sleep(0.1)  # Simulate async setup
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection...")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        return False

async def use_database():
    async with AsyncDatabaseConnection() as conn:
        print("Using database...")
\`\`\`

**4. Async Generators:**
\`\`\`python
async def async_range (start, stop):
    """Async generator example"""
    for i in range (start, stop):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i

async def consume_async_generator():
    async for value in async_range(0, 5):
        print(value)
\`\`\`

**Real-World Example - Web Scraping:**
\`\`\`python
import aiohttp
import asyncio

async def fetch_url (session, url):
    """Fetch single URL"""
    async with session.get (url) as response:
        return await response.text()

async def scrape_websites (urls):
    """Scrape multiple websites concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url (session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Scrape 10 websites concurrently (much faster than sequential!)
urls = [f"https://example.com/page{i}" for i in range(10)]
results = asyncio.run (scrape_websites (urls))
\`\`\`

**Common Pitfalls:**

**1. Forgetting await:**
\`\`\`python
# WRONG - returns coroutine object, doesn't execute
result = fetch_data("url")  

# CORRECT - awaits the coroutine
result = await fetch_data("url")
\`\`\`

**2. Using blocking operations:**
\`\`\`python
# WRONG - blocks event loop
time.sleep(1)

# CORRECT - async version
await asyncio.sleep(1)
\`\`\`

**3. Not using asyncio.run() properly:**
\`\`\`python
# WRONG - can't await outside async function
result = await fetch_data("url")  

# CORRECT - use asyncio.run()
result = asyncio.run (fetch_data("url"))
\`\`\`

**When to Use Async:**
- ✅ I/O-bound operations (network, file I/O, database)
- ✅ Many concurrent connections (web servers, chat apps)
- ✅ Real-time applications (websockets, streaming)
- ❌ CPU-bound operations (use multiprocessing instead)
- ❌ Simple scripts with few I/O operations

**Best Practices:**
- Use \`asyncio.gather()\` for concurrent tasks
- Always await async functions
- Use async libraries (aiohttp, asyncpg) not blocking ones
- Handle exceptions in async code properly
- Don't mix blocking and async code`,
};
