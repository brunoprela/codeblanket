export const asyncContextManagersGenerators = {
  title: 'Async Context Managers & Generators',
  id: 'async-context-managers-generators',
  content: `
# Async Context Managers & Generators

## Introduction

**Async context managers** and **async generators** are essential tools for managing resources and streaming data in async applications. They bring the power of \`with\` statements and generators to the async world, ensuring proper resource cleanup and efficient data streaming.

### Why Async Context Managers Matter

\`\`\`python
"""
Problem: Resource Management in Async Code
"""

import aiohttp
import asyncio

# ❌ Bad: Manual resource management
async def fetch_bad(url):
    session = aiohttp.ClientSession()
    response = await session.get(url)
    data = await response.text()
    await response.close()
    await session.close()  # Easy to forget!
    return data

# ✅ Good: Async context manager
async def fetch_good(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
    # Automatically closed, even if exception occurs!

# Context manager guarantees cleanup
asyncio.run(fetch_good('https://api.example.com'))
\`\`\`

By the end of this section, you'll understand:
- Creating async context managers with \`__aenter__\` and \`__aexit__\`
- Using \`async with\` for resource management
- Async generators for streaming data
- Combining async context managers and generators
- Best practices for resource lifecycle management
- Production patterns for database connections, file handles, and network resources

---

## Async Context Managers

Async context managers use \`async with\` instead of \`with\` and support asynchronous setup and cleanup.

### The async with Statement

\`\`\`python
"""
Basic Async Context Manager Usage
"""

import asyncio
import aiohttp

async def fetch_users():
    # Async context manager for HTTP session
    async with aiohttp.ClientSession() as session:
        # Async context manager for response
        async with session.get('https://api.example.com/users') as response:
            users = await response.json()
            return users
    # Session and response automatically closed here

# What happens:
# 1. Create ClientSession (async operation)
# 2. Execute code block
# 3. Close session (async operation) - GUARANTEED

asyncio.run(fetch_users())
\`\`\`

### Creating Custom Async Context Managers

\`\`\`python
"""
Defining Your Own Async Context Manager
"""

import asyncio

class AsyncDatabaseConnection:
    """Async context manager for database connections"""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    async def __aenter__(self):
        """Called when entering 'async with' block"""
        print(f"Connecting to {self.host}:{self.port}...")
        await asyncio.sleep(0.5)  # Simulate connection
        self.connection = f"Connection to {self.host}"
        print("Connected!")
        return self  # Return value is assigned to 'as' variable
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'async with' block"""
        print("Closing connection...")
        await asyncio.sleep(0.2)  # Simulate cleanup
        self.connection = None
        print("Connection closed!")
        
        # Return False to propagate exceptions, True to suppress
        return False
    
    async def query(self, sql):
        """Execute query on connection"""
        if not self.connection:
            raise RuntimeError("Not connected")
        await asyncio.sleep(0.1)
        return f"Results for: {sql}"

async def main():
    async with AsyncDatabaseConnection('localhost', 5432) as db:
        result = await db.query("SELECT * FROM users")
        print(result)
    # Connection automatically closed

asyncio.run(main())

# Output:
# Connecting to localhost:5432...
# Connected!
# Results for: SELECT * FROM users
# Closing connection...
# Connection closed!
\`\`\`

### Exception Handling in Context Managers

\`\`\`python
"""
Context Managers Always Cleanup (Even on Exceptions)
"""

import asyncio

class ResourceManager:
    async def __aenter__(self):
        print("Acquiring resource...")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Releasing resource...")
        print(f"  Exception type: {exc_type}")
        print(f"  Exception value: {exc_val}")
        await asyncio.sleep(0.1)
        
        if exc_type is ValueError:
            print("  Suppressing ValueError")
            return True  # Suppress this exception
        
        return False  # Propagate other exceptions

async def test_exceptions():
    # Test 1: No exception
    print("Test 1: No exception")
    async with ResourceManager() as rm:
        print("Working with resource")
    print()
    
    # Test 2: ValueError (suppressed)
    print("Test 2: ValueError (suppressed)")
    try:
        async with ResourceManager() as rm:
            raise ValueError("Something went wrong!")
    except ValueError:
        print("ValueError was not caught (shouldn't happen)")
    print("Continued after suppressed exception\\n")
    
    # Test 3: RuntimeError (propagated)
    print("Test 3: RuntimeError (propagated)")
    try:
        async with ResourceManager() as rm:
            raise RuntimeError("Critical error!")
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")

asyncio.run(test_exceptions())

# Output:
# Test 1: No exception
# Acquiring resource...
# Working with resource
# Releasing resource...
#   Exception type: None
#   Exception value: None
#
# Test 2: ValueError (suppressed)
# Acquiring resource...
# Releasing resource...
#   Exception type: <class 'ValueError'>
#   Exception value: Something went wrong!
#   Suppressing ValueError
# Continued after suppressed exception
#
# Test 3: RuntimeError (propagated)
# Acquiring resource...
# Releasing resource...
#   Exception type: <class 'RuntimeError'>
#   Exception value: Critical error!
# Caught RuntimeError: Critical error!
\`\`\`

### Using @asynccontextmanager Decorator

\`\`\`python
"""
Simpler Async Context Managers with Decorator
"""

from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def database_transaction(connection):
    """Context manager for database transactions"""
    print("BEGIN TRANSACTION")
    await connection.execute("BEGIN")
    
    try:
        yield connection  # Provide connection to 'as' variable
        
        # If we get here, commit
        print("COMMIT")
        await connection.execute("COMMIT")
    except Exception as e:
        # On exception, rollback
        print(f"ROLLBACK due to {e}")
        await connection.execute("ROLLBACK")
        raise

class MockConnection:
    async def execute(self, sql):
        await asyncio.sleep(0.1)
        print(f"  Executing: {sql}")

async def main():
    conn = MockConnection()
    
    # Successful transaction
    print("Successful transaction:")
    async with database_transaction(conn):
        await conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        await conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    
    print("\\nFailed transaction:")
    # Failed transaction (automatic rollback)
    try:
        async with database_transaction(conn):
            await conn.execute("INSERT INTO users VALUES (3, 'Charlie')")
            raise ValueError("Something went wrong!")
    except ValueError:
        print("Exception handled")

asyncio.run(main())

# Output:
# Successful transaction:
# BEGIN TRANSACTION
#   Executing: BEGIN
#   Executing: INSERT INTO users VALUES (1, 'Alice')
#   Executing: INSERT INTO users VALUES (2, 'Bob')
# COMMIT
#   Executing: COMMIT
#
# Failed transaction:
# BEGIN TRANSACTION
#   Executing: BEGIN
#   Executing: INSERT INTO users VALUES (3, 'Charlie')
# ROLLBACK due to Something went wrong!
#   Executing: ROLLBACK
# Exception handled
\`\`\`

---

## Async Generators

Async generators combine \`async def\` with \`yield\` to stream values asynchronously.

### Basic Async Generators

\`\`\`python
"""
Creating and Using Async Generators
"""

import asyncio

async def async_range(start, stop):
    """Async version of range()"""
    current = start
    while current < stop:
        await asyncio.sleep(0.1)  # Simulate async operation
        yield current
        current += 1

async def main():
    print("Iterating async generator:")
    async for value in async_range(0, 5):
        print(f"  Got: {value}")

asyncio.run(main())

# Output:
# Iterating async generator:
#   Got: 0  (after 0.1s)
#   Got: 1  (after 0.2s)
#   Got: 2  (after 0.3s)
#   Got: 3  (after 0.4s)
#   Got: 4  (after 0.5s)
\`\`\`

### Async Generators for Data Streaming

\`\`\`python
"""
Streaming Large Datasets with Async Generators
"""

import asyncio
import aiohttp

async def fetch_pages(urls):
    """
    Fetch pages and yield them as they complete
    Memory-efficient: Don't load all pages at once
    """
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.get(url) as response:
                    content = await response.text()
                    yield {
                        'url': url,
                        'content': content,
                        'status': response.status
                    }
            except Exception as e:
                yield {
                    'url': url,
                    'error': str(e),
                    'status': None
                }

async def process_pages():
    """Process pages as they arrive (streaming)"""
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/status/404',
        'https://httpbin.org/delay/1',
    ]
    
    processed_count = 0
    
    async for page in fetch_pages(urls):
        processed_count += 1
        
        if 'error' in page:
            print(f"[{processed_count}] Error fetching {page['url']}: {page['error']}")
        else:
            print(f"[{processed_count}] Fetched {page['url']}: {page['status']} ({len(page['content'])} bytes)")
    
    print(f"\\nProcessed {processed_count} pages")

# asyncio.run(process_pages())

# Key benefit: Process each page as it arrives
# Don't wait for all 100 pages before processing
# Memory-efficient: Only one page in memory at a time
\`\`\`

### Async Generator with Context Manager

\`\`\`python
"""
Combining Async Generator with Context Manager
"""

import asyncio

@asynccontextmanager
async def file_reader(filename):
    """Async context manager for file reading"""
    print(f"Opening {filename}")
    file = await asyncio.to_thread(open, filename, 'r')
    try:
        yield file
    finally:
        print(f"Closing {filename}")
        await asyncio.to_thread(file.close)

async def read_lines_async(filename):
    """Async generator that yields lines from file"""
    async with file_reader(filename) as f:
        while True:
            line = await asyncio.to_thread(f.readline)
            if not line:
                break
            await asyncio.sleep(0.01)  # Simulate processing
            yield line.strip()

async def process_large_file():
    """Process large file line-by-line (streaming)"""
    async for line in read_lines_async('large_file.txt'):
        # Process each line as it's read
        print(f"Processing: {line[:50]}...")  # First 50 chars

# Memory-efficient: Don't load entire file
# Responsive: Can cancel/timeout while processing
\`\`\`

### Async Generator Methods

\`\`\`python
"""
Async Generator Methods: asend(), athrow(), aclose()
"""

import asyncio

async def echo_generator():
    """Generator that echoes values sent to it"""
    value = None
    while True:
        try:
            # yield can receive values via send()
            received = yield value
            print(f"Received: {received}")
            value = f"Echo: {received}"
        except GeneratorExit:
            print("Generator closing")
            break
        except Exception as e:
            print(f"Exception in generator: {e}")
            value = f"Error: {e}"

async def main():
    gen = echo_generator()
    
    # Start generator
    await gen.asend(None)  # First send must be None
    
    # Send values
    result = await gen.asend("Hello")
    print(f"Got: {result}")
    
    result = await gen.asend("World")
    print(f"Got: {result}")
    
    # Throw exception into generator
    result = await gen.athrow(ValueError("Something wrong"))
    print(f"Got: {result}")
    
    # Close generator
    await gen.aclose()

asyncio.run(main())

# Output:
# Received: Hello
# Got: Echo: Hello
# Received: World
# Got: Echo: World
# Exception in generator: Something wrong
# Got: Error: Something wrong
# Generator closing
\`\`\`

---

## Real-World Patterns

### Database Connection Pool

\`\`\`python
"""
Production Pattern: Database Connection Pool
"""

import asyncio
from contextlib import asynccontextmanager
import asyncpg

class AsyncDatabasePool:
    def __init__(self, dsn, min_size=10, max_size=20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
    
    async def __aenter__(self):
        """Initialize connection pool"""
        print(f"Creating connection pool ({self.min_size}-{self.max_size} connections)")
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close connection pool"""
        print("Closing connection pool")
        await self.pool.close()
        return False
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self):
        """Acquire connection and start transaction"""
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection

async def main():
    dsn = "postgresql://user:password@localhost/mydb"
    
    async with AsyncDatabasePool(dsn) as db:
        # Simple query with auto-managed connection
        async with db.acquire() as conn:
            users = await conn.fetch("SELECT * FROM users")
            print(f"Found {len(users)} users")
        
        # Transaction with auto-commit/rollback
        async with db.transaction() as conn:
            await conn.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
            await conn.execute("INSERT INTO orders (user_id) VALUES ($1)", 1)
            # Automatically commits if no exception
        
        # Multiple concurrent queries (pool manages connections)
        async def fetch_user(user_id):
            async with db.acquire() as conn:
                return await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        
        users = await asyncio.gather(*[fetch_user(i) for i in range(1, 11)])

# Connection pool automatically created and closed
# asyncio.run(main())
\`\`\`

### API Rate Limiter with Context Manager

\`\`\`python
"""
Production Pattern: Rate-Limited API Client
"""

import asyncio
import time
from contextlib import asynccontextmanager

class RateLimiter:
    """Rate limiter using token bucket algorithm"""
    
    def __init__(self, rate, per):
        """
        rate: Number of requests allowed
        per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until request is allowed"""
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            # Add tokens for time passed
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                # Not enough tokens, wait
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

class RateLimitedClient:
    """API client with rate limiting"""
    
    def __init__(self, rate=10, per=1.0):
        self.limiter = RateLimiter(rate, per)
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        return False
    
    async def get(self, url):
        """Rate-limited GET request"""
        await self.limiter.acquire()
        async with self.session.get(url) as response:
            return await response.json()

async def main():
    # Allow 10 requests per second
    async with RateLimitedClient(rate=10, per=1.0) as client:
        urls = [f"https://api.example.com/data/{i}" for i in range(50)]
        
        # Makes 50 requests, automatically rate-limited to 10/second
        results = await asyncio.gather(*[client.get(url) for url in urls])
        
        print(f"Fetched {len(results)} items (rate-limited)")

# Takes ~5 seconds (50 requests / 10 per second)
# asyncio.run(main())
\`\`\`

### Streaming Data Processing Pipeline

\`\`\`python
"""
Production Pattern: Streaming ETL Pipeline
"""

import asyncio
from typing import AsyncIterator

async def extract_data(source_urls) -> AsyncIterator[dict]:
    """Extract: Fetch data from sources"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        for url in source_urls:
            async with session.get(url) as response:
                data = await response.json()
                for record in data['records']:
                    yield record

async def transform_data(records: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Transform: Process and clean records"""
    async for record in records:
        # Transform record
        transformed = {
            'id': record['id'],
            'name': record['name'].upper(),
            'email': record['email'].lower(),
            'created_at': record.get('created_at'),
        }
        
        # Validate
        if transformed['email'] and '@' in transformed['email']:
            yield transformed

async def load_data(records: AsyncIterator[dict], batch_size=100):
    """Load: Insert records into database in batches"""
    batch = []
    count = 0
    
    async for record in records:
        batch.append(record)
        
        if len(batch) >= batch_size:
            # Insert batch
            await insert_batch(batch)
            count += len(batch)
            print(f"Loaded {count} records")
            batch = []
    
    # Insert remaining
    if batch:
        await insert_batch(batch)
        count += len(batch)
        print(f"Loaded {count} records (final)")

async def insert_batch(batch):
    """Insert batch into database"""
    await asyncio.sleep(0.1)  # Simulate database insert
    # await db.executemany("INSERT INTO users ...", batch)

async def etl_pipeline(source_urls):
    """Complete ETL pipeline with streaming"""
    records = extract_data(source_urls)
    transformed = transform_data(records)
    await load_data(transformed, batch_size=100)

# Memory-efficient: Process one record at a time
# Responsive: Can monitor/cancel during processing
# Scalable: Can handle millions of records

async def main():
    sources = [
        "https://api.example.com/users?page=1",
        "https://api.example.com/users?page=2",
        "https://api.example.com/users?page=3",
    ]
    
    await etl_pipeline(sources)

# asyncio.run(main())
\`\`\`

---

## Best Practices

### Always Use Async Context Managers for Resources

\`\`\`python
# ✅ Good: Automatic cleanup
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
    # Session and response automatically closed

# ❌ Bad: Manual cleanup (easy to forget, especially with exceptions)
async def fetch_data_bad():
    session = aiohttp.ClientSession()
    response = await session.get(url)
    data = await response.json()
    await response.close()  # Forgotten if exception occurs
    await session.close()   # Forgotten if exception occurs
    return data
\`\`\`

### Use @asynccontextmanager for Simple Cases

\`\`\`python
# ✅ Simple async context manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def temp_resource():
    resource = await acquire_resource()
    try:
        yield resource
    finally:
        await release_resource(resource)

# ❌ Verbose class-based (only needed for complex cases)
class TempResource:
    async def __aenter__(self):
        self.resource = await acquire_resource()
        return self.resource
    
    async def __aexit__(self, *args):
        await release_resource(self.resource)
\`\`\`

### Stream Large Datasets with Async Generators

\`\`\`python
# ✅ Memory-efficient streaming
async def process_large_dataset():
    async for record in fetch_records_streaming():
        await process_record(record)
    # Memory: One record at a time

# ❌ Load everything (high memory)
async def process_all_at_once():
    records = await fetch_all_records()  # Load 1M records in memory!
    for record in records:
        await process_record(record)
\`\`\`

---

## Summary

### Key Takeaways

1. **Async Context Managers**
   - Use \`async with\` for resource management
   - Define with \`__aenter__\` and \`__aexit__\`
   - Or use \`@asynccontextmanager\` decorator
   - Guarantees cleanup even on exceptions

2. **Async Generators**
   - Combine \`async def\` with \`yield\`
   - Iterate with \`async for\`
   - Memory-efficient streaming
   - Process data as it arrives

3. **Real-World Patterns**
   - Database connection pools
   - Transaction management
   - Rate limiting
   - ETL pipelines
   - Resource lifecycle management

4. **Best Practices**
   - Always use context managers for resources
   - Prefer \`@asynccontextmanager\` for simplicity
   - Stream large datasets with async generators
   - Combine both for powerful patterns

5. **Common Use Cases**
   - Database connections
   - HTTP sessions
   - File handles
   - Locks and semaphores
   - Transaction boundaries
   - Data streaming

### Next Steps

Now that you master async context managers and generators, we'll explore:
- Built-in asyncio functions for common patterns
- Async HTTP clients with aiohttp
- Async database operations
- Production async patterns

**Remember**: Context managers and generators are essential for resource management and data streaming in production async applications. Master them for robust, memory-efficient code.
`,
};
