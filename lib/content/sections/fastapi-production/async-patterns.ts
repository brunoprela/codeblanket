export const asyncPatterns = {
  title: 'Async FastAPI Patterns',
  id: 'async-patterns',
  content: `
# Async FastAPI Patterns

## Introduction

FastAPI is built on Starlette, an async web framework. Understanding async/await is critical for building high-performance APIs that handle thousands of concurrent connections efficiently.

**Why async matters:**
- **Concurrency**: Handle many requests simultaneously
- **I/O efficiency**: Don't block on database/network operations
- **Performance**: Better resource utilization
- **Scalability**: Serve more users with same hardware

**Common misconceptions:**
- ❌ "Async makes everything faster" - Not for CPU-bound tasks
- ❌ "Always use async" - Sync is simpler when appropriate
- ❌ "Async is parallel" - It\'s concurrent, not parallel

In this section, you'll master:
- Async vs sync: when to use each
- AsyncIO event loop fundamentals
- Async database operations
- Concurrent request patterns
- Async HTTP clients
- Common async pitfalls
- Performance optimization
- Production async patterns

---

## Async vs Sync

### Understanding the Difference

\`\`\`
Synchronous (Blocking):
Request 1: [===========]          Done
Request 2:             [===========]          Done
Request 3:                         [===========]          Done

Total time: 30 seconds (10s each)

Asynchronous (Non-blocking):
Request 1: [======I/O wait======]
Request 2:    [======I/O wait======]
Request 3:       [======I/O wait======]

Total time: ~10 seconds (concurrent I/O)

Key: Async shines when waiting for I/O (database, network, files)
\`\`\`

### When to Use Each

\`\`\`python
"""
Choosing between async and sync
"""

# ✅ Use ASYNC for I/O-bound operations
@app.get("/users")
async def get_users():
    """
    Database query - I/O bound
    Multiple users can query concurrently
    """
    async with db.session() as session:
        users = await session.execute (select(User))
        return users.scalars().all()

@app.get("/external-api")
async def call_external_api():
    """
    External HTTP request - I/O bound
    Don't block while waiting for response
    """
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# ✅ Use SYNC for CPU-bound operations
@app.post("/process-data")
def process_data (data: list):
    """
    Data processing - CPU bound
    Runs in thread pool, doesn't block event loop
    """
    result = []
    for item in data:
        # CPU-intensive calculation
        processed = complex_calculation (item)
        result.append (processed)
    return result

# ❌ DON'T mix blocking code in async functions
@app.get("/bad-example")
async def bad_async():
    """
    BAD: Blocking operation in async function
    Blocks event loop, prevents other requests
    """
    time.sleep(5)  # ❌ BLOCKS EVENT LOOP!
    return {"done": True}

# ✅ DO wrap blocking code
@app.get("/good-example")
async def good_async():
    """
    GOOD: Run blocking code in thread pool
    """
    import asyncio
    
    def blocking_operation():
        time.sleep(5)
        return {"done": True}
    
    # Run in thread pool
    result = await asyncio.to_thread (blocking_operation)
    return result
\`\`\`

---

## AsyncIO Fundamentals

### Event Loop Basics

\`\`\`python
"""
Understanding the AsyncIO event loop
"""

import asyncio

# The event loop
"""
1. Event loop runs in single thread
2. Schedules and executes async tasks
3. When task awaits, loop switches to another task
4. No task runs until previous yields control (await)
"""

async def task1():
    print("Task 1 start")
    await asyncio.sleep(1)  # Yield control
    print("Task 1 end")

async def task2():
    print("Task 2 start")
    await asyncio.sleep(1)  # Yield control
    print("Task 2 end")

# Run concurrently
async def main():
    await asyncio.gather (task1(), task2())
    # Output:
    # Task 1 start
    # Task 2 start
    # (both sleep concurrently)
    # Task 1 end
    # Task 2 end

# Total time: ~1 second (not 2!)
\`\`\`

### Async Context Managers

\`\`\`python
"""
Async context managers for resource management
"""

class AsyncDatabaseConnection:
    """
    Async context manager for database connections
    """
    async def __aenter__(self):
        """Called on 'async with' entry"""
        self.connection = await create_connection()
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Called on 'async with' exit"""
        await self.connection.close()

# Usage
async def query_database():
    async with AsyncDatabaseConnection() as conn:
        result = await conn.execute("SELECT * FROM users")
        return result
    # Connection automatically closed

# FastAPI example
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_connection():
    """Reusable async context manager"""
    conn = await db_pool.acquire()
    try:
        yield conn
    finally:
        await db_pool.release (conn)

@app.get("/users")
async def get_users():
    async with get_db_connection() as conn:
        users = await conn.fetch("SELECT * FROM users")
        return users
\`\`\`

---

## Async Database Operations

### SQLAlchemy Async

\`\`\`python
"""
Async SQLAlchemy with FastAPI
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

# Create async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True,
    pool_size=20,
    max_overflow=0
)

# Create async session maker
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dependency
async def get_db():
    async with async_session() as session:
        yield session

# Async CRUD operations
@app.get("/users/{user_id}")
async def get_user (user_id: int, db: AsyncSession = Depends (get_db)):
    """Async database query"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    
    return user

@app.post("/users")
async def create_user(
    user: UserCreate,
    db: AsyncSession = Depends (get_db)
):
    """Async database insert"""
    new_user = User(**user.dict())
    
    db.add (new_user)
    await db.commit()
    await db.refresh (new_user)
    
    return new_user

@app.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends (get_db)
):
    """Async database update"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException (status_code=404)
    
    for key, value in user_update.dict (exclude_unset=True).items():
        setattr (user, key, value)
    
    await db.commit()
    await db.refresh (user)
    
    return user
\`\`\`

### Async Transactions

\`\`\`python
"""
Async transactions for data consistency
"""

@app.post("/transfer")
async def transfer_money(
    from_account: int,
    to_account: int,
    amount: float,
    db: AsyncSession = Depends (get_db)
):
    """
    Transfer money between accounts (atomic)
    """
    async with db.begin():  # Transaction
        # Debit from account
        result = await db.execute(
            select(Account).where(Account.id == from_account).with_for_update()
        )
        from_acc = result.scalar_one()
        
        if from_acc.balance < amount:
            raise HTTPException(400, "Insufficient funds")
        
        from_acc.balance -= amount
        
        # Credit to account
        result = await db.execute(
            select(Account).where(Account.id == to_account).with_for_update()
        )
        to_acc = result.scalar_one()
        to_acc.balance += amount
        
        # Commit happens automatically on context exit
        # Rollback on exception
    
    return {"transferred": amount}
\`\`\`

---

## Concurrent Requests

### asyncio.gather()

\`\`\`python
"""
Execute multiple operations concurrently
"""

import asyncio
import httpx

@app.get("/user-dashboard/{user_id}")
async def get_user_dashboard(
    user_id: int,
    db: AsyncSession = Depends (get_db)
):
    """
    Fetch user data, posts, and comments concurrently
    """
    # Sequential (slow): 3 seconds
    # user = await fetch_user (user_id)           # 1s
    # posts = await fetch_user_posts (user_id)    # 1s
    # comments = await fetch_user_comments (user_id)  # 1s
    
    # Concurrent (fast): 1 second
    user, posts, comments = await asyncio.gather(
        fetch_user (user_id, db),
        fetch_user_posts (user_id, db),
        fetch_user_comments (user_id, db)
    )
    
    return {
        "user": user,
        "posts": posts,
        "comments": comments
    }

async def fetch_user (user_id: int, db: AsyncSession):
    result = await db.execute (select(User).where(User.id == user_id))
    return result.scalar_one()

async def fetch_user_posts (user_id: int, db: AsyncSession):
    result = await db.execute (select(Post).where(Post.user_id == user_id))
    return result.scalars().all()

async def fetch_user_comments (user_id: int, db: AsyncSession):
    result = await db.execute (select(Comment).where(Comment.user_id == user_id))
    return result.scalars().all()
\`\`\`

### Error Handling with gather()

\`\`\`python
"""
Handle errors in concurrent operations
"""

@app.get("/aggregate-data")
async def aggregate_data():
    """
    Fetch from multiple sources, handle failures gracefully
    """
    async def fetch_source1():
        # May fail
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api1.example.com/data")
            return response.json()
    
    async def fetch_source2():
        # May fail
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api2.example.com/data")
            return response.json()
    
    # Option 1: Fail if any fails
    try:
        data1, data2 = await asyncio.gather(
            fetch_source1(),
            fetch_source2()
        )
    except Exception as e:
        raise HTTPException(500, f"Data fetch failed: {e}")
    
    # Option 2: Continue on error
    results = await asyncio.gather(
        fetch_source1(),
        fetch_source2(),
        return_exceptions=True  # Return exceptions instead of raising
    )
    
    data = []
    for result in results:
        if isinstance (result, Exception):
            # Handle error
            logger.error (f"Source failed: {result}")
            data.append(None)
        else:
            data.append (result)
    
    return {"sources": data}
\`\`\`

---

## Async HTTP Clients

### httpx for Async Requests

\`\`\`python
"""
Make async HTTP requests with httpx
"""

import httpx

@app.get("/weather/{city}")
async def get_weather (city: str):
    """
    Fetch weather from external API
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.weather.com/v1/{city}",
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()

# Reusable client (connection pooling)
http_client = httpx.AsyncClient(
    timeout=10.0,
    limits=httpx.Limits (max_keepalive_connections=20, max_connections=100)
)

@app.on_event("startup")
async def startup():
    """Initialize HTTP client on startup"""
    global http_client
    http_client = httpx.AsyncClient (timeout=10.0)

@app.on_event("shutdown")
async def shutdown():
    """Close HTTP client on shutdown"""
    await http_client.aclose()

@app.get("/api-call")
async def make_api_call():
    """Use shared client"""
    response = await http_client.get("https://api.example.com/data")
    return response.json()
\`\`\`

---

## Common Async Pitfalls

### Pitfall 1: Blocking Operations

\`\`\`python
"""
DON'T block the event loop
"""

# ❌ BAD: Blocks event loop
@app.get("/bad")
async def blocking_async():
    time.sleep(5)  # Blocks ALL requests!
    return {"done": True}

# ✅ GOOD: Use asyncio.sleep
@app.get("/good")
async def non_blocking_async():
    await asyncio.sleep(5)  # Yields control
    return {"done": True}

# ✅ GOOD: Move blocking to thread
@app.get("/better")
async def blocking_in_thread():
    result = await asyncio.to_thread (blocking_operation)
    return result
\`\`\`

### Pitfall 2: Mixing Sync/Async

\`\`\`python
"""
Be careful mixing sync and async code
"""

# ❌ BAD: Sync DB call in async function
@app.get("/bad-db")
async def bad_database():
    # Blocks event loop!
    users = db.query(User).all()
    return users

# ✅ GOOD: Async DB call
@app.get("/good-db")
async def good_database (db: AsyncSession = Depends (get_db)):
    result = await db.execute (select(User))
    return result.scalars().all()
\`\`\`

### Pitfall 3: Shared Mutable State

\`\`\`python
"""
Async doesn't protect against race conditions
"""

# ❌ BAD: Race condition
counter = 0

@app.post("/increment")
async def increment():
    global counter
    temp = counter
    await asyncio.sleep(0)  # Simulate async work
    counter = temp + 1  # Race condition!
    return counter

# ✅ GOOD: Use locks
from asyncio import Lock

counter = 0
counter_lock = Lock()

@app.post("/increment-safe")
async def increment_safe():
    global counter
    async with counter_lock:
        temp = counter
        await asyncio.sleep(0)
        counter = temp + 1
    return counter
\`\`\`

---

## Performance Optimization

### Connection Pooling

\`\`\`python
"""
Reuse database connections for better performance
"""

# Configure connection pool
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,          # Number of connections to keep
    max_overflow=10,       # Additional connections if needed
    pool_timeout=30,       # Wait time for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True     # Check connection health
)
\`\`\`

### Caching Async Results

\`\`\`python
"""
Cache expensive async operations
"""

from functools import lru_cache
import aioredis

redis = aioredis.from_url("redis://localhost")

async def get_cached_data (key: str):
    """Get from cache or compute"""
    # Try cache
    cached = await redis.get (key)
    if cached:
        return json.loads (cached)
    
    # Compute
    data = await expensive_operation()
    
    # Cache result
    await redis.setex (key, 3600, json.dumps (data))
    
    return data
\`\`\`

---

## Summary

✅ **Async for I/O**: Database, HTTP, file operations  
✅ **Sync for CPU**: Calculations, data processing  
✅ **AsyncIO fundamentals**: Event loop, async/await  
✅ **Async database**: SQLAlchemy async engine  
✅ **Concurrent requests**: asyncio.gather() for parallel operations  
✅ **Async HTTP**: httpx AsyncClient with connection pooling  
✅ **Avoid pitfalls**: Don't block event loop, handle race conditions  
✅ **Performance**: Connection pooling, caching  

### Best Practices

**1. Use async for I/O, sync for CPU**
**2. Never block the event loop** (no time.sleep, blocking I/O)
**3. Use asyncio.gather()** for concurrent operations
**4. Connection pooling** for databases and HTTP clients
**5. Handle errors** in concurrent operations
**6. Use locks** for shared mutable state
**7. Profile and measure** - async isn't always faster

### Next Steps

In the next section, we'll explore **Production Deployment**: deploying FastAPI with Uvicorn, Gunicorn, Docker, and Kubernetes.
`,
};
