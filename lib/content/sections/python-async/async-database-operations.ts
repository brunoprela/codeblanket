export const asyncDatabaseOperations = {
  title: 'Async Database Operations',
  id: 'async-database-operations',
  content: `
# Async Database Operations

## Introduction

Async database operations are critical for high-performance applications. Traditional blocking database drivers freeze the entire event loop, but async drivers like **asyncpg** (PostgreSQL), **aiomysql** (MySQL), and **motor** (MongoDB) enable thousands of concurrent queries.

### Why Async Databases Matter

\`\`\`python
"""
Blocking vs Async Database Performance
"""

import asyncio
import asyncpg
import psycopg2  # Blocking driver

# Blocking: Sequential queries
def fetch_users_blocking(count):
    conn = psycopg2.connect("postgresql://...")
    cursor = conn.cursor()
    users = []
    for i in range(count):
        cursor.execute("SELECT * FROM users WHERE id = %s", (i,))
        users.append(cursor.fetchone())
    conn.close()
    return users

# Time: 100 queries × 10ms = 1 second

# Async: Concurrent queries
async def fetch_users_async(count):
    conn = await asyncpg.connect("postgresql://...")
    queries = [
        conn.fetchrow("SELECT * FROM users WHERE id = $1", i)
        for i in range(count)
    ]
    users = await asyncio.gather(*queries)
    await conn.close()
    return users

# Time: ~10ms (all queries concurrent!)

# 100× speedup with async!
\`\`\`

By the end of this section, you'll master:
- asyncpg for PostgreSQL (fastest async driver)
- Connection pooling for performance
- Transactions and error handling
- Bulk operations and optimization
- SQLAlchemy async (2.0+)
- Query patterns and best practices
- Production deployment strategies

---

## asyncpg: PostgreSQL Async Driver

asyncpg is the fastest PostgreSQL driver for Python, built in Cython.

### Basic Connection and Queries

\`\`\`python
"""
Basic asyncpg Usage
"""

import asyncio
import asyncpg

async def basic_queries():
    # Connect to database
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='password',
        database='mydb'
    )
    
    # Simple query (returns single row)
    user = await conn.fetchrow(
        "SELECT * FROM users WHERE id = $1",
        123
    )
    print(f"User: {user['name']}, {user['email']}")
    
    # Fetch multiple rows
    users = await conn.fetch(
        "SELECT * FROM users WHERE created_at > $1",
        '2024-01-01'
    )
    print(f"Found {len(users)} users")
    
    # Fetch single value
    count = await conn.fetchval(
        "SELECT COUNT(*) FROM users"
    )
    print(f"Total users: {count}")
    
    # Execute (no return value)
    await conn.execute(
        "INSERT INTO users (name, email) VALUES ($1, $2)",
        'Alice', 'alice@example.com'
    )
    
    # Close connection
    await conn.close()

asyncio.run(basic_queries())
\`\`\`

### Connection Pooling

\`\`\`python
"""
Connection Pool: Essential for Production
"""

import asyncio
import asyncpg

async def with_connection_pool():
    # Create connection pool
    pool = await asyncpg.create_pool(
        host='localhost',
        database='mydb',
        user='postgres',
        password='password',
        min_size=10,    # Minimum connections
        max_size=20,    # Maximum connections
        command_timeout=60.0
    )
    
    # Acquire connection from pool
    async with pool.acquire() as conn:
        users = await conn.fetch("SELECT * FROM users")
        print(f"Fetched {len(users)} users")
    
    # Connection automatically returned to pool
    
    # Execute query directly on pool (acquires internally)
    count = await pool.fetchval("SELECT COUNT(*) FROM orders")
    print(f"Total orders: {count}")
    
    # Close pool
    await pool.close()

# Best practice: Create pool once at startup, reuse everywhere
class Database:
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            "postgresql://user:pass@localhost/db",
            min_size=10,
            max_size=20
        )
    
    async def close(self):
        await self.pool.close()
    
    async def fetch_users(self):
        async with self.pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM users")

asyncio.run(with_connection_pool())
\`\`\`

---

## Transactions

### Basic Transactions

\`\`\`python
"""
Database Transactions with asyncpg
"""

import asyncpg
import asyncio

async def transaction_example():
    pool = await asyncpg.create_pool("postgresql://...")
    
    async with pool.acquire() as conn:
        # Start transaction
        async with conn.transaction():
            # All operations in this block are atomic
            await conn.execute(
                "INSERT INTO accounts (user_id, balance) VALUES ($1, $2)",
                123, 1000
            )
            
            await conn.execute(
                "INSERT INTO transactions (account_id, amount) VALUES ($1, $2)",
                123, 1000
            )
            
            # If any exception occurs, automatic rollback
            # Otherwise, commits when exiting block
    
    await pool.close()

# Automatic commit on success, rollback on exception!
\`\`\`

### Manual Transaction Control

\`\`\`python
"""
Manual Transaction Management
"""

import asyncpg
import asyncio

async def manual_transaction():
    pool = await asyncpg.create_pool("postgresql://...")
    
    async with pool.acquire() as conn:
        # Begin transaction
        transaction = conn.transaction()
        await transaction.start()
        
        try:
            # Transfer money between accounts
            await conn.execute(
                "UPDATE accounts SET balance = balance - $1 WHERE id = $2",
                100, 1  # Deduct from account 1
            )
            
            await conn.execute(
                "UPDATE accounts SET balance = balance + $1 WHERE id = $2",
                100, 2  # Add to account 2
            )
            
            # Verify balances
            balance1 = await conn.fetchval(
                "SELECT balance FROM accounts WHERE id = $1", 1
            )
            
            if balance1 < 0:
                raise ValueError("Insufficient funds")
            
            # Commit transaction
            await transaction.commit()
            print("Transfer successful")
        
        except Exception as e:
            # Rollback on error
            await transaction.rollback()
            print(f"Transfer failed: {e}")
    
    await pool.close()

asyncio.run(manual_transaction())
\`\`\`

### Nested Transactions (Savepoints)

\`\`\`python
"""
Savepoints for Partial Rollback
"""

import asyncpg
import asyncio

async def nested_transactions():
    pool = await asyncpg.create_pool("postgresql://...")
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Main transaction
            await conn.execute(
                "INSERT INTO users (name) VALUES ($1)", "Alice"
            )
            
            try:
                # Nested transaction (savepoint)
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO orders (user_id) VALUES ($1)", 999
                    )
                    # This will fail (invalid user_id)
                    raise Exception("Order failed")
            
            except Exception:
                # Inner transaction rolled back
                # But outer transaction continues
                print("Order failed, but user still created")
            
            await conn.execute(
                "INSERT INTO audit_log (message) VALUES ($1)",
                "User creation completed"
            )
            
            # Outer transaction commits
    
    await pool.close()

asyncio.run(nested_transactions())
\`\`\`

---

## Bulk Operations

### Batch Inserts

\`\`\`python
"""
Efficient Bulk Inserts
"""

import asyncpg
import asyncio

async def bulk_insert_example():
    pool = await asyncpg.create_pool("postgresql://...")
    
    # Prepare data (1000 users)
    users = [
        (f"user{i}", f"user{i}@example.com")
        for i in range(1000)
    ]
    
    async with pool.acquire() as conn:
        # Method 1: executemany (slower)
        await conn.executemany(
            "INSERT INTO users (name, email) VALUES ($1, $2)",
            users
        )
        
        # Method 2: copy_records_to_table (fastest!)
        await conn.copy_records_to_table(
            'users',
            records=users,
            columns=['name', 'email']
        )
        # Up to 100× faster than executemany!
    
    await pool.close()

# Benchmark (1M rows):
# executemany: ~60 seconds
# copy_records_to_table: ~0.6 seconds (100× faster!)

asyncio.run(bulk_insert_example())
\`\`\`

### Batch Updates

\`\`\`python
"""
Efficient Batch Updates
"""

import asyncpg
import asyncio

async def bulk_update_example():
    pool = await asyncpg.create_pool("postgresql://...")
    
    async with pool.acquire() as conn:
        # Update multiple rows efficiently
        user_ids = list(range(1, 1001))
        
        # Use unnest for batch update
        await conn.execute("""
            UPDATE users
            SET last_seen = NOW()
            WHERE id = ANY($1::int[])
        """, user_ids)
        
        # Or use temporary table for complex updates
        async with conn.transaction():
            # Create temp table
            await conn.execute("""
                CREATE TEMP TABLE user_updates (
                    user_id INT,
                    new_status TEXT
                )
            """)
            
            # Bulk insert to temp table
            updates = [(i, 'active') for i in range(1, 1001)]
            await conn.copy_records_to_table(
                'user_updates',
                records=updates,
                columns=['user_id', 'new_status']
            )
            
            # Update from temp table
            await conn.execute("""
                UPDATE users u
                SET status = ut.new_status
                FROM user_updates ut
                WHERE u.id = ut.user_id
            """)
    
    await pool.close()

asyncio.run(bulk_update_example())
\`\`\`

---

## Concurrent Queries

### Multiple Queries in Parallel

\`\`\`python
"""
Execute Multiple Queries Concurrently
"""

import asyncpg
import asyncio
import time

async def concurrent_queries_example():
    pool = await asyncpg.create_pool("postgresql://...")
    
    start = time.time()
    
    # Execute queries concurrently
    results = await asyncio.gather(
        pool.fetch("SELECT * FROM users LIMIT 1000"),
        pool.fetch("SELECT * FROM orders LIMIT 1000"),
        pool.fetch("SELECT * FROM products LIMIT 1000"),
        pool.fetchval("SELECT COUNT(*) FROM users"),
        pool.fetchval("SELECT COUNT(*) FROM orders"),
    )
    
    users, orders, products, user_count, order_count = results
    
    elapsed = time.time() - start
    print(f"Fetched all data in {elapsed:.2f}s")
    
    # Sequential would take: 5 queries × 100ms = 500ms
    # Concurrent takes: max(100ms) = 100ms
    # 5× speedup!
    
    await pool.close()

asyncio.run(concurrent_queries_example())
\`\`\`

---

## SQLAlchemy Async (2.0+)

### Async Engine and Sessions

\`\`\`python
"""
SQLAlchemy 2.0 Async Support
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import select, insert, update, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import asyncio

# Define models
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str]

# Create async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True,  # Log SQL
    pool_size=10,
    max_overflow=20
)

# Create session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def sqlalchemy_example():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Use session
    async with async_session() as session:
        # Query
        stmt = select(User).where(User.id == 1)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user:
            print(f"User: {user.name}, {user.email}")
        
        # Insert
        new_user = User(name="Alice", email="alice@example.com")
        session.add(new_user)
        await session.commit()
        
        # Update
        stmt = (
            update(User)
            .where(User.id == 1)
            .values(name="Updated Name")
        )
        await session.execute(stmt)
        await session.commit()
    
    await engine.dispose()

asyncio.run(sqlalchemy_example())
\`\`\`

---

## Error Handling

\`\`\`python
"""
Handling Database Errors
"""

import asyncpg
import asyncio

async def error_handling_example():
    pool = await asyncpg.create_pool("postgresql://...")
    
    async with pool.acquire() as conn:
        try:
            # Unique constraint violation
            await conn.execute(
                "INSERT INTO users (email) VALUES ($1)",
                "duplicate@example.com"
            )
        except asyncpg.UniqueViolationError as e:
            print(f"Duplicate email: {e}")
        
        try:
            # Foreign key violation
            await conn.execute(
                "INSERT INTO orders (user_id) VALUES ($1)",
                999999  # Non-existent user
            )
        except asyncpg.ForeignKeyViolationError as e:
            print(f"Invalid user_id: {e}")
        
        try:
            # Query timeout
            await conn.execute(
                "SELECT pg_sleep(10)",
                timeout=1.0
            )
        except asyncio.TimeoutError:
            print("Query timed out")
        
        try:
            # Syntax error
            await conn.execute("SELCT * FROM users")
        except asyncpg.PostgresSyntaxError as e:
            print(f"SQL syntax error: {e}")
    
    await pool.close()

asyncio.run(error_handling_example())
\`\`\`

---

## Production Patterns

### Database Manager Class

\`\`\`python
"""
Production-Ready Database Manager
"""

import asyncpg
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

class DatabaseManager:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=10,
            max_size=20,
            command_timeout=60.0,
            max_inactive_connection_lifetime=300.0
        )
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        async with self.pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def fetch_one(self, query: str, *args):
        """Fetch single row"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args):
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def execute(self, query: str, *args):
        """Execute query"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

# Usage
db = DatabaseManager("postgresql://user:pass@localhost/db")

async def main():
    await db.connect()
    
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", 123)
    print(f"User: {user}")
    
    async with db.transaction() as conn:
        await conn.execute("INSERT INTO users ...")
        await conn.execute("INSERT INTO orders ...")
    
    await db.close()

asyncio.run(main())
\`\`\`

---

## Summary

### Key Concepts

1. **asyncpg**: Fastest PostgreSQL driver
2. **Connection Pooling**: Reuse connections (10-20 pool size)
3. **Transactions**: Atomic operations with auto rollback
4. **Bulk Operations**: copy_records_to_table for speed
5. **Concurrent Queries**: Use gather() for parallel execution
6. **Error Handling**: Catch specific asyncpg exceptions

### Performance Tips

- Use connection pool (not individual connections)
- Batch operations with copy_records_to_table
- Execute independent queries concurrently
- Use prepared statements for repeated queries
- Set appropriate timeouts
- Monitor pool usage

### Next Steps

Now that you master async databases, we'll explore:
- Async file I/O operations
- Error handling patterns
- Building complete async applications
- Production deployment

**Remember**: Async databases enable thousands of concurrent queries with minimal overhead!
\`,
};
