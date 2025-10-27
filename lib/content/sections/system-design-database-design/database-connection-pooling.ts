/**
 * Database Connection Pooling Section
 */

export const databaseconnectionpoolingSection = {
  id: 'database-connection-pooling',
  title: 'Database Connection Pooling',
  content: `Database connection pooling is a critical performance optimization technique that reuses database connections instead of creating new ones for each request. Understanding connection pooling is essential for building scalable, high-performance applications.

## The Problem: Connection Overhead

### Creating a New Connection is Expensive

**What happens when you open a database connection:**1. **TCP handshake:** Client establishes TCP connection to database server (3-way handshake)
2. **Authentication:** Username/password verification
3. **Session initialization:** Database allocates memory, creates session context
4. **SSL/TLS negotiation:** If encryption is enabled (most production systems)

**Time cost:**
- Local connection: 5-10ms
- Same datacenter: 10-20ms
- Cross-region: 50-100ms+

**Resource cost:**
- Each connection consumes memory on both client and server
- PostgreSQL: ~10MB per connection
- MySQL: ~2-5MB per connection
- Server has connection limits (PostgreSQL default: 100, MySQL default: 151)

### Without Connection Pooling (Naive Approach)

\`\`\`python
def handle_request():
    # Open new connection
    conn = psycopg2.connect(
        host="db.example.com",
        database="mydb",
        user="user",
        password="password"
    )  # 10-20ms overhead

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    result = cursor.fetchone()  # Actual query: 1-2ms

    cursor.close()
    conn.close()  # Cleanup

    return result

# For 1000 requests/sec:
# - 1000 new connections/sec
# - 10-20 seconds total overhead (10-20ms × 1000)
# - Database server overwhelmed
\`\`\`

**Problems:**
- **High latency:** Connection setup >> query execution time
- **Resource exhaustion:** Database runs out of connections
- **CPU overhead:** Context switching, memory allocation/deallocation
- **Poor scalability:** Can't handle high concurrency

## What is Connection Pooling?

**Connection pooling** maintains a pool of reusable database connections that are shared across requests.

### How It Works:

\`\`\`
Application Request → Connection Pool → Database
                     ↑              ↓
                     └──────────────┘
                   Reuse connection
\`\`\`

**Lifecycle:**1. **Initialization:** Create pool with min connections (e.g., 10)
2. **Request arrives:** Checkout idle connection from pool
3. **Execute query:** Use connection
4. **Return to pool:** Release connection back to pool (don't close)
5. **Reuse:** Next request uses same connection (no setup overhead)

### With Connection Pooling:

\`\`\`python
# Initialize pool once (application startup)
pool = psycopg2.pool.SimpleConnectionPool(
    minconn=10,
    maxconn=100,
    host="db.example.com",
    database="mydb",
    user="user",
    password="password"
)

def handle_request():
    # Get connection from pool (instant)
    conn = pool.getconn()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()  # 1-2ms
        return result
    finally:
        # Return to pool (don't close!)
        pool.putconn (conn)

# For 1000 requests/sec:
# - 0 new connections (reuse from pool)
# - ~1-2 seconds total query time
# - Massive performance improvement!
\`\`\`

## Connection Pool Configuration

### Key Parameters

#### 1. Minimum Connections (minconn)

**Definition:** Number of connections to keep open at all times.

\`\`\`python
pool = ConnectionPool(
    minconn=10,  # Always maintain 10 open connections
    maxconn=100
)
\`\`\`

**Trade-offs:**
- **Higher min:** Faster for sudden traffic spikes, but more idle connections
- **Lower min:** Saves resources, but slower initial requests

**Guideline:** Set to average concurrent requests.

#### 2. Maximum Connections (maxconn)

**Definition:** Maximum number of connections pool can create.

\`\`\`python
pool = ConnectionPool(
    minconn=10,
    maxconn=100  # Never exceed 100 connections
)
\`\`\`

**Choosing maxconn:**

\`\`\`
maxconn = (number_of_app_instances) × (connections_per_instance)
         ≤ database_max_connections × 0.8
\`\`\`

**Example:**
- Database max connections: 200
- App instances: 4
- Safe maxconn per instance: (200 × 0.8) / 4 = 40

**Too high:** Database connection exhaustion
**Too low:** Requests wait for available connection

#### 3. Connection Timeout

**Definition:** How long to wait for an available connection.

\`\`\`python
pool = ConnectionPool(
    minconn=10,
    maxconn=100,
    timeout=30  # Wait up to 30 seconds
)

conn = pool.get_connection (timeout=5)  # Or override per request
if conn is None:
    raise TimeoutError("No connection available")
\`\`\`

**Recommendations:**
- **Web APIs:** 5-10 seconds (fail fast)
- **Background jobs:** 30-60 seconds (can wait)
- **Batch processing:** No timeout (blocking)

#### 4. Idle Timeout

**Definition:** Close connections idle for too long.

\`\`\`python
pool = ConnectionPool(
    minconn=10,
    maxconn=100,
    max_idle_time=300  # Close connections idle > 5 minutes
)
\`\`\`

**Why it matters:**
- Databases close idle connections after timeout
- Network issues can stale connections
- Prevents accumulation of dead connections

**Guideline:** Set slightly less than database's idle timeout.

#### 5. Max Lifetime

**Definition:** Close connections after maximum age.

\`\`\`python
pool = ConnectionPool(
    minconn=10,
    maxconn=100,
    max_lifetime=3600  # Refresh connections every hour
)
\`\`\`

**Why it matters:**
- Database restarts/failovers
- Credential rotation
- Memory leaks in long-lived connections

**Guideline:** 1-24 hours depending on stability needs.

## Connection Pool Implementations

### Python: psycopg2 (PostgreSQL)

\`\`\`python
import psycopg2
from psycopg2 import pool

# Create pool
db_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=10,
    maxconn=100,
    host="localhost",
    database="mydb",
    user="user",
    password="password"
)

def execute_query (query, params):
    conn = None
    try:
        # Get connection from pool
        conn = db_pool.getconn()

        cursor = conn.cursor()
        cursor.execute (query, params)
        result = cursor.fetchall()
        conn.commit()

        return result
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            # Return connection to pool
            db_pool.putconn (conn)
\`\`\`

### Python: SQLAlchemy (Universal)

\`\`\`python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:password@localhost/mydb",
    poolclass=QueuePool,
    pool_size=10,          # minconn
    max_overflow=90,       # additional connections beyond pool_size (total max = 100)
    pool_timeout=30,       # wait timeout
    pool_recycle=3600,     # max lifetime
    pool_pre_ping=True     # validate connections before use
)

# Usage
def execute_query (query):
    with engine.connect() as conn:
        result = conn.execute (query)
        return result.fetchall()
    # Connection automatically returned to pool
\`\`\`

### Java: HikariCP (Best in class)

\`\`\`java
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
config.setUsername("user");
config.setPassword("password");

// Pool configuration
config.setMinimumIdle(10);
config.setMaximumPoolSize(100);
config.setConnectionTimeout(30000);      // 30 seconds
config.setIdleTimeout(600000);           // 10 minutes
config.setMaxLifetime(1800000);          // 30 minutes
config.setConnectionTestQuery("SELECT 1");

HikariDataSource dataSource = new HikariDataSource (config);

// Usage
try (Connection conn = dataSource.getConnection()) {
    PreparedStatement stmt = conn.prepareStatement("SELECT * FROM users WHERE id = ?");
    stmt.setInt(1, userId);
    ResultSet rs = stmt.executeQuery();
    // Process results...
}
// Connection automatically returned to pool
\`\`\`

### Node.js: pg-pool (PostgreSQL)

\`\`\`javascript
const { Pool } = require('pg');

const pool = new Pool({
    host: 'localhost',
    database: 'mydb',
    user: 'user',
    password: 'password',
    min: 10,              // minconn
    max: 100,             // maxconn
    connectionTimeoutMillis: 30000,  // 30 seconds
    idleTimeoutMillis: 300000,       // 5 minutes
    maxUses: 7500         // close connection after 7500 queries
});

// Usage
async function executeQuery (query, params) {
    const client = await pool.connect();
    try {
        const result = await client.query (query, params);
        return result.rows;
    } finally {
        client.release();  // Return to pool
    }
}

// Graceful shutdown
process.on('SIGINT', async () => {
    await pool.end();
    process.exit(0);
});
\`\`\`

### Go: database/sql (Built-in)

\`\`\`go
import (
    "database/sql"
    _ "github.com/lib/pq"
)

db, err := sql.Open("postgres",
    "host=localhost dbname=mydb user=user password=password")
if err != nil {
    log.Fatal (err)
}

// Configure pool
db.SetMaxOpenConns(100)          // maxconn
db.SetMaxIdleConns(10)           // minconn
db.SetConnMaxLifetime (time.Hour) // max lifetime
db.SetConnMaxIdleTime(5 * time.Minute)  // idle timeout

// Usage
func executeQuery (query string, args ...interface{}) error {
    // Connection automatically checked out and returned
    rows, err := db.Query (query, args...)
    if err != nil {
        return err
    }
    defer rows.Close()

    // Process rows...
    return nil
}
\`\`\`

## Best Practices

### 1. Size Your Pool Correctly

**Formula for web applications:**

\`\`\`
optimal_pool_size = ((core_count × 2) + effective_spindle_count)
\`\`\`

For SSDs (no spinning disks):
\`\`\`
optimal_pool_size = core_count × 2
\`\`\`

**Example:**
- 8-core database server with SSD
- Optimal pool size per application instance: 16

**Common mistake:** "More connections = better performance"
- **Reality:** Too many connections cause contention and context switching
- **Sweet spot:** Usually 10-50 connections per app instance

### 2. Validate Connections (Health Checks)

**Problem:** Connections can become stale (network issues, database restart).

\`\`\`python
# SQLAlchemy: pre-ping
engine = create_engine(
    "postgresql://...",
    pool_pre_ping=True  # Test connection before use
)

# HikariCP: test query
config.setConnectionTestQuery("SELECT 1");

# Manual validation
def get_connection():
    conn = pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return conn
    except Exception:
        # Connection is dead, remove from pool
        pool.putconn (conn, close=True)
        return get_connection()  # Retry
\`\`\`

### 3. Handle Connection Leaks

**Connection leak:** Connection checked out but never returned to pool.

\`\`\`python
# BAD: Connection leak if exception occurs
def bad_function():
    conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    # Exception here → connection never returned!
    result = cursor.fetchall()
    pool.putconn (conn)
    return result

# GOOD: Always return connection
def good_function():
    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        result = cursor.fetchall()
        return result
    finally:
        if conn:
            pool.putconn (conn)

# BEST: Use context manager
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn (conn)

def best_function():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        return cursor.fetchall()
    # Connection automatically returned
\`\`\`

### 4. Monitor Pool Metrics

**Key metrics to track:**

\`\`\`python
# Pool health metrics
pool_metrics = {
    'total_connections': pool.size,
    'idle_connections': pool.available,
    'active_connections': pool.size - pool.available,
    'wait_count': pool.wait_count,          # Requests waiting for connection
    'wait_time_avg': pool.avg_wait_time,    # Average wait time
    'connection_errors': pool.error_count
}

# Alerts
if pool.available == 0:
    alert("Connection pool exhausted!")

if pool.avg_wait_time > 1.0:
    alert("High connection wait time")
\`\`\`

### 5. Graceful Shutdown

\`\`\`python
import signal
import sys

def shutdown_handler (signum, frame):
    print("Shutting down gracefully...")

    # Stop accepting new requests
    server.stop_accepting()

    # Wait for active connections to finish
    pool.wait_for_active_connections (timeout=30)

    # Close all connections
    pool.close_all()

    sys.exit(0)

signal.signal (signal.SIGTERM, shutdown_handler)
signal.signal (signal.SIGINT, shutdown_handler)
\`\`\`

## Advanced Patterns

### 1. Multiple Connection Pools

**Use case:** Separate read and write traffic.

\`\`\`python
# Write pool (smaller, points to primary)
write_pool = ConnectionPool(
    minconn=5,
    maxconn=20,
    host="primary.db.example.com"
)

# Read pool (larger, points to replicas)
read_pool = ConnectionPool(
    minconn=20,
    maxconn=100,
    host="replica.db.example.com"
)

def write_data (query, params):
    with write_pool.get_connection() as conn:
        conn.execute (query, params)

def read_data (query, params):
    with read_pool.get_connection() as conn:
        return conn.execute (query, params).fetchall()
\`\`\`

### 2. Priority Queues

**Use case:** Critical requests get connections first.

\`\`\`python
class PriorityConnectionPool:
    def __init__(self):
        self.pool = ConnectionPool (minconn=10, maxconn=50)
        self.high_priority_queue = queue.PriorityQueue()
        self.normal_queue = queue.Queue()

    def get_connection (self, priority='normal'):
        if self.pool.available > 0:
            return self.pool.getconn()

        # No available connection, queue the request
        if priority == 'high':
            self.high_priority_queue.put (get_connection_waiter())
        else:
            self.normal_queue.put (get_connection_waiter())

        # Wait for connection
        return wait_for_connection()
\`\`\`

### 3. Dynamic Pool Sizing

**Use case:** Scale pool based on load.

\`\`\`python
class DynamicConnectionPool:
    def __init__(self):
        self.min_size = 10
        self.max_size = 100
        self.current_size = self.min_size

    def adjust_pool_size (self):
        # Monitor metrics
        wait_time = self.get_avg_wait_time()
        utilization = self.active_connections / self.current_size

        # Scale up if high utilization
        if utilization > 0.8 and wait_time > 0.1:
            self.current_size = min (self.current_size + 10, self.max_size)
            self.add_connections(10)

        # Scale down if low utilization
        elif utilization < 0.2 and self.current_size > self.min_size:
            self.current_size = max (self.current_size - 10, self.min_size)
            self.remove_connections(10)
\`\`\`

## Common Mistakes

### 1. Creating Pool Per Request

\`\`\`python
# ❌ WRONG: Creates new pool for every request
def handle_request():
    pool = ConnectionPool (minconn=10, maxconn=100)
    conn = pool.getconn()
    # ... use connection
    pool.putconn (conn)

# ✅ CORRECT: Create pool once at startup
pool = ConnectionPool (minconn=10, maxconn=100)  # Global

def handle_request():
    conn = pool.getconn()
    # ... use connection
    pool.putconn (conn)
\`\`\`

### 2. Pool Size Too Large

\`\`\`python
# ❌ WRONG: Pool size exceeds database capacity
# Database max_connections = 100
# 10 app instances × 100 connections/instance = 1000 connections
pool = ConnectionPool (minconn=50, maxconn=100)  # Per instance

# ✅ CORRECT: Pool size accounts for all instances
# 10 instances × 8 connections/instance = 80 connections (safe)
pool = ConnectionPool (minconn=4, maxconn=8)  # Per instance
\`\`\`

### 3. Not Returning Connections

\`\`\`python
# ❌ WRONG: Connection leak
def query_users():
    conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()
    # Connection never returned!

# ✅ CORRECT: Always return in finally block
def query_users():
    conn = None
    try:
        conn = pool.getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        return cursor.fetchall()
    finally:
        if conn:
            pool.putconn (conn)
\`\`\`

### 4. Ignoring Pool Exhaustion

\`\`\`python
# ❌ WRONG: Block indefinitely waiting for connection
conn = pool.getconn()  # Hangs forever if pool exhausted

# ✅ CORRECT: Use timeout and handle gracefully
try:
    conn = pool.getconn (timeout=5)
except TimeoutError:
    return {"error": "Database busy, please try again"}
\`\`\`

## Debugging Connection Pool Issues

### Symptom: Requests Timing Out

**Possible causes:**1. Pool size too small
2. Slow queries holding connections
3. Connection leaks

**Debug:**
\`\`\`python
# Check pool status
print(f"Total: {pool.size}, Available: {pool.available}, Active: {pool.size - pool.available}")

# Identify slow queries
SELECT pid, now() - query_start as duration, query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC;

# Find connection leaks (long-running idle connections)
SELECT pid, usename, application_name, client_addr,
       now() - state_change as idle_time, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
ORDER BY idle_time DESC;
\`\`\`

### Symptom: Database Running Out of Connections

**Possible causes:**1. Too many application instances
2. Pool sizes too large
3. Zombie connections

**Debug:**
\`\`\`sql
-- PostgreSQL: Check connection count
SELECT count(*) FROM pg_stat_activity;
SELECT max_connections FROM pg_settings WHERE name = 'max_connections';

-- MySQL: Check connection count
SHOW STATUS WHERE variable_name = 'Threads_connected';
SHOW VARIABLES WHERE variable_name = 'max_connections';
\`\`\`

**Fix:**
- Reduce pool size per instance
- Increase database max_connections (with caution)
- Scale vertically (more database resources)

## Interview Tips

**Q: "Why use connection pooling?"**
- Eliminates expensive connection setup overhead (10-20ms per connection)
- Reuses existing connections (faster, less resource intensive)
- Prevents database connection exhaustion
- Enables handling high concurrency with limited database connections

**Q: "How do you size a connection pool?"**
- Start with formula: core_count × 2 per instance
- Consider: number of app instances × pool size ≤ database max × 0.8
- Monitor and adjust based on: wait times, utilization, query latency
- Too large causes contention; too small causes waits

**Q: "What problems can occur with connection pooling?"**
- **Connection leaks:** Not returning connections to pool
- **Stale connections:** Network issues, database restarts
- **Pool exhaustion:** Pool size too small for load
- **Resource contention:** Pool size too large causes context switching

**Q: "Describe connection pool lifecycle"**1. Initialization: Create min connections at startup
2. Checkout: Request gets idle connection from pool
3. Use: Execute queries
4. Return: Connection returned to pool (not closed)
5. Health check: Validate connection periodically
6. Expire: Close old/idle connections, create new ones

## Key Takeaways

1. **Connection pooling eliminates expensive setup overhead** (10-20ms per connection)
2. **Pool size formula: core_count × 2 per instance**3. **Total connections across all instances must not exceed database max**4. **Always return connections to pool (use finally blocks or context managers)**5. **Monitor pool metrics: active, idle, wait time, errors**6. **Validate connections before use (pre-ping, test queries)**7. **Set appropriate timeouts: idle timeout, max lifetime, checkout timeout**8. **Connection leaks are the most common issue** (always use try/finally)
9. **More connections ≠ better performance** (causes contention beyond optimal size)
10. **Different pools for different purposes** (read vs write, high vs low priority)

## Summary

Connection pooling is essential for performant database access. It eliminates the expensive overhead of creating new connections (10-20ms) by reusing existing ones. Properly configured pools balance resource efficiency with request throughput. Key parameters include min/max connections, timeouts, and health checks. Common pitfalls include oversized pools (resource contention), undersized pools (request waits), and connection leaks (not returning connections). Monitoring pool metrics and using context managers ensures reliable, high-performance database access at scale.
`,
};
