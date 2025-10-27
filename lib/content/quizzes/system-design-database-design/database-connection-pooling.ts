/**
 * Quiz questions for Database Connection Pooling section
 */

export const databaseconnectionpoolingQuiz = [
  {
    id: 'pool-disc-1',
    question:
      'Design a connection pooling strategy for a high-traffic web application with 20 application instances connecting to a primary database (max 500 connections) and 3 read replicas (max 500 connections each). The application handles both transactional writes and high-volume reads. Discuss pool sizing, separate pools for reads/writes, health checks, monitoring, and how to handle failover scenarios.',
    sampleAnswer: `Comprehensive connection pooling strategy for high-traffic application:

**System Overview:**
- 20 application instances
- 1 primary database (writes): max 500 connections
- 3 read replicas: max 500 connections each
- Mixed workload: 80% reads, 20% writes
- Peak load: 10,000 requests/second

**1. Pool Sizing Strategy**

**Write Pool (Primary Database):**
\`\`\`python
# Per application instance
write_pool = ConnectionPool(
    host="primary.db.example.com",
    minconn=5,
    maxconn=20,
    timeout=10,
    max_idle_time=300,
    max_lifetime=3600
)

# Total: 20 instances × 20 connections = 400 connections
# Database capacity: 500 connections
# Utilization: 80% (safe margin for admin, monitoring)
\`\`\`

**Rationale:**
- 20 connections per instance: Handles write load (20% of traffic)
- 400 total connections: Leaves 100 for admin/failover
- Smaller pool than reads: Writes are less frequent but more critical

**Read Pool (Replicas - Load Balanced):**
\`\`\`python
# Per application instance (distributed across 3 replicas)
read_pools = [
    ConnectionPool(
        host="replica1.db.example.com",
        minconn=10,
        maxconn=50,
        timeout=5,
        max_idle_time=300,
        max_lifetime=3600
    ),
    ConnectionPool(
        host="replica2.db.example.com",
        minconn=10,
        maxconn=50,
        timeout=5,
        max_idle_time=300,
        max_lifetime=3600
    ),
    ConnectionPool(
        host="replica3.db.example.com",
        minconn=10,
        maxconn=50,
        timeout=5,
        max_idle_time=300,
        max_lifetime=3600
    )
]

# Round-robin load balancing
current_replica = 0

def get_read_connection():
    global current_replica
    pool = read_pools[current_replica]
    current_replica = (current_replica + 1) % len (read_pools)
    return pool.getconn()

# Total per replica: 20 instances × 50 connections = 1000 max
# But load balanced, so average per replica: ~333 connections (66% capacity)
\`\`\`

**Rationale:**
- 50 connections per replica per instance: Handles high read volume
- Load balanced across 3 replicas: Distributes load, provides redundancy
- Larger pool than writes: Reads are 80% of traffic

**2. Connection Pool Manager**

\`\`\`python
class DatabaseConnectionManager:
    def __init__(self):
        self.write_pool = self._create_write_pool()
        self.read_pools = self._create_read_pools()
        self.read_pool_index = 0
        self.failed_replicas = set()
        
        # Health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
    
    def _create_write_pool (self):
        return ConnectionPool(
            host=os.environ['PRIMARY_DB_HOST',],
            database=os.environ['DB_NAME',],
            user=os.environ['DB_USER',],
            password=os.environ['DB_PASSWORD',],
            minconn=5,
            maxconn=20,
            timeout=10,
            max_idle_time=300,
            max_lifetime=3600,
            connect_timeout=5
        )
    
    def _create_read_pools (self):
        replica_hosts = os.environ['REPLICA_DB_HOSTS',].split(',')
        return [
            ConnectionPool(
                host=host,
                database=os.environ['DB_NAME',],
                user=os.environ['DB_USER',],
                password=os.environ['DB_PASSWORD',],
                minconn=10,
                maxconn=50,
                timeout=5,
                max_idle_time=300,
                max_lifetime=3600,
                connect_timeout=3
            )
            for host in replica_hosts
        ]
    
    @contextmanager
    def get_write_connection (self):
        conn = None
        try:
            conn = self.write_pool.getconn (timeout=10)
            # Validate connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.write_pool.putconn (conn)
    
    @contextmanager
    def get_read_connection (self):
        max_retries = len (self.read_pools)
        
        for attempt in range (max_retries):
            # Get next healthy replica (round-robin)
            pool_idx = self._get_next_healthy_replica()
            
            if pool_idx is None:
                # All replicas failed - fall back to primary
                logger.warning("All replicas unhealthy, using primary for reads")
                with self.get_write_connection() as conn:
                    yield conn
                return
            
            pool = self.read_pools[pool_idx]
            conn = None
            
            try:
                conn = pool.getconn (timeout=5)
                
                # Validate connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                
                yield conn
                return
                
            except Exception as e:
                logger.error (f"Read replica {pool_idx} failed: {e}")
                self.failed_replicas.add (pool_idx)
                
                if conn:
                    # Remove bad connection from pool
                    pool.putconn (conn, close=True)
                
                # Retry with next replica
                continue
            
            finally:
                if conn:
                    pool.putconn (conn)
        
        raise Exception("All database connections failed")
    
    def _get_next_healthy_replica (self):
        healthy_replicas = [
            i for i in range (len (self.read_pools))
            if i not in self.failed_replicas
        ]
        
        if not healthy_replicas:
            return None
        
        # Round-robin among healthy replicas
        self.read_pool_index = (self.read_pool_index + 1) % len (healthy_replicas)
        return healthy_replicas[self.read_pool_index]
    
    def _health_check_loop (self):
        while True:
            try:
                self._check_replica_health()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error (f"Health check error: {e}")
    
    def _check_replica_health (self):
        for idx, pool in enumerate (self.read_pools):
            if idx in self.failed_replicas:
                # Try to reconnect failed replicas
                try:
                    conn = pool.getconn (timeout=3)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    pool.putconn (conn)
                    
                    # Replica is healthy again
                    logger.info (f"Replica {idx} recovered")
                    self.failed_replicas.discard (idx)
                    
                except Exception:
                    # Still unhealthy
                    pool.putconn (conn, close=True) if conn else None

# Global instance
db_manager = DatabaseConnectionManager()
\`\`\`

**3. Usage Patterns**

\`\`\`python
# Write operation
def create_order (user_id, items):
    with db_manager.get_write_connection() as conn:
        cursor = conn.cursor()
        
        # Insert order
        cursor.execute("""
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (%s, %s, %s)
            RETURNING order_id
        """, (user_id, calculate_total (items), 'pending'))
        
        order_id = cursor.fetchone()[0]
        
        # Insert order items
        for item in items:
            cursor.execute("""
                INSERT INTO order_items (order_id, product_id, quantity, price)
                VALUES (%s, %s, %s, %s)
            """, (order_id, item['product_id',], item['quantity',], item['price',]))
        
        return order_id

# Read operation
def get_product (product_id):
    with db_manager.get_read_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE product_id = %s", (product_id,))
        return cursor.fetchone()

# Read-after-write (use primary to avoid replication lag)
def get_fresh_order (order_id):
    with db_manager.get_write_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,))
        return cursor.fetchone()
\`\`\`

**4. Monitoring and Metrics**

\`\`\`python
class PoolMetrics:
    def __init__(self, pool, name):
        self.pool = pool
        self.name = name
    
    def get_metrics (self):
        return {
            f"{self.name}_total_connections": self.pool.size,
            f"{self.name}_active_connections": self.pool.size - self.pool.available,
            f"{self.name}_idle_connections": self.pool.available,
            f"{self.name}_wait_count": self.pool.wait_count,
            f"{self.name}_avg_wait_time_ms": self.pool.avg_wait_time * 1000,
            f"{self.name}_connection_errors": self.pool.error_count
        }

# Collect metrics
def collect_pool_metrics():
    metrics = {}
    
    # Write pool metrics
    write_metrics = PoolMetrics (db_manager.write_pool, "write_pool")
    metrics.update (write_metrics.get_metrics())
    
    # Read pool metrics (per replica)
    for idx, pool in enumerate (db_manager.read_pools):
        read_metrics = PoolMetrics (pool, f"read_pool_{idx}")
        metrics.update (read_metrics.get_metrics())
    
    # Failed replicas
    metrics['failed_replicas_count',] = len (db_manager.failed_replicas)
    
    return metrics

# Expose metrics endpoint
@app.route('/metrics')
def metrics():
    return jsonify (collect_pool_metrics())

# Alert rules
def check_alerts():
    metrics = collect_pool_metrics()
    
    # Write pool exhaustion
    if metrics['write_pool_idle_connections',] == 0:
        alert("CRITICAL: Write pool exhausted!")
    
    # High wait times
    if metrics['write_pool_avg_wait_time_ms',] > 100:
        alert("WARNING: High write pool wait time")
    
    # All replicas failed
    if metrics['failed_replicas_count',] == len (db_manager.read_pools):
        alert("CRITICAL: All read replicas failed!")
\`\`\`

**5. Failover Scenarios**

**Scenario 1: Primary Database Fails**

\`\`\`python
# Automatic failover using database proxy (PgBouncer, ProxySQL)
# Or manual promotion:

def promote_replica_to_primary (replica_idx):
    # 1. Stop all writes
    db_manager.write_pool.disable()
    
    # 2. Wait for replication lag to catch up
    wait_for_replication_sync (replica_idx)
    
    # 3. Promote replica
    promote_replica_command (replica_idx)
    
    # 4. Reconfigure write pool to new primary
    db_manager.write_pool = ConnectionPool(
        host=db_manager.read_pools[replica_idx].host,
        # ... same config
    )
    
    # 5. Remove promoted replica from read pools
    del db_manager.read_pools[replica_idx]
    
    # 6. Resume writes
    logger.info("Failover complete")
\`\`\`

**Scenario 2: Read Replica Fails**

Handled automatically by health checks:
1. Health check detects failure
2. Replica marked as failed
3. Traffic routes to remaining healthy replicas
4. Periodic health checks attempt reconnection
5. When recovered, replica returns to rotation

**Scenario 3: All Replicas Fail**

\`\`\`python
# Fallback to primary for reads
if all replicas unhealthy:
    route_reads_to_primary()
    alert("Using primary for reads - degraded performance")
\`\`\`

**6. Performance Benchmarks**

| Scenario | Without Pooling | With Pooling | Improvement |
|----------|----------------|--------------|-------------|
| Simple SELECT | 15ms | 2ms | 7.5x faster |
| Complex JOIN | 50ms | 37ms | 1.35x faster |
| INSERT | 20ms | 5ms | 4x faster |
| 10,000 req/sec throughput | Can't sustain | Sustained | Scalable |

**7. Best Practices Summary**

✅ Separate pools for reads and writes
✅ Size based on database capacity and instance count
✅ Load balance reads across replicas
✅ Health checks with automatic failover
✅ Monitor pool metrics and set alerts
✅ Use context managers (no connection leaks)
✅ Validate connections before use
✅ Set appropriate timeouts
✅ Graceful degradation (fallback to primary)
✅ Regular load testing and capacity planning

This architecture provides high performance, reliability, and automatic failover for a production-grade system.`,
    keyPoints: [
      'Separate pools for reads (replicas) and writes (primary)',
      'Pool sizing: core_count × 2, use 80% of database max_connections',
      'Health checks with automatic failover to backup replicas',
      'Context managers prevent connection leaks',
      'Monitor pool metrics: utilization, wait times, connection age',
    ],
  },
  {
    id: 'pool-disc-2',
    question:
      'You notice that during peak traffic hours, your application experiences intermittent 5-second delays on database queries, and monitoring shows connection pool utilization spiking to 100%. However, the database CPU and memory are only at 40% utilization. Diagnose the problem and propose solutions. Consider both immediate fixes and long-term architectural improvements.',
    sampleAnswer: `**Problem Diagnosis:**

The issue is **connection pool exhaustion** - not database capacity problems.

**Root Causes:**1. **Pool too small** for peak traffic
2. **Long-running queries** holding connections
3. **Connection leaks** (connections not returned to pool)
4. **Thundering herd** during traffic spikes

**Evidence Analysis:**

\`\`\`python
# Current metrics show:
pool_size = 20  # Per application instance
active_instances = 10
peak_requests_per_second = 5000
average_query_time = 50ms  # milliseconds

# Theoretical capacity:
total_connections = 20 × 10 = 200
queries_per_second_capacity = 200 / 0.05 = 4000 req/sec
# We need 5000 req/sec but can only handle 4000
# Pool is undersized by 25%
\`\`\`

**Immediate Fixes (Within 1 hour):**

**1. Increase Pool Size**

\`\`\`python
# Before
connection_pool = ConnectionPool(
    minconn=5,
    maxconn=20,  # Too small
    timeout=10
)

# After (temporary fix)
connection_pool = ConnectionPool(
    minconn=10,
    maxconn=40,  # Increased to handle peak
    timeout=5,  # Reduced timeout to fail fast
    max_overflow=10  # Allow temporary burst beyond maxconn
)

# For 10 instances: 10 × 40 = 400 connections
# Database max: 500 connections (80% utilization is safe)
\`\`\`

**2. Implement Connection Timeouts**

\`\`\`python
@contextmanager
def get_connection_with_timeout (timeout=5):
    start_time = time.time()
    
    try:
        # Try to get connection with timeout
        conn = pool.getconn (timeout=timeout)
        
        elapsed = time.time() - start_time
        
        if elapsed > 1.0:
            # Log slow connection acquisition
            logger.warning (f"Slow connection acquisition: {elapsed:.2f}s")
            metrics.increment('db.connection.slow_acquisition')
        
        yield conn
        
    except PoolTimeout:
        # Log and fail fast instead of hanging
        logger.error("Connection pool exhausted")
        metrics.increment('db.connection.pool_exhausted')
        raise ServiceUnavailable("Database connection unavailable")
    
    finally:
        if conn:
            pool.putconn (conn)
\`\`\`

**3. Add Connection Leak Detection**

\`\`\`python
import traceback
import time

class LeakDetectionPool:
    def __init__(self, *args, **kwargs):
        self.pool = ConnectionPool(*args, **kwargs)
        self.active_connections = {}
        self.lock = threading.Lock()
    
    def getconn (self, timeout=None):
        conn = self.pool.getconn (timeout=timeout)
        
        with self.lock:
            # Track where connection was acquired
            self.active_connections[id (conn)] = {
                'acquired_at': time.time(),
                'stack_trace': '.join (traceback.format_stack()),
                'thread': threading.current_thread().name
            }
        
        return conn
    
    def putconn (self, conn, close=False):
        with self.lock:
            if id (conn) in self.active_connections:
                del self.active_connections[id (conn)]
        
        self.pool.putconn (conn, close=close)
    
    def check_for_leaks (self):
        """Call this periodically (every 30 seconds)"""
        with self.lock:
            current_time = time.time()
            
            for conn_id, info in self.active_connections.items():
                age = current_time - info['acquired_at',]
                
                if age > 60:  # Connection held > 60 seconds
                    logger.error (f"Connection leak detected!")
                    logger.error (f"Age: {age:.2f}s")
                    logger.error (f"Thread: {info['thread',]}")
                    logger.error (f"Stack trace:\\n{info['stack_trace',]}")
                    
                    metrics.increment('db.connection.leak_detected')
\`\`\`

**4. Query Timeout Enforcement**

\`\`\`python
def execute_with_timeout (cursor, query, params=None, timeout=5):
    """Enforce query timeout at application level"""
    
    # Set statement timeout (PostgreSQL)
    cursor.execute (f"SET statement_timeout = {timeout * 1000}")  # milliseconds
    
    try:
        if params:
            cursor.execute (query, params)
        else:
            cursor.execute (query)
        
        return cursor.fetchall()
        
    except QueryTimeout:
        logger.warning (f"Query timeout after {timeout}s: {query[:100]}")
        metrics.increment('db.query.timeout')
        raise
    
    finally:
        # Reset timeout
        cursor.execute("SET statement_timeout = 0")
\`\`\`

**Short-term Solutions (Within 1 week):**

**1. Identify and Optimize Slow Queries**

\`\`\`sql
-- Find queries holding connections longest
SELECT pid, usename, application_name, client_addr,
       state, query_start, state_change,
       now() - query_start AS query_duration,
       query
FROM pg_stat_activity
WHERE state = 'active'
  AND now() - query_start > interval '5 seconds'
ORDER BY query_duration DESC
LIMIT 20;

-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_orders_user_created 
ON orders (user_id, created_at DESC);

-- Use connection for read-only queries
EXEC query ON REPLICA;
\`\`\`

**2. Implement Request Prioritization**

\`\`\`python
# Priority-based connection pool
class PriorityConnectionPool:
    def __init__(self):
        self.high_priority_pool = ConnectionPool (maxconn=30)
        self.low_priority_pool = ConnectionPool (maxconn=10)
    
    def get_connection (self, priority='normal'):
        if priority == 'high':
            # Critical writes, payments, user registration
            return self.high_priority_pool.getconn()
        else:
            # Analytics, reports, background jobs
            return self.low_priority_pool.getconn (timeout=2)
\`\`\`

**Long-term Architectural Solutions:**

**1. Read Replicas for Read-Heavy Workload**

\`\`\`python
class SmartConnectionManager:
    def __init__(self):
        self.write_pool = ConnectionPool (host='primary', maxconn=20)
        self.read_pools = [
            ConnectionPool (host='replica1', maxconn=50),
            ConnectionPool (host='replica2', maxconn=50),
            ConnectionPool (host='replica3', maxconn=50)
        ]
    
    def get_write_connection (self):
        return self.write_pool.getconn()
    
    def get_read_connection (self):
        # Load balance across replicas
        replica_idx = random.randint(0, len (self.read_pools) - 1)
        return self.read_pools[replica_idx].getconn()
\`\`\`

**2. Caching Layer (Redis)**

\`\`\`python
from functools import lru_cache

@lru_cache (maxsize=10000)
def get_user (user_id):
    # Check Redis first
    cached = redis_client.get (f"user:{user_id}")
    if cached:
        metrics.increment('cache.hit')
        return json.loads (cached)
    
    # Cache miss - query database
    metrics.increment('cache.miss')
    with get_read_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
    
    # Store in Redis (TTL: 5 minutes)
    redis_client.setex (f"user:{user_id}", 300, json.dumps (user))
    
    return user
\`\`\`

**3. Connection Pool Monitoring Dashboard**

\`\`\`python
# Expose metrics endpoint
@app.get("/metrics/pool")
def pool_metrics():
    return {
        "pool_size": pool.maxconn,
        "active_connections": pool.size - pool.idle,
        "idle_connections": pool.idle,
        "waiting_requests": pool.waiting,
        "utilization_percent": ((pool.size - pool.idle) / pool.maxconn) * 100,
        "total_connections_created": pool.connections_created,
        "connection_timeouts": pool.timeouts,
        "avg_wait_time_ms": pool.avg_wait_time * 1000
    }
\`\`\`

**Expected Results:**

| Metric | Before | After Immediate Fix | After Long-term |
|--------|--------|-------------------|----------------|
| P95 latency | 5000ms | 100ms | 50ms |
| Pool utilization | 100% | 70% | 50% |
| Requests/sec capacity | 4000 | 8000 | 15000 |
| Connection timeouts/min | 50 | 2 | 0 |

**Key Takeaways:**
- Pool exhaustion != database capacity problems
- Monitor pool metrics, not just database metrics
- Separate read and write workloads
- Implement caching for frequently accessed data
- Add connection leak detection in development`,
    keyPoints: [
      'Pool exhaustion can occur even when database has spare capacity',
      'Increase pool size based on request rate and query duration',
      'Implement connection timeouts to fail fast and prevent cascading failures',
      'Use connection leak detection to identify code not returning connections',
      'Long-term: separate read/write pools, add caching, use read replicas',
    ],
  },
  {
    id: 'pool-disc-3',
    question:
      'Compare the trade-offs between using a single large connection pool shared across all application threads versus creating separate connection pools for different types of database operations (e.g., transactional writes, analytical queries, background jobs). Which approach would you choose and why?',
    sampleAnswer: `**Comparison of Connection Pool Strategies:**

**Approach 1: Single Shared Connection Pool**

\`\`\`python
# All operations share one pool
connection_pool = ConnectionPool(
    host='database.example.com',
    minconn=10,
    maxconn=100,
    timeout=30
)

# All operations use the same pool
def process_payment():
    with connection_pool.getconn() as conn:
        # Critical write
        pass

def generate_analytics_report():
    with connection_pool.getconn() as conn:
        # Slow analytical query
        pass

def background_cleanup():
    with connection_pool.getconn() as conn:
        # Bulk delete
        pass
\`\`\`

**Pros:**
✅ **Simpler to manage** - one pool configuration
✅ **Better resource utilization** - connections shared across all operations
✅ **Lower total connections** - database sees fewer total connections
✅ **Easier to reason about** - single point of configuration

**Cons:**
❌ **No priority control** - slow queries block critical operations
❌ **"Noisy neighbor" problem** - analytical queries starve transactional queries
❌ **Harder to debug** - can't isolate performance issues by operation type
❌ **No isolation** - one operation type can exhaust entire pool

**Real-world Problem:**

\`\`\`python
# Critical payment processing waits for slow analytics
2023-12-15 14:30:15 [ERROR] Payment failed: Connection pool timeout
2023-12-15 14:30:15 [INFO] Pool status: 100/100 connections in use
2023-12-15 14:30:15 [INFO] 85 connections running analytics queries (60+ seconds)
2023-12-15 14:30:15 [INFO] 15 connections for all other operations
# Analytics queries are hogging the pool!
\`\`\`

---

**Approach 2: Separate Connection Pools**

\`\`\`python
class DatabasePoolManager:
    def __init__(self):
        # High-priority: Payments, user auth, critical writes
        self.transactional_pool = ConnectionPool(
            host='primary.db.example.com',
            minconn=20,
            maxconn=50,
            timeout=5,  # Fail fast
            max_idle_time=300
        )
        
        # Medium-priority: Normal application queries
        self.application_pool = ConnectionPool(
            host='replica1.db.example.com',
            minconn=10,
            maxconn=30,
            timeout=10
        )
        
        # Low-priority: Analytics, reports, background jobs
        self.analytics_pool = ConnectionPool(
            host='replica2.db.example.com',
            minconn=5,
            maxconn=20,
            timeout=60  # Allow longer queries
        )
        
        # Background jobs: Cleanup, exports, migrations
        self.background_pool = ConnectionPool(
            host='replica3.db.example.com',
            minconn=2,
            maxconn=10,
            timeout=120
        )

pool_manager = DatabasePoolManager()

# Usage
def process_payment():
    with pool_manager.transactional_pool.getconn() as conn:
        # Guaranteed fast access to connection
        pass

def generate_analytics_report():
    with pool_manager.analytics_pool.getconn() as conn:
        # Isolated from critical operations
        pass
\`\`\`

**Pros:**
✅ **Priority isolation** - critical ops not blocked by slow queries
✅ **Better performance guarantees** - SLA for critical operations
✅ **Easier debugging** - isolate issues by operation type
✅ **Targeted optimization** - tune each pool for its workload
✅ **Graceful degradation** - analytics can fail without affecting payments
✅ **Different database targets** - route reads to replicas

**Cons:**
❌ **More complex** - multiple pools to configure and monitor
❌ **Higher total connections** - database sees more total connections
❌ **Potential underutilization** - one pool idle while another is saturated
❌ **More configuration** - need to choose correct pool in code

---

**Detailed Comparison Table:**

| Aspect | Single Pool | Separate Pools | Winner |
|--------|-------------|----------------|--------|
| **Simplicity** | Very simple | More complex | Single |
| **Resource Efficiency** | Better (shared) | Lower (isolated) | Single |
| **Priority Control** | None | Excellent | Separate |
| **Fault Isolation** | None | Excellent | Separate |
| **Debugging** | Harder | Easier | Separate |
| **Performance SLAs** | Cannot guarantee | Can guarantee | Separate |
| **Total DB Connections** | Lower | Higher | Single |
| **Noisy Neighbor Protection** | No | Yes | Separate |

---

**Recommended Approach: Hybrid Strategy**

For production systems, I recommend **separate pools with shared overflow**:

\`\`\`python
class HybridPoolManager:
    def __init__(self):
        # Critical operations - dedicated pool
        self.critical_pool = ConnectionPool(
            host='primary.db',
            minconn=30,
            maxconn=50,
            timeout=5
        )
        
        # General operations - shared pool for reads
        self.general_pool = ConnectionPool(
            host='replica1.db',
            minconn=20,
            maxconn=60,
            timeout=10
        )
        
        # Low-priority - separate pool with longer timeout
        self.low_priority_pool = ConnectionPool(
            host='replica2.db',
            minconn=5,
            maxconn=20,
            timeout=60
        )
    
    @contextmanager
    def get_connection (self, priority='normal'):
        if priority == 'critical':
            # Use dedicated critical pool
            pool = self.critical_pool
        elif priority == 'low':
            # Use low-priority pool
            pool = self.low_priority_pool
        else:
            # Use general pool
            pool = self.general_pool
        
        conn = None
        try:
            conn = pool.getconn()
            yield conn
        finally:
            if conn:
                pool.putconn (conn)

# Usage with clear priority declaration
with pool_manager.get_connection (priority='critical') as conn:
    process_payment (conn)

with pool_manager.get_connection (priority='low') as conn:
    generate_report (conn)
\`\`\`

**When to Use Each Approach:**

**Single Shared Pool:**
- Small applications (< 10,000 req/hour)
- Uniform query patterns (all queries similar duration)
- Limited database connection capacity
- Team prefers simplicity

**Separate Pools:**
- Large applications (> 100,000 req/hour)
- Mixed workloads (fast + slow queries)
- Need SLA guarantees for critical operations
- Multiple database replicas available
- High-traffic payment/financial systems

**Real-World Example (E-commerce):**

\`\`\`python
# Stripe uses separate pools for different operations
pools = {
    'checkout': ConnectionPool (maxconn=100),      # Payment processing
    'api': ConnectionPool (maxconn=200),           # API requests
    'dashboard': ConnectionPool (maxconn=50),      # Customer dashboard
    'analytics': ConnectionPool (maxconn=30),      # Internal analytics
    'background': ConnectionPool (maxconn=20)      # Background jobs
}

# Result:
# - Checkout never blocked by analytics
# - API requests isolated from dashboard
# - Background jobs don't impact user-facing operations
\`\`\`

**My Recommendation:**

For most production systems, use **separate pools** because:
1. **User-facing performance** is more important than simplicity
2. **Fault isolation** prevents cascading failures
3. **Debugging** is much easier with isolated pools
4. **Database connections** are relatively cheap (can afford more)
5. **SLA requirements** usually demand guaranteed performance for critical paths

The small increase in complexity is worth the significant improvement in reliability and performance guarantees.`,
    keyPoints: [
      'Single pool is simpler but suffers from "noisy neighbor" problem',
      'Separate pools provide priority isolation and better performance guarantees',
      'Critical operations should have dedicated pools with fail-fast timeouts',
      'Hybrid approach: separate pools for critical, shared pool for general operations',
      'For production systems with SLA requirements, separate pools are worth the complexity',
    ],
  },
];
