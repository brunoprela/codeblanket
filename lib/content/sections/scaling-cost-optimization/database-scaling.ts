export const databaseScaling = {
  title: 'Database Scaling',
  content: `

# Database Scaling for LLM Applications

## Introduction

As your LLM application scales from thousands to millions of users, the database becomes a critical bottleneck. Databases in LLM applications store:

- **User Data**: Authentication, profiles, preferences
- **Conversation History**: Multi-turn dialogues requiring fast retrieval
- **Cached Responses**: LLM outputs for performance
- **Usage Analytics**: Tracking costs, performance, user behavior
- **Vector Embeddings**: Semantic search capabilities
- **Application State**: Sessions, workflows, background jobs

A slow database means slow responses, poor user experience, and wasted LLM API calls. This section covers production database scaling strategies that enable handling millions of users while maintaining sub-100ms query times.

---

## Read Replicas for Read-Heavy Workloads

Most LLM applications are read-heavy (80-95% reads). Read replicas distribute read load across multiple database instances.

### PostgreSQL Read Replicas

\`\`\`python
import psycopg2
from psycopg2 import pool
import random

class DatabaseCluster:
    """Manage primary and read replicas"""
    
    def __init__(
        self,
        primary_dsn: str,
        replica_dsns: list[str],
        min_conn: int = 5,
        max_conn: int = 20
    ):
        # Primary for writes
        self.primary_pool = pool.ThreadedConnectionPool(
            min_conn, max_conn, primary_dsn
        )
        
        # Replicas for reads
        self.replica_pools = [
            pool.ThreadedConnectionPool(min_conn, max_conn, dsn)
            for dsn in replica_dsns
        ]
        
        self.replica_count = len(replica_dsns)
    
    def execute_read(self, query: str, params: tuple = None):
        """Execute read query on a replica"""
        
        # Round-robin or random selection
        replica_pool = random.choice(self.replica_pools)
        
        conn = replica_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            replica_pool.putconn(conn)
    
    def execute_write(self, query: str, params: tuple = None):
        """Execute write query on primary"""
        
        conn = self.primary_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                return cur.fetchone()
        finally:
            self.primary_pool.putconn(conn)
    
    def get_conversation_history(self, user_id: str):
        """Read from replica"""
        return self.execute_read(
            "SELECT * FROM conversations WHERE user_id = %s ORDER BY created_at DESC LIMIT 10",
            (user_id,)
        )
    
    def save_message(self, user_id: str, role: str, content: str):
        """Write to primary"""
        return self.execute_write(
            "INSERT INTO messages (user_id, role, content) VALUES (%s, %s, %s) RETURNING id",
            (user_id, role, content)
        )

# Usage
db = DatabaseCluster(
    primary_dsn="postgresql://primary:5432/mydb",
    replica_dsns=[
        "postgresql://replica1:5432/mydb",
        "postgresql://replica2:5432/mydb",
        "postgresql://replica3:5432/mydb"
    ]
)

# Reads go to replicas (distributed load)
history = db.get_conversation_history("user_123")

# Writes go to primary
message_id = db.save_message("user_123", "user", "Hello!")
\`\`\`

**Benefits**:
- Distribute read load across multiple servers
- 3 replicas → 3x read capacity
- Primary handles only writes
- Can add more replicas without downtime

**Replication Lag**: Be aware replicas are slightly behind primary (typically <100ms). For critical reads-after-writes, query the primary.

---

## Connection Pooling

Database connections are expensive to create. Connection pooling reuses connections efficiently.

### Production Connection Pool

\`\`\`python
from psycopg2 import pool
from contextlib import contextmanager
import time

class ConnectionPoolManager:
    """Production-grade connection pool"""
    
    def __init__(
        self,
        dsn: str,
        min_connections: int = 5,
        max_connections: int = 20,
        max_idle_time: int = 300,  # 5 minutes
        health_check_interval: int = 60
    ):
        self.pool = pool.ThreadedConnectionPool(
            min_connections,
            max_connections,
            dsn
        )
        
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "connection_errors": 0,
            "avg_query_time": 0
        }
        
        # Start health check
        import threading
        self.health_check_thread = threading.Thread(
            target=self._periodic_health_check,
            daemon=True
        )
        self.health_check_thread.start()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic return"""
        conn = None
        start_time = time.time()
        
        try:
            conn = self.pool.getconn()
            self.stats["active_connections"] += 1
            
            # Test connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            
            yield conn
            
            # Record query time
            query_time = time.time() - start_time
            self._update_avg_query_time(query_time)
            
        except Exception as e:
            self.stats["connection_errors"] += 1
            if conn:
                # Connection is bad, close it
                conn.close()
                conn = None
            raise
        finally:
            if conn:
                self.stats["active_connections"] -= 1
                self.pool.putconn(conn)
    
    def _update_avg_query_time(self, query_time: float):
        """Update rolling average query time"""
        alpha = 0.1  # Smoothing factor
        self.stats["avg_query_time"] = (
            alpha * query_time +
            (1 - alpha) * self.stats["avg_query_time"]
        )
    
    def _periodic_health_check(self):
        """Periodically check pool health"""
        while True:
            time.sleep(60)
            
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                print("✅ Pool health check passed")
            except Exception as e:
                print(f"❌ Pool health check failed: {e}")
    
    def get_pool_stats(self) -> dict:
        """Get pool statistics"""
        return {
            **self.stats,
            "pool_size": self.pool.maxconn,
            "available": self.pool.maxconn - self.stats["active_connections"]
        }

# Usage
pool_manager = ConnectionPoolManager(
    dsn="postgresql://localhost:5432/mydb",
    min_connections=10,
    max_connections=50
)

# Use connection
with pool_manager.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()

# Connection automatically returned to pool

# Monitor pool health
stats = pool_manager.get_pool_stats()
print(f"Active: {stats['active_connections']}/{stats['pool_size']}")
print(f"Avg query time: {stats['avg_query_time']:.3f}s")
\`\`\`

**Benefits**:
- Reuse connections (saves 100-200ms per request)
- Limit total connections (prevent overwhelming DB)
- Health monitoring built-in
- Automatic recovery from connection failures

**Rule of Thumb**: 
- Min connections: 2 × (number of app servers)
- Max connections: 10 × (number of app servers)
- Never exceed database's max_connections setting

---

## Sharding for Write-Heavy Workloads

When a single database can't handle write load, shard (partition) data across multiple databases.

### Hash-Based Sharding

\`\`\`python
import hashlib
from typing import Dict, List

class ShardedDatabase:
    """Shard data across multiple databases"""
    
    def __init__(self, shard_dsns: List[str]):
        self.shards = []
        
        for dsn in shard_dsns:
            pool = pool.ThreadedConnectionPool(5, 20, dsn)
            self.shards.append(pool)
        
        self.shard_count = len(self.shards)
    
    def get_shard_index(self, user_id: str) -> int:
        """Determine which shard to use for a user"""
        # Hash user_id to get consistent shard
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return hash_value % self.shard_count
    
    def get_shard(self, user_id: str):
        """Get shard for a user"""
        index = self.get_shard_index(user_id)
        return self.shards[index]
    
    def save_conversation(self, user_id: str, conversation: Dict):
        """Save conversation to appropriate shard"""
        shard = self.get_shard(user_id)
        conn = shard.getconn()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversations (user_id, messages, created_at)
                    VALUES (%s, %s, NOW())
                """, (user_id, json.dumps(conversation)))
                conn.commit()
        finally:
            shard.putconn(conn)
    
    def get_conversations(self, user_id: str) -> List[Dict]:
        """Get conversations from appropriate shard"""
        shard = self.get_shard(user_id)
        conn = shard.getconn()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT messages, created_at 
                    FROM conversations 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (user_id,))
                return cur.fetchall()
        finally:
            shard.putconn(conn)
    
    def get_all_conversations(self, user_ids: List[str]) -> Dict:
        """Get conversations for multiple users (scatter-gather)"""
        
        # Group users by shard
        users_by_shard = {}
        for user_id in user_ids:
            shard_idx = self.get_shard_index(user_id)
            if shard_idx not in users_by_shard:
                users_by_shard[shard_idx] = []
            users_by_shard[shard_idx].append(user_id)
        
        # Query each shard in parallel
        results = {}
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.shard_count) as executor:
            futures = {}
            
            for shard_idx, users in users_by_shard.items():
                future = executor.submit(
                    self._query_shard_for_users,
                    shard_idx,
                    users
                )
                futures[future] = users
            
            for future in concurrent.futures.as_completed(futures):
                results.update(future.result())
        
        return results
    
    def _query_shard_for_users(self, shard_idx: int, user_ids: List[str]) -> Dict:
        """Query single shard for multiple users"""
        shard = self.shards[shard_idx]
        conn = shard.getconn()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT user_id, messages, created_at
                    FROM conversations
                    WHERE user_id = ANY(%s)
                    ORDER BY created_at DESC
                """, (user_ids,))
                
                results = {}
                for row in cur.fetchall():
                    user_id = row[0]
                    if user_id not in results:
                        results[user_id] = []
                    results[user_id].append({
                        "messages": row[1],
                        "created_at": row[2]
                    })
                
                return results
        finally:
            shard.putconn(conn)

# Usage
sharded_db = ShardedDatabase([
    "postgresql://shard1:5432/mydb",
    "postgresql://shard2:5432/mydb",
    "postgresql://shard3:5432/mydb",
    "postgresql://shard4:5432/mydb"
])

# Save to appropriate shard
sharded_db.save_conversation("user_123", {"messages": [...]})

# Retrieve from appropriate shard
conversations = sharded_db.get_conversations("user_123")

# Cross-shard query (scatter-gather)
all_convs = sharded_db.get_all_conversations(["user_1", "user_2", "user_3"])
\`\`\`

**Benefits**:
- Distribute write load across multiple databases
- Each shard handles 1/N of traffic
- Can scale to billions of users
- Isolates failures (one shard down doesn't affect others)

**Challenges**:
- Cross-shard queries are slow (scatter-gather)
- Resharding is complex (need to migrate data)
- Joins across shards not possible

---

## Query Optimization

Slow queries kill performance. Optimize queries for LLM applications.

### Index Strategy

\`\`\`python
# SLOW QUERY (no index):
# SELECT * FROM messages WHERE user_id = 'user_123' ORDER BY created_at DESC LIMIT 10;
# Query time: 2000ms on 10M rows

# CREATE INDEX idx_messages_user_created ON messages(user_id, created_at DESC);

# FAST QUERY (with index):
# Query time: 5ms on 10M rows

class QueryOptimizer:
    """Optimize database queries"""
    
    def analyze_slow_queries(self, conn):
        """Find slow queries"""
        with conn.cursor() as cur:
            # PostgreSQL slow query log
            cur.execute("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    max_time
                FROM pg_stat_statements
                WHERE mean_time > 100  -- > 100ms average
                ORDER BY mean_time DESC
                LIMIT 20
            """)
            
            slow_queries = cur.fetchall()
            
            print("🐌 Slow Queries:")
            for query, calls, total, mean, max_time in slow_queries:
                print(f"  {query[:100]}...")
                print(f"  Calls: {calls}, Avg: {mean:.2f}ms, Max: {max_time:.2f}ms")
                print()
    
    def explain_query(self, conn, query: str, params: tuple = None):
        """Analyze query execution plan"""
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN ANALYZE {query}", params)
            plan = cur.fetchall()
            
            print("📊 Query Plan:")
            for line in plan:
                print(f"  {line[0]}")
    
    def suggest_indexes(self, conn):
        """Suggest missing indexes"""
        with conn.cursor() as cur:
            # Find tables with many sequential scans
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch
                FROM pg_stat_user_tables
                WHERE seq_scan > 1000  -- Many sequential scans
                AND idx_scan < seq_scan * 0.1  -- Few index scans
                ORDER BY seq_scan DESC
                LIMIT 10
            """)
            
            tables = cur.fetchall()
            
            print("💡 Tables that might need indexes:")
            for schema, table, seq, seq_reads, idx, idx_reads in tables:
                print(f"  {schema}.{table}")
                print(f"    Sequential scans: {seq} ({seq_reads} rows)")
                print(f"    Index scans: {idx} ({idx_reads} rows)")
                print(f"    Consider adding indexes on frequently filtered columns")
                print()

# Usage
optimizer = QueryOptimizer()

# Find slow queries
optimizer.analyze_slow_queries(conn)

# Analyze specific query
optimizer.explain_query(
    conn,
    "SELECT * FROM conversations WHERE user_id = %s ORDER BY created_at DESC",
    ("user_123",)
)

# Get index suggestions
optimizer.suggest_indexes(conn)
\`\`\`

### Common Optimizations

1. **Add Indexes**: On WHERE, JOIN, ORDER BY columns
2. **Use LIMIT**: Don't fetch all rows if you need only a few
3. **Avoid SELECT ***: Fetch only needed columns
4. **Use Prepared Statements**: Faster query parsing
5. **Batch Inserts**: INSERT multiple rows at once

---

## Caching Layer (Redis)

Add Redis between application and database to reduce load.

\`\`\`python
import redis
import json
from typing import Optional

class CachedDatabase:
    """Database with Redis caching layer"""
    
    def __init__(self, db_pool, redis_url: str):
        self.db = db_pool
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutes
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user with caching"""
        
        # Check cache first
        cache_key = f"user:{user_id}"
        cached = self.redis.get(cache_key)
        
        if cached:
            print("✅ Cache hit")
            return json.loads(cached)
        
        # Cache miss - query database
        print("❌ Cache miss - querying DB")
        conn = self.db.getconn()
        
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, email, preferences FROM users WHERE id = %s",
                    (user_id,)
                )
                row = cur.fetchone()
                
                if row:
                    user = {
                        "id": row[0],
                        "name": row[1],
                        "email": row[2],
                        "preferences": row[3]
                    }
                    
                    # Cache for next time
                    self.redis.setex(
                        cache_key,
                        self.default_ttl,
                        json.dumps(user)
                    )
                    
                    return user
                
                return None
        finally:
            self.db.putconn(conn)
    
    def update_user(self, user_id: str, updates: Dict):
        """Update user and invalidate cache"""
        
        conn = self.db.getconn()
        
        try:
            with conn.cursor() as cur:
                # Build UPDATE query
                set_clause = ", ".join(f"{k} = %s" for k in updates.keys())
                values = list(updates.values()) + [user_id]
                
                cur.execute(
                    f"UPDATE users SET {set_clause} WHERE id = %s",
                    values
                )
                conn.commit()
            
            # Invalidate cache
            cache_key = f"user:{user_id}"
            self.redis.delete(cache_key)
            print("🗑️  Cache invalidated")
            
        finally:
            self.db.putconn(conn)

# Usage
cached_db = CachedDatabase(db_pool, "redis://localhost:6379")

# First call - DB query (slow)
user = cached_db.get_user("user_123")  # 50ms

# Second call - cache hit (fast)
user = cached_db.get_user("user_123")  # 1ms

# Update invalidates cache
cached_db.update_user("user_123", {"name": "New Name"})

# Next call queries DB again
user = cached_db.get_user("user_123")  # 50ms
\`\`\`

**Cache Hit Rate**: Aim for 80%+ hit rate for frequently accessed data.

---

## Vector Databases for Embeddings

LLM applications need semantic search. Use specialized vector databases.

### Pgvector (PostgreSQL Extension)

\`\`\`python
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

class VectorDatabase:
    """PostgreSQL with pgvector for embeddings"""
    
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self._setup_pgvector()
    
    def _setup_pgvector(self):
        """Enable pgvector extension"""
        with self.conn.cursor() as cur:
            # Enable extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table with vector column
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536),  -- OpenAI embedding size
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            self.conn.commit()
    
    def insert_document(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: dict = None
    ) -> int:
        """Insert document with embedding"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (content, embedding, metadata)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (
                content,
                embedding.tolist(),
                json.dumps(metadata or {})
            ))
            
            self.conn.commit()
            return cur.fetchone()[0]
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Find similar documents using cosine similarity"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (
                query_embedding.tolist(),
                query_embedding.tolist(),
                threshold,
                query_embedding.tolist(),
                limit
            ))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "similarity": row[3]
                })
            
            return results

# Usage
vector_db = VectorDatabase("postgresql://localhost/vectordb")

# Insert documents with embeddings
embedding = get_embedding("Machine learning is amazing")  # OpenAI embedding
doc_id = vector_db.insert_document(
    content="Machine learning is amazing",
    embedding=embedding,
    metadata={"source": "blog", "author": "Alice"}
)

# Search for similar documents
query_embedding = get_embedding("AI and ML")
similar_docs = vector_db.search_similar(query_embedding, limit=5)

for doc in similar_docs:
    print(f"Similarity: {doc['similarity']:.2f}")
    print(f"Content: {doc['content']}")
    print()
\`\`\`

**Alternatives**:
- **Pinecone**: Managed vector database (easy, expensive)
- **Weaviate**: Open-source vector database (flexible)
- **Qdrant**: Fast vector similarity search
- **FAISS**: Facebook's library (local, very fast)

---

## Monitoring Database Performance

\`\`\`python
import psycopg2
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DatabaseMetrics:
    active_connections: int
    idle_connections: int
    waiting_connections: int
    slow_queries: int
    cache_hit_rate: float
    index_usage_rate: float
    avg_query_time: float
    timestamp: datetime

class DatabaseMonitor:
    """Monitor database performance"""
    
    def __init__(self, dsn: str):
        self.dsn = dsn
    
    def get_metrics(self) -> DatabaseMetrics:
        """Collect database metrics"""
        conn = psycopg2.connect(self.dsn)
        
        try:
            with conn.cursor() as cur:
                # Active connections
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE state = 'active'),
                        COUNT(*) FILTER (WHERE state = 'idle'),
                        COUNT(*) FILTER (WHERE wait_event IS NOT NULL)
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """)
                active, idle, waiting = cur.fetchone()
                
                # Cache hit rate
                cur.execute("""
                    SELECT 
                        sum(heap_blks_hit) / 
                        (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_rate
                    FROM pg_statio_user_tables
                """)
                cache_hit_rate = cur.fetchone()[0] or 0
                
                # Index usage rate
                cur.execute("""
                    SELECT 
                        sum(idx_scan) / (sum(seq_scan) + sum(idx_scan)) as index_usage
                    FROM pg_stat_user_tables
                    WHERE seq_scan + idx_scan > 0
                """)
                index_usage = cur.fetchone()[0] or 0
                
                return DatabaseMetrics(
                    active_connections=active,
                    idle_connections=idle,
                    waiting_connections=waiting,
                    slow_queries=0,  # Would need pg_stat_statements
                    cache_hit_rate=cache_hit_rate,
                    index_usage_rate=index_usage,
                    avg_query_time=0,
                    timestamp=datetime.now()
                )
        finally:
            conn.close()
    
    def alert_if_issues(self, metrics: DatabaseMetrics):
        """Alert on performance issues"""
        issues = []
        
        if metrics.cache_hit_rate < 0.90:
            issues.append(f"Low cache hit rate: {metrics.cache_hit_rate:.1%}")
        
        if metrics.index_usage_rate < 0.80:
            issues.append(f"Low index usage: {metrics.index_usage_rate:.1%}")
        
        if metrics.waiting_connections > 10:
            issues.append(f"Many waiting connections: {metrics.waiting_connections}")
        
        if issues:
            print("🚨 DATABASE ISSUES:")
            for issue in issues:
                print(f"  - {issue}")

# Usage
monitor = DatabaseMonitor("postgresql://localhost/mydb")

# Collect metrics
metrics = monitor.get_metrics()
print(f"Active connections: {metrics.active_connections}")
print(f"Cache hit rate: {metrics.cache_hit_rate:.1%}")
print(f"Index usage: {metrics.index_usage_rate:.1%}")

# Alert on issues
monitor.alert_if_issues(metrics)
\`\`\`

---

## Best Practices

### 1. Use Read Replicas
- Route reads to replicas (80-95% of queries)
- Keep primary for writes only
- Add replicas as read load grows

### 2. Implement Connection Pooling
- Reuse connections (saves 100-200ms)
- Size pool appropriately (10-20 per app server)
- Monitor pool utilization

### 3. Add Indexes Strategically
- Index WHERE, JOIN, ORDER BY columns
- Don't over-index (slows writes)
- Monitor slow queries

### 4. Cache Aggressively
- Use Redis for frequently accessed data
- Achieve 80%+ cache hit rate
- Invalidate on writes

### 5. Consider Sharding
- When single DB can't handle writes
- Shard by user_id (natural partitioning)
- Plan for cross-shard queries

### 6. Optimize Queries
- Use EXPLAIN ANALYZE
- Add missing indexes
- Avoid N+1 queries

### 7. Monitor Performance
- Track slow queries
- Monitor cache hit rate
- Alert on connection saturation

---

## Summary

Database scaling for LLM applications requires:

- **Read Replicas**: Distribute read load (3x capacity with 3 replicas)
- **Connection Pooling**: Reuse connections efficiently
- **Sharding**: Partition data across multiple databases for writes
- **Query Optimization**: Indexes, LIMIT, and query analysis
- **Caching Layer**: Redis for frequently accessed data
- **Vector Databases**: Pgvector, Pinecone for embeddings
- **Monitoring**: Track performance and alert on issues

With proper database scaling, handle millions of users with sub-100ms query times.

`,
  exercises: [
    {
      prompt:
        'Implement a production database cluster with primary and 3 read replicas. Route 90% of queries to replicas and measure performance improvement.',
      solution: `Use DatabaseCluster class with round-robin routing to replicas. Measure query latency before/after. Expected: 3x read capacity, 70% latency reduction.`,
    },
    {
      prompt:
        'Add pgvector to PostgreSQL and implement semantic search for 100K documents. Benchmark query time vs traditional keyword search.',
      solution: `Use VectorDatabase class with ivfflat index. Compare pgvector cosine similarity vs traditional LIKE queries. Expected: pgvector 10-100x faster for semantic search.`,
    },
    {
      prompt:
        'Implement database sharding across 4 shards and measure write throughput improvement. Handle cross-shard queries with scatter-gather.',
      solution: `Use ShardedDatabase class with hash-based sharding. Measure writes/second before (1 DB) and after (4 DBs). Expected: ~3-4x write throughput.`,
    },
  ],
};
