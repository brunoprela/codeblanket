export const databaseIntegrationQuiz = [
  {
    id: 'pllm-q-9-1',
    question:
      'Design a database schema for an LLM application that stores conversations, messages, user data, and usage metrics. Include indexes, relationships, and how you would optimize for the most common queries.',
    sampleAnswer:
      'Schema: Users (id PK, email UNIQUE, tier, api_key INDEXED, created_at), Conversations (id PK, user_id FK INDEXED, system_prompt TEXT, metadata JSONB, created_at INDEXED, updated_at), Messages (id PK, conversation_id FK INDEXED, role VARCHAR(50), content TEXT, tokens_used INT, cost NUMERIC(10,6), created_at INDEXED), UsageRecords (id PK, user_id FK, date DATE INDEXED, requests_count INT, tokens_used INT, cost NUMERIC). Composite indexes: (user_id, created_at) on Conversations for "get user conversations", (conversation_id, created_at) on Messages for "get conversation history", (user_id, date) on UsageRecords for "get user usage". Optimization: Partition Messages table by created_at (monthly partitions) to keep active data small, use JSONB for flexible metadata with GIN index for queries, archive conversations older than 90 days to cold storage, use materialized views for dashboard queries (daily rollups), connection pooling (20 connections), query optimization with EXPLAIN ANALYZE, read replicas for heavy read operations. Common query patterns: Get last N conversations with message count, Get conversation with all messages (JOIN), Get user usage for billing period (aggregation), Search conversations by content (full-text search).',
    keyPoints: [
      'Well-indexed schema with appropriate foreign keys and composite indexes',
      'Partitioning and archival strategy for growing data',
      'Optimization through caching, read replicas, and materialized views',
    ],
  },
  {
    id: 'pllm-q-9-2',
    question:
      'Compare using PostgreSQL with pgvector vs dedicated vector databases (Pinecone, Weaviate) for semantic search in LLM applications. When would you choose each approach?',
    sampleAnswer:
      'PostgreSQL + pgvector: Single database for all data, simpler architecture, transactional consistency between conversations and embeddings, good for <1M vectors, cosine similarity queries via SQL, lower cost (existing infrastructure). Limitations: Slower at scale (>10M vectors), limited filtering capabilities, requires more indexing knowledge. Dedicated vector DBs (Pinecone/Weaviate): Optimized for high-dimensional vectors, faster similarity search at scale (>10M vectors), advanced filtering/metadata, built-in sharding, managed service. Downsides: Additional service to manage, data duplication (store conversation in Postgres AND embeddings in vector DB), eventual consistency issues, higher cost. Choose PostgreSQL + pgvector for: MVP/prototype, <1M documents, tight budget, simple semantic search, need transactional guarantees, already using PostgreSQL. Choose dedicated vector DB for: >10M documents, millisecond latency requirements, complex metadata filtering, need managed solution, building semantic search as core feature, can handle eventual consistency. Hybrid approach: Store recent conversations in pgvector (fast, consistent), archive old ones to Pinecone (cost-effective), query both and merge results. Implementation: pgvector CREATE INDEX USING ivfflat ON embeddings USING vector_cosine_ops, Pinecone upsert with metadata filtering.',
    keyPoints: [
      'PostgreSQL+pgvector: simpler, cheaper, good for <1M vectors',
      'Dedicated vector DBs: faster at scale, advanced features, higher cost',
      'Choice based on scale, budget, and complexity requirements',
    ],
  },
  {
    id: 'pllm-q-9-3',
    question:
      'Explain connection pooling for database access in LLM applications. How do you size the pool, handle connection leaks, and monitor pool health? What happens when the pool is exhausted?',
    sampleAnswer:
      'Connection pooling reuses database connections instead of creating new ones (expensive: 100ms+ per connection). Configuration: pool_size=20 (concurrent requests), max_overflow=10 (temporary extra), pool_timeout=30s (wait for connection), pool_recycle=3600s (refresh hourly), pool_pre_ping=True (verify connection health). Sizing: pool_size = expected_concurrent_requests, start with 20, monitor usage, increase if seeing timeouts. Rule: workers * concurrent_requests_per_worker ≈ pool_size. Handle leaks: Use context managers (with get_db_session()), set statement_timeout=30s in PostgreSQL, monitor long-running queries, automatic connection cleanup on timeout, track connection lifetime, alert on connections open >5min. Monitor health: Track active connections, pool exhaustion events, connection wait time, connection errors, query duration. Grafana dashboard showing pool utilization over time. Pool exhausted: Requests wait for pool_timeout (30s) then fail with OperationalError, return 503 to client with retry-after, scale up pool size or reduce request rate, investigate slow queries holding connections, consider read replicas to distribute load. Prevention: Connection limits per user, kill long queries automatically, scale workers based on pool capacity, use async connections for non-blocking I/O.',
    keyPoints: [
      'Proper pool sizing based on concurrent requests with overflow buffer',
      'Context managers and timeouts to prevent connection leaks',
      'Comprehensive monitoring and graceful handling of pool exhaustion',
    ],
  },
];
