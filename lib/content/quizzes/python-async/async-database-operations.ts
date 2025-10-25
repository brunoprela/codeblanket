export const asyncDatabaseOperationsQuiz = [
  {
    id: 'ado-q-1',
    question:
      'Design a high-throughput data ingestion system using asyncpg that: (1) Inserts 1M records/minute from API, (2) Uses transactions for data consistency, (3) Handles duplicate key errors gracefully, (4) Monitors database connection pool health, (5) Implements backpressure when database slower than API. Explain why copy_records_to_table is 100× faster than executemany and when to use each.',
    sampleAnswer:
      'High-throughput data ingestion: class DataIngestion: def __init__(self): self.pool = None; self.stats = {"inserted": 0, "duplicates": 0, "errors": 0}. async def setup (self): self.pool = await asyncpg.create_pool (dsn, min_size=20, max_size=50, command_timeout=30). async def ingest_batch (self, records): async with self.pool.acquire() as conn: async with conn.transaction(): try: await conn.copy_records_to_table("events", records=records, columns=["id", "data", "timestamp"]); self.stats["inserted"] += len (records); except asyncpg.UniqueViolationError: # Handle duplicates individually; for record in records: try: await conn.execute("INSERT INTO events VALUES ($1, $2, $3)", *record); except asyncpg.UniqueViolationError: self.stats["duplicates"] += 1. Why copy_records_to_table faster: Uses PostgreSQL COPY protocol (binary format). Bypasses query parser and planner. Single round-trip for entire batch. executemany: N round-trips for N rows. Use copy for bulk inserts (1000+ rows), executemany for small batches (<100 rows). Backpressure: semaphore = Semaphore(10); async with semaphore: await ingest_batch(). Pool monitoring: pool.get_size(), pool.get_idle_size(), alert if idle < 5.',
    keyPoints: [
      'copy_records_to_table: binary protocol, 100× faster, single round-trip, use for 1000+ rows',
      'Transaction per batch: atomic, rollback on error, handle UniqueViolationError individually',
      'Backpressure: Semaphore limits concurrent batches, prevents overwhelming database',
      'Pool monitoring: track size/idle, alert on low idle connections, scale pool if needed',
      'Batch size: 1000-10000 records optimal, balance memory vs throughput',
    ],
  },
  {
    id: 'ado-q-2',
    question:
      'Compare asyncpg connection pool vs creating connections per-request. Explain: (1) Connection establishment overhead, (2) When pool exhaustion occurs and how to handle, (3) pool.acquire() vs pool.fetchrow() differences, (4) Monitoring pool health metrics. Why is min_size=10, max_size=20 typical configuration?',
    sampleAnswer:
      'Connection pool vs per-request: (1) Overhead: New connection: TCP handshake (50ms) + SSL (50ms) + auth (20ms) = 120ms. Pooled connection: ~0ms (reuse existing). At 1000 req/s: New: 120s wasted overhead. Pooled: ~0s. (2) Pool exhaustion: Occurs when all max_size connections in use. Symptoms: acquire() blocks, requests queue. Handling: Set command_timeout to prevent infinite wait. Monitor queue depth. Scale pool if consistently exhausted. (3) pool.acquire() vs pool.fetchrow(): acquire(): Explicit connection from pool, manual release. Use for multiple operations on same connection. fetchrow(): Acquires internally, auto-releases. Use for single query. (4) Pool health metrics: pool.get_size(): current connections. pool.get_idle_size(): available connections. Alert if idle < 20% of size. Why min_size=10, max_size=20: min_size=10: Keep 10 connections warm (no cold start). max_size=20: Allow burst to 20 for traffic spikes. Ratio: 2:1 gives headroom. Too small: pool exhaustion. Too large: waste DB resources.',
    keyPoints: [
      'Overhead: new connection 120ms, pooled 0ms, critical at scale (1000 req/s saves 120s)',
      'Pool exhaustion: all max_size busy, acquire() blocks, set timeout, monitor queue depth',
      'acquire() for multiple ops on same connection, fetchrow() for single query (auto-releases)',
      'Pool health: track size/idle ratio, alert if idle <20%, scale if consistently low',
      'Configuration: min_size=10 (warm connections), max_size=20 (burst capacity), 2:1 ratio typical',
    ],
  },
  {
    id: 'ado-q-3',
    question:
      'Implement database transactions with proper error handling for a money transfer: (1) Deduct from account A, (2) Add to account B, (3) Check sufficient balance, (4) Rollback on any error, (5) Handle concurrent transfers (race conditions). Explain why conn.transaction() automatic commit/rollback is superior to manual BEGIN/COMMIT.',
    sampleAnswer:
      'Money transfer with transactions: async def transfer (pool, from_id, to_id, amount): async with pool.acquire() as conn: async with conn.transaction(): # Check balance; balance = await conn.fetchval("SELECT balance FROM accounts WHERE id = $1", from_id); if balance < amount: raise ValueError("Insufficient funds"); # Deduct from source; await conn.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from_id); # Add to destination; await conn.execute("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to_id); # Auto-commits here if no exception, rolls back if exception. Race condition handling: Use SELECT FOR UPDATE for locking: async with conn.transaction(): balance = await conn.fetchval("SELECT balance FROM accounts WHERE id = $1 FOR UPDATE", from_id); # Locks row until transaction completes. Prevents concurrent modifications. Why conn.transaction() better: Automatic: Commits on success, rolls back on exception. Exception-safe: Even if forgot to commit, rolls back. Context manager: Cleanup guaranteed. Manual BEGIN/COMMIT risks: Forgot COMMIT: changes not saved. Forgot ROLLBACK: partial updates. Exception during COMMIT: unclear state. conn.transaction() eliminates these bugs.',
    keyPoints: [
      'Transaction wraps all operations: deduct + add atomic, all-or-nothing',
      'Check balance first, raise ValueError if insufficient, automatic rollback',
      'Race conditions: SELECT FOR UPDATE locks row, prevents concurrent modifications',
      'conn.transaction() automatic: commits on success, rolls back on exception, exception-safe',
      'Manual BEGIN/COMMIT error-prone: forgot commit/rollback, unclear state on error',
    ],
  },
];
