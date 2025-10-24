/**
 * Quiz questions for Design Pastebin section
 */

export const pastebinQuiz = [
  {
    id: 'q1',
    question:
      'Explain your strategy for handling paste expiration at scale. Compare lazy deletion vs active deletion, and describe how you would implement cleanup without impacting user-facing performance.',
    sampleAnswer:
      'EXPIRATION STRATEGIES: (1) LAZY DELETION: Check expires_at during read requests. If current_time > expires_at, return 404 and mark for deletion. PROS: Zero background job overhead, simple implementation. CONS: Expired pastes take storage until accessed, storage grows unnecessarily. (2) ACTIVE DELETION: Background job runs periodically (every hour/day) and deletes expired pastes: DELETE FROM pastes WHERE expires_at < NOW() LIMIT 10000. PROS: Reclaims storage promptly, predictable storage usage. CONS: Database load from bulk deletes, can impact performance if not throttled. PRODUCTION APPROACH (Hybrid): (a) Lazy deletion at read time (returns 404 immediately). (b) Background cleanup job runs daily at 3 AM (low traffic): Uses rate-limited batched deletes (10K rows per batch, sleep 1 second between batches). Indexes on expires_at make queries efficient. (c) Archive expired pastes to S3 before deletion (for analytics/recovery). (d) Vacuum/optimize tables weekly to reclaim space. METRICS: Track daily deletion rate vs expiration rate to ensure cleanup keeps pace. For 30% 1-day expiry from 1M daily pastes = 300K deletions needed per day. Batched cleanup of 10K every 5 minutes = 288K per day, sufficient. KEY INSIGHT: Never delete synchronously during user requests - always use background jobs with rate limiting to protect database.',
    keyPoints: [
      'Lazy deletion: Check expires_at at read time, return 404',
      'Active deletion: Background job with batched deletes (10K limit)',
      'Run cleanup during low-traffic hours (3 AM)',
      'Index on expires_at critical for efficient cleanup queries',
      'Never delete synchronously in user request path',
    ],
  },
  {
    id: 'q2',
    question:
      'Your Pastebin has 1 million pastes per day, with average size 10 KB. Some power users paste 50 MB log files. How do you handle storage efficiently? Explain your strategy for large pastes, compression, and deduplication.',
    sampleAnswer:
      'MULTI-TIER STORAGE STRATEGY: (1) SMALL PASTES (< 10 MB): Store directly in database MEDIUMTEXT column (16 MB max). Fast queries, simple architecture. 99% of pastes are <10 MB, so most pastes stay in DB. (2) LARGE PASTES (10-50 MB): Upload to S3, store only metadata + S3 key in database. Schema: s3_key VARCHAR(255) → "s3://bucket/pastes/abc123.txt". Serve via CloudFront CDN for global low-latency access. (3) COMPRESSION: Apply gzip compression before storing content. Typical compression: 5:1 for code/logs (JSON, XML). 10 KB paste → 2 KB compressed = 80% storage savings. Decompress on read (CPU cost acceptable). Store compressed_size to calculate true storage. (4) DEDUPLICATION: Compute SHA-256 hash of content before insert. Query: SELECT paste_id FROM pastes WHERE content_hash = ? AND (expires_at IS NULL OR expires_at > NOW()). If duplicate found and not expired, return existing paste_id (saves insert + storage). Deduplication savings: 20-30% for common errors, stack traces, boilerplate code. (5) STORAGE CALCULATION: 1M pastes/day × 10 KB avg = 10 GB/day raw. After compression (5:1): 2 GB/day. After deduplication (30%): 1.4 GB/day → 511 GB/year → 5.1 TB over 10 years (very manageable). Large pastes (1% of 1M = 10K/day × 50 MB) = 500 GB/day → Use S3 lifecycle to archive to Glacier after 30 days (10x cost reduction). KEY INSIGHT: Different storage tiers optimize cost: hot DB for small/frequent, S3 for large, Glacier for archived.',
    keyPoints: [
      'Small pastes (<10 MB): Store in database with compression',
      'Large pastes (>10 MB): Upload to S3, serve via CDN',
      'Gzip compression: 5:1 savings for code/logs',
      'SHA-256 deduplication: 20-30% storage savings',
      'S3 lifecycle: Archive large old pastes to Glacier',
    ],
  },
  {
    id: 'q3',
    question:
      'Design the view counter feature for Pastebin. Explain why updating view_count synchronously is a bad idea, and propose an efficient solution for tracking millions of paste views per day.',
    sampleAnswer:
      'WHY SYNCHRONOUS UPDATES ARE BAD: Naive approach: UPDATE pastes SET view_count = view_count + 1 WHERE paste_id = ? on every read. PROBLEMS: (1) Write amplification: 120 reads/sec = 120 DB writes/sec (10x write load increase). (2) Row-level locking: Updates lock the paste row, blocking concurrent reads (defeats purpose of read replicas). (3) Replication lag: Writes go to primary, reads from replicas, counters become inconsistent. (4) Slow queries: Index lookup + write for every read adds 10-20ms latency. EFFICIENT SOLUTION - Redis + Batch Sync: (1) ON READ: Increment counter in Redis: INCR views:paste_id (in-memory, sub-millisecond, no locking). Redis handles millions of increments/sec easily. (2) BACKGROUND JOB (Hourly): Query all Redis keys: KEYS views:*. For each key, read count: GET views:paste_id. Batch update DB: UPDATE pastes SET view_count = view_count + ? WHERE paste_id = ?. DELETE Redis key after sync. (3) DISPLAY COUNT: Current count = DB count + Redis count (if exists). This gives real-time accuracy without synchronous writes. ADVANCED: Use Redis sorted set for top pastes: ZINCRBY trending:daily paste_id 1. Query leaderboard: ZREVRANGE trending:daily 0 99 WITHSCORES. TRADE-OFFS: (a) View counts may be lost if Redis crashes before hourly sync (acceptable - analytics data, not critical). (b) Slight inconsistency between Redis and DB during the hour (acceptable). (c) Need to combine Redis + DB counts when displaying (small complexity). RESULT: Zero write load on paste reads, 120 reads/sec remains pure reads. Batch DB updates: 2M unique views/day ÷ 24 hours = 83K updates/hour (easily handled). KEY INSIGHT: Analytics counters should NEVER block user-facing requests - use async aggregation.',
    keyPoints: [
      'Synchronous updates kill read performance (write amplification)',
      'Redis INCR for real-time counting (millions of ops/sec)',
      'Hourly batch sync from Redis to database',
      'Display count = DB count + Redis delta',
      'Acceptable to lose hourly counts if Redis crashes (not critical data)',
    ],
  },
];
