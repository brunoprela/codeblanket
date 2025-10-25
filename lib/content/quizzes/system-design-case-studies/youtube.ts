/**
 * Quiz questions for Design YouTube section
 */

export const youtubeQuiz = [
  {
    id: 'q1',
    question:
      'YouTube allows resumable uploads for large video files (up to 256 GB). Explain how chunked resumable uploads work, why they are critical for UX, and how you handle chunk failures.',
    sampleAnswer:
      'RESUMABLE UPLOAD PROTOCOL: (1) INITIATE: Client requests POST /api/v1/videos/upload/init with metadata (title, size). Server generates upload_session_id and pre-signed URL, determines chunk_size (10 MB), returns: {upload_url, session_id, chunk_size}. (2) CHUNKED UPLOAD: Client splits 1.5 GB video into 150 chunks (1.5 GB / 10 MB). Uploads chunks sequentially or in parallel (5 concurrent): PUT {upload_url}?chunk=5 with header Content-Range: bytes 50000000-59999999/1500000000. Server validates chunk order, writes to temp storage, updates progress: Redis SET upload:session_123 "chunks_uploaded: 5/150". Responds: 308 Resume Incomplete (if more chunks) or 200 OK (if complete). (3) FAILURE HANDLING: If chunk 67 fails (network timeout), client queries: GET {upload_url}?session=123 for progress. Server responds: "Last successful chunk: 66". Client resumes from chunk 67 (skips 1-66). (4) COMPLETION: After chunk 150 uploaded, server assembles chunks into full video file, moves from temp to permanent storage (GCS), publishes "video_uploaded" event to Pub/Sub, returns video_id. WHY CRITICAL? Without resumable: 1.5 GB upload fails at 99% → User must restart (terrible UX, wastes 1.4 GB bandwidth). With resumable: Failure at 99% → Resume from chunk 149 (uploads last 10 MB). MOBILE USERS: Cellular network unstable (frequent disconnects). Resumable ensures upload completes over multiple sessions (hours/days). IMPLEMENTATION: Store upload state in Redis (fast, TTL 7 days). After 7 days, incomplete uploads expire (cleanup). KEY INSIGHT: Large file uploads MUST be resumable - non-negotiable for modern web apps. Protocol: Google Resumable Upload (RFC 5789), similar to TUS protocol.',
    keyPoints: [
      'Split large file into 10 MB chunks, upload sequentially/parallel',
      'Server tracks progress in Redis (chunk_uploaded: 67/150)',
      'On failure, query progress and resume from last successful chunk',
      'Critical for mobile (unstable networks) and large files (hours to upload)',
      'Google Resumable Upload protocol (308 Resume Incomplete)',
    ],
  },
  {
    id: 'q2',
    question:
      "Explain YouTube\'s recommendation algorithm in detail. How does it go from 500 million videos to the 20 videos shown on your homepage? Discuss candidate generation, ranking, and the optimization objective.",
    sampleAnswer:
      'TWO-STAGE RECOMMENDATION: (1) CANDIDATE GENERATION (500M → 1000 candidates): Goal: Reduce search space from entire catalog to manageable subset. METHODS: (a) Collaborative Filtering: Train matrix factorization model on (user, video, watch) triples. User embedding (200-dim vector) learned from watch history. Video embedding (200-dim vector) learned from user interactions. Candidate score = dot product (user_embedding, video_embedding). Select top 500 videos by score. (b) Content-Based: Find videos similar to recently watched. Use video embeddings (trained on co-watch patterns: "users who watched X also watched Y"). Nearest neighbor search in embedding space. (c) Subscription Feed: Latest uploads from subscribed channels (top 100). (d) Trending/Popular: Globally trending videos (top 50). TOTAL: ~1000 candidates from all sources. (2) RANKING (1000 → 20 displayed): Goal: Rank candidates by expected watch time (optimization metric). FEATURES (100+): Video features (age, duration, view count, like ratio, thumbnail CTR), User features (watch history on similar content, engagement rate, session behavior), Context (device type, time of day, previously impressed), Cross features (user × video interaction predictions). MODEL: Gradient Boosted Trees or Deep Neural Network trained on billions of examples. Label: Actual watch time (seconds). Prediction: Expected watch time if shown. RANKING SCORE: P(click) × P(watch | click) × expected_watch_time. P(click): CTR model (will user click thumbnail?). P(watch | click): User won\'t immediately back out. expected_watch_time: Regression prediction (180 seconds). Example: Video A: 0.05 × 0.8 × 300s = 12.0 points. Video B: 0.10 × 0.4 × 200s = 8.0 points. Rank A higher despite lower CTR (more watch time). (3) RE-RANKING & DIVERSITY: Remove duplicates (same video, same channel in top 20). Inject diversity: Mix categories (educational, music, gaming). Insert fresh content: Recent uploads from subscriptions. Final order: Personalized top 20 videos for homepage. OPTIMIZATION OBJECTIVE: Maximize total watch time on YouTube (not clicks, not views). Why watch time? Leads to higher engagement, more ad revenue, better user retention. TRAINING PIPELINE: Batch job: Process 1B user interactions/day. Train model daily on last 30 days of data (billions of examples). Deploy to TensorFlow Serving (real-time inference < 100ms). Caching: Pre-compute recommendations for active users, cache in Redis (5-min TTL). KEY INSIGHT: Recommendations are machine learning problem at massive scale (billions of examples, 100M+ models). Two-stage funnel (candidate generation → ranking) necessary to handle 500M video corpus in real-time.',
    keyPoints: [
      'Stage 1: Candidate generation (500M → 1K using collaborative filtering, content-based, subscriptions)',
      'Stage 2: Ranking (1K → 20 using GBDT/DNN with 100+ features)',
      'Optimization objective: Maximize expected watch time (not clicks)',
      'Scoring: P(click) × P(watch|click) × expected_watch_time',
      'Models trained daily on billions of interactions, cached in Redis',
    ],
  },
  {
    id: 'q3',
    question:
      "Design YouTube\'s view counting system. 10 million users watch a viral video simultaneously. How do you increment the view counter without overwhelming the database? Explain your solution with Redis and discuss consistency trade-offs.",
    sampleAnswer:
      'VIEW COUNTING CHALLENGES: Naive approach: UPDATE videos SET view_count = view_count + 1 WHERE video_id = X on every view. PROBLEMS: (1) Database bottleneck: 10M concurrent viewers = potential 10M writes/sec to single row (impossible). (2) Row locking: UPDATE locks row, blocks concurrent writes and reads. (3) Replication lag: Primary overwhelmed, replicas lag behind. (4) Hotspot: If sharded by video_id, viral video creates shard hotspot. SOLUTION - REDIS AGGREGATION: (1) VIEW EVENT: User watches video for 30 seconds (validated view). Client sends: POST /api/v1/view {video_id, user_id, timestamp}. API server increments Redis counter: INCR views:dQw4w9WgXcQ. Redis handles millions of increments/sec per node (in-memory, no disk). Returns 200 OK immediately (no DB write). (2) DISPLAY COUNT: User requests video page. Query Redis: GET views:dQw4w9WgXcQ → 1,425,673 (real-time count). Fallback: If Redis miss, query database (base count) + sum pending Redis deltas. Display to user: "1.4M views" (rounded). (3) BATCH SYNC TO DATABASE: Background job runs every 10 minutes. Scans all Redis keys: KEYS views:* (or scan with cursor). For each key: Read count (GET views:dQw4w9WgXcQ), batch update DB: UPDATE videos SET view_count = view_count + 1425673 WHERE video_id = \'dQw4w9WgXcQ\', delete Redis key (DEL views:dQw4w9WgXcQ) or reset to 0 (SET views:dQw4w9WgXcQ 0). Database receives ~10K batch updates instead of 10M individual writes (1000x reduction). (4) REDIS PERSISTENCE: Enable AOF (Append-Only File) or RDB snapshots. If Redis crashes, rebuild from snapshot (may lose last few minutes of counts). Alternatively: Never delete Redis keys, keep as delta forever, always display: DB count + Redis delta. CONSISTENCY TRADE-OFFS: (a) Eventual consistency: View count may lag by up to 10 minutes in database. Displayed count is real-time (Redis) so users see accurate numbers. (b) Lost counts: If Redis crashes before sync, lose up to 10 minutes of views (acceptable for analytics data). (c) Double counting: If sync job runs twice due to failure, views may be over-counted (use idempotency key to prevent). SCALING: For extremely viral video (100M views/hour): Shard Redis by video_id: Use Redis Cluster with 10 shards. Each shard handles 10M views/hour (manageable). Aggregate across shards: Total views = SUM(views from all shards). ALTERNATIVE - KAFKA: Write view events to Kafka instead of direct Redis. Consumer aggregates views in sliding windows (Kafka Streams). More complex but better for real-time analytics (latency breakdowns, traffic sources). KEY INSIGHT: Analytics counters (views, likes, shares) should NEVER go directly to database - always use Redis/Kafka aggregation layer. Batch writes are mandatory at scale.',
    keyPoints: [
      'Redis INCR for view counting (millions of ops/sec, in-memory)',
      'Background job syncs Redis → DB every 10 minutes (batch updates)',
      'Display count: Redis (real-time) + DB (base) for accuracy',
      'Trade-off: Up to 10-min lag in DB, potential lost counts if Redis crashes',
      'Shard Redis for viral videos (100M+ views/hour)',
    ],
  },
];
