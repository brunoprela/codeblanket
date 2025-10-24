/**
 * Quiz questions for Design Instagram section
 */

export const instagramQuiz = [
  {
    id: 'q1',
    question:
      'Instagram stores 100 million photos per day with 456 PB total over 5 years. Walk through your storage strategy: multiple image sizes, S3 storage classes, CDN integration, and cost optimization.',
    sampleAnswer:
      'STORAGE STRATEGY: (1) MULTIPLE SIZES: Generate 3 versions per photo: Thumbnail (150×150, 50 KB) for feed preview, Medium (1080×1080, 500 KB) for full view, Original (as uploaded, ~2 MB) for zoom/download. Total: 2.5 MB per photo. Why? 95% of views only need thumbnail or medium - saves bandwidth and improves load time. (2) S3 STORAGE CLASSES: Recent photos (<30 days): S3 Standard ($0.023/GB/month) - frequently accessed. Photos 30-90 days: S3 Standard-IA (Infrequent Access, $0.0125/GB, 45% savings) - less accessed but still available. Photos >90 days: S3 Intelligent-Tiering (auto-moves between tiers based on access patterns, 40-70% savings). Very old photos (>2 years): S3 Glacier ($0.004/GB, 83% savings) - archived, retrieval takes minutes (acceptable for old photos). (3) CDN INTEGRATION: All image URLs point to CloudFront, not S3 directly. CloudFront caches at 200+ edge locations globally. Popular photos cached at edge (< 50ms latency vs 200ms from S3). CDN cache hit rate: 80-90% → Only 10-20% requests hit S3. CDN cost: $0.085/GB vs S3 egress $0.09/GB (similar), but CDN provides better latency. (4) COST CALCULATION: 456 PB over 5 years at Standard pricing: 456,000 TB × $23/TB = $10.5M/month. With Intelligent-Tiering (50% avg savings): $5.2M/month over lifecycle. With Glacier for 2+ year old photos (60% savings): $4.2M/month. OPTIMIZATION: Delete unpopular photos after 5 years (< 10 views), compress more aggressively (WebP 30% smaller). KEY INSIGHT: Storage costs dominate at petabyte scale - tiering is mandatory, not optional.',
    keyPoints: [
      'Generate 3 sizes: thumbnail (50KB), medium (500KB), original (2MB)',
      'S3 Intelligent-Tiering: Auto-moves between Standard/IA/Glacier',
      'CloudFront CDN: 80-90% cache hit rate, <50ms global latency',
      'Cost: $10.5M/month at scale, reduce to $4-5M with tiering',
      'Old/unpopular photos move to Glacier (83% cost savings)',
    ],
  },
  {
    id: 'q2',
    question:
      'Design the photo upload pipeline for Instagram. Explain why you use async processing, how you generate multiple image sizes, and how you handle upload failures.',
    sampleAnswer:
      'PHOTO UPLOAD PIPELINE: (1) CLIENT UPLOAD: Mobile app compresses image to 2 MB max (client-side). Uploads to API server via POST /api/v1/photos. Server generates unique photo_id (UUID/base62). (2) IMMEDIATE RESPONSE: Server uploads original to S3: s3://photos/user_123/photo_abc_orig.jpg, publishes "new_photo" event to Kafka, returns photo_id to user immediately (< 1 second). User sees "processing" indicator. (3) ASYNC PROCESSING WORKER: Consumes Kafka event, downloads original from S3, generates 3 sizes using ImageMagick/Pillow: Thumbnail (150×150, 50 KB), Medium (1080×1080, 500 KB), Original (compressed, 2 MB). Uploads all to S3, updates photos table with URLs, publishes "processing_complete" event. (4) CLIENT POLLING: App polls GET /api/v1/photos/abc/status every 2 seconds until status="ready". WHY ASYNC? Image processing takes 3-5 seconds (resize, compress, upload). Users should not wait - better UX to show immediate upload success. Decouples upload from processing - if processing fails, upload already succeeded. (5) FAILURE HANDLING: If processing worker crashes: Kafka retries message (exponential backoff, 3 retries). Dead letter queue for failed jobs. Monitoring alerts if processing lag > 1 minute. User experience: Show "still processing" if taking too long. If fails after 5 minutes, show retry button. (6) PRE-SIGNED URLS (Advanced): Generate S3 pre-signed URL, return to client. Client uploads directly to S3 (faster, no proxy through server). Publish event when upload completes. TRADE-OFF: Faster upload response (1s vs 5s) for slight eventual consistency (photo appears after 5s processing). KEY INSIGHT: Async processing is standard for anything > 1 second - never block user on slow operations.',
    keyPoints: [
      'Immediate response (<1s): Upload original to S3, return photo_id',
      'Async worker: Generate 3 sizes, takes 3-5 seconds',
      'Kafka for reliability: Retries, dead letter queue',
      'Client polls for status until processing complete',
      'Pre-signed URLs: Client uploads directly to S3 (no proxy)',
    ],
  },
  {
    id: 'q3',
    question:
      'A popular photo gets 1 million likes in one hour. Explain why updating like_count synchronously is infeasible, and design a system using Redis that can handle this scale.',
    sampleAnswer:
      'WHY SYNCHRONOUS UPDATES FAIL: Naive approach: UPDATE photos SET like_count = like_count + 1 WHERE photo_id = X. At 1M likes/hour = 278 likes/sec, this creates: (1) Database bottleneck: 278 writes/sec to single row, (2) Row-level locking: Locks photo row, blocks all other operations, (3) Replication lag: Replicas fall behind primary, (4) Hotspot: All writes hit same shard (if sharded by photo_id). REDIS SOLUTION: (1) LIKE ACTION: User likes photo → Check Redis SET: SISMEMBER liked:{user_id} {photo_id}. If already liked, return error. Else: Add to SET: SADD liked:{user_id} {photo_id} (prevents double-like), Increment counter: INCR likes:{photo_id} (in-memory, sub-ms, no locking). Return success immediately. (2) DISPLAY COUNT: Query Redis: GET likes:{photo_id}, fallback to DB if not in Redis. Add DB count + Redis count for total. (3) BATCH SYNC TO DB: Background job runs every hour: Scans Redis keys likes:*, reads count, batch updates DB: UPDATE photos SET like_count = like_count + ? WHERE photo_id = ?, deletes Redis key after sync. (4) REDIS PERSISTENCE: Enable AOF (Append-Only File) persistence in Redis. If Redis crashes, rebuild from AOF on restart (may lose last few seconds). Alternatively: Keep Redis as cache only, query DB for authoritative count (acceptable staleness). (5) SCALING REDIS: Use Redis Cluster (sharded by photo_id) for horizontal scaling. Each shard handles subset of photos - distributes load. RESULT: Redis handles millions of increments/sec per node. Batched DB updates: 1M likes/hour ÷ 60 min = 17K updates/min (manageable). Like count may lag by up to 1 hour (acceptable for social media). TRADE-OFF: Real-time accuracy vs system scalability. Perfect accuracy would require distributed transactions (slow). Eventual consistency with <1 hour lag is acceptable. KEY INSIGHT: Analytics counters (likes, views, shares) should NEVER be synchronous database writes at scale - use Redis + batch sync.',
    keyPoints: [
      'Synchronous DB updates: 278 writes/sec create hotspots and locking',
      'Redis INCR: Millions of ops/sec, sub-ms latency, no locking',
      'Batch sync to DB hourly: 17K updates/min (manageable)',
      'Display count: Redis (real-time) + DB (base) for accuracy',
      'Redis Cluster for horizontal scaling of hot photos',
    ],
  },
];
