/**
 * Multiple choice questions for Design Instagram section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const instagramMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Instagram stores 100 million photos per day, with an average photo requiring 2.5 MB (thumbnail + medium + original). How much storage is needed per day?',
    options: [
      '100 MB (100M photos × 1 KB metadata)',
      '250 GB (100M × 2.5 KB)',
      '250 TB (100M × 2.5 MB)',
      '250 PB (100M × 2.5 GB)',
    ],
    correctAnswer: 2,
    explanation:
      '100 million photos × 2.5 MB = 250,000,000 MB = 250,000 GB = 250 TB per day. Over 5 years: 250 TB/day × 365 days × 5 years = 456 PB (petabytes). This is why Instagram needs massive distributed object storage (S3) and cannot use traditional databases. At $0.023/GB/month for S3 Standard, this would cost $10.5M/month without optimization. This scale requires S3 Intelligent-Tiering and Glacier for cost management.',
  },
  {
    id: 'mc2',
    question:
      'Why does Instagram generate multiple image sizes (thumbnail, medium, original) instead of serving the original and letting clients resize?',
    options: [
      'To use more storage (better for cloud providers)',
      'To reduce client-side CPU and bandwidth - 95% of views only need smaller sizes',
      'To make the system more complex',
      'Thumbnails are required by law',
    ],
    correctAnswer: 1,
    explanation:
      'BANDWIDTH & PERFORMANCE: Feed view shows 10-20 thumbnails (50 KB each). If serving originals: 10 × 2 MB = 20 MB download vs 10 × 50 KB = 500 KB (40x savings). Mobile data cost and load time matter. CLIENT CPU: Resizing 2 MB image on phone drains battery and slows app. LAZY LOADING: Load thumbnail first (fast), then medium when viewing (good UX), original only if zooming (rare). COST: Storage is cheap ($0.023/GB/month), bandwidth is expensive ($0.09/GB egress). Storing 3 sizes (3x storage) saves 20-40x bandwidth. TRADE-OFF: 2.5x storage cost for 20-40x bandwidth savings - clear win.',
  },
  {
    id: 'mc3',
    question:
      'Instagram uses CloudFront CDN with 80% cache hit rate for image delivery. How does this reduce S3 load?',
    options: [
      'CDN has no effect on S3 load',
      'S3 requests reduced by 80% (only cache misses hit S3)',
      'CDN stores all images permanently',
      'S3 requests increase (CDN adds overhead)',
    ],
    correctAnswer: 1,
    explanation:
      'With 80% CDN cache hit rate, only 20% of image requests reach S3 (cache misses). Example: 1 billion image requests/day → 800M served from CDN edge (< 50ms), 200M hit S3 (200ms+). S3 load reduced by 5x. BENEFITS: (1) Latency: 800M requests at <50ms vs 200ms (4x faster), (2) S3 costs: Reduced GET requests (fewer operations), (3) Global performance: Edge locations near users, (4) S3 bandwidth: Reduced egress from S3. CDN cache hit rate depends on popularity (hot images stay cached 24+ hours). This is why CDN is mandatory for image-heavy services at scale.',
  },
  {
    id: 'mc4',
    question:
      'Instagram uses async image processing via Kafka after upload. What happens if the processing worker crashes before completing?',
    options: [
      'User loses the photo (must re-upload)',
      'Kafka retries the message; worker picks it up again after restart',
      'Photo stays in "processing" state forever',
      'Upload was never completed',
    ],
    correctAnswer: 1,
    explanation:
      'KAFKA RELIABILITY: Messages are persisted to disk until acknowledged. If worker crashes mid-processing: (1) Message not acknowledged → Kafka retries. (2) Worker restarts, consumes same message again. (3) Processing completes, worker acknowledges, Kafka deletes message. IDEMPOTENCY: Processing must be idempotent (safe to run twice). Check if image sizes already exist before regenerating. DEAD LETTER QUEUE: After 3 retries, message goes to DLQ for manual investigation. USER EXPERIENCE: Photo already uploaded to S3, shows as "processing" until worker completes. This decoupling is why async processing is resilient.',
  },
  {
    id: 'mc5',
    question:
      'Instagram shards the photos table by user_id. A celebrity with 100M followers posts a photo. What performance issue arises?',
    options: [
      'No issue - sharding solves all problems',
      "All 100M followers try to view the photo, creating a hotspot on the celebrity's shard",
      'The photo cannot be stored due to size',
      'Followers cannot see the photo',
    ],
    correctAnswer: 1,
    explanation:
      "HOTSPOT PROBLEM: Celebrity's shard receives millions of read requests (SELECT * FROM photos WHERE photo_id = X). This shard's read replicas get overloaded while other shards idle. SOLUTIONS: (1) Aggressive caching: Cache celebrity photos in Redis with longer TTL (24+ hours). Cache at application layer too. (2) Read replicas: Add 5-10 read replicas specifically for hot shards. (3) CDN for images: Photo metadata from DB, but images from CDN (distributes load). (4) Separate shards for celebrities: Move VIP users to dedicated infrastructure. REAL-WORLD: Instagram/Twitter use all these techniques. Hotspots are inevitable with sharding + skewed access patterns. KEY INSIGHT: Sharding solves write scaling but introduces new problems (hotspots, scatter-gather). Requires additional techniques (caching, replication) to fully solve.",
  },
];
