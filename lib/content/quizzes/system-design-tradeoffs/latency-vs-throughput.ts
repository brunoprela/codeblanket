/**
 * Quiz questions for Latency vs Throughput section
 */

export const latencyvsthroughputQuiz = [
  {
    id: 'q1',
    question:
      'You are designing a notification system that sends push notifications to mobile apps. Should you optimize for latency or throughput for: (a) User-triggered notification ("John liked your post"), (b) Daily digest email to 10M users? Justify your answer.',
    sampleAnswer:
      "Notification system latency vs throughput analysis: (a) USER-TRIGGERED NOTIFICATION - Optimize for LATENCY: Context: User A likes User B's post → User B should receive notification quickly. Reasoning: Real-time user expectation. User B expects to see notification within seconds. This is a user-facing, interactive flow. Each notification is independent (not batch). Requirements: Latency < 3 seconds from trigger to mobile push. User perceives > 10 seconds as \"broken.\" Implementation: Direct push to Firebase Cloud Messaging (FCM) or APNs immediately. Use message queue with multiple consumers for parallelism (maintain low latency). In-memory cache for user device tokens (avoid DB lookup latency). Metrics: P99 latency < 3 seconds, throughput secondary concern (user-triggered is low volume). Trade-off: Sending immediately means higher cost per notification (can't batch), but latency critical for UX. (b) DAILY DIGEST EMAIL - Optimize for THROUGHPUT: Context: Send 10M emails in 2-hour window (8am-10am). Reasoning: Background batch job. No user waiting for individual email. Goal: Send all 10M emails efficiently. Individual email latency doesn't matter. Requirements: Throughput: 10M emails / 2 hours = 83,333 emails/minute = 1,389 emails/second. Individual email latency not measured (user not waiting). Implementation: Batch processing: Collect 10,000 emails, send in bulk to email service (SendGrid, SES). Parallel workers: 100 worker threads, each sending batches. Connection reuse: Keep persistent connections to email service (avoid handshake overhead). Metrics: Throughput 1,500+ emails/second, latency per email irrelevant. Trade-off: Individual emails may take 10+ minutes to send (wait for batch), but total job completes faster and cheaper. Summary: User-triggered real-time notifications → Latency critical. Batch email campaigns → Throughput critical.",
    keyPoints: [
      'User-triggered notifications: Latency critical (< 3 seconds), user waiting',
      'Daily digest: Throughput critical (1,500 emails/sec), batch job',
      'Real-time user-facing → optimize for latency per operation',
      'Background batch jobs → optimize for total work completed',
      'Batching trades latency for throughput (good for digest, bad for real-time)',
    ],
  },
  {
    id: 'q2',
    question:
      "Explain Little's Law and how you would use it to determine the required concurrency for an API server that needs to handle 5,000 requests/second with a P99 latency of 100ms.",
    sampleAnswer:
      "Little's Law application for API server capacity planning: LITTLE'S LAW FORMULA: Average Number of Requests in System (L) = Arrival Rate (λ) × Average Time in System (W). For capacity planning, rearranged as: Required Concurrency = Throughput × Latency. GIVEN REQUIREMENTS: Throughput: 5,000 requests/second. P99 latency: 100ms = 0.1 seconds. CALCULATION: Required Concurrency = 5,000 req/s × 0.1s = 500 concurrent requests. INTERPRETATION: To sustain 5,000 req/s at 100ms latency, system must handle 500 concurrent requests in flight simultaneously. Each request takes 100ms, so at any moment, 500 requests are being processed. ARCHITECTURE IMPLICATIONS: (1) Thread pool sizing: If using thread-per-request model, need 500+ threads. Reality: Use async I/O (Node.js, Go) to handle thousands of concurrent connections with fewer threads. (2) Connection pooling: Database connection pool should be sized appropriately. Rule of thumb: 1 connection per 10 concurrent requests = 50 DB connections. (3) Load balancing: If single server can handle 100 concurrent requests, need 5 servers (500/100). (4) Resource allocation: CPU, memory, network bandwidth must support 500 concurrent operations. WHAT IF REQUIREMENTS CHANGE? (a) Reduce latency to 50ms: Required Concurrency = 5,000 × 0.05 = 250 concurrent requests (easier to handle). (b) Increase throughput to 10,000 req/s: Required Concurrency = 10,000 × 0.1 = 1,000 concurrent requests (need more resources). VALIDATION: Monitor in production: Actual throughput, actual P99 latency, actual concurrent requests. If actual < target, bottleneck exists (CPU, DB, network). PRACTICAL USE: Before launch, use Little's Law to estimate infrastructure needs. Cost estimation: Know how many servers/resources needed.",
    keyPoints: [
      "Little's Law: Concurrency = Throughput × Latency",
      '5,000 req/s at 100ms latency = 500 concurrent requests',
      'Use for capacity planning: thread pools, connection pools, server count',
      'Lower latency or higher throughput requires more concurrency',
      'Validate assumptions by monitoring production metrics',
    ],
  },
  {
    id: 'q3',
    question:
      'You are optimizing a database query that runs in a batch job processing 100M records. The current implementation processes one record at a time with 1ms latency per record (total time: 100,000 seconds = 28 hours). How would you optimize for throughput? What trade-offs would you make?',
    sampleAnswer:
      "Database batch job optimization for throughput: CURRENT STATE: Processing: 1 record at a time (serial). Latency per record: 1ms. Total time: 100M records × 1ms = 100,000 seconds = 28 hours. Throughput: 1,000 records/second (1/0.001s). Problem: Too slow for overnight job (need < 8 hours). OPTIMIZATION STRATEGY - Optimize for throughput: APPROACH 1: BATCHING. Instead of 100M individual queries, batch into chunks. Implementation: Process 10,000 records per query using IN clause or batch SELECT. SELECT * FROM table WHERE id IN (1,2,3,...,10000); Latency per batch: 50ms (slightly higher than 1ms × 10,000 due to overhead). Batches needed: 100M / 10,000 = 10,000 batches. Total time: 10,000 batches × 50ms = 500 seconds = 8.3 minutes (200x faster!). Throughput: 100M / 500s = 200,000 records/second. Trade-off: Higher memory usage (load 10K records at once), slightly higher latency per individual record (but nobody cares in batch job). APPROACH 2: PARALLEL PROCESSING. Run multiple workers simultaneously. Implementation: Partition data by ID ranges: Worker 1: IDs 1-10M, Worker 2: IDs 10M-20M, ..., Worker 10: IDs 90M-100M. Each worker processes its partition (10M records each). With batching (10K per batch): Each worker takes 500 seconds / 10 = 50 seconds. Total time: 50 seconds (parallel) vs 500 seconds (serial) = 10x faster. Throughput: 100M / 50s = 2M records/second. Trade-off: More database load (10 concurrent connections), more complex coordination (ensure no overlap). APPROACH 3: STREAMING (CURSOR). For very large result sets, use cursor to avoid loading all data into memory. Implementation: Open cursor, fetch 10K records at a time. Process batch, fetch next batch. Benefit: Constant memory usage (don't load 100M records). Trade-off: Slightly slower than bulk batch (cursor overhead). FINAL ARCHITECTURE: 10 parallel workers, each processing 10M records in batches of 10K. Database connection pool: 10 connections (one per worker). Result: 50 seconds total (was 28 hours = 2,000x faster!). Cost: Higher DB load, more complexity. In batch job, latency per record increased from 1ms to ~50ms, but throughput increased 200x.",
    keyPoints: [
      'Batching: Process 10K records per query (200x faster)',
      'Parallel processing: 10 workers (10x faster)',
      'Combined: 2,000x speedup (28 hours → 50 seconds)',
      'Trade-off: Higher memory, more DB load, higher per-record latency',
      'Batch jobs: Sacrifice per-item latency for total throughput',
    ],
  },
];
