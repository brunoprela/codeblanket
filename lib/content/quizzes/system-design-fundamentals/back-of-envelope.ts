/**
 * Quiz questions for Back-of-the-Envelope Estimations section
 */

export const backofenvelopeQuiz = [
  {
    id: 'q1',
    question:
      'Walk through a complete storage estimation for Instagram, including assumptions, calculations, and architectural implications.',
    sampleAnswer:
      "INSTAGRAM STORAGE ESTIMATION: ASSUMPTIONS: 500M DAU, Each user uploads 0.5 photos/day (some upload many, most upload none), Average photo: 2MB (compressed), 2% users upload stories (30 sec video, 5MB), Data retention: Permanent, Replication factor: 3× for redundancy. CALCULATIONS: Daily uploads: Photos: 500M × 0.5 = 250M photos/day. Photo storage: 250M × 2MB = 500 TB/day. Stories: 500M × 2% × 1 story = 10M videos/day. Story storage: 10M × 5MB = 50 TB/day. Total daily: 500 + 50 = 550 TB/day. With replication: 550 TB × 3 = 1.65 PB/day. Yearly: 1.65 PB × 365 ≈ 600 PB/year. 5-year projection: 3,000 PB = 3 EB. COST CALCULATION: S3 storage: 3000 PB × $0.023/GB/month = $69M/month (after 5 years) ARCHITECTURAL IMPLICATIONS: (1) Need distributed object storage (S3, or build own like Facebook Haystack). (2) Multiple tiers: Hot storage (recent, SSD), Cold storage (old, HDD), Archive (rarely accessed, glacier). (3) CDN essential: Can't serve 3 EB from origin. Need edge caching globally. (4) Compression: Aggressive image compression to reduce storage 30-40%. (5) Deduplication: Same photo uploaded multiple times → store once. (6) Consider deletion: Delete after X years or for inactive users? (7) Geographic distribution: Store in regions close to users. REALITY CHECK: Instagram actually stores exabytes of data. Our estimation is reasonable. Facebook built Haystack specifically for this scale.",
    keyPoints: [
      'Start with clear assumptions: users, upload rate, file sizes',
      'Calculate daily volume, then extrapolate to yearly and multi-year',
      'Include replication factor (typically 3×)',
      'Estimate costs using cloud storage pricing',
      'Derive architecture: distributed storage, tiered storage, CDN, compression',
    ],
  },
  {
    id: 'q2',
    question:
      "You're designing a real-time messaging app like WhatsApp. Estimate the bandwidth and server requirements, and explain how these numbers drive your architecture.",
    sampleAnswer:
      "WHATSAPP BANDWIDTH & SERVER ESTIMATION: ASSUMPTIONS: 2B registered users, 500M DAU (25%), Average 50 messages/user/day, Average message: 100 bytes text, 5% messages have images (200KB avg), Peak traffic: 3× average (messaging is bursty). MESSAGE THROUGHPUT: Total messages/day: 500M × 50 = 25B messages/day. Messages/second: 25B / 86,400 ≈ 290K msg/sec (average). Peak: 290K × 3 ≈ 900K msg/sec. BANDWIDTH: Text only: 290K msg/sec × 100 bytes = 29 MB/sec = 232 Mbps. Images: 290K × 5% × 200KB = 2.9 GB/sec = 23 Gbps. Total: ~24 Gbps average, 72 Gbps peak. SERVER COUNT ESTIMATION: Assume WebSocket server handles 50K concurrent connections. Total concurrent users (assume 10% online): 500M × 10% = 50M concurrent. Servers needed: 50M / 50K = 1,000 servers minimum. Add 50% buffer: 1,500 servers. DATABASE WRITES: Message metadata: 290K writes/sec. Single MySQL: ~1K writes/sec → Need 290 shards. Better: Cassandra (10K writes/sec/node) → 30 nodes. ARCHITECTURAL IMPLICATIONS: (1) WebSocket connections: Need connection servers separate from business logic. (2) Message routing: Users connect to different servers → need routing layer (Redis pub/sub or Kafka). (3) Database: NoSQL for writes (Cassandra), eventually consistent OK. (4) Media: Store images in S3/blob storage, send URLs not files. (5) Geographic distribution: Deploy servers globally, route to nearest datacenter. (6) Optimization: Message batching reduces overhead, Connection pooling for database. REALITY CHECK: WhatsApp famously handled 900M users with 50 engineers using Erlang's massive concurrency. Our estimates align with real scale.",
    keyPoints: [
      'Calculate message throughput: total messages/day → msg/sec',
      'Estimate bandwidth: message size × frequency',
      'Server count: concurrent users ÷ connections per server',
      'Database: write rate determines sharding needs',
      'Architecture: WebSocket servers, message routing, distributed DB',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain why "peak vs average" traffic matters and how you would estimate peak multipliers for different types of applications.',
    sampleAnswer:
      'PEAK VS AVERAGE TRAFFIC: WHY IT MATTERS: Systems must be designed for PEAK load, not average. If you design for average: (1) System crashes during peak. (2) Users see errors, timeouts. (3) Poor user experience, lost revenue. (4) Database overwhelm, cascading failures. Example: E-commerce on Black Friday can see 10-20× normal traffic. If designed for average, site goes down when it matters most. PEAK MULTIPLIERS BY APPLICATION TYPE: (1) SOCIAL MEDIA (2-3× average): Twitter during major events: 2-3× normal. Instagram during holidays: 2× normal. Reason: Events cause spikes, but usage spreads throughout day. (2) NEWS/MEDIA (5-10× average): Breaking news: Massive spike for hours. CNN during election night: 10× traffic. Reason: Everyone checks simultaneously. (3) E-COMMERCE (10-20× average): Black Friday, Cyber Monday, Prime Day. Reason: Concentrated shopping window, everyone rushes. (4) VIDEO STREAMING (2-4× average): Netflix at 8PM: Peak viewing hours. Reason: People watch after work/dinner. (5) ENTERPRISE APPS (1.5-2× average): Email, Slack during business hours. Reason: Workday has start/end, but otherwise steady. (6) GAMING (3-5× average): Weekends, new releases, special events. Reason: Leisure activity, concentrated play times. HOW TO ESTIMATE: (1) Analyze historical data if available. (2) Industry benchmarks. (3) Event-driven: Consider what drives spikes. (4) Time-based: Hourly, daily, seasonal patterns. (5) Safety factor: Add 20-30% buffer beyond estimated peak. ARCHITECTURAL IMPLICATIONS: Auto-scaling: Must scale servers based on load. Caching: Absorb read spikes without hitting database. Queue systems: Smooth out write spikes. CDN: Handle content delivery spikes. Rate limiting: Protect system from overload. Graceful degradation: If peak exceeds capacity, degrade non-critical features. EXAMPLE CALCULATION: Average QPS: 10K. Peak multiplier: 3×. Peak QPS: 30K. Servers needed: 30K QPS / 5K QPS per server = 6 servers at peak. Add buffer: 8 servers deployed. Auto-scale: Start with 3 servers (average), scale to 8 at peak.',
    keyPoints: [
      'Design for peak load, not average (or system crashes)',
      'Peak multipliers vary by application type: 2-3× (social), 10-20× (e-commerce)',
      'Estimate peaks using historical data, industry benchmarks, event analysis',
      'Architecture must support scaling: auto-scaling, caching, queues',
      'Add safety buffer (20-30%) beyond estimated peak',
    ],
  },
];
