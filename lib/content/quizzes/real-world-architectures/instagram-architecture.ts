/**
 * Quiz questions for Instagram Architecture section
 */

export const instagramarchitectureQuiz = [
    {
        id: 'q1',
        question: 'Explain Instagram\'s hybrid fanout approach for feed generation. Why does Instagram use fanout-on-write for regular users but fanout-on-read for celebrities?',
        sampleAnswer: 'Instagram uses hybrid fanout to balance write performance and read latency. Fanout-on-write for regular users (<1M followers): When user posts, write to all followers\' feeds immediately in Cassandra. Read is fast (pre-computed feed), write is acceptable (most users have <1K followers = 1K writes). Fanout-on-read for celebrities (>1M followers): Don\'t write to followers\' feeds. When follower requests feed, fetch celebrity posts on-demand and merge with pre-computed feed. Avoids 100M+ writes per celebrity post. Trade-offs: Regular users get instant feed updates (good UX), celebrities have slightly slower feed generation (acceptable since it\'s still <500ms). Threshold typically 1M followers. This scales to billions of users while maintaining performance.',
        keyPoints: [
            'Fanout-on-write for regular users: Write to followers\' feeds, fast reads',
            'Fanout-on-read for celebrities: Fetch on-demand, avoid 100M+ writes',
            'Threshold: ~1M followers determines which approach',
            'Hybrid provides scalability while maintaining performance',
        ],
    },
    {
        id: 'q2',
        question: 'How did Instagram migrate from PostgreSQL to TAO (Facebook\'s data store), and what were the benefits?',
        sampleAnswer: 'After Facebook acquisition, Instagram migrated to TAO over 18 months. Migration strategy: (1) Dual writes to both PostgreSQL and TAO during transition. (2) Read from TAO, validate against PostgreSQL. (3) Gradually shift read traffic to TAO. (4) Fully cut over once validation complete. Benefits: (1) Performance - TAO optimized for social graph queries with 99%+ cache hit rate. (2) Scalability - TAO handles billions of objects and associations. (3) Unified platform - Share infrastructure with Facebook. (4) Expertise - Leverage Facebook\'s operational experience. TAO uses MySQL + Memcached underneath but provides graph-oriented API (objects and associations). Trade-offs: Migration complexity, new operational model, but worth it for performance gains.',
        keyPoints: [
            'Dual writes during migration (PostgreSQL + TAO)',
            'Read from TAO, validate against PostgreSQL',
            'TAO benefits: Performance (99%+ cache hits), scalability, unified platform',
            'TAO provides graph-oriented API over MySQL + Memcached',
        ],
    },
    {
        id: 'q3',
        question: 'Describe Instagram\'s approach to handling millions of photo uploads per day, including storage, processing, and delivery.',
        sampleAnswer: 'Instagram\'s photo pipeline: (1) Upload: Client uploads to Instagram server, multiple sizes generated (thumbnail 236x, medium 564x, full size). (2) Processing: Resize, compress (JPEG optimization reduces 40-60%), extract metadata. (3) Storage: S3 for master and processed versions, organized by photo_id. (4) Metadata: Store in database (photo_id, user_id, timestamp, captions). (5) Delivery: CloudFront CDN with 95%+ cache hit rate, lazy loading as user scrolls. (6) Optimization: Progressive JPEG (loads blurry→clear), dominant color extraction, blurhash placeholders. Scale: 100M+ photos daily. Challenges: Storage costs (petabytes), processing latency, global delivery. Solutions: Aggressive compression, CDN, async processing queue.',
        keyPoints: [
            'Multiple sizes generated: thumbnail, medium, full',
            'S3 storage with CloudFront CDN (95%+ hit rate)',
            'Progressive JPEG and lazy loading for performance',
            'Async processing queue for resizing/compression',
        ],
    },
];

