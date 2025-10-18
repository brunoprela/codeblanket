/**
 * Quiz questions for System Design Basics section
 */

export const systemdesignbasicsQuiz = [
  {
    id: 'q1',
    question:
      'In Parking Lot design, why do we use a min heap for available spots instead of a simple list?',
    sampleAnswer:
      'Min heap gives us O(log N) access to the nearest spot (by level, then row). With a list, we\'d need O(N) to find the nearest available spot each time. As parking lot scales (1000+ spots), this matters significantly. Heap maintains spots sorted by (level, row), so heap[0] is always the closest available spot. When we park, we pop from heap (O(log N)). When a spot becomes available, we push to heap (O(log N)). Alternative: sorted list with binary search would be O(N) for insertion/deletion. Heap is the right tool for "repeatedly get min from dynamic set" pattern. This improves user experience - customers get nearest parking.',
    keyPoints: [
      'Min heap: O(log N) get nearest spot',
      'List would be O(N) linear search',
      'Heap maintains sorted order dynamically',
      'Pop gives closest spot instantly',
      'Scales well (1000+ spots)',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is Base62 encoding better than using random strings for URL shortener?',
    sampleAnswer:
      'Base62 with counter is deterministic, collision-free, and shorter. With counter: URL 1 -> "1", URL 1000 -> "g8", URL 1M -> "4c92". No collisions ever because counter is unique. Random strings need collision checking: generate random, check if exists, retry if collision - could take multiple attempts. Base62 uses 62 chars (0-9,a-z,A-Z) vs UUID\'s 36 chars (0-9,a-f), so shorter codes for same space: 62^7 = 3.5T URLs vs 36^7 = 78B. Random also needs cryptographically secure RNG (slower). Base62 is: guaranteed unique, fast (just math), shortest possible, predictable length. Only downside: sequential codes are guessable, but usually not a security concern for public URLs.',
    keyPoints: [
      'Deterministic: no collision checking',
      'Counter guarantees uniqueness',
      'Shorter: 62 chars vs 36 (UUID)',
      'Fast: just math, no random generation',
      'Predictable: know how many URLs from code length',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you scale URL shortener to handle 10,000 requests per second?',
    sampleAnswer:
      'For 10K req/s: (1) Multiple application servers behind load balancer for horizontal scaling. (2) Redis cache for hot URLs (top 10% get 90% of traffic) - check cache before DB, O(1) lookups. (3) Database: Use master-replica setup - master for writes (shorten), replicas for reads (expand). Reads are 90% of traffic so this helps. (4) Generate short codes in advance and store in queue - servers pop from queue when needed, separate service refills queue. This avoids counter bottleneck. (5) Database sharding by hash of short code (e.g., short_code[0] determines shard). (6) CDN for static content and caching. (7) Rate limiting per IP (Token Bucket). (8) Monitoring and auto-scaling. Key: cache everything, scale horizontally, separate reads from writes.',
    keyPoints: [
      'Load balancer + multiple app servers',
      'Redis cache for hot URLs',
      'Master-replica DB (reads vs writes)',
      'Pre-generate codes (avoid counter bottleneck)',
      'Database sharding by short code',
    ],
  },
];
