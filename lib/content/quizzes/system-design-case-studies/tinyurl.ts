/**
 * Quiz questions for Design TinyURL section
 */

export const tinyurlQuiz = [
  {
    id: 'q1',
    question:
      'Compare the hash-based approach vs base62 encoding for generating short URLs. What are the key trade-offs, and which would you choose for a production system?',
    sampleAnswer:
      'Hash-based approach (MD5/SHA256): Takes long URL, generates hash, uses first 6-8 characters. PROS: (1) Idempotent - same URL always generates same short URL (prevents duplicates). (2) No database dependency to generate key. (3) Fast computation. CONS: (1) Collisions are inevitable and require handling (check DB, append counter, retry). (2) Extra database queries on collision add latency. (3) Collision rate increases as table fills. Base62 encoding: Uses auto-increment database ID encoded to base62 (0-9, a-z, A-Z). PROS: (1) Guaranteed unique - no collision handling needed. (2) Simple implementation. (3) Predictable key length (6 chars = 56B URLs, 7 chars = 3.5T URLs). CONS: (1) Sequential IDs are predictable (security concern). (2) Requires database write to get ID. (3) Cannot generate key before DB insert. PRODUCTION CHOICE: Base62 with random offset (e.g., start IDs at 1 billion + random). This gives uniqueness guarantee of base62 while breaking predictable pattern. For ultra-high scale (trillions of URLs), consider hash-based with robust collision handling. The key insight: collision-free guarantee is worth the database dependency for most use cases.',
    keyPoints: [
      'Hash-based: Idempotent but requires collision handling',
      'Base62: Guaranteed unique but sequential/predictable',
      'Production recommendation: Base62 + random offset',
      'Collisions increase over time in hash-based approach',
      'Base62: 6 chars = 56B URLs, 7 chars = 3.5T URLs',
    ],
  },
  {
    id: 'q2',
    question:
      'Your URL shortener is serving 20,000 redirects/second. Walk through your caching strategy: what to cache, cache size, eviction policy, and how to handle cache failures.',
    sampleAnswer:
      'CACHING STRATEGY: (1) WHAT TO CACHE: Cache short_url → long_url mappings (the hot 20% serves 80% of traffic per Zipf distribution). Also cache 404 responses to prevent repeated DB queries for non-existent URLs. (2) CACHE SIZE: 20K requests/sec × 86,400 sec/day = 1.7B requests/day. Assuming 20% are unique URLs = 340M unique URLs cached. 340M × 500 bytes = 170 GB Redis cache. (3) CACHE KEY: short_url (e.g., "abc123") → VALUE: {long_url, created_at, expiry_date}. (4) EVICTION POLICY: LRU (Least Recently Used) since URL popularity follows power law - old unpopular URLs should be evicted first. (5) TTL: 24-48 hours to allow cache to refresh and handle deleted/expired URLs. (6) CACHE PATTERN: Cache-Aside (lazy loading) - check cache first, if miss, query DB and populate cache. (7) CACHE FAILURE: If Redis goes down, all traffic hits database. MITIGATION: (a) Use Redis Cluster with replication (3 replicas). (b) Circuit breaker pattern - if Redis is down, serve from DB with degraded performance. (c) Application-level cache (in-memory LRU cache per app server) as fallback. (8) CACHE WARMING: Pre-populate cache with top 1M URLs during deployment. KEY INSIGHT: Caching is critical because 20K QPS directly to database would overload it, but with 80% cache hit rate, only 4K QPS hits DB (manageable with read replicas).',
    keyPoints: [
      'Cache 20% of hot URLs = 170 GB for 340M URLs',
      'LRU eviction policy matches power-law access pattern',
      'Cache-aside pattern with 24-48 hour TTL',
      'Redis replication + circuit breaker for failure handling',
      ' 80% cache hit rate reduces DB load from 20K to 4K QPS',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the difference between 301 and 302 redirects in the context of URL shorteners. Which should you use and why?',
    sampleAnswer:
      '301 (Permanent Redirect): Tells browser "this URL has permanently moved". Browser CACHES the redirect and subsequent visits go directly to long URL without hitting our servers. 302 (Temporary Redirect): Tells browser "this URL is temporarily at this location". Browser DOES NOT cache, so every visit hits our servers. TRADE-OFFS: 301 PROS: (1) Reduces server load dramatically (only first visit hits server). (2) Faster for users (no extra hop after first visit). 301 CONS: (1) Cannot track analytics after first visit per user. (2) Cannot change destination URL (cached in browser). (3) Cannot implement A/B testing or dynamic routing. 302 PROS: (1) Full analytics - track every click, timestamp, referrer, location. (2) Can change destination URL anytime. (3) Can implement features like geo-routing, A/B testing, time-based routing. 302 CONS: (1) Higher server load (every click hits servers). (2) Slightly slower for users (extra hop every time). PRODUCTION CHOICE: Use 302 by default because analytics are critical for URL shorteners (how many clicks? when? from where?). Users expect click tracking. Only use 301 if explicitly optimizing for scale and analytics are not needed. HYBRID: Some services (Bitly) use 302 initially, then switch to 301 after URL "stabilizes" (no changes expected). KEY INSIGHT: This is a classic availability vs observability trade-off - 302 costs more but provides essential product functionality.',
    keyPoints: [
      '301: Browser caches, reduces load but kills analytics',
      '302: No caching, enables full analytics and flexibility',
      'Production default: 302 for analytics (tracking is core feature)',
      'URL shorteners are analytics products, not just redirectors',
      'Hybrid approach: 302 → 301 after URL stabilizes',
    ],
  },
];
