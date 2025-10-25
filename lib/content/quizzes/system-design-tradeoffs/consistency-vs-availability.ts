/**
 * Quiz questions for Consistency vs Availability section
 */

export const consistencyvsavailabilityQuiz = [
  {
    id: 'q1',
    question:
      'You are designing a ride-sharing app (like Uber). Should you prioritize consistency or availability for: (a) Driver location updates, (b) Ride payment processing, (c) Ride history? Justify each decision.',
    sampleAnswer:
      'Consistency vs Availability for ride-sharing app: (a) DRIVER LOCATION UPDATES - Prioritize AVAILABILITY (AP): Reasoning: Need real-time location updates even if slightly stale. If network partition occurs between driver and server, better to show last known location (2 seconds old) than "location unavailable." Impact: Users see slightly delayed location, which is acceptable for UX. AP system uses eventual consistency—locations sync when partition resolves. Implementation: Cache last-known location on client, continue showing map. (b) RIDE PAYMENT PROCESSING - Prioritize CONSISTENCY (CP): Reasoning: Cannot charge twice or fail to charge. Payment must be atomic and exactly-once. If network partition occurs, better to fail payment and ask user to retry than to have inconsistent payment state. Impact: User might see "Payment failed, please retry" but money is never lost. CP system uses strong consistency—payment transaction must commit fully or rollback. Implementation: Database transactions, idempotency keys, retry logic with same key. (c) RIDE HISTORY - Prioritize AVAILABILITY (AP): Reasoning: Ride history is read-heavy, historical data. Slightly stale history (missing last 5 minutes) is acceptable. If partition occurs, better to show history (missing latest ride) than "service unavailable." Impact: User might briefly not see their just-completed ride, but it appears soon. AP system uses eventual consistency—history syncs across regions. Implementation: Replicated data stores, cache-aside pattern. Summary: Critical transactions (payments) → CP. Real-time data where stale acceptable (location) → AP. Historical data (ride history) → AP.',
    keyPoints: [
      'Driver location: AP (real-time, stale acceptable for UX)',
      'Payment processing: CP (must be exactly-once, no double-charging)',
      'Ride history: AP (read-heavy, eventual consistency fine)',
      'Different data types require different consistency guarantees',
      'Use hybrid approach: CP for money, AP for everything else',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a global e-commerce system (like Amazon) that operates in multiple regions. How would you balance consistency and availability? What data should be strongly consistent vs eventually consistent?',
    sampleAnswer:
      'Global e-commerce consistency design: ARCHITECTURE: Multi-region active-active deployment. User routed to nearest region for low latency. STRONGLY CONSISTENT (CP) - Single source of truth: (1) Inventory count: Prevent overselling. Use distributed lock or consensus. Write to primary region, sync replicas. Accept higher latency (100-200ms). Example: Product has 5 units left → Two users checkout simultaneously → Lock prevents double-allocation. (2) Order state: Order processing, payment status. Use database transactions. Order must be atomically created with payment. (3) Payment processing: Idempotency keys, exactly-once semantics. Better to fail payment than double-charge. (4) Shopping cart checkout: During checkout, lock inventory. EVENTUALLY CONSISTENT (AP) - Optimized for availability: (1) Product catalog: Description, images, price. Replicated globally with CDN. Stale price briefly acceptable (price changes are rare). If price changes, replicate in ~seconds. (2) Product reviews: User-generated content. Reviews from Asia might take seconds to appear in US. Acceptable trade-off for global availability. (3) Shopping cart browsing: Cart stored locally, synced async. If sync fails, cart persists locally. (4) Recommendation engine: Personalized recommendations. Slight staleness acceptable. (5) Wish lists: Non-critical feature. IMPLEMENTATION: Use different databases for different consistency needs. Strong consistency: PostgreSQL (primary region) + sync replication. Eventual consistency: DynamoDB, Cassandra (multi-region replication). RESULT: 99% of requests (browsing) are fast and always available. 1% of requests (checkout) have strong consistency guarantees.',
    keyPoints: [
      'Inventory & payments: CP with distributed locks/consensus',
      'Product catalog & reviews: AP with multi-region replication',
      'Different databases for different consistency needs',
      'Optimize common case (browsing) for availability',
      'Strong consistency only for critical transactions',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how Cassandra achieves tunable consistency and how you would configure it for a social media application with posts, likes, and user profiles.',
    sampleAnswer:
      'Cassandra tunable consistency for social media: CASSANDRA QUORUM MODEL: Replication Factor (RF) = N total replicas per partition. Write Consistency Level (W) = replicas that must acknowledge write. Read Consistency Level (R) = replicas queried for read. STRONG CONSISTENCY: W + R > N. Example: RF=3, W=2, R=2 (quorum) → Strong consistency. CONFIGURATION FOR SOCIAL MEDIA: (1) USER PROFILES (email, password hash, account status): Use QUORUM writes and reads (W=2, R=2 with RF=3). Reasoning: Account data must be accurate. Can\'t have two different passwords.User expectations: Login should use correct credentials.Trade - off: Slightly higher latency(~50ms) acceptable for login. (2) POSTS(user creates post): Use QUORUM write(W = 2) but ONE read(R = 1).Reasoning: Post must be durably written (prevent data loss).But reading from any replica is fine (eventual consistency for visibility).User expectations: "Post published" must be guaranteed.Seeing own post immediately is nice- to - have.Implementation: Write with W = QUORUM → Ensure durability.Read with R = ONE → Fast feed loading. (3) LIKES(user likes post): Use ONE for both writes and reads(W = 1, R = 1).Reasoning: Likes are high- volume, low-value.Exact like count in real - time not critical.Like count 1, 234 vs 1, 237 is acceptable.Trade - off: Very fast writes(~10ms) at cost of brief inconsistency.User expectations: Like registers immediately (client optimistic update).Actual count syncs eventually.RESULT: Critical data (user profiles) → Strong consistency(QUORUM).Important data (posts) → Write durability(QUORUM) + fast reads(ONE).High - volume data (likes) → Fast and available(ONE).This balances consistency, availability, and performance based on data criticality.',
    keyPoints: [
      'Tunable consistency: Configure W and R per request',
      'Strong consistency: W + R > N (use QUORUM)',
      'User profiles: QUORUM reads/writes (accuracy critical)',
      'Posts: QUORUM writes (durability) + ONE reads (speed)',
      'Likes: ONE/ONE (high volume, eventual consistency acceptable)',
    ],
  },
];
