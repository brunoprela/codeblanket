/**
 * Quiz questions for Strong Consistency vs Eventual Consistency section
 */

export const strongvseventualconsistencyQuiz = [
  {
    id: 'q1',
    question:
      'Explain the "read-your-writes" consistency guarantee and why it matters for user experience. Provide an example where lack of this guarantee causes poor UX.',
    sampleAnswer:
      "Read-Your-Writes Consistency: DEFINITION: After a user writes data, any subsequent reads by that same user will see that write (or a newer write). This guarantees users see their own changes immediately, even if global system uses eventual consistency. WHY IT MATTERS: Without read-your-writes, user makes a change (write), immediately checks (read), and doesn't see their change → appears broken. BAD UX EXAMPLE - Social Media Post: User posts \"I got engaged! 💍\" on Facebook. Write goes to US-East datacenter. User immediately refreshes feed. Load balancer routes read to EU-West datacenter (not yet replicated). User doesn't see their own post → \"App is broken!\" frustration. User reposts → duplicate post. Reality: System is working (eventual consistency), but UX is broken. SOLUTION: Implement read-your-writes consistency. Track where user's write went (store in cookie/session: last_write_datacenter=US-East). Route user's subsequent reads to same datacenter until replication completes. After 2-5 seconds (replication done), resume normal load balancing. Result: User sees own post immediately. Other users see it within seconds (eventual). IMPLEMENTATION APPROACHES: (1) Session Stickiness: Pin user to datacenter for ~5 seconds after write. (2) Version Tracking: Store write version in session, read from replica with >= that version. (3) Write-Through Cache: Cache user's own writes locally, overlay on reads. (4) Redirect Reads: After write, redirect reads to primary (master) for ~5 seconds. TRADE-OFFS: Adds complexity (tracking, routing logic). Slight increase in read latency (routing to specific datacenter). But critical for UX in eventually consistent systems. REAL-WORLD: Facebook, Instagram, Twitter all implement read-your-writes for user's own content.",
    keyPoints: [
      'Read-your-writes: User always sees their own changes immediately',
      "Without it: User doesn't see their post/like → appears broken",
      'Implementation: Route user reads to datacenter where write happened',
      'Applies for ~5 seconds until replication completes',
      'Critical for UX in eventually consistent systems (social media)',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a global shopping cart system for an e-commerce site. Should you use strong or eventual consistency? How would you handle conflicts if two devices add items simultaneously?',
    sampleAnswer:
      "Global Shopping Cart Consistency Design: CONSISTENCY CHOICE: Use EVENTUAL CONSISTENCY with application-level conflict resolution. REASONING: (1) Shopping cart is non-critical (not a payment). User adds items across devices (laptop, mobile, work computer). Strong consistency would require coordinating writes globally (high latency). Cart must work offline (mobile app on subway). User expectation: Items never lost. Slight delay seeing cart sync is acceptable. (2) Availability critical: If cart service down, user can't shop (bad business). ARCHITECTURE: Multi-region active-active deployment. User writes to nearest region (low latency). Async replication between regions. CONFLICT SCENARIO: User on laptop (US-East): Adds Item A at 10:00:00. User on mobile (EU-West): Adds Item B at 10:00:01. Both writes happen concurrently to different datacenters. CONFLICT RESOLUTION STRATEGY: Use SET MERGE (union of all items). Never delete items due to conflict (Amazon\'s approach). Implementation: Cart data structure: { user_id, items: [ { item_id, quantity, added_timestamp, device_id } ] }. When carts sync and conflict detected: Merge operation: Union all items. If same item_id from different devices: Sum quantities (Item X qty 2 from laptop + Item X qty 1 from mobile = 3 total). If same item_id from same device: Last-write-wins by timestamp (duplicate detection). Result after merge: Cart = [Item A from laptop, Item B from mobile]. User sees both items on next cart view. DATABASE CHOICE: DynamoDB (eventual consistency, auto-conflict resolution) or Cassandra (LWW, but application merges). LOCAL STORAGE: Mobile app stores cart locally (offline capability). Syncs when online. If sync fails, cart persists locally (never lost). EDGE CASES: (1) User adds Item A on laptop, removes Item A on mobile simultaneously. Resolution: Removal wins (safer to remove than duplicate). Implement tombstone (deleted_at timestamp). (2) Price change during shopping: Cart stores price at add-time, checkout uses current price (inform user if changed). RESULT: High availability, low latency, works offline, items never lost.",
    keyPoints: [
      'Shopping cart: Eventual consistency (availability & offline support critical)',
      'Conflict resolution: SET MERGE (union all items, never lose items)',
      'Same item from multiple devices: Sum quantities',
      'Offline support: Store cart locally, sync when online',
      "Better to have duplicate items than lose customer's item (Amazon approach)",
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the performance implications of strong vs eventual consistency for a global application. Include latency, availability, and throughput in your analysis.',
    sampleAnswer:
      "Strong vs Eventual Consistency Performance Analysis: LATENCY COMPARISON: STRONG CONSISTENCY - Single Region: Primary database write: 5-10ms (disk write + transaction log). Read: 5-10ms (if read from primary) or 10-20ms (if read from replica with sync replication). Total latency: 10-30ms. Multi-Region: Write must replicate to multiple regions before confirming (2PC or consensus). Latency = max (latency to all regions) = 100-500ms (cross-continental). Read: Must read from primary or wait for sync replication = 100-500ms. Total latency: 200-1000ms (poor UX for global users). EVENTUAL CONSISTENCY - Single Region: Write: 5-10ms (write to local primary, confirm immediately). Read: 5-10ms (read from local replica, might be stale). Total latency: 10-20ms. Multi-Region: Write: 5-10ms (write to nearest datacenter, confirm immediately, async replication). Read: 5-10ms (read from nearest datacenter, might be stale). Convergence: 100ms - 5 seconds (async replication in background). Total latency: 10-20ms (50x faster than strong!). AVAILABILITY COMPARISON: STRONG CONSISTENCY: If primary datacenter down → System unavailable for writes (no failover until manual). If any replica down during sync replication → Writes fail or latency spikes. Multi-region: If any region unreachable → Writes fail (can't reach quorum). Availability: 99.9% (three 9s, downtime during failures). EVENTUAL CONSISTENCY: If primary datacenter down → Writes route to secondary (auto-failover). If replica down → Other replicas serve reads (multi-master). Multi-region: If region down → Other regions continue operating (partition tolerance). Availability: 99.99%+ (four 9s, remains available during failures). THROUGHPUT COMPARISON: STRONG CONSISTENCY: Writes: Limited by coordination overhead. Single-master: ~1,000-10,000 writes/sec (one datacenter bottleneck). Multi-region: ~100-1,000 writes/sec (consensus overhead). Reads: Can scale with read replicas, but sync replication adds load. Throughput: 10,000 - 100,000 reads/sec. EVENTUAL CONSISTENCY: Writes: Multi-master → writes scale across regions. Each region handles 10,000 writes/sec → 3 regions = 30,000 writes/sec. No coordination → higher throughput. Reads: Fully scalable (read from any replica, eventual is fine). Throughput: 100,000 - 1,000,000+ reads/sec (add more replicas). COST COMPARISON: Strong Consistency: More expensive compute (coordination overhead). More expensive network (sync replication). Lower resource utilization (idle during coordination). Eventual Consistency: Lower compute cost (no coordination). Cheaper async replication. Higher resource utilization (always processing). EXAMPLE METRICS: Global app with users in US, Europe, Asia. Strong Consistency: Latency: P95 = 500ms, P99 = 1000ms. Availability: 99.9%. Throughput: 10,000 writes/sec, 100,000 reads/sec. Eventual Consistency: Latency: P95 = 20ms, P99 = 50ms. Availability: 99.99%. Throughput: 30,000 writes/sec, 500,000 reads/sec. Trade-off: Eventual consistency is 25x lower latency, 3x higher write throughput, but temporary stale reads.",
    keyPoints: [
      'Strong consistency: 100-500ms multi-region (coordination overhead)',
      'Eventual consistency: 10-20ms (no coordination, async replication)',
      'Strong: Lower availability (99.9%), eventual: higher (99.99%)',
      'Strong: Lower throughput (coordination bottleneck)',
      'Eventual: 25x lower latency, 3x higher throughput for global apps',
    ],
  },
];
