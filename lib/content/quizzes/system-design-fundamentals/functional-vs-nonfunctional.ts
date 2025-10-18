/**
 * Quiz questions for Functional vs. Non-functional Requirements section
 */

export const functionalvsnonfunctionalQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental difference between functional and non-functional requirements, and give examples of how each type affects system design decisions.',
    sampleAnswer:
      'FUNCTIONAL REQUIREMENTS define WHAT the system does - the features and capabilities users can see and interact with. Examples: "Users can post tweets," "Users can upload photos," "Users can search." These are testable, specific actions. NON-FUNCTIONAL REQUIREMENTS define HOW WELL the system performs - quality attributes like scalability, performance, reliability, availability. Examples: "Handle 10K requests/second," "99.99% uptime," "Latency <100ms." IMPACT ON DESIGN: Consider Instagram: Functional requirement "users can post photos" doesn\'t tell you much about architecture. But non-functional requirements drive major decisions: (1) If scale is 500M users → Need CDN, distributed storage, sharding. (2) If latency must be <200ms → Need aggressive caching, edge servers. (3) If availability target is 99.9% → Need load balancers, redundancy, failover. (4) If eventual consistency is OK → Can use NoSQL, async replication for better performance. Same functional requirements + different non-functional requirements = completely different architectures. A Twitter clone for 100 users can run on a single server with MySQL. But 300M users requires microservices, Cassandra, Kafka, CDN, etc. This is why clarifying non-functional requirements is CRITICAL in interviews - they determine your entire architecture.',
    keyPoints: [
      'Functional: WHAT system does (features, user-facing actions)',
      'Non-functional: HOW WELL it performs (scale, latency, availability)',
      'Non-functional requirements drive architectural decisions',
      'Same features + different scale = completely different architectures',
      'Must clarify both types early in system design interviews',
    ],
  },
  {
    id: 'q2',
    question:
      "You're designing a payment processing system like Stripe. What are the critical non-functional requirements you must clarify, and how would different answers change your design?",
    sampleAnswer:
      'CRITICAL NON-FUNCTIONAL REQUIREMENTS FOR PAYMENT SYSTEM: (1) CONSISTENCY: Must be STRONG consistency. You cannot have eventual consistency - double charging or lost payments is unacceptable. Design impact: Must use ACID-compliant database (PostgreSQL), synchronous replication, distributed transactions. Cannot use eventually consistent NoSQL. (2) RELIABILITY/DURABILITY: ZERO data loss tolerance. Every transaction must be recorded. Design impact: Write-ahead logs, synchronous replication to multiple regions, transaction logs never deleted, multiple backup systems. (3) SECURITY: Highest level - PCI DSS compliance required. Design impact: Encryption at rest and in transit, tokenization of card data, strict access controls, audit logs, no storing sensitive data. (4) AVAILABILITY: 99.99% minimum (52 min downtime/year). Design impact: Multi-region active-active, automatic failover, redundant everything. (5) LATENCY: <500ms acceptable (users tolerate slight delay for payments). Design impact: Can sacrifice some latency for consistency and reliability - acceptable trade-off. (6) IDEMPOTENCY: Must handle retries - cannot charge twice for same request. Design impact: Idempotency keys, deduplication layer, request tracking. DIFFERENT ANSWERS CHANGE DESIGN: If requirements were social media likes instead: Eventual consistency OK → Could use Cassandra. Some data loss acceptable → Async replication OK. Lower security needs → Simpler architecture. Result: Payment systems prioritize consistency, reliability, security over performance. Social media prioritizes availability, performance over strong consistency.',
    keyPoints: [
      'Payments require strong consistency (no double-charging)',
      'Zero data loss tolerance requires synchronous replication',
      'Security requirements drive encryption, PCI compliance',
      'High availability requires multi-region active-active',
      'Different requirements → completely different architecture choices',
    ],
  },
  {
    id: 'q3',
    question:
      'How do you balance conflicting non-functional requirements like high availability, strong consistency, and low latency in system design?',
    sampleAnswer:
      'Balancing conflicting requirements requires understanding trade-offs and making pragmatic choices based on business needs. KEY CONFLICTS: (1) HIGH AVAILABILITY vs STRONG CONSISTENCY (CAP Theorem): Cannot have both during network partition. Must choose: Option A: Prioritize availability (AP system) - eventual consistency, always accept writes. Use case: Social media feeds, recommendation systems. Option B: Prioritize consistency (CP system) - reject writes during partition, guarantee correctness. Use case: Banking, payment processing. (2) LOW LATENCY vs STRONG CONSISTENCY: Strong consistency requires coordination → higher latency. Trade-off: Strong consistency: Read from master, wait for write replication = higher latency. Eventual consistency: Write returns immediately, read from any replica = lower latency. Solution: Hybrid approach - strong consistency for critical data (bank balance), eventual for non-critical (profile views). (3) HIGH AVAILABILITY vs LOW COST: Redundancy costs money. Must find balance. BALANCING STRATEGY: (1) Prioritize by business impact: "What causes most damage: downtime, incorrect data, or slow response?" Payment system: Correctness > Availability > Latency. Social feed: Availability > Latency > Correctness. (2) Use hybrid approaches: Strong consistency for writes, eventual for reads. Relaxed consistency during peak load. (3) Set realistic SLAs: 99.99% not always necessary. Instagram can tolerate 99.9% (8.7 hours/year downtime). Saves cost of over-engineering. (4) Communicate trade-offs in interview: "I recommend eventual consistency here because availability is more important for a social feed. The business can tolerate a few seconds delay in like counts, but cannot tolerate the entire feed being down."',
    keyPoints: [
      'CAP theorem: cannot have all three (consistency, availability, partition tolerance)',
      'Strong consistency increases latency due to coordination',
      'Business priorities determine which requirements matter most',
      'Hybrid approaches: strong consistency for critical paths, eventual for non-critical',
      "Communicate trade-offs explicitly - show you understand there's no perfect solution",
    ],
  },
];
