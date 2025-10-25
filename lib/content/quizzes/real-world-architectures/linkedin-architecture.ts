/**
 * Quiz questions for LinkedIn Architecture section
 */

export const linkedinarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain LinkedIn's Espresso database system. Why did LinkedIn build a custom NoSQL database instead of using existing solutions?",
    sampleAnswer:
      "Espresso is LinkedIn's distributed document database, built to replace Oracle. Motivations: (1) Oracle couldn't scale horizontally (vertical scaling limits). (2) Needed multi-datacenter replication with low latency. (3) Existing NoSQL (Cassandra, MongoDB) didn't meet consistency + performance requirements. Espresso features: (1) Document model with secondary indexes (unlike Cassandra). (2) Timeline consistency - reads reflect recent writes. (3) Cross-datacenter replication with configurable consistency (read local, replicate async). (4) MySQL storage engine underneath (leverages MySQL durability, adds distributed layer). (5) Horizontal sharding with consistent hashing. Use case: Member profiles, connections graph, feed data. Scale: 1000+ nodes, 100TB+ data, millions of QPS. Trade-offs: Operational complexity (custom database) vs performance (optimized for LinkedIn workload).",
    keyPoints: [
      'Custom NoSQL built on MySQL storage engine with distributed layer',
      'Document model + secondary indexes + timeline consistency',
      'Cross-datacenter async replication, read-local for low latency',
      'Scale: 1000+ nodes, 100TB+, millions of QPS',
    ],
  },
  {
    id: 'q2',
    question:
      "How does LinkedIn's feed ranking algorithm work? What signals does it use to determine which posts to show members?",
    sampleAnswer:
      'LinkedIn feed is two-stage ranking. Stage 1 - Candidate generation: From 1000s of possible posts (connections, followed pages, viral posts), select ~500 candidates. Sources: (1) Posts from 1st-degree connections. (2) Posts from followed hashtags/pages. (3) Viral posts (many comments/shares). (4) Sponsored content. Stage 2 - Ranking: ML model scores 500 candidates. Features: (1) Post signals - engagement (likes, comments, shares, clicks), recency, author (connection degree, influence). (2) Member signals - industry, seniority, past engagement, interests. (3) Member-post affinity - has member engaged with author before, topic relevance. (4) Dwell time prediction - how long will member spend on this post. Model: Gradient boosted trees → deep neural networks (embeddings for member/post). Training: Logistic regression on engagement labels (click, like, share = positive). Infrastructure: Offline training on Hadoop, online serving via Kafka+Samza for real-time updates. Result: 10% increase in feed engagement after ML ranking deployed.',
    keyPoints: [
      'Two-stage: Candidate generation (1000s→500) + ML ranking',
      'Signals: Post engagement, recency, member-post affinity, dwell time prediction',
      'Model evolution: Gradient boosted trees → deep neural networks',
      'Real-time: Kafka+Samza update member feeds, offline training on Hadoop',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe LinkedIn's approach to data consistency in its distributed architecture. How does it balance consistency, availability, and latency?",
    sampleAnswer:
      'LinkedIn uses differentiated consistency models based on use case. (1) Strong consistency - Critical operations (accept connection request, post job). Use synchronous replication across datacenters, wait for quorum (2/3 DCs). Trade-off: Higher latency (50-100ms cross-DC) for correctness. (2) Timeline consistency - Most reads (view profile, feed). Read from local datacenter (low latency <10ms), async replication. Guarantees: reads reflect recent writes from same DC, eventual consistency across DCs. Typically consistent within seconds. (3) Eventual consistency - Analytics, counters (profile views). High availability, low latency, okay if slightly stale. Implementation: Espresso for timeline consistency, Kafka for event propagation, Brooklin for cross-DC replication. Monitoring: Track replication lag (p99 <5 seconds), detect split-brain. Result: 99.99% availability with acceptable consistency for most use cases.',
    keyPoints: [
      'Differentiated consistency: Strong (critical ops), timeline (most reads), eventual (analytics)',
      'Timeline consistency: Read-local for latency, async replication (seconds lag)',
      'Strong consistency: Synchronous cross-DC replication, quorum writes (higher latency)',
      'Infrastructure: Espresso (storage), Kafka (events), Brooklin (cross-DC replication)',
    ],
  },
];
