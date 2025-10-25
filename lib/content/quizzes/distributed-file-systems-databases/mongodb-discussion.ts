/**
 * Quiz questions for MongoDB section
 */

export const mongodbQuiz = [
  {
    id: 'q1',
    question:
      'When should you embed documents vs reference documents in MongoDB? Provide examples and discuss trade-offs.',
    sampleAnswer:
      "Embedding: Store related data within document. Example: User with addresses - embed addresses array in user document. Benefits: (1) Single read operation (no joins). (2) Atomic updates (entire document). (3) Better performance (no multiple queries). (4) Data accessed together stored together. Use when: One-to-few relationships, data accessed together, data doesn't change independently. Referencing: Store related data in separate collections with IDs. Example: User and Orders - separate collections, orders store user_id. Benefits: (1) Avoid duplication. (2) Smaller documents. (3) Independent updates. (4) Many-to-many relationships. Use when: One-to-many or many-to-many, data accessed independently, document size would exceed 16 MB. Example: Blog posts with comments. Embed if: Few comments per post, comments always shown with post. Reference if: Thousands of comments, comments queried independently. Trade-offs: Embedding = performance + simplicity vs duplication + doc size. Referencing = normalized + flexible vs multiple queries + no joins. MongoDB shines with embedding (leverage document model), but don't abuse it!",
    keyPoints: [
      'Embedding: Store related data within document (one-to-few)',
      'Referencing: Separate collections with IDs (one-to-many)',
      'Embedding: Single read, atomic, better performance',
      'Referencing: No duplication, independent updates',
      'Choose based on: access patterns, relationship cardinality, doc size',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain MongoDB sharding. What is a shard key and why is it critical?',
    sampleAnswer:
      "MongoDB sharding distributes data across shards (replica sets). Shard key determines distribution. mongos (router) routes queries to appropriate shards. Config servers store metadata. Shard key: Field (s) that determine which shard stores document. hash (shard_key) → chunk → shard. Example: user_id as shard key - documents with same user_id on same shard. Critical because: (1) Determines data distribution (balance). (2) Query routing (targeted vs scatter-gather). (3) Performance (hot vs cold shards). Bad shard keys: (1) Sequential (created_at) - all writes to last shard. (2) Low cardinality (status) - poor distribution. (3) Monotonic (_id) - hot shard. Good shard keys: (1) High cardinality (user_id UUID). (2) Even distribution. (3) Frequently queried (targeted queries). (4) Compound key for bucketing. Hashed shard key: {_id: 'hashed'} - even distribution, but no range queries. Choosing wrong shard key = poor performance, can't change later! Must get it right initially.",
    keyPoints: [
      'Shard key determines data distribution across shards',
      'Bad: Sequential, low cardinality, monotonic',
      'Good: High cardinality, even distribution, query-aligned',
      'Hashed shard key: Even distribution, no range queries',
      'Cannot change shard key after creation!',
    ],
  },
  {
    id: 'q3',
    question:
      'How does MongoDB replica set failover work? What happens during primary failure?',
    sampleAnswer:
      'MongoDB replica set: 1 primary (read/write) + N secondaries (read-only, replicate from primary). Automatic failover on primary failure. Process: (1) Secondaries detect primary failure (heartbeat timeout, typically 10 seconds). (2) Election initiated - secondaries vote. (3) Majority votes required (prevents split-brain). (4) Secondary with highest priority and most recent oplog becomes new primary. (5) New primary starts accepting writes. (6) Clients automatically reconnect to new primary. (7) Old primary rejoins as secondary when recovered. Failover time: 10-30 seconds typically. Requirements: (1) Odd number of nodes (3, 5, 7) for majority. (2) Or use arbiter (voting member, no data). (3) Network connectivity between nodes. Applications must handle: (1) Brief write unavailability (10-30 sec). (2) Retry logic for failed operations. (3) Read concern for consistency. Best practices: (1) Use 3+ nodes (or 2 + arbiter). (2) Deploy across availability zones. (3) Configure appropriate timeouts. (4) Test failover regularly.',
    keyPoints: [
      'Primary failure → secondaries elect new primary (majority vote)',
      'Failover time: 10-30 seconds',
      'Need odd number of nodes for majority',
      'Applications need retry logic',
      'Deploy across AZs for fault tolerance',
    ],
  },
];
