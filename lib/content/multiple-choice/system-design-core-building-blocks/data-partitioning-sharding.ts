/**
 * Multiple choice questions for Data Partitioning & Sharding section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const datapartitioningshardingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'You have a users table with 100M rows. You shard by user_id using hash(user_id) % 10. Which query is MOST efficient?',
      options: [
        'SELECT * FROM users WHERE user_id = 12345',
        'SELECT * FROM users WHERE country = "USA"',
        'SELECT * FROM users WHERE created_at > "2023-01-01"',
        'SELECT COUNT(*) FROM users',
      ],
      correctAnswer: 0,
      explanation:
        'Query by user_id is most efficient: hash(12345) % 10 = X determines exact shard, query hits only 1 shard. Other queries: (1) country = "USA": Must query ALL 10 shards (country not in partition key). (2) created_at > "2023-01-01": Must query ALL 10 shards (date not in partition key). (3) COUNT(*): Must query ALL 10 shards and sum results. General rule: Queries filtering by partition key are efficient (single shard). Queries not filtering by partition key require querying all shards (scatter-gather).',
    },
    {
      id: 'mc2',
      question:
        'What is the main advantage of consistent hashing over simple hash (hash % N) for sharding?',
      options: [
        'Consistent hashing provides better data distribution',
        'Consistent hashing is simpler to implement',
        'Adding/removing shards requires minimal data movement',
        'Consistent hashing enables efficient range queries',
      ],
      correctAnswer: 2,
      explanation:
        'Minimal data movement when adding/removing shards is the main advantage. Simple hash (hash % N): Changing N requires rehashing almost all data (~80% moves). Consistent hashing: Only data from neighboring shards moves (~5-10% of total). This is critical for production systems that need to scale without downtime. Option 1: Both provide similar distribution (with virtual nodes). Option 2: Consistent hashing is actually more complex. Option 4: Neither enables efficient range queries (both use hashing).',
    },
    {
      id: 'mc3',
      question:
        "You're sharding a logs table by created_at (timestamp). What problem will you likely encounter?",
      options: [
        'Uneven distribution of data across shards',
        'Recent logs go to one shard (hotspot), old shards are idle',
        'Difficult to query logs by timestamp',
        'Too much data movement when adding shards',
      ],
      correctAnswer: 1,
      explanation:
        "Sharding by timestamp creates hotspot: all recent logs go to one shard (the current time range). Example: 10 shards by date ranges. Today's logs: ALL writes hit Shard 10 (today's shard). Old shards (1-9): Idle (no writes, only rare historical reads). Result: Uneven load, Shard 10 overloaded. Better solution: Shard by hash(log_id) or composite key hash(source_id + timestamp) for uniform distribution. Lesson: Avoid sharding by sequential/time-based keys unless data is truly immutable and access patterns are historical.",
    },
    {
      id: 'mc4',
      question:
        'What is the recommended approach for handling cross-shard joins in a sharded database?',
      options: [
        'Use distributed transactions (2-phase commit)',
        'Query all shards and join results in application',
        'Avoid cross-shard joins by denormalizing or colocating related data',
        'Use a global secondary index spanning all shards',
      ],
      correctAnswer: 2,
      explanation:
        'Avoid cross-shard joins by denormalizing or colocating related data. Example: If Users and Posts often joined, shard both by user_id (colocate). Or: Denormalize user info into Posts table (duplicate data, avoid join). Option 1 (2PC): Slow, complex, blocks on failure. Option 2 (app-level join): Works but slow (query all shards). Option 4 (global index): Possible but adds complexity and latency. Best practice: Design schema to avoid cross-shard operations. Denormalization is often acceptable trade-off in distributed systems.',
    },
    {
      id: 'mc5',
      question:
        'Your sharded database uses range-based sharding by user_id: Shard 1 (1-1M), Shard 2 (1M-2M), Shard 3 (2M-3M). User IDs are sequential (auto-increment). What problem will you encounter?',
      options: [
        'Uneven read distribution across shards',
        'All new users go to the latest shard (hotspot for writes)',
        'Difficult to query users by ID range',
        'Cannot add more shards without rehashing',
      ],
      correctAnswer: 1,
      explanation:
        'Sequential IDs with range sharding creates write hotspot: all new users go to the latest shard. Example: User IDs 1-2M exist. Shard 1: Users 1-1M (idle for writes). Shard 2: Users 1M-2M (idle for writes). Shard 3: Users 2M-3M (ALL writes go here). Result: Shard 3 overloaded, Shards 1-2 underutilized. Solution: Use hash-based sharding (uniform distribution) instead of range-based for sequential IDs. Or: Use UUIDs/random IDs instead of sequential IDs with range sharding.',
    },
  ];
