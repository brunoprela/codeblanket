/**
 * Multiple choice questions for Design Twitter section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const twitterMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Twitter has 300M users, average 200 follows per user. The follows table stores (follower_id, followee_id) pairs. How much storage is required?',
    options: [
      '300M × 200 × 8 bytes = 480 GB',
      '300M × 200 × 16 bytes = 960 GB ≈ 1 TB',
      '300M × 200 × 100 bytes = 6 TB',
      '300M × 8 bytes = 2.4 GB',
    ],
    correctAnswer: 1,
    explanation:
      '300M users × 200 follows = 60 billion follow relationships. Each row stores two BIGINTs (8 bytes each): follower_id (8 bytes) + followee_id (8 bytes) = 16 bytes per relationship. 60B × 16 bytes = 960 GB ≈ 1 TB. This demonstrates that social graphs are storage-heavy at scale. Adding indexes (follower_id, followee_id) would double storage to ~2 TB. This is why Twitter shards the follows table.',
  },
  {
    id: 'mc2',
    question:
      "Using fanout-on-write, Elon Musk (100M followers) posts a tweet. How many writes are required to update all followers' timelines?",
    options: [
      '1 write (tweet insert)',
      '100 writes (batched)',
      '100 million writes (one per follower)',
      '100,000 writes (sampled subset)',
    ],
    correctAnswer: 2,
    explanation:
      "Fanout-on-write means inserting tweet_id into EACH follower's timeline. 100M followers = 100M writes. Even with batching (10K inserts per batch), that's 10,000 batches. At 100ms per batch, that's 1,000 seconds (16 minutes) - completely infeasible. This is the \"celebrity problem\" that breaks fanout-on-write. This is why hybrid approach uses fanout-on-read for celebrities (>10K followers). For celebrities, skip fanout - let timeline generation query their tweets on-demand.",
  },
  {
    id: 'mc3',
    question:
      'Twitter generates timelines using a hybrid approach: fanout-on-write for normal users, fanout-on-read for celebrities. Why is 10K followers a good threshold?',
    options: [
      '10K is arbitrary, any number works',
      '10K writes is fast enough (~1 second) while covering 99% of users',
      '10K is the database limit per transaction',
      '10K is the Redis write limit',
    ],
    correctAnswer: 1,
    explanation:
      "10K writes can complete in ~1 second (assuming 100 writes/sec per worker with parallel workers). This keeps tweet posting responsive for 99%+ of users (<10K followers). Higher threshold (100K) would make posts slow for more users. Lower threshold (1K) would increase fanout-on-read load (more celebrities). 10K is empirically validated sweet spot: fast enough to complete quickly, rare enough to not affect many users. Twitter\'s actual threshold is proprietary but likely 10-50K range. Key is choosing threshold where fanout completes in <2 seconds.",
  },
  {
    id: 'mc4',
    question:
      'Why does Twitter shard the tweets table by user_id instead of tweet_id?',
    options: [
      'user_id is smaller than tweet_id',
      'User timeline queries (common) hit single shard; tweet_id sharding would require scatter-gather',
      'tweet_id sharding would create hotspots',
      'Regulations require keeping user data together',
    ],
    correctAnswer: 1,
    explanation:
      'MOST COMMON QUERY: "Get user X\'s tweets" (user profile page). Query: SELECT * FROM tweets WHERE user_id = X ORDER BY created_at DESC. With user_id sharding: Hits SINGLE shard (fast). With tweet_id sharding: User\'s tweets scattered across ALL shards, requires scatter-gather (slow). Sharding principle: Choose shard key that makes YOUR MOST COMMON query hit single shard. Less common queries (e.g., "get tweet by ID") may require scatter-gather, but that\'s acceptable if rare. User timeline >> individual tweet lookup frequency.',
  },
  {
    id: 'mc5',
    question:
      'Twitter caches timelines in Redis with 5-minute TTL. What happens when cache hit rate drops from 80% to 50%?',
    options: [
      'No impact, database can handle it',
      'Database query load doubles (from 23K to 57K QPS), potentially causing degradation',
      'Application servers run out of memory',
      'Redis automatically increases cache size',
    ],
    correctAnswer: 1,
    explanation:
      'CALCULATION: 115K timeline reads/sec total. 80% cache hit rate → 20% miss → 23K DB queries/sec (manageable). 50% cache hit rate → 50% miss → 57K DB queries/sec (2.5x increase). If DB read replicas sized for 30K QPS, this would overload them → queries slow down → latency spikes → timeouts → user-facing errors. CAUSES: (1) TTL too short, (2) Many new users (cold cache), (3) Celebrity posts causing cache invalidations, (4) Cache size too small (evictions). FIX: (1) Increase cache size, (2) Increase read replicas, (3) Adjust TTL. KEY INSIGHT: Small changes in cache hit rate have MASSIVE impact on backend load at scale. Monitor cache hit rate closely.',
  },
];
