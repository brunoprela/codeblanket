/**
 * Multiple choice questions for Twitter Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const twitterarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the structure of a Twitter Snowflake ID?',
    options: [
      '32-bit timestamp + 16-bit machine ID + 16-bit sequence',
      '1-bit unused + 41-bit timestamp + 10-bit machine ID + 12-bit sequence',
      '64-bit random UUID with timestamp embedded',
      '48-bit timestamp + 16-bit datacenter ID',
    ],
    correctAnswer: 1,
    explanation:
      'Snowflake IDs are 64-bit: 1 bit unused (always 0), 41 bits timestamp (milliseconds since epoch, good until 2079), 10 bits machine ID (1024 machines), 12 bits sequence (4096 IDs per machine per millisecond). This structure ensures time-ordered IDs, global uniqueness without coordination, and high throughput. Each machine generates IDs independently without bottlenecks.',
  },
  {
    id: 'mc2',
    question:
      'Why does Twitter use fanout-on-read for celebrity accounts in timeline generation?',
    options: [
      'Celebrity tweets require special compliance checks before fanout',
      'To reduce storage costs for celebrity follower lists',
      'To avoid writing to 100M+ follower timelines per tweet',
      'Celebrities post less frequently so caching is ineffective',
    ],
    correctAnswer: 2,
    explanation:
      'Twitter uses fanout-on-read for celebrities (>1M followers) because writing a tweet to 100M+ follower timelines would overwhelm the system. Instead, celebrity tweets are fetched on-demand when followers request timelines and merged with their pre-computed feed. This adds 20-50ms latency but avoids massive write amplification. The threshold is typically around 1M followers.',
  },
  {
    id: 'mc3',
    question: "What is Twitter's Earlybird and what does it do?",
    options: [
      'A caching layer for frequently accessed tweets',
      'A real-time search engine built on Lucene for indexing tweets',
      'A message queue for tweet delivery',
      'A recommendation system for trending topics',
    ],
    correctAnswer: 1,
    explanation:
      "Earlybird is Twitter's real-time search engine built on Lucene. It indexes tweets within 5 seconds of posting, enabling immediate searchability for billions of searches. Earlybird uses time-based sharding (recent tweets in hot shards) and inverted indexes. Search results are ranked by keyword relevance (BM25), recency, engagement (likes, retweets), and social graph signals. The system handles 500M+ tweets per day and billions of searches.",
  },
  {
    id: 'mc4',
    question:
      'Which database did Twitter use before Manhattan, and why did they migrate?',
    options: [
      'PostgreSQL; needed better horizontal scalability',
      'MongoDB; needed stronger consistency guarantees',
      'MySQL; needed better write scalability and multi-datacenter replication',
      'Cassandra; needed better read performance',
    ],
    correctAnswer: 2,
    explanation:
      "Twitter originally used MySQL but migrated to Manhattan (their custom distributed database). MySQL struggled with write scalability and multi-datacenter replication at Twitter's scale. Manhattan provides horizontal scalability, multi-datacenter replication, tunable consistency, and is optimized for Twitter's access patterns (timeline storage, user graphs, tweet metadata). It handles millions of operations per second across multiple datacenters.",
  },
  {
    id: 'mc5',
    question:
      'How does Twitter ensure tweet delivery consistency in its distributed architecture?',
    options: [
      'Strong consistency with synchronous replication to all datacenters',
      'Eventual consistency with conflict resolution after 30 seconds',
      'Timeline consistency - reads reflect recent writes in same datacenter, async cross-DC replication',
      'Last-write-wins with version vectors',
    ],
    correctAnswer: 2,
    explanation:
      'Twitter uses timeline consistency for tweet delivery. Reads reflect recent writes within the same datacenter (low latency <10ms), while cross-datacenter replication happens asynchronously (typically consistent within seconds). This balances consistency with availability and latency. Critical operations like tweet creation use quorum writes for durability, but timeline reads are served from the local datacenter for performance.',
  },
];
