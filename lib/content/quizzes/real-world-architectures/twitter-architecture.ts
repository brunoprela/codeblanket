/**
 * Quiz questions for Twitter Architecture section
 */

export const twitterarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Twitter\'s Snowflake ID generation system. Why is it better than auto-increment IDs for distributed systems?",
    sampleAnswer:
      "Snowflake generates unique 64-bit IDs without coordination. Structure: 1 bit unused, 41 bits timestamp (milliseconds), 10 bits machine ID, 12 bits sequence (counter per machine per ms). Benefits: (1) Time-ordered - higher ID = newer tweet, enables time-based sorting without timestamp column. (2) Unique globally - machine ID ensures no collisions across datacenters. (3) No coordination - each machine generates independently, high throughput (4096 IDs/machine/ms). (4) Decentralized - no single point of failure. (5) Decodes to timestamp - useful for analytics. Auto-increment problems: Requires central coordinator (bottleneck), single point of failure, doesn't scale across datacenters. Snowflake trades: Uses more bits than auto-increment (64 vs 32), but worth it for distributed systems.",
    keyPoints: [
      'Structure: timestamp (41 bits) + machine ID (10 bits) + sequence (12 bits)',
      'Time-ordered, globally unique, no coordination needed',
      'High throughput: 4096 IDs per machine per millisecond',
      'Better than auto-increment for distributed systems (no bottleneck)',
    ],
  },
  {
    id: 'q2',
    question:
      "How does Twitter\'s hybrid timeline fanout work, and why is it necessary for handling celebrity accounts?",
    sampleAnswer:
      "Twitter uses hybrid fanout. Regular users (<1M followers): Fanout-on-write. When user tweets, write to followers' timelines in Redis immediately. Fast reads (timeline pre-computed). Celebrities (>1M followers): Fanout-on-read. Don't write to followers' timelines. When follower requests timeline, fetch celebrity tweets on-demand from celebrity's outbox, merge with pre-computed timeline. Why necessary: Lady Gaga tweets → 100M followers → 100M writes is infeasible (overloads system, takes minutes). Fanout-on-read avoids this. Trade-off: Celebrity followers' timelines slightly slower (fetch on-demand adds 20-50ms) but acceptable. Threshold: ~1M followers. Result: Scales to 330M users while maintaining <500ms timeline generation.",
    keyPoints: [
      'Regular users: Fanout-on-write (pre-compute timelines)',
      'Celebrities: Fanout-on-read (fetch on-demand, merge)',
      'Avoids 100M+ writes per celebrity tweet',
      'Trade-off: Slight latency increase for celebrity followers (acceptable)',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Twitter\'s real-time search system (Earlybird). How does it index tweets within seconds for billions of searches?",
    sampleAnswer:
      'Earlybird is Twitter\'s real-time search engine built on Lucene. Indexing: (1) User tweets. (2) Tweet stored in Manhattan (database). (3) Published to Kafka. (4) Earlybird consumes Kafka, indexes tweet. (5) Tweet searchable within 5 seconds. Index structure: Inverted index (keyword → tweet IDs), sharded by time (recent tweets = hot shard). Search query: User searches "machine learning" → Query all shards, retrieve matching tweet IDs, rank by relevance + recency + engagement. Ranking: ML model considers keyword relevance (BM25), recency (newer boosted), engagement (likes, retweets), social graph (followed accounts boosted). Challenges: 500M tweets/day to index, billions of searches, trending topics spike traffic. Solutions: Horizontal sharding (100s of shards), time-based sharding (old tweets archived), caching popular queries.',
    keyPoints: [
      'Built on Lucene, sharded by time (recent tweets prioritized)',
      'Kafka for real-time indexing pipeline (tweets searchable in 5 seconds)',
      'Ranking: relevance + recency + engagement + social graph',
      'Horizontal scaling: 100s of shards, cache popular queries',
    ],
  },
];
