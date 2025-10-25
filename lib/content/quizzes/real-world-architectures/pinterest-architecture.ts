/**
 * Quiz questions for Pinterest Architecture section
 */

export const pinterestarchitectureQuiz = [
  {
    id: 'q1',
    question:
      "Explain Pinterest\'s sharding strategy for its graph data (pins, boards, users). Why did they choose object-based sharding over hash-based sharding?",
    sampleAnswer:
      'Pinterest uses object-based sharding (not user-based). Shard by pin_id, board_id, user_id separately. Example: All pins sharded by pin_id → 4096 shards. All boards sharded by board_id → 4096 shards. Benefits: (1) Query "get pin by pin_id" hits one shard (O(1) lookup). (2) Adding shards easier - rehash only affected objects. (3) Hot users don\'t overload single shard. Hash-based alternative: Shard by user_id → all user\'s data on one shard. Problem: Hot users (celebrities with 10M pins) overload shard, can\'t distribute load. Object-based solves this: Celebrity\'s pins distributed across many shards. Trade-off: Queries like "get all pins for user" require scatter-gather (query multiple shards), but Pinterest uses caching + denormalization to optimize. Implementation: Shard ID = (object_id mod 4096). Use consistent hashing for adding/removing shards.',
    keyPoints: [
      'Object-based sharding: Shard by pin_id, board_id, user_id separately',
      'Benefits: O(1) lookups, no hot shard problem, easier resharding',
      'Trade-off: Scatter-gather for some queries (mitigated by caching)',
      'Shard ID = (object_id mod 4096), consistent hashing for flexibility',
    ],
  },
  {
    id: 'q2',
    question:
      "How does Pinterest\'s visual search work? Describe the ML pipeline from image upload to returning similar pins.",
    sampleAnswer:
      'Pinterest visual search (Pinterest Lens): (1) User uploads image or clicks pin. (2) Image sent to visual search service. (3) Object detection - identify objects in image (ML model detects dress, shoes, furniture). (4) Feature extraction - for each object, extract feature vector (embeddings from CNN trained on millions of pins). Vector = 128-512 dimensions capturing visual features. (5) Similarity search - query embedding database for nearest neighbors (cosine similarity). Use approximate nearest neighbor (FAISS/Annoy) for speed. (6) Retrieve top 100 similar pins. (7) Ranking - ML model reranks considering: visual similarity, engagement (clicks, saves), freshness, personalization (user preferences). (8) Return top 20 pins. Infrastructure: TensorFlow for CNN models, FAISS for ANN search, Redis for caching popular searches. Scale: Billions of pins indexed, millions of searches daily. Result: 600M visual searches per month.',
    keyPoints: [
      'Pipeline: Object detection → Feature extraction (CNN embeddings) → ANN search → Ranking',
      'FAISS/Annoy for approximate nearest neighbor search (sub-100ms)',
      'Ranking: Visual similarity + engagement + personalization',
      'Infrastructure: TensorFlow (models), FAISS (search), Redis (cache)',
    ],
  },
  {
    id: 'q3',
    question:
      "Describe Pinterest\'s home feed ranking system. How does it personalize the feed for 450M users with diverse interests?",
    sampleAnswer:
      'Pinterest home feed is personalized board. Candidate generation: (1) Followed boards - pins from boards user follows. (2) Followed topics - pins tagged with topics user follows. (3) Related pins - similar to pins user recently saved. (4) Trending pins - viral content. (5) Collaborative filtering - pins users with similar taste enjoyed. Result: ~1000 candidates. Ranking: Two-stage. (1) Lightweight scoring - filter to top 500 using simple features (recency, board popularity). (2) Deep ranking - ML model scores 500 candidates. Features: Pin features (image quality, text, engagement), user features (interests, demographics, past saves), user-pin features (topic match, similar pins saved before). Model: Deep neural network with embeddings (user embedding, pin embedding, learn similarity). Training: Positive labels = saves/clicks, negative labels = impressions without engagement. Infrastructure: Offline training on Spark, online serving via low-latency service (TensorFlow Serving). Real-time updates: User saves pin → update user embedding in real-time via Flink streaming. Result: 70% of engagement from personalized home feed.',
    keyPoints: [
      'Candidate generation: Follows + topics + related + trending + collaborative filtering',
      'Two-stage ranking: Lightweight (1000→500) + Deep ML model (500→ordered)',
      'Deep model: User/pin embeddings, predict save/click probability',
      'Real-time: Flink updates user embeddings when user saves pins',
    ],
  },
];
