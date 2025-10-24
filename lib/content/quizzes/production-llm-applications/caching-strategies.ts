export const cachingStrategiesQuiz = [
  {
    id: 'pllm-q-5-1',
    question:
      'Design a multi-tier caching strategy for an LLM application that achieves 80%+ cache hit rate while handling 100K requests/day. Include exact match caching, semantic caching, and cost analysis showing ROI of implementation.',
    sampleAnswer:
      'Three-tier architecture: L1 in-memory LRU cache (1000 items, <1ms), L2 Redis (1M items, <10ms), L3 PostgreSQL (unlimited, <100ms). Exact match (L1+L2): Hash prompt+model+params, 40% hit rate, saves $0.002/request. Semantic caching (L2+L3): Embed prompts (384-dim), FAISS similarity search, cosine_similarity >0.95, 40% additional hits. Implementation: 1) Check L1 exact, 2) Check L2 exact, 3) Query L2 semantic (embed + search <10 similar), 4) If no hit, call LLM, 5) Store in all tiers. Cost analysis: 100K requests/day, $0.002/request = $200/day without cache. With 80% hit rate: 20K API calls = $40/day ($160 saved). Cache infrastructure: Redis $50/month, embeddings $10/month = $60/month vs $4800/month saved = 80x ROI. Implementation costs: 40 hours dev time = $6K, pays for itself in 1.5 months. Invalidation: TTL 24hrs for trending topics, 7 days for evergreen, manual invalidation on model updates. Monitor: cache hit rate per tier, avg latency by tier, cost savings, memory usage.',
    keyPoints: [
      'Multi-tier caching with exact match and semantic similarity',
      '80% cache hit rate with 80x ROI in cost savings',
      'Proper TTL strategy and comprehensive monitoring',
    ],
  },
  {
    id: 'pllm-q-5-2',
    question:
      'Explain semantic caching for LLM applications in detail. How do you determine similarity thresholds, handle edge cases, and prevent serving incorrect cached responses? Provide implementation specifics.',
    sampleAnswer:
      'Semantic caching matches prompts by meaning, not exact text. Implementation: 1) Generate embedding (384-dim from sentence-transformers), 2) Store in vector DB with metadata, 3) Query similar vectors (FAISS/Pinecone), 4) Calculate cosine similarity, 5) If similarity >threshold, return cached response. Threshold selection: Test 0.90-0.99 range, measure false positives (wrong response) vs cache hit rate, optimal usually 0.95-0.97. Higher threshold = fewer hits but safer, lower = more hits but riskier. Validation: A/B test cached vs fresh responses, measure user satisfaction, review mismatches manually. Edge cases: 1) Negation - "tell me about X" vs "don\'t tell me about X" can be similar embeddings, solution: boost negation words in embedding, 2) Numbers - "calculate 5+5" vs "calculate 5+6" very similar, solution: extract numbers as metadata filter, 3) Time-sensitive - "current president" requires date check, solution: tag with expiry, 4) Context-dependent - same question, different conversation history, solution: include conversation hash. Prevention: Store prompt + response + timestamp + model, require model match, add cache metadata (quality score, usage count), allow user to request fresh generation, log cache mismatches for review. Monitor: similarity distribution of hits, user feedback on cached responses, manual review samples.',
    keyPoints: [
      'Embedding-based similarity with carefully tuned thresholds (0.95-0.97)',
      'Handle edge cases like negation, numbers, and time-sensitivity',
      'Validation through A/B testing and user feedback',
    ],
  },
  {
    id: 'pllm-q-5-3',
    question:
      'Compare Redis, in-memory (LRU cache), and database caching for LLM responses. When would you use each tier, and how do you implement cache warming and invalidation strategies?',
    sampleAnswer:
      'In-memory LRU (cachetools): Fastest (<1ms), limited size (1K-10K items), per-instance (not shared), best for frequently accessed items, lost on restart. Use for: top 1000 hot prompts accessed multiple times per second. Redis: Fast (<10ms), large capacity (GBs), shared across instances, persistent, supports TTL. Use for: all responses, semantic cache, session data, distributed locking. Database (PostgreSQL): Slowest (<100ms), unlimited storage, permanent record, queryable, enables analytics. Use for: long-term storage, audit trail, analytics, rarely accessed historical data. Cache warming: 1) Identify top 100 common prompts from analytics, 2) Pre-generate responses during deploy, 3) Load into all cache tiers, 4) Schedule periodic refresh (daily/weekly), 5) Monitor hit rates to adjust warm set. Invalidation strategies: 1) TTL-based: 24hrs for normal, 1hr for trending topics, instant for breaking news, 2) Manual: trigger on model update or prompt changes, 3) LRU: automatic eviction of least recently used, 4) Tag-based: tag caches with version, invalidate all with old version, 5) Semantic: if new response significantly different (embeddings distance >0.3), update cache. Implement with cache.setex(key, ttl, value) in Redis, monitor stale hits.',
    keyPoints: [
      'Three-tier strategy: LRU for hot items, Redis for shared cache, DB for permanent storage',
      'Cache warming with common prompts during deployment',
      'Multiple invalidation strategies: TTL, manual, LRU, and semantic diff',
    ],
  },
];
