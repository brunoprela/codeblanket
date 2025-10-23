/**
 * Quiz questions for Caching & Performance section
 */

export const cachingperformanceQuiz = [
  {
    id: 'q1',
    question:
      'Your application has a 30% cache hit rate which saves $300/month. Management suggests implementing semantic caching to increase the hit rate to 60%. Explain the trade-offs and whether this makes economic sense.',
    sampleAnswer:
      "Semantic caching trade-off analysis: Current state: 30% cache hit rate saves $300/month, meaning 30% of $1000/month total = $300 saved, $700 spent on API calls. Target: 60% hit rate would save $600/month, $400 on API, saving additional $300/month. Sounds great! But consider costs: (1) Embedding costs: Semantic cache requires computing embeddings for every query (to compare similarity). At ~$0.02 per 1M tokens (OpenAI embeddings), if you have 100K queries/month averaging 100 tokens each = 10M tokens = $0.20/month for embeddings. Negligible. (2) Vector database: Storing and querying embeddings requires vector DB (Pinecone, Weaviate). Costs: ~$70-100/month for reasonable scale. (3) Additional latency: Semantic comparison takes 50-100ms per query (embedding + similarity search). For 100K queries, adds ~0.1s average latency. Acceptable for most use cases. (4) False positives: Semantic matching at 0.95 similarity might cache slightly different queries. If 5% of cached responses are not quite right, user satisfaction drops. Need to tune threshold carefully. Economics: Additional monthly cost: $70 (vector DB) + $0.20 (embeddings) = ~$70. Additional savings: $300/month (doubling cache hit rate). Net benefit: $230/month = 330% ROI. Makes economic sense IF: (1) Quality doesn't degrade (tune similarity threshold carefully, A/B test first), (2) Latency increase is acceptable (<100ms is usually fine), (3) Scale justifies complexity (at 10K queries/month, $70 fixed cost might not be worth it). Alternative consideration: Before semantic caching, try: (1) Normalize queries (lowercase, remove punctuation) to increase exact matches, (2) Implement query reformulation to standard form, (3) Add user-level caching (same user asking similar things). These might get you to 45-50% hit rate for free. Decision framework: If current cost >$500/month and hit rate <40%, semantic caching probably worth it. If current cost <$100/month, optimize exact matching first. At $700/month API spend, semantic caching justified - implement with careful quality monitoring.",
    keyPoints: [
      'Calculate ROI including vector database costs',
      'Semantic caching adds latency (50-100ms)',
      'Risk of false positives degrading quality',
      'Worth it at scale (>$500/month), maybe not at small scale',
      'Try optimizing exact matching first',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between client-side caching, application-level caching (Redis), and LLM provider-level caching (Claude prompt caching). When would you use each?',
    sampleAnswer:
      'Three caching layers serve different purposes: Client-side caching (browser/app): Caches responses in user\'s device. Use when: (1) Same user makes identical requests within session (FAQ page, repeated questions), (2) Responses are user-specific and not shareable, (3) Want instant response without network round trip, (4) Privacy sensitive data that should not leave device. Limitations: Only helps individual user, not shared across users, lost when user clears cache or switches devices. Example: Chat app caches user\'s conversation history locally. Application-level caching (Redis/Memcached): Caches responses server-side, shared across all users. Use when: (1) Multiple users ask same questions ("What are your hours?"), (2) Responses are not user-specific, (3) Want to reduce API calls and costs significantly, (4) Can control cache invalidation centrally. Benefits: 30-80% cache hit rate possible, saves costs across all users, reduces latency for all users. Limitations: Redis costs money ($50-200/month), adds complexity, must handle cache invalidation. Example: Support chatbot caches common questions - 1000 users asking "reset password" hits cache 999 times. Provider-level caching (Claude Prompt Caching): LLM provider caches your prompts/context at their end. Use when: (1) Large, repeated context (documentation, code files) sent with many requests, (2) Context is identical across requests but questions change, (3) Using Claude (other providers may add similar features). Benefits: 90% cost reduction on cached portions, no infrastructure needed, automatic by provider. Limitations: Only works with providers supporting it, context must be identical (any change breaks cache), 5-minute TTL on Claude. Example: AI coding assistant sends same 50K token documentation with every query. First query pays full price, subsequent queries within 5 minutes pay 10% for cached portion. Optimal strategy - use all three: (1) Client-side for user-specific, repeated requests (conversation history), (2) Application-level for shared, common requests (FAQs, product info), (3) Provider-level for large context (documentation, knowledge bases). Example: Coding assistant with docs: Provider cache (docs), Application cache (common code questions), Client cache (user\'s chat history). Each layer catches different patterns, compound savings.',
    keyPoints: [
      'Client-side for user-specific repeated requests',
      'Application-level for shared requests across users',
      'Provider-level for large repeated context',
      'Each layer serves different use case',
      'Use multiple layers for maximum benefit',
    ],
  },
  {
    id: 'q3',
    question:
      'Your cache hit rate is only 15% despite implementing caching. Diagnose potential causes and describe how you would investigate and improve this.',
    sampleAnswer:
      'Low hit rate investigation process: Step 1 - Measure current state: (1) Log cache key generation for sample requests, (2) Analyze what makes requests "different" (parameters, user input variations), (3) Check cache key distribution - are keys unique even for similar requests? (4) Measure cache size - is cache getting full and evicting entries? Step 2 - Common causes of low hit rate: (1) Cache key too specific - includes timestamp, request_id, or other varying fields that should not be there. Fix: Hash only model + messages + core parameters, exclude metadata. (2) User input variations - users ask "What is Python?" vs "what is python?" vs "Tell me about Python". These should hit cache but do not with exact matching. Fix: Normalize inputs (lowercase, remove punctuation, trim whitespace). (3) Temperature >0 in cache key - if including temperature 0.7 in key but model generates differently each time anyway, cache misses. Fix: For temperature >0, might not cache at all (outputs vary). (4) Short TTL - cache expires too quickly, entries evicted before reuse. Fix: Increase TTL from 5min to 1hr for stable content. (5) Low query volume on specific topics - with 100K diverse questions, each question asked only once. Fix: Semantic caching to catch similar questions. (6) Cache size too small - LRU eviction removes entries before reuse. Fix: Increase cache size or monitor eviction rate. Step 3 - Investigation queries: (1) "What percentage of cache misses are near-duplicates?" - check if normalization would help, (2) "What is average time between cache access for same key?" - if >TTL, increase TTL, (3) "What is cache eviction rate?" - if high, increase size, (4) "What percentage of requests are unique?" - if >80%, caching might not help much. Step 4 - Improvement strategy: (1) Implement input normalization - expect 5-10% improvement, (2) Increase TTL appropriately based on content staleness tolerance, (3) Review cache key construction - remove varying fields, (4) Consider semantic caching if many similar but not identical queries, (5) Add query reformulation - convert variations to canonical form. Result: With these fixes, 15% →30-40% hit rate is typical. If still low after fixes, application might have naturally diverse queries where caching cannot help much - focus on other optimizations.',
    keyPoints: [
      'Analyze what makes requests different',
      'Normalize input to catch variations',
      'Review cache key construction',
      'Check TTL and eviction rates',
      'Some applications naturally have low cache potential',
    ],
  },
];
