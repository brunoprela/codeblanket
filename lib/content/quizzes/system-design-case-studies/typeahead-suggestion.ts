/**
 * Design Typeahead Suggestion Quiz
 */

export const typeaheadQuiz = [
    {
        question: "Explain why a Trie with pre-computed top suggestions at each node is superior to traversing the Trie on every request for a typeahead system serving 1 million QPS. Calculate the performance difference for a prefix 'new' that has 10,000 matching queries in the Trie, and discuss the memory trade-offs of pre-computation.",
        sampleAnswer: "A Trie with pre-computed top suggestions stores the top 10 results at each node during build time, enabling O(k) lookup where k is the prefix length. For prefix 'new' (k=3), we traverse 3 nodes and return the pre-stored top 10 suggestions—total operations: 3. In contrast, traversing the Trie on every request requires: (1) navigate to the 'new' node (3 operations), (2) DFS traversal of the entire subtree to find all matching queries (10,000 operations), (3) sort by frequency (10,000 log 10,000 = 130,000 operations), (4) return top 10. Total: ~140,000 operations—roughly 47,000× slower. At 1M QPS, traversal approach would require 140 billion operations/second, far exceeding server capacity. Pre-computation reduces this to 3M operations/second (manageable with caching). Memory trade-off: Each node stores 10 suggestions (10 × 50 bytes/query = 500 bytes). For a Trie with 150M nodes (10M queries, 15 chars average), this adds 75 GB memory. However, only ~10% of nodes are internal nodes with suggestions (15M nodes), so realistic overhead is ~7.5 GB. This is acceptable for modern servers (128-256 GB RAM) and enables sub-10ms latencies. The key insight: pre-computation trades modest memory (7.5 GB) for 47,000× speedup, making it essential for real-time typeahead at scale.",
        keyPoints: [
            "Pre-computed suggestions enable O(k) lookup (k = prefix length) vs O(n log n) for DFS + sort (n = matching queries)",
            "Performance difference: 3 operations vs 140,000 operations for prefix 'new' with 10K matches (47,000× faster)",
            "Memory overhead: ~7.5 GB for 10M queries (10 suggestions × 50 bytes × 15M internal nodes)",
            "Trade-off justified: 7.5 GB memory cost enables handling 1M QPS with sub-10ms Trie lookup latencies",
            "Without pre-computation, system would need 47,000× more CPU capacity—economically infeasible"
        ]
    },
    {
        question: "Your typeahead system serves global users. A user in New York types 'new' and should see 'new york' as the top suggestion, while a user in New Mexico types 'new' and should see 'new mexico' first. Design a solution that balances personalization with performance, considering that: (1) the Trie is shared across all users, (2) personalizing every request adds latency, and (3) you need to maintain sub-100ms response times.",
        sampleAnswer: "I would implement a hybrid approach with pre-computed base suggestions and real-time personalization scoring. Architecture: (1) The Trie stores global top 20 suggestions at each node (not just 10), sorted by global frequency. (2) When a request arrives with prefix 'new', fetch the top 20 from the Trie (10ms), then apply real-time scoring based on user context (5-10ms total). Personalization factors: location boost (user in New York: multiply 'new york' score by 2.0), search history boost (if user previously searched 'new mexico', multiply by 1.5), and recency (time-decay factor for trending terms). Implementation: `personalized_score = base_frequency × location_boost × history_boost × recency_factor`. Sort the 20 suggestions by personalized_score and return top 10. This keeps latency at ~20ms (10ms Trie + 10ms scoring) while providing relevant suggestions. For extreme personalization needs, cache the top 5 personalized suggestions per user per common prefix in Redis: key = `suggest:user:{user_id}:prefix:{prefix}`, TTL = 1 hour. On cache hit (60%+ rate for active users), latency drops to 2-3ms (Redis lookup). On cache miss, compute using Trie + scoring, then cache. Privacy considerations: store location at city-level granularity (not precise coordinates), and search history in client-side local storage with only hashed identifiers sent to server. This approach provides personalized results while maintaining sub-100ms latencies even at 1M QPS.",
        keyPoints: [
            "Hybrid approach: fetch global top 20 from Trie (10ms), apply real-time personalization scoring (10ms additional)",
            "Personalization factors: location boost (2×), search history boost (1.5×), recency decay for trending terms",
            "User-level caching in Redis: cache top 5 personalized suggestions per prefix (1-hour TTL, 60%+ hit rate)",
            "Cache hit: 2-3ms latency (Redis). Cache miss: 20ms (Trie + scoring), acceptable for 40% of requests",
            "Privacy: city-level location (not precise), hashed user identifiers, client-side history storage where possible"
        ]
    },
    {
        question: "Your typeahead system notices that 'taylor swift tickets' suddenly spikes from 100 searches/day to 100,000 searches/day (concert announcement). The Trie is rebuilt daily via batch jobs, so suggestions lag by 24 hours. Design a real-time trending query detection and injection system that can surface this query within 5 minutes while maintaining the integrity of the base Trie data structure.",
        sampleAnswer: "I would implement a real-time trending detection pipeline using Kafka and Redis. Architecture: (1) Stream all search queries to Kafka topic 'search-events' (10GB/day volume). (2) Spark Streaming (or Flink) consumes events and aggregates query counts over 5-minute tumbling windows. (3) Compare current window count to 24-hour average: if ratio > 10× (e.g., 'taylor swift tickets' went from 10/5min to 1000/5min), flag as trending. (4) Store trending queries in Redis sorted set: `ZADD trending:current {timestamp} {query}`, with 1-hour expiration. (5) Modify autocomplete logic to merge trending results with Trie results. Implementation: when serving autocomplete for prefix 'tay', first check `ZRANGEBYLEX trending:current [tay [taz` to find trending queries starting with 'tay'. If 'taylor swift tickets' is in trending set and matches prefix, inject it at position 2-3 (not #1, to avoid displacing stable suggestions). Final result: [global_top_1, global_top_2, trending_1, global_top_3, ...]. This ensures trending queries appear quickly (5-min lag) without disrupting the entire suggestion order. For high-confidence trending (ratio > 100×), inject at position 1. Anti-spam measures: require query appears from at least 100 unique IPs (prevent single user spamming), and apply ML model to detect malicious patterns (profanity, spam). Monitor trending dashboard: if false positives >5%, increase threshold from 10× to 20×. This system detected the Taylor Swift spike within 5 minutes and surfaced suggestions to millions of users before the daily Trie rebuild, dramatically improving relevance during breaking events.",
        keyPoints: [
            "Real-time pipeline: Kafka streaming → Spark/Flink 5-minute windows → detect 10× spike vs 24-hour baseline",
            "Redis sorted set stores trending queries (1-hour TTL): `ZADD trending:current {timestamp} {query}`",
            "Hybrid autocomplete: merge trending queries (Redis) with global suggestions (Trie) at positions 2-3",
            "Anti-spam: require 100 unique IPs, ML filter for profanity/malicious patterns, adjust threshold if false positives >5%",
            "Trending lag reduced from 24 hours (batch rebuild) to 5 minutes (real-time), critical for breaking events"
        ]
    }
];

