/**
 * Design Typeahead Suggestion Section
 */

export const typeaheadSection = {
    id: 'typeahead-suggestion',
    title: 'Design Typeahead Suggestion',
    content: `Typeahead (autocomplete) suggests search queries as users type. Google search shows 10 suggestions after typing just a few characters. The challenges are: ultra-low latency (< 100ms), handling typos, ranking by relevance/popularity, updating suggestions with trending queries, and supporting billions of queries across millions of users.

## Problem Statement

Design a typeahead system that:
- **Real-Time Suggestions**: Show suggestions within 100ms
- **Prefix Matching**: "new y" → ["new york", "new year", "new york times"]
- **Ranking**: Most popular/relevant suggestions first
- **Personalization**: User's search history influences suggestions
- **Typo Tolerance**: "nwe york" → "new york"
- **Trending Queries**: "taylor swift concert" spikes during ticket sales
- **Multi-Language**: Support English, Spanish, Chinese, etc.
- **Data Collection**: Learn from user searches to improve suggestions

**Scale**: 1 billion users, 10 billion searches/day, 10 million unique search terms

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **Autocomplete**: Suggest queries matching prefix
2. **Ranking**: Order by popularity, relevance
3. **Fast**: Return suggestions < 100ms
4. **Typo Handling**: Fuzzy matching ("twiter" → "twitter")
5. **Personalization**: User's location, history
6. **Real-Time Updates**: Trending queries appear quickly

### Non-Functional Requirements

1. **Low Latency**: p95 < 100ms
2. **High Availability**: 99.99% uptime
3. **Scalability**: Handle billions of queries
4. **Consistency**: Eventual consistency acceptable (suggestions lag by minutes)

---

## Step 2: Capacity Estimation

**Queries Per Second** (QPS):
- 10 billion searches/day ÷ 86,400 seconds = 115K QPS average
- Peak (2× average) = 230K QPS

**Keystrokes**:
- Average query length: 15 characters
- User types 1 char/sec → Typeahead triggered ~10 times per search (after 3+ chars)
- Typeahead QPS: 115K × 10 = 1.15M QPS

**Storage**:
- 10 million unique queries
- Average query: 30 bytes
- Query + count: 30 bytes + 8 bytes (frequency) = 38 bytes
- Total: 10M × 38 bytes = 380 MB (small enough for in-memory)

**Trie Size**:
- 10 million queries, average 15 characters → ~150M nodes
- Each node: character (1 byte) + children pointers (8 bytes × 26) + frequency (8 bytes) = ~217 bytes
- Optimized Trie: ~10 GB (compressed)

---

## Step 3: Trie Data Structure (Core Algorithm)

**Trie (Prefix Tree)**: Efficient for prefix matching.

\`\`\`
Example queries: ["new york", "new year", "news"]

Trie:
         root
          |
          n
          |
          e
          |
          w
         / \\
        (space) s [freq=5K]  ("news")
        |
        y
       / \\
      o   e
      |   |
      r   a
      |   |
      k   r  [freq=2K] ("new year")
   [freq=10K] ("new york")
\`\`\`

**Trie Node Structure**:

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}  # char → TrieNode
        self.is_end_of_word = False
        self.frequency = 0  # How many times this query searched
        self.top_queries = []  # Pre-computed top suggestions from this node

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, query, frequency):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = frequency
    
    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Traverse subtree, collect all words
        results = []
        self._dfs(node, prefix, results)
        return sorted(results, key=lambda x: x[1], reverse=True)[:10]  # Top 10
    
    def _dfs(self, node, current_word, results):
        if node.is_end_of_word:
            results.append((current_word, node.frequency))
        for char, child in node.children.items():
            self._dfs(child, current_word + char, results)
\`\`\`

**Problem**: DFS traversal on every request is slow (1000s of nodes).

---

## Step 4: Optimization - Pre-Computed Top Suggestions

**Idea**: Store top 10 suggestions at each node (precomputed during build).

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.top_suggestions = []  # List of (query, frequency), sorted by frequency

class Trie:
    def insert(self, query, frequency):
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Update top suggestions at this node
            self._update_top_suggestions(node, query, frequency)
        
        node.is_end_of_word = True
        node.frequency = frequency
    
    def _update_top_suggestions(self, node, query, frequency):
        # Add new query to suggestions
        node.top_suggestions.append((query, frequency))
        
        # Sort by frequency, keep top 10
        node.top_suggestions.sort(key=lambda x: x[1], reverse=True)
        node.top_suggestions = node.top_suggestions[:10]
    
    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Return pre-computed top suggestions
        return [query for query, freq in node.top_suggestions]

# Example:
trie = Trie()
trie.insert("new york", 10000)
trie.insert("new year", 2000)
trie.insert("news", 5000)

print(trie.autocomplete("new"))  # ["new york", "news", "new year"]
\`\`\`

**Benefits**:
- ✅ O(k) time, where k = length of prefix (fast)
- ✅ No traversal needed
- ✅ Pre-sorted results

**Trade-off**:
- ❌ More memory (10 suggestions × 10M nodes = 100M entries)
- ❌ Updates more expensive (rebuild top suggestions)

---

## Step 5: Handling Billions of Queries (Distributed Trie)

**Challenge**: Trie is 10 GB. Single server can handle, but we need redundancy + global distribution.

**Solution**: Partition Trie by prefix.

\`\`\`
Server 1: Handles prefixes starting with "a", "b", "c"
Server 2: Handles "d", "e", "f"
...
Server 9: Handles "x", "y", "z"

User types "new" → Route to server handling "n"
\`\`\`

**Routing**:

\`\`\`python
def get_server(prefix):
    first_char = prefix[0].lower()
    server_id = ord(first_char) - ord('a')  # 0-25
    server_id = server_id % num_servers  # Distribute evenly
    return f"trie-server-{server_id}"

# User types "new"
server = get_server("new")  # "trie-server-13" (n = 13)
suggestions = request(f"{server}/autocomplete?prefix=new")
\`\`\`

**Load Balancing**:
- Popular prefixes (e.g., "a", "s", "t") get more traffic
- Solution: Partition by query hash (more even distribution)

\`\`\`python
def get_server_hash(prefix):
    return hash(prefix) % num_servers
\`\`\`

---

## Step 6: Ranking Suggestions

**Factors**:

1. **Frequency**: How many users searched this query?
2. **Recency**: Recent queries ranked higher (trending)
3. **Personalization**: User's location, history
4. **Click-Through Rate (CTR)**: Did users click suggested query?

**Ranking Formula**:

\`\`\`python
def calculate_score(query, user):
    base_score = query.frequency
    
    # Boost recent queries (exponential decay)
    days_old = (now - query.last_updated).days
    recency_factor = math.exp(-0.1 * days_old)  # 90% decay in ~23 days
    
    # Personalization (location)
    location_boost = 1.0
    if user.location in query.popular_locations:
        location_boost = 1.5
    
    # User history (previously searched?)
    history_boost = 1.0
    if query.text in user.search_history:
        history_boost = 2.0
    
    # CTR (users click this suggestion?)
    ctr_boost = 1.0 + (query.ctr / 100)  # CTR=20% → 1.2x boost
    
    score = base_score * recency_factor * location_boost * history_boost * ctr_boost
    return score
\`\`\`

**Pre-compute vs Real-Time**:
- **Pre-compute**: Base score (frequency, recency) → stored in Trie
- **Real-Time**: Personalization (location, history) → applied during request

---

## Step 7: Typo Tolerance (Fuzzy Matching)

**Problem**: User types "nwe york" (typo), should suggest "new york".

**Approach 1: Edit Distance (Levenshtein)**

\`\`\`python
def edit_distance(s1, s2):
    # Number of insertions, deletions, substitutions to transform s1 → s2
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

# edit_distance("nwe", "new") = 2 (swap 'w' and 'e')
\`\`\`

**Fuzzy Search in Trie**:

\`\`\`python
def fuzzy_search(prefix, max_edit_distance=2):
    results = []
    
    def dfs(node, current_word, remaining_edits):
        if node.is_end_of_word and edit_distance(prefix, current_word) <= max_edit_distance:
            results.append((current_word, node.frequency))
        
        if remaining_edits == 0:
            return  # No more edits allowed
        
        # Try all children (exploration)
        for char, child in node.children.items():
            dfs(child, current_word + char, remaining_edits - 1)
    
    dfs(trie.root, "", max_edit_distance)
    return sorted(results, key=lambda x: x[1], reverse=True)[:10]
\`\`\`

**Problem**: Slow (explores many paths).

**Approach 2: Spelling Correction (Better)**

\`\`\`python
# Pre-built dictionary of common misspellings
typo_map = {
    "nwe": "new",
    "teh": "the",
    "recieve": "receive"
}

def correct_spelling(prefix):
    return typo_map.get(prefix, prefix)

# User types "nwe"
corrected = correct_spelling("nwe")  # "new"
suggestions = trie.autocomplete(corrected)  # ["new york", "news", ...]
\`\`\`

**Approach 3: Keyboard Proximity (Advanced)**

\`\`\`python
# Common typos based on keyboard layout (QWERTY)
keyboard_neighbors = {
    'a': ['q', 's', 'w', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    # ...
}

def generate_typo_variants(query):
    variants = [query]
    for i, char in enumerate(query):
        for neighbor in keyboard_neighbors.get(char, []):
            variant = query[:i] + neighbor + query[i+1:]
            variants.append(variant)
    return variants

# "nwe" → ["nwe", "mwe", "bwe", "nee", "nqe", ...]
\`\`\`

---

## Step 8: Trending Queries

**Problem**: "taylor swift tickets" suddenly searched 1000× more (concert announcement).

**Approach**: Real-time aggregation of recent searches.

\`\`\`python
# Stream of search events
search_stream = Kafka("search-events")

# Aggregate queries over 5-minute windows
for event in search_stream:
    query = event.query
    timestamp = event.timestamp
    window = timestamp // 300  # 5-minute buckets
    
    redis.zincrby(f"trending:{window}", 1, query)

# Top 100 trending queries in last 5 minutes
trending = redis.zrevrange("trending:1672531200", 0, 99, withscores=True)

# Boost trending queries in autocomplete
def autocomplete_with_trending(prefix):
    normal_suggestions = trie.autocomplete(prefix)
    trending_suggestions = [q for q in trending if q.startswith(prefix)]
    
    # Merge: 50% normal, 50% trending
    merged = []
    for i in range(5):
        if i < len(trending_suggestions):
            merged.append(trending_suggestions[i])
        if i < len(normal_suggestions):
            merged.append(normal_suggestions[i])
    
    return merged[:10]
\`\`\`

---

## Step 9: Personalization

**User Context**:

\`\`\`python
user = {
    "id": 12345,
    "location": "New York",
    "search_history": ["pizza near me", "weather nyc", "taylor swift"],
    "language": "en"
}

def personalized_autocomplete(prefix, user):
    # Get base suggestions
    suggestions = trie.autocomplete(prefix)
    
    # Re-rank based on user context
    scores = []
    for query in suggestions:
        score = calculate_score(query, user)
        scores.append((query, score))
    
    # Sort by personalized score
    scores.sort(key=lambda x: x[1], reverse=True)
    return [query for query, score in scores[:10]]
\`\`\`

**Privacy**: Store user history locally (browser), send only necessary context (location, language) to server.

---

## Step 10: High-Level Architecture

\`\`\`
┌──────────────┐
│   Browser    │
│ (User types) │
└──────┬───────┘
       │ HTTP: GET /autocomplete?prefix=new&user_id=123
       ▼
┌──────────────┐
│ Load Balancer│
└──────┬───────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│ Trie Servers │───▶│    Redis     │
│ (Partitioned)│    │(Query Cache) │
└──────┬───────┘    └──────────────┘
       │
       │ Fetch from Trie (if cache miss)
       ▼
┌──────────────┐
│ Trie Storage │
│  (In-Memory) │
└──────────────┘

┌──────────────┐    ┌──────────────┐
│ Kafka Stream │───▶│ Aggregation  │
│(Search Logs) │    │ Service      │
└──────────────┘    │ (Trending)   │
                    └──────┬───────┘
                           │ Update Trie
                           ▼
                    ┌──────────────┐
                    │ Trie Builder │
                    │(Batch Update)│
                    └──────────────┘
\`\`\`

---

## Step 11: Update Strategy

**Challenge**: 10 billion new searches/day, Trie needs updating.

**Approach 1: Batch Rebuild** (Simple)

\`\`\`
1. Collect search logs for 24 hours
2. Aggregate: Count frequency of each query
3. Build new Trie offline (takes 10 minutes)
4. Swap Trie atomically (zero downtime)
5. Repeat daily
\`\`\`

**Approach 2: Incremental Update** (Real-Time)

\`\`\`
1. Stream search events (Kafka)
2. Update Trie incrementally (thread-safe writes)
3. Re-sort top suggestions after each insert
4. Lock-free reads (queries don't block updates)
\`\`\`

**Hybrid** (Best):
- Batch rebuild weekly (full refresh)
- Incremental updates every 5 minutes (trending queries)

---

## Step 12: Caching

**Cache Frequent Prefixes** (Pareto Principle: 20% of prefixes account for 80% of queries)

\`\`\`python
# Redis cache
def autocomplete_with_cache(prefix):
    cache_key = f"autocomplete:{prefix}"
    
    # Try cache first
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss: Query Trie
    suggestions = trie.autocomplete(prefix)
    
    # Store in cache (10-minute TTL)
    redis.setex(cache_key, 600, json.dumps(suggestions))
    
    return suggestions
\`\`\`

**Cache Hit Rate**: 90%+ for popular prefixes ("a", "the", "how").

---

## Step 13: Latency Optimization

**Target**: < 100ms response time

**Optimizations**:

1. **Pre-computed Top Suggestions**: O(k) lookup (no traversal)
2. **In-Memory Trie**: No disk I/O
3. **CDN for Static Suggestions**: Cache common prefixes at edge
4. **Compression**: gzip response (reduce bandwidth)
5. **Early Termination**: Return after 3 characters typed (not 1-2)

**Latency Breakdown**:
- Network: 20-50ms
- Trie lookup: 1-5ms
- Ranking/personalization: 5-10ms
- Redis cache: 1-2ms
- **Total**: 30-70ms ✅

---

## Step 14: Multi-Language Support

**Challenge**: Chinese queries have different tokenization (no spaces).

**Solution**:

\`\`\`python
# Tokenizer based on language
def tokenize(query, language):
    if language == "zh":  # Chinese
        return jieba.cut(query)  # Chinese word segmentation
    elif language == "ja":  # Japanese
        return mecab.parse(query)
    else:
        return query.split()  # Space-separated

# Build separate Tries per language
trie_en = Trie()  # English
trie_zh = Trie()  # Chinese
trie_es = Trie()  # Spanish

def autocomplete_multi_lang(prefix, language="en"):
    trie = get_trie(language)
    return trie.autocomplete(prefix)
\`\`\`

---

## Step 15: Monitoring & Metrics

**Key Metrics**:

1. **Latency**: p50, p95, p99 response time
2. **Cache Hit Rate**: % of queries served from cache
3. **Autocomplete Usage**: % of searches that used suggestion
4. **CTR**: Click-through rate on suggestions
5. **Typo Rate**: % of queries with spelling errors

**Alerts**:
- p95 latency > 100ms → Scale Trie servers
- Cache hit rate < 80% → Adjust cache TTL, add more popular prefixes
- Trie update lag > 10 minutes → Speed up batch rebuild

---

## Trade-offs

**Pre-computed vs Real-Time**:
- Pre-computed: Fast, but stale (lag in trending queries)
- Real-time: Fresh, but slower (compute ranking on every request)
- **Choice**: Pre-compute base suggestions, real-time personalization

**Memory vs Latency**:
- Store top 10 suggestions at each node → More memory, faster
- Traverse Trie on each request → Less memory, slower
- **Choice**: Pre-compute (10 GB memory acceptable)

**Consistency vs Availability**:
- Strong consistency: All users see same suggestions (slow updates)
- Eventual consistency: Users see slightly different suggestions (fast updates)
- **Choice**: Eventual consistency (OK if suggestions lag by minutes)

---

## Interview Tips

**Clarify**:
1. Scale: Millions or billions of queries?
2. Personalization: Generic or user-specific?
3. Language: English only or multi-language?
4. Latency: < 100ms required?

**Emphasize**:
1. **Trie Data Structure**: Efficient prefix matching
2. **Pre-computed Top Suggestions**: Fast O(k) lookup
3. **Distributed Trie**: Partition by prefix for scale
4. **Caching**: Redis for frequent prefixes
5. **Ranking**: Frequency + recency + personalization

**Common Mistakes**:
- Using database LIKE query (too slow: 100-500ms)
- Not caching (overwhelm Trie servers)
- Ignoring personalization (generic suggestions less useful)
- No typo handling (poor UX)

**Follow-up Questions**:
- "How to handle long-tail queries? (Less popular, not in top 10)"
- "What if user has no search history? (Use location, trending)"
- "How to prevent offensive suggestions? (Blacklist, ML filter)"
- "How to support voice search? (Phonetic matching, not prefix)"

---

## Summary

**Core Algorithm**: **Trie with Pre-Computed Top Suggestions**

**Data Structure**:
- Trie: 10M queries, 10 GB in-memory
- Each node: Top 10 suggestions (pre-sorted by score)
- Partitioned by prefix (distribute across servers)

**Ranking Factors**:
- Frequency (base score)
- Recency (trending queries)
- Personalization (location, history)
- CTR (user clicks)

**Key Optimizations**:
- ✅ Pre-computed suggestions (no traversal)
- ✅ Redis caching (90% hit rate)
- ✅ CDN edge caching (popular prefixes)
- ✅ Incremental updates (trending queries)
- ✅ Fuzzy matching (typo tolerance)

**Capacity**:
- 1.15M autocomplete QPS
- < 100ms latency (p95)
- 10 GB Trie in-memory
- 90%+ cache hit rate
- Support multi-language, personalization

A production typeahead system balances **speed** (< 100ms), **relevance** (ranking), and **freshness** (trending queries) to provide instant, helpful suggestions for billions of users worldwide.`,
};

