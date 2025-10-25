import { QuizQuestion } from '@/lib/types';

export const searchOptimizationDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'search-opt-discussion-1',
    question:
      'Your e-commerce search application is experiencing performance issues. Users report that searches for "laptop" return in 50ms, but searches for "smartphone case" take 2-3 seconds. After investigation, you discover that multi-word queries are significantly slower. The "title" field uses standard analysis, and you have 10 million products across 20 shards. Diagnose the likely causes and propose a comprehensive optimization strategy that addresses relevance, performance, and maintainability.',
    sampleAnswer: `This is a classic phrase query performance problem. Let me analyze the root causes and provide a systematic solution:

**Root Cause Analysis:**

1. **Phrase Matching Overhead**: "smartphone case" likely uses phrase matching, which requires:
   - Finding all documents with "smartphone"
   - Finding all documents with "case"
   - Checking positional data to verify adjacency
   - This is much more expensive than single-term queries

2. **High Document Frequency**: "case" is likely a very common term (phone case, laptop case, etc.), appearing in millions of documents
   - Larger postings lists to scan
   - More positional data to check

3. **Cross-Shard Coordination**: With 20 shards:
   - Query broadcast to all 20 shards
   - Each shard processes phrase match independently
   - Coordinating node merges 20 result sets
   - Overhead scales with shard count

4. **Lack of Optimization**: Standard analyzer doesn't optimize for phrase queries

**Comprehensive Optimization Strategy:**

**Phase 1: Immediate Query Optimization (Week 1)**

**1. Implement Query Strategy Based on Terms:**

\`\`\`javascript
function buildQuery (userQuery) {
  const terms = userQuery.split(' ');
  
  if (terms.length === 1) {
    // Single term: Fast match query
    return {
      multi_match: {
        query: userQuery,
        fields: ["title^3", "description"],
        type: "best_fields"
      }
    };
  } else {
    // Multi-term: Combine match + phrase with boost
    return {
      bool: {
        should: [
          {
            match: {
              title: {
                query: userQuery,
                boost: 2
              }
            }
          },
          {
            match_phrase: {
              title: {
                query: userQuery,
                boost: 5,
                slop: 2  // Allow 2 words between terms
              }
            }
          }
        ],
        minimum_should_match: 1
      }
    };
  }
}
\`\`\`

**Benefits:**
- \`match\` query: Fast, finds docs with any terms
- \`match_phrase\` with \`slop: 2\`: Finds exact/near matches, highly boosted
- Documents with both terms (even non-adjacent) still returned
- Exact phrase matches score highest

**2. Add Query Timeout:**

\`\`\`json
{
  "timeout": "500ms",
  "query": { ... }
}
\`\`\`

Returns partial results rather than waiting forever.

**Phase 2: Index Optimization (Week 2)**

**1. Implement Shingles for Common Phrases:**

\`\`\`json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "shingle_analyzer": {
          "tokenizer": "standard",
          "filter": ["lowercase", "shingle_filter"]
        }
      },
      "filter": {
        "shingle_filter": {
          "type": "shingle",
          "min_shingle_size": 2,
          "max_shingle_size": 3,
          "output_unigrams": true
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "shingles": {
            "type": "text",
            "analyzer": "shingle_analyzer"
          }
        }
      }
    }
  }
}
\`\`\`

**Effect:**
- "smartphone case" indexed as: ["smartphone", "case", "smartphone case"]
- Query "smartphone case" → exact match on "smartphone case" term (fast!)
- Converts expensive phrase query into cheap term query

**Query Update:**
\`\`\`json
{
  "multi_match": {
    "query": "smartphone case",
    "fields": ["title.shingles^3", "title^1"]
  }
}
\`\`\`

**Trade-off:** Index size increases 30-50%, but phrase queries become 10x faster.

**2. Add Multi-Field Optimization:**

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "index_options": "offsets",  // Fast phrase queries
        "fields": {
          "shingles": {
            "type": "text",
            "analyzer": "shingle_analyzer"
          },
          "keyword": {
            "type": "keyword"
          },
          "edge": {
            "type": "text",
            "analyzer": "edge_ngram_analyzer"
          }
        }
      }
    }
  }
}
\`\`\`

**Phase 3: Caching and Architecture (Week 3)**

**1. Implement Intelligent Caching:**

\`\`\`json
{
  "query": {
    "bool": {
      "must": {
        "match": { "title": "smartphone case" }
      },
      "filter": [  // Filters are cached
        { "term": { "in_stock": true } },
        { "range": { "price": { "gte": 0 } } }
      ]
    }
  }
}
\`\`\`

**2. Query Result Caching for Popular Queries:**

Application-level cache:
\`\`\`javascript
const cacheKey = \`search:\${query}:\${filters}\`;
const cached = await redis.get (cacheKey);
if (cached) return cached;

const results = await elasticsearch.search(...);
await redis.setex (cacheKey, 300, results);  // 5 min TTL
return results;
\`\`\`

**3. Category-Based Index Splitting:**

If "smartphone case" searches are common:
\`\`\`
products-electronics
products-cases-accessories
products-computers
products-other
\`\`\`

Query "smartphone case" → Route to \`products-cases-accessories\` + \`products-electronics\`
- Hits 4 shards instead of 20 (5x reduction)
- Smaller indices = faster queries

**Phase 4: Shard Optimization (Week 4)**

**Current**: 10M products, 20 shards = 500k products/shard

**Analysis:**
- 20 shards might be excessive for 10M docs
- Each query hits all 20 shards (coordination overhead)

**Recommendation**: Consolidate to 10 shards (1M products each)
- Reindex with 10 primary shards
- Reduces coordination overhead by 50%
- Maintains good shard size (if 10M products ≈ 100GB, each shard = 10GB)

**Migration Strategy:**
\`\`\`
1. Create products_v2 with 10 shards
2. Reindex in background
3. Use index alias for zero-downtime switch
4. Monitor performance improvements
\`\`\`

**Phase 5: Relevance and User Experience (Week 5)**

**1. Implement Query Analysis:**

\`\`\`javascript
const queryAnalytics = {
  "smartphone case": { count: 50000, avgLatency: 2300ms },
  "laptop": { count: 30000, avgLatency: 50ms },
  "iphone charger": { count: 25000, avgLatency: 1800ms }
};
\`\`\`

**2. Pre-compute Popular Queries:**

For top 1000 queries, pre-compute and cache results:
\`\`\`javascript
// Nightly job
for (const query of topQueries) {
  const results = await elasticsearch.search (query);
  await redis.set(\`precache:\${query}\`, results, 'EX', 86400);
}
\`\`\`

**3. Implement Search-as-you-type:**

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "search_as_you_type",
        "max_shingle_size": 3
      }
    }
  }
}
\`\`\`

Auto-generates shingles and edge n-grams for instant search.

**Performance Targets and Measurements:**

**Before Optimization:**
- Single word ("laptop"): 50ms
- Multi-word ("smartphone case"): 2,300ms
- User satisfaction: 65%

**After Optimization:**
- Single word: 30ms (faster with shingles)
- Multi-word: 150ms (15x improvement!)
- User satisfaction target: 90%

**Measurement Strategy:**
\`\`\`javascript
// Track query performance
{
  query: "smartphone case",
  latency: 150,
  results_count: 1247,
  clicked_result: 3,
  timestamp: "2024-01-15T10:30:00"
}

// Metrics
- P50 latency: 100ms
- P95 latency: 300ms
- P99 latency: 800ms
- Zero results rate: <2%
- Click-through rate: >40%
\`\`\`

**Trade-offs Analysis:**

1. **Shingles: Performance vs Index Size**
   - Pros: 10-15x faster phrase queries
   - Cons: 30-50% larger index
   - Decision: Worth it for user experience

2. **Fewer Shards: Simplicity vs Parallelization**
   - Pros: Less coordination overhead, simpler ops
   - Cons: Less parallelization per query
   - Decision: 10 shards optimal for current scale

3. **Caching: Speed vs Freshness**
   - Pros: Sub-millisecond for cached queries
   - Cons: Stale results for 5 minutes
   - Decision: Acceptable for most e-commerce

4. **Category Split: Performance vs Complexity**
   - Pros: 5x fewer shards per query
   - Cons: Application routing complexity
   - Decision: Implement if query patterns support it

**Expected Results:**

Week 1: 40% improvement (query optimization)
Week 2: 70% improvement (shingles)
Week 3: 85% improvement (caching)
Week 4: 90% improvement (shard consolidation)
Week 5: 95% improvement + better UX

**Final Recommendation:**

Prioritize shingles (Week 2) for maximum impact with acceptable cost. Implement query-level caching (Week 3) for popular queries. Consider shard consolidation if coordination overhead is significant. This balanced approach delivers excellent performance without over-engineering.`,
    keyPoints: [
      'Phrase queries are expensive due to positional data checking across many documents',
      'Shingles convert phrase queries into term queries (10x faster) at cost of index size',
      'Fewer shards reduce coordination overhead for queries broadcast to all shards',
      'Multi-field strategy: shingles for phrases, edge n-grams for prefix, standard for full-text',
      'Application-level caching for popular queries provides massive performance gains',
      "Measure query patterns to inform optimization decisions (don't optimize blindly)",
    ],
  },
  {
    id: 'search-opt-discussion-2',
    question:
      "You're implementing autocomplete for a product search with 5 million products. Users expect results within 50ms as they type. You're debating between three approaches: (1) completion suggester, (2) edge n-grams on product titles, (3) prefix queries on keyword fields. Compare these approaches in terms of performance, relevance, index size, and maintenance complexity. Which would you choose and why?",
    sampleAnswer: `This is an excellent question about choosing the right autocomplete strategy. Let me analyze each approach comprehensively:

**Approach 1: Completion Suggester**

**Implementation:**
\`\`\`json
{
  "mappings": {
    "properties": {
      "suggest": {
        "type": "completion",
        "analyzer": "simple",
        "preserve_separators": true,
        "preserve_position_increments": true,
        "max_input_length": 50
      }
    }
  }
}

// Index document
{
  "title": "Apple iPhone 15 Pro Max",
  "suggest": {
    "input": [
      "Apple iPhone 15 Pro Max",
      "iPhone 15 Pro Max",
      "15 Pro Max",
      "iPhone 15",
      "Apple iPhone"
    ],
    "weight": 100
  }
}

// Query
POST /products/_search
{
  "suggest": {
    "my-suggestion": {
      "prefix": "ipho",
      "completion": {
        "field": "suggest",
        "size": 10,
        "skip_duplicates": true
      }
    }
  }
}
\`\`\`

**Performance:**
- ⭐⭐⭐⭐⭐ **Excellent** (1-5ms)
- Uses in-memory FST (Finite State Transducer)
- Constant time O(k) where k is prefix length
- Sub-millisecond for most queries

**Relevance:**
- ⭐⭐⭐⭐ **Good**
- Supports weights for popularity ranking
- Returns suggestions in weight order
- No fuzzy matching (must match prefix exactly)
- Limited scoring flexibility

**Index Size:**
- ⭐⭐⭐ **Moderate**
- +20-30% per completion field
- FST is memory-efficient
- Must store all input variants (5 variants per product = 25M total)

**Maintenance:**
- ⭐⭐⭐ **Moderate**
- Must manually define input variants at index time
- No automatic phrase generation
- Need to update weights periodically
- Requires reindexing to change inputs

**Code Example for Weight Updates:**
\`\`\`javascript
// Update weights based on popularity
async function updateSuggestWeights() {
  const popular = await getPopularProducts();  // From click data
  
  for (const product of popular) {
    await es.update({
      id: product.id,
      body: {
        doc: {
          suggest: {
            input: product.suggest.input,
            weight: product.clickCount  // Higher = appears first
          }
        }
      }
    });
  }
}
\`\`\`

**Pros:**
- Blazing fast (1-5ms)
- Low memory overhead
- Purpose-built for autocomplete
- Weighted suggestions

**Cons:**
- Must define all input variants upfront
- No fuzzy matching
- Limited to prefix matching
- Requires careful input planning

**Approach 2: Edge N-grams**

**Implementation:**
\`\`\`json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "autocomplete": {
          "tokenizer": "standard",
          "filter": ["lowercase", "autocomplete_filter"]
        },
        "autocomplete_search": {
          "tokenizer": "lowercase"
        }
      },
      "filter": {
        "autocomplete_filter": {
          "type": "edge_ngram",
          "min_gram": 2,
          "max_gram": 20
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "autocomplete",
        "search_analyzer": "autocomplete_search"
      }
    }
  }
}

// Index: "iPhone" → ["ip", "iph", "ipho", "iphon", "iphone"]
// Query
{
  "query": {
    "match": {
      "title": {
        "query": "ipho",
        "operator": "and"
      }
    }
  }
}
\`\`\`

**Performance:**
- ⭐⭐⭐⭐ **Good** (10-30ms)
- Term lookup (fast)
- Scales well with data volume
- Slower than completion but still fast

**Relevance:**
- ⭐⭐⭐⭐⭐ **Excellent**
- Full BM25 scoring available
- Can combine with other queries
- Field boosting, function scores
- Fuzzy matching possible

**Index Size:**
- ⭐ **Large**
- +200-300% per field (!)
- "Apple iPhone 15" → 50+ ngram tokens
- 5M products × 300% = significant storage

**Maintenance:**
- ⭐⭐⭐⭐⭐ **Simple**
- Automatic at index time
- No manual input variants
- Works with any text
- Easy to modify analyzer settings

**Storage Calculation:**
\`\`\`
Original title: "Apple iPhone 15 Pro Max" (6 words)
Average word length: 6 characters
Total characters: 36

Edge n-grams (min=2, max=20):
"Apple" (5 chars) → ["ap", "app", "appl", "apple"] = 4 tokens
"iPhone" (6 chars) → ["ip", "iph", "ipho", "iphon", "iphone"] = 5 tokens
"15" (2 chars) → ["15"] = 1 token
...

Total ngrams per product: ~30 tokens
5M products × 30 tokens = 150M tokens

Compared to original:
5M products × 6 tokens = 30M tokens

Storage overhead: 5x !
\`\`\`

**Pros:**
- Automatic, no manual inputs
- Full relevance scoring
- Flexible querying
- Easy to maintain

**Cons:**
- Huge index size (2-3x)
- Slower than completion
- High disk I/O
- Expensive at scale

**Approach 3: Prefix Query on Keyword Field**

**Implementation:**
\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "keyword"
      }
    }
  }
}

// Query
{
  "query": {
    "prefix": {
      "title": {
        "value": "ipho"
      }
    }
  }
}
\`\`\`

**Performance:**
- ⭐⭐ **Poor** (100-500ms)
- Must scan term dictionary
- O(n) where n is number of unique terms
- Very slow for short prefixes ("i" → millions of terms)
- Can timeout on large indices

**Relevance:**
- ⭐⭐ **Poor**
- No relevance scoring
- Alphabetical order only
- Can't incorporate popularity
- Returns all matches (not top N)

**Index Size:**
- ⭐⭐⭐⭐⭐ **Minimal**
- No overhead (baseline)
- Just stores original values

**Maintenance:**
- ⭐⭐⭐⭐⭐ **Simple**
- No configuration needed
- Works out of box

**Pros:**
- No index overhead
- Simple to implement
- No special configuration

**Cons:**
- Very slow (100-500ms+)
- No relevance ranking
- Poor user experience
- Doesn't scale

**Performance Testing Results:**

Simulated 5M products, query "ipho":

\`\`\`
Completion Suggester:
- 50th percentile: 2ms
- 95th percentile: 4ms
- 99th percentile: 8ms
- Index size: +25% (125% total)

Edge N-grams:
- 50th percentile: 15ms
- 95th percentile: 35ms
- 99th percentile: 80ms
- Index size: +250% (350% total)

Prefix Query:
- 50th percentile: 180ms
- 95th percentile: 450ms
- 99th percentile: 1200ms
- Index size: baseline (100%)
\`\`\`

**My Recommendation: Hybrid Approach**

Use **completion suggester** as primary with **edge n-grams** as fallback:

**Implementation:**

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "suggest": {
            "type": "completion",
            "analyzer": "simple"
          },
          "edge": {
            "type": "text",
            "analyzer": "edge_ngram_analyzer"
          }
        }
      }
    }
  }
}
\`\`\`

**Query Strategy:**

\`\`\`javascript
async function autocomplete (prefix) {
  // Phase 1: Try completion suggester (fast, limited)
  const suggestions = await es.search({
    suggest: {
      my_suggest: {
        prefix: prefix,
        completion: {
          field: "title.suggest",
          size: 10
        }
      }
    }
  });
  
  if (suggestions.hits.length >= 5) {
    return suggestions;  // Enough results, return immediately
  }
  
  // Phase 2: Fallback to edge n-grams (slower, more flexible)
  const searchResults = await es.search({
    query: {
      bool: {
        should: [
          {
            match: {
              "title.edge": {
                query: prefix,
                boost: 2
              }
            }
          },
          {
            match: {
              "title": {
                query: prefix,
                fuzziness: "AUTO"  // Handle typos
              }
            }
          }
        ]
      }
    },
    size: 10
  });
  
  return merge (suggestions, searchResults);
}
\`\`\`

**Benefits of Hybrid:**

1. **Fast common queries**: Completion suggester handles 80% of queries in 2-5ms
2. **Flexible rare queries**: Edge n-grams handle edge cases
3. **Fuzzy matching**: Can implement typo tolerance with n-grams
4. **Reasonable index size**: Only +50-70% total (completion + partial n-grams)

**Cost-Benefit Analysis:**

For 5M products at 100KB avg per product = 500GB base

**Completion only:** 625GB (+25%), <5ms queries, limited flexibility
**Edge n-grams only:** 1.75TB (+250%), <30ms queries, full flexibility
**Hybrid:** 850GB (+70%), <10ms for 80% queries, best of both
**Prefix query:** 500GB (baseline), >100ms queries, poor UX

**Final Recommendation: Completion Suggester**

For pure autocomplete use case:

**Why:**
- Performance is critical (50ms target, completion achieves 2-5ms)
- 10x faster than edge n-grams
- 7x smaller index than edge n-grams
- Purpose-built for autocomplete
- Weighted suggestions handle popularity

**Implementation Plan:**

1. **Generate input variants intelligently:**
\`\`\`javascript
function generateInputs (title, brand, category) {
  return [
    title,                                    // "Apple iPhone 15 Pro"
    title.replace(/^\\S+\\s/, ''),            // "iPhone 15 Pro"
    \`\${brand} \${category}\`,                   // "Apple Smartphone"
    ...title.split(' ').slice(0, 3).join(' ') // "Apple iPhone 15"
  ];
}
\`\`\`

2. **Update weights periodically** based on clicks/sales
3. **Add fallback** for zero-results: fuzzy match query
4. **Cache** top 1000 queries in Redis

This provides optimal performance for the autocomplete use case while keeping infrastructure costs reasonable.`,
    keyPoints: [
      'Completion suggester is fastest (1-5ms) using in-memory FST, ideal for autocomplete',
      'Edge n-grams provide flexibility and scoring but increase index size 2-3x',
      'Prefix queries on keyword fields are too slow (100-500ms) for user-facing autocomplete',
      'Hybrid approach: completion suggester for common queries, edge n-grams for fallback',
      'Index size vs performance: completion (+25%), edge n-grams (+250%)',
      'Weighted suggestions in completion type enable popularity-based ranking',
    ],
  },
  {
    id: 'search-opt-discussion-3',
    question:
      'Your Elasticsearch cluster is experiencing high heap usage (85-90% consistently) and frequent long GC pauses (5-10 seconds), causing query timeouts and poor user experience. You have 10 data nodes with 32GB RAM each (16GB heap). The cluster stores 2TB of product data with heavy aggregation queries on high-cardinality fields. Diagnose the root causes and provide a detailed remediation plan covering both immediate fixes and long-term architectural improvements.',
    sampleAnswer: `This is a critical production issue involving heap pressure and GC problems. Let me provide a comprehensive diagnosis and remediation strategy:

**Immediate Diagnosis (First Hour):**

**1. Check Heap Usage Breakdown:**

\`\`\`bash
GET /_nodes/stats/jvm?filter_path=nodes.*.jvm.mem

# Check what's using heap
GET /_nodes/stats/indices?filter_path=nodes.*.indices.fielddata,nodes.*.indices.segments,nodes.*.indices.query_cache

# Common culprits:
# - Fielddata: text field aggregations/sorting (BAD!)
# - Segments: Too many small segments
# - Query cache: Overly large cached queries
# - Parent circuit breaker: Heavy aggregations
\`\`\`

**Likely Findings:**
\`\`\`
Node-1:
  JVM Heap: 85% (13.6GB / 16GB)
  Fielddata: 4.2GB (26% of heap!)  ⚠️ RED FLAG
  Segments: 3.8GB (24%)
  Query cache: 2.1GB (13%)
  Request circuit breaker: 75% ⚠️
\`\`\`

**2. Identify Problematic Queries:**

\`\`\`bash
# Check circuit breaker trips
GET /_nodes/stats/breaker

# Enable slow log if not already
PUT /products/_settings
{
  "index.search.slowlog.threshold.query.warn": "1s"
}

# Review logs for patterns
tail -f /var/log/elasticsearch/slowlog.log
\`\`\`

**Root Cause Analysis:**

**Problem 1: Fielddata on Text Fields (Most Likely Culprit)**

Aggregating/sorting on analyzed text fields loads entire inverted index into heap:

\`\`\`json
// BAD QUERY causing heap issues
{
  "aggs": {
    "by_product_name": {
      "terms": { "field": "product_name" }  // "product_name" is text type!
    }
  }
}
\`\`\`

**Effect:**
- Elasticsearch loads ALL terms from "product_name" into heap
- For 2TB data, could be 4-8GB of fielddata
- Fielddata stays in heap until evicted (often never)
- Causes heap pressure and GC thrashing

**Problem 2: High-Cardinality Aggregations**

\`\`\`json
{
  "aggs": {
    "by_user_id": {
      "terms": {
        "field": "user_id",    // 10M unique users
        "size": 1000
      }
    }
  }
}
\`\`\`

**Effect:**
- Must track millions of buckets in memory
- Coordinator node must merge results from all shards
- Heavy heap allocation during query
- Triggers frequent GC

**Problem 3: Segment Proliferation**

With 2TB data and heavy indexing:
- Many small segments (if refresh_interval is fast)
- Each segment has overhead (file handles, memory)
- Too many segments = high heap usage

**Immediate Remediation (Hours 1-4):**

**Step 1: Clear Fielddata Cache (Immediate Relief)**

\`\`\`bash
# Clear fielddata cache (frees 4.2GB)
POST /_cache/clear?fielddata=true

# WARNING: Next query will trigger reload
# This is temporary fix only!
\`\`\`

**Step 2: Restrict Fielddata Usage**

\`\`\`bash
# Temporarily disable problematic queries
# Add application-level circuit breaker
if (query.includes('aggs') && query.includes('text_field')) {
  return { error: "Aggregations temporarily disabled" };
}
\`\`\`

**Step 3: Force Merge Segments (Off-Peak Hours)**

\`\`\`bash
# Reduce segments to free up heap
POST /products/_forcemerge?max_num_segments=1

# WARNING: CPU intensive, do during low traffic
\`\`\`

**Step 4: Increase Circuit Breaker Limits (Temporary)**

\`\`\`json
PUT /_cluster/settings
{
  "transient": {
    "indices.breaker.fielddata.limit": "50%",        // Was 40%
    "indices.breaker.request.limit": "70%",           // Was 60%
    "indices.breaker.total.limit": "85%"              // Was 70%
  }
}
\`\`\`

⚠️ **WARNING**: This doesn't solve the problem, just prevents circuit breaker trips. Must fix root cause!

**Short-Term Fixes (Days 1-7):**

**Fix 1: Correct Field Mappings (Critical!)**

The #1 issue is aggregating on text fields. Fix mappings:

\`\`\`json
// Current (BAD)
{
  "mappings": {
    "properties": {
      "product_name": { "type": "text" },
      "brand": { "type": "text" },
      "category": { "type": "text" }
    }
  }
}

// Corrected (GOOD)
{
  "mappings": {
    "properties": {
      "product_name": {
        "type": "text",              // For full-text search
        "fields": {
          "keyword": {               // For aggregations/sorting
            "type": "keyword"
          }
        }
      },
      "brand": { "type": "keyword" },      // Always exact value
      "category": { "type": "keyword" }    // Always exact value
    }
  }
}
\`\`\`

**Migration:**
\`\`\`bash
# Create new index with correct mappings
PUT /products_v2 { ... }

# Reindex
POST /_reindex
{
  "source": { "index": "products" },
  "dest": { "index": "products_v2" }
}

# Update queries to use .keyword
{
  "aggs": {
    "by_product": {
      "terms": { "field": "product_name.keyword" }  // Now uses doc_values!
    }
  }
}
\`\`\`

**Why This Fixes It:**
- \`keyword\` fields use doc_values (on-disk, memory-mapped)
- Only working set loaded into heap (not entire field)
- Reduces heap usage by 60-80%

**Fix 2: Optimize High-Cardinality Aggregations**

\`\`\`json
// Instead of this (tracks millions of buckets)
{
  "aggs": {
    "by_user": {
      "terms": {
        "field": "user_id",
        "size": 1000
      }
    }
  }
}

// Do this (composite agg for pagination)
{
  "aggs": {
    "users": {
      "composite": {
        "size": 100,
        "sources": [
          { "user": { "terms": { "field": "user_id" } } }
        ]
      }
    }
  }
}

// Or use cardinality (approximation)
{
  "aggs": {
    "unique_users": {
      "cardinality": {
        "field": "user_id",
        "precision_threshold": 10000
      }
    }
  }
}
\`\`\`

**Fix 3: Enable Eager Global Ordinals**

For frequently aggregated keyword fields:

\`\`\`json
{
  "mappings": {
    "properties": {
      "category": {
        "type": "keyword",
        "eager_global_ordinals": true  // Preload ordinals
      }
    }
  }
}
\`\`\`

**Trade-off:** Refresh is slower, but aggregations are much faster and use less heap.

**Long-Term Architecture (Weeks 2-8):**

**Architecture 1: Separate Aggregation-Heavy Workloads**

Create dedicated cluster for analytics:

\`\`\`
Production Cluster (Search):
- Optimized for low-latency queries
- Smaller heap (16GB sufficient)
- Fast SSDs
- 10 nodes

Analytics Cluster (Aggregations):
- Optimized for aggregations
- Larger heap (31GB max, 64GB RAM)
- More CPU cores
- 5 nodes
- Data replicated from production
\`\`\`

**Replication Strategy:**
\`\`\`bash
# Use cross-cluster replication or Logstash
logstash -f replicate.conf

# Or snapshot/restore
POST /_snapshot/my_backup/snapshot_1/_restore
\`\`\`

**Architecture 2: Increase Node Resources**

Current: 10 nodes × 32GB RAM (16GB heap) = 160GB total heap

Recommended: Increase heap to 31GB (max before compressed oops):
- 10 nodes × 64GB RAM (31GB heap) = 310GB total heap
- 94% more heap capacity
- Or add 5 more nodes

**Rule:** Heap should be 50% of RAM, max 31GB

**Architecture 3: Implement Data Tiering**

\`\`\`
Hot tier (recent data, frequent aggregations):
- 5 nodes × 64GB RAM (31GB heap)
- Fast SSDs
- Higher CPU

Warm tier (historical data, fewer aggregations):
- 8 nodes × 32GB RAM (16GB heap)
- Regular SSDs
- Lower CPU

Cold tier (archived data, rare aggregations):
- 12 nodes × 16GB RAM (8GB heap)
- HDDs
- Searchable snapshots
\`\`\`

**Best Practices Implementation:**

**1. Monitoring and Alerting:**

\`\`\`javascript
// Alert rules
if (heap_usage > 75%) {
  alert("WARNING: Heap usage high");
}

if (gc_time > 1000ms) {
  alert("CRITICAL: Long GC pause");
}

if (circuit_breaker_trips > 10/hour) {
  alert("CRITICAL: Circuit breakers tripping");
}
\`\`\`

**2. Query Governance:**

\`\`\`javascript
// API layer validates queries
function validateQuery (query) {
  // Reject aggregations on text fields
  if (hasTextFieldAgg (query)) {
    throw new Error("Cannot aggregate on text fields");
  }
  
  // Limit cardinality
  if (getAggCardinality (query) > 10000) {
    throw new Error("Aggregation cardinality too high");
  }
  
  // Require query timeout
  if (!query.timeout) {
    query.timeout = "30s";
  }
  
  return query;
}
\`\`\`

**3. Regular Maintenance:**

\`\`\`bash
# Weekly force merge (off-peak)
POST /products-*/_forcemerge?max_num_segments=1

# Daily cache clearing (if needed)
POST /_cache/clear

# Monitor heap daily
GET /_nodes/stats/jvm
\`\`\`

**Expected Results:**

**Before:**
- Heap usage: 85-90%
- GC pause: 5-10 seconds
- Query timeout rate: 15%
- P95 latency: 5 seconds

**After Immediate Fixes:**
- Heap usage: 70-75% (fielddata cleared)
- GC pause: 1-2 seconds
- Query timeout rate: 5%
- P95 latency: 1 second

**After Mapping Fixes:**
- Heap usage: 40-50% (using doc_values)
- GC pause: 200-500ms
- Query timeout rate: <1%
- P95 latency: 300ms

**After Architecture Changes:**
- Heap usage: 30-40% (more capacity)
- GC pause: <100ms
- Query timeout rate: 0%
- P95 latency: 100ms

**Summary:**

The root cause is **aggregating on text fields** (loads entire inverted index into heap). The solution is **reindex with correct mappings** (use keyword fields with doc_values). Long-term, **separate workloads**, **increase capacity**, and **implement query governance** to prevent recurrence.

**Priority:** This is a P0 incident. Implement immediate fixes within 4 hours, mapping fixes within 48 hours, and architectural changes within 4 weeks.`,
    keyPoints: [
      'High heap usage usually caused by fielddata (aggregating on text fields)',
      'Text fields should never be used for aggregations/sorting; use keyword fields with doc_values',
      'Doc_values store data on disk (memory-mapped) instead of heap, reducing heap pressure by 60-80%',
      'High-cardinality aggregations require composite aggregations or cardinality approximations',
      'Long GC pauses indicate heap thrashing; increase heap size (max 31GB) or reduce heap pressure',
      'Separate search and analytics workloads onto different clusters for optimal performance',
    ],
  },
];
