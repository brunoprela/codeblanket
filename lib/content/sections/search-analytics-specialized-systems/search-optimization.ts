import { ModuleSection } from '@/lib/types';

const searchOptimizationSection: ModuleSection = {
  id: 'search-optimization',
  title: 'Search Optimization',
  content: `
# Search Optimization

## Introduction

Having a functional Elasticsearch cluster is just the beginning. To deliver exceptional search experiences at scale, you need to optimize for **relevance**, **performance**, and **cost**. This section covers advanced techniques for making your search fast, accurate, and cost-effective.

Search optimization is an iterative process involving:
- **Index design**: Structure data for efficient querying
- **Query optimization**: Write queries that execute fast
- **Relevance tuning**: Return the most useful results
- **Caching strategies**: Reduce redundant computation
- **Resource management**: Balance cost and performance

## Index Design Optimization

### Field Mapping Strategy

**Problem**: Dynamic mapping is convenient but suboptimal for production.

**Solution**: Explicit mappings tailored to your access patterns.

**Example: E-commerce Product**

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "keyword": {
            "type": "keyword"           // For sorting
          },
          "suggest": {
            "type": "completion"        // For autocomplete
          },
          "search_as_you_type": {
            "type": "search_as_you_type", // For instant search
            "max_shingle_size": 3
          }
        }
      },
      "sku": {
        "type": "keyword",              // Exact match only
        "doc_values": true,             // For sorting/aggs
        "store": false                  // Don't need separately
      },
      "price": {
        "type": "scaled_float",         // Efficient for currency
        "scaling_factor": 100           // 2 decimal places
      },
      "description": {
        "type": "text",
        "analyzer": "english",
        "index_options": "offsets",     // For highlighting
        "norms": false                  // Don't need length norm
      },
      "category": {
        "type": "keyword",              // For filtering, facets
        "eager_global_ordinals": true   // Faster aggregations
      },
      "tags": {
        "type": "keyword",
        "eager_global_ordinals": true
      },
      "created_at": {
        "type": "date",
        "format": "strict_date_time"
      },
      "in_stock": {
        "type": "boolean"
      },
      "reviews": {
        "type": "nested",               // Independent review queries
        "properties": {
          "rating": { "type": "integer" },
          "comment": { "type": "text" }
        }
      }
    }
  }
}
\`\`\`

**Key Optimizations:**1. **Multi-fields for different use cases**:
   - \`title\`: Full-text search
   - \`title.keyword\`: Exact match, sorting
   - \`title.suggest\`: Autocomplete
   - \`title.search_as_you_type\`: Instant search

2. **Appropriate data types**:
   - \`scaled_float\` for currency (more efficient than \`float\`)
   - \`keyword\` for exact values (tags, categories, IDs)
   - \`nested\` for objects that need independent querying

3. **Disable unnecessary features**:
   - \`norms: false\`: If you don't need length normalization scoring
   - \`doc_values: false\`: If you never sort or aggregate on field
   - \`store: false\`: If you don't need to retrieve separately
   - \`index: false\`: If you never search this field

4. **Performance tuning**:
   - \`eager_global_ordinals: true\`: Preload ordinals for faster aggregations on high-cardinality fields
   - \`index_options: "offsets"\`: Enable fast highlighting

### Disable _source for Large Documents

If you don't need the original document returned:

\`\`\`json
{
  "mappings": {
    "_source": {
      "enabled": false     // Saves 30-50% disk space
    },
    "properties": {
      "title": { "type": "text", "store": true },
      "price": { "type": "float", "store": true }
    }
  }
}
\`\`\`

**Trade-off**: Can't reindex, can't update, must \`store: true\` for fields you need to return.

**Use case**: Log aggregation where you only need metrics, not raw logs.

### Index Sorting

Pre-sort documents in index for faster range queries:

\`\`\`json
{
  "settings": {
    "index.sort.field": ["created_at", "price"],
    "index.sort.order": ["desc", "asc"]
  }
}
\`\`\`

**Benefits**:
- Faster range queries on sorted field
- Early termination for top N queries
- Cons: Slower indexing, can't change after index creation

## Shard Sizing and Distribution

### Optimal Shard Size

**Rule**: 10-50GB per shard is optimal.

**Why?**
- **Too small** (<10GB): Coordination overhead, too many shards
- **Too large** (>50GB): Slow recovery, difficult rebalancing

**Calculation for New Index:**

\`\`\`
Expected data: 500GB
Target shard size: 30GB
Primary shards needed: 500GB / 30GB ≈ 17 shards

Round to: 15 or 20 primary shards
\`\`\`

### Shard Allocation Awareness

Distribute shards across failure domains (racks, availability zones):

\`\`\`yaml
# elasticsearch.yml
node.attr.rack: rack1
cluster.routing.allocation.awareness.attributes: rack
\`\`\`

**Effect**: Elasticsearch ensures primary and replica shards are on different racks.

**Benefit**: Survive rack failure without data loss.

### Index Lifecycle Management (ILM)

Automate index transitions through lifecycle phases:

\`\`\`json
{
  "policy": "logs_policy",
  "phases": {
    "hot": {
      "actions": {
        "rollover": {
          "max_size": "50GB",
          "max_age": "1d"
        }
      }
    },
    "warm": {
      "min_age": "2d",
      "actions": {
        "forcemerge": {
          "max_num_segments": 1    // Optimize for read
        },
        "shrink": {
          "number_of_shards": 1    // Consolidate small shards
        }
      }
    },
    "cold": {
      "min_age": "7d",
      "actions": {
        "freeze": {}               // Rarely searched, minimal memory
      }
    },
    "delete": {
      "min_age": "30d",
      "actions": {
        "delete": {}
      }
    }
  }
}
\`\`\`

**hot → warm → cold → delete** lifecycle optimizes performance and cost.

## Query Optimization

### Use Filters Instead of Queries When Possible

**Queries**: Calculate relevance score, not cached
**Filters**: No scoring, cached, faster

\`\`\`json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "laptop" } }      // Scored
      ],
      "filter": [                               // Not scored, cached
        { "term": { "category": "electronics" } },
        { "range": { "price": { "lte": 1000 } } },
        { "term": { "in_stock": true } }
      ]
    }
  }
}
\`\`\`

**When to use filter**:
- Yes/no conditions (in_stock, category)
- Range queries (price, date)
- Term matching (exact values)

**When to use query**:
- Full-text search
- Relevance matters
- Need scoring

### Avoid Deep Pagination

**Problem**: \`GET /products/_search?from=10000&size=100\` is expensive.

**Why?** Coordinating node must:
1. Get top 10,100 results from each shard
2. Sort all results
3. Return only 10,000-10,100

For 5 shards: Must sort 50,500 results to return 100!

**Solutions:**

**Option 1: Search After (Recommended)**

\`\`\`json
// First request
{
  "size": 100,
  "sort": [{ "created_at": "desc" }, { "_id": "asc" }],
  "query": { "match_all": {} }
}

// Response includes last document sort values
// Next request
{
  "size": 100,
  "sort": [{ "created_at": "desc" }, { "_id": "asc" }],
  "search_after": ["2024-01-15T10:30:00", "product-123"],
  "query": { "match_all": {} }
}
\`\`\`

**Benefits**: Constant performance regardless of page depth.

**Option 2: Scroll API (For Batch Processing)**

\`\`\`json
// Initial request
POST /products/_search?scroll=5m
{
  "size": 1000,
  "query": { "match_all": {} }
}

// Returns scroll_id, use for subsequent requests
POST /_search/scroll
{
  "scroll": "5m",
  "scroll_id": "..."
}
\`\`\`

**Use case**: Exporting data, batch processing, not for user-facing pagination.

**Option 3: Limit Pagination**

\`\`\`json
{
  "settings": {
    "index.max_result_window": 1000  // Default: 10000
  }
}
\`\`\`

Reject deep pagination requests to protect cluster.

### Optimize Aggregations

**Problem**: Aggregations on high-cardinality fields are expensive.

**Example: Bad**

\`\`\`json
{
  "aggs": {
    "products_by_user": {
      "terms": {
        "field": "user_id",          // 10M unique users
        "size": 100
      }
    }
  }
}
\`\`\`

Must track 10M buckets in memory!

**Solution 1: Reduce Cardinality**

\`\`\`json
{
  "aggs": {
    "products_by_user": {
      "terms": {
        "field": "user_id",
        "size": 100,
        "execution_hint": "map",     // Use coordinating node memory
        "shard_size": 500            // Fetch more per shard
      }
    }
  }
}
\`\`\`

**Solution 2: Composite Aggregation**

For paginating through buckets:

\`\`\`json
{
  "aggs": {
    "my_buckets": {
      "composite": {
        "size": 1000,
        "sources": [
          { "category": { "terms": { "field": "category" } } },
          { "brand": { "terms": { "field": "brand" } } }
        ]
      }
    }
  }
}
\`\`\`

**Solution 3: Eager Global Ordinals**

\`\`\`json
{
  "mappings": {
    "properties": {
      "category": {
        "type": "keyword",
        "eager_global_ordinals": true    // Preload at refresh
      }
    }
  }
}
\`\`\`

First aggregation is slow (builds ordinals), subsequent queries are fast (ordinals cached).

### Minimize Fields in _source

**Problem**: Returning large \`_source\` wastes network bandwidth.

**Solution: Source Filtering**

\`\`\`json
{
  "query": { "match": { "title": "laptop" } },
  "_source": ["title", "price", "sku"]     // Only return these fields
}
\`\`\`

Or exclude fields:

\`\`\`json
{
  "_source": {
    "excludes": ["description", "reviews"]
  }
}
\`\`\`

**Benefit**: 50-90% reduction in response size for documents with large fields.

## Relevance Tuning

### Field Boosting

Give more weight to certain fields:

\`\`\`json
{
  "query": {
    "multi_match": {
      "query": "elasticsearch guide",
      "fields": ["title^3", "description^1", "author^2"]
    }
  }
}
\`\`\`

- \`title^3\`: Title matches are 3x more important
- \`author^2\`: Author matches are 2x important
- \`description^1\`: Description is baseline

### Function Score Query

Combine text relevance with business logic:

\`\`\`json
{
  "query": {
    "function_score": {
      "query": {
        "match": { "title": "laptop" }
      },
      "functions": [
        {
          "filter": { "term": { "in_stock": true } },
          "weight": 2                              // 2x boost for in-stock
        },
        {
          "gauss": {
            "created_at": {
              "origin": "now",
              "scale": "30d",
              "decay": 0.5
            }
          }
        },
        {
          "field_value_factor": {
            "field": "sales_count",
            "modifier": "log1p",                   // log(1 + sales_count)
            "factor": 0.1
          }
        }
      ],
      "score_mode": "sum",                        // How to combine scores
      "boost_mode": "multiply"                    // How to combine with query score
    }
  }
}
\`\`\`

**Scoring Functions:**
- \`weight\`: Fixed boost
- \`field_value_factor\`: Use document field value
- \`gauss/exp/linear\`: Decay function (recency, distance)
- \`script_score\`: Custom scoring logic

### Minimum Should Match

Control how many terms must match:

\`\`\`json
{
  "query": {
    "match": {
      "title": {
        "query": "quick brown fox jumps",
        "minimum_should_match": "75%"     // 3 out of 4 terms
      }
    }
  }
}
\`\`\`

**Values**:
- Percentage: \`"75%"\`
- Fixed: \`3\`
- Conditional: \`"3<75%"\` (if 3 or fewer terms, all must match; otherwise 75%)

**Effect**: Reduces noise from docs matching only 1 term.

### Synonym Handling

Expand queries with synonyms:

\`\`\`json
// Create synonym filter
PUT /products
{
  "settings": {
    "analysis": {
      "filter": {
        "synonym_filter": {
          "type": "synonym",
          "synonyms": [
            "laptop, notebook, computer",
            "phone, mobile, smartphone",
            "tv, television"
          ]
        }
      },
      "analyzer": {
        "synonym_analyzer": {
          "tokenizer": "standard",
          "filter": ["lowercase", "synonym_filter"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "synonym_analyzer"
      }
    }
  }
}
\`\`\`

Now searching "notebook" also finds "laptop" and "computer".

**Trade-off**: Index size increases (more terms per doc).

## Caching Strategies

### Query Cache

Caches entire query results:

\`\`\`json
{
  "query": {
    "bool": {
      "must": { "match": { "title": "laptop" } },
      "filter": {
        "term": { "category": "electronics" }    // Cached
      }
    }
  }
}
\`\`\`

**Cached**: Filter clauses
**Not cached**: Scoring queries, queries with \`now\`, queries with scripts

**Cache size**: 10% of heap by default

**Invalidation**: Cache cleared on refresh (new data)

### Request Cache

Caches entire search request response:

\`\`\`json
GET /products/_search?request_cache=true
{
  "size": 0,                            // Only aggregations
  "aggs": {
    "popular_categories": {
      "terms": { "field": "category" }
    }
  }
}
\`\`\`

**Use case**: Dashboards, analytics queries that don't change often

**Invalidation**: On index refresh

**Enable by default**:
\`\`\`json
{
  "settings": {
    "index.requests.cache.enable": true
  }
}
\`\`\`

### Fielddata Cache

For sorting and aggregating on text fields (not recommended):

\`\`\`json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "fielddata": true       // DON'T DO THIS!
      }
    }
  }
}
\`\`\`

**Problem**: Loads entire inverted index into heap (expensive!)

**Solution**: Use \`keyword\` field for sorting/aggregations instead.

## Autocomplete and Suggestions

### Completion Suggester

Optimized for autocomplete:

\`\`\`json
// Mapping
{
  "mappings": {
    "properties": {
      "suggest": {
        "type": "completion"
      }
    }
  }
}

// Indexing
{
  "title": "Elasticsearch Guide",
  "suggest": {
    "input": ["Elasticsearch Guide", "ES Guide", "Elastic"],
    "weight": 10
  }
}

// Query
POST /products/_search
{
  "suggest": {
    "product-suggest": {
      "prefix": "elas",
      "completion": {
        "field": "suggest",
        "size": 5,
        "skip_duplicates": true
      }
    }
  }
}
\`\`\`

**Performance**: Sub-millisecond (in-memory FST data structure)

### Edge N-grams for Search-as-you-type

\`\`\`json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "autocomplete": {
          "tokenizer": "standard",
          "filter": ["lowercase", "autocomplete_filter"]
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
        "search_analyzer": "standard"    // Don't ngram the query!
      }
    }
  }
}
\`\`\`

**Indexing**: "laptop" → ["la", "lap", "lapt", "lapto", "laptop"]

**Searching**: "lap" → exact term "lap" matches indexed "lap"

### Term Suggester (Did You Mean?)

\`\`\`json
{
  "suggest": {
    "text": "elasticseerch",
    "term-suggestion": {
      "term": {
        "field": "title",
        "suggest_mode": "popular",     // Suggest popular corrections
        "min_word_length": 4
      }
    }
  }
}
\`\`\`

Returns: "elasticsearch" (edit distance = 1)

## Performance Monitoring

### Key Metrics to Track

**1. Query Latency**
\`\`\`
GET /_nodes/stats/indices/search
\`\`\`
- p50, p95, p99 latency
- Target: <100ms for p95

**2. Indexing Rate**
\`\`\`
GET /_nodes/stats/indices/indexing
\`\`\`
- Documents per second
- Indexing time

**3. Shard Stats**
\`\`\`
GET /_cat/shards?v&s=store:desc
\`\`\`
- Shard sizes
- Unassigned shards

**4. Heap Usage**
\`\`\`
GET /_nodes/stats/jvm
\`\`\`
- Heap utilization (target: <75%)
- GC frequency and duration

**5. Cache Hit Rates**
\`\`\`
GET /_nodes/stats/indices/query_cache
GET /_nodes/stats/indices/request_cache
\`\`\`
- Hit rate (target: >50%)
- Evictions (lower is better)

### Slow Query Log

Enable slow query logging:

\`\`\`json
{
  "settings": {
    "index.search.slowlog.threshold.query.warn": "10s",
    "index.search.slowlog.threshold.query.info": "5s",
    "index.search.slowlog.threshold.query.debug": "2s",
    "index.search.slowlog.threshold.fetch.warn": "1s"
  }
}
\`\`\`

Logs written to: \`/var/log/elasticsearch/[cluster]_index_search_slowlog.log\`

## Common Mistakes

### 1. Using Text Fields for Aggregations

\`\`\`json
// BAD
{
  "aggs": {
    "by_category": {
      "terms": { "field": "category" }    // "category" is text type
    }
  }
}
\`\`\`

**Error**: "Fielddata is disabled on text fields by default"

**Solution**: Use \`category.keyword\` or map as keyword type.

### 2. Not Using Filters for Exact Matches

\`\`\`json
// BAD: Scores unnecessarily
{
  "query": {
    "bool": {
      "must": [
        { "term": { "status": "active" } },
        { "term": { "in_stock": true } }
      ]
    }
  }
}

// GOOD: Filters are cached and faster
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "status": "active" } },
        { "term": { "in_stock": true } }
      ]
    }
  }
}
\`\`\`

### 3. Wildcard Queries on Large Fields

\`\`\`json
// VERY SLOW
{
  "query": {
    "wildcard": {
      "description": "*keyword*"
    }
  }
}
\`\`\`

**Problem**: Must scan ALL terms in field.

**Solution**: Use n-grams or prefix queries instead.

### 4. Over-Sharding

Creating 100 shards for 50GB of data:
- Coordination overhead
- Memory waste
- Slower queries

**Rule**: 10-50GB per shard

### 5. Deep Pagination

\`\`\`json
// VERY EXPENSIVE
GET /products/_search?from=100000&size=100
\`\`\`

**Solution**: Use search_after or scroll API.

## Best Practices Summary

### Index Design
- Use explicit mappings in production
- Disable unused features (\`norms\`, \`doc_values\`, \`store\`)
- Use appropriate data types (\`scaled_float\` for currency)
- Implement multi-fields for different use cases
- Optimize shard size: 10-50GB per shard

### Queries
- Use filters for exact matches (cached)
- Avoid deep pagination (use search_after)
- Minimize returned fields with \`_source\` filtering
- Use bulk API for indexing
- Implement query timeouts

### Relevance
- Field boosting for important fields
- Function score for business logic
- Minimum should match to reduce noise
- Synonyms for query expansion
- A/B test relevance changes

### Performance
- Monitor slow queries
- Track cache hit rates
- Keep heap usage <75%
- Use ILM for lifecycle management
- Separate hot/warm/cold data

### Caching
- Enable request cache for dashboards
- Leverage query cache for filters
- Avoid fielddata on text fields
- Cache frequently accessed data

## Interview Tips

When discussing search optimization:

1. **Start with the bottleneck**: Identify what's slow (queries, indexing, relevance)
2. **Match solution to problem**:
   - Slow queries → Better indexing, caching, fewer shards
   - Poor relevance → Field boosting, function scores, synonyms
   - High cost → ILM, better shard sizing, compression
3. **Discuss trade-offs**: Every optimization has trade-offs
   - Index size vs query speed (ngrams)
   - Indexing speed vs search speed (force merge)
   - Cost vs performance (hot/warm/cold)
4. **Mention specific features**: completion suggester, edge n-grams, function_score
5. **Show monitoring awareness**: Track latency, cache hit rates, slow queries

**Example question**: "Search queries are slow. How would you optimize?"

**Strong answer**: "First, I'd identify the bottleneck using slow query logs and metrics. If queries are scanning many shards, I'd check shard sizing (target 10-50GB). I'd move exact-match conditions from 'must' to 'filter' clauses since filters are cached. For high-cardinality aggregations, I'd enable eager_global_ordinals. I'd implement source filtering to reduce network overhead. For pagination beyond 10k results, I'd switch to search_after. I'd add field boosting to improve relevance (title^3, description^1). Finally, I'd enable request caching for dashboard queries and monitor cache hit rates. If the workload is read-heavy, I'd add replicas and distribute query load."

## Summary

Search optimization involves:
- **Efficient index design**: Right mappings, appropriate data types, optimal sharding
- **Query optimization**: Filters over queries, avoid deep pagination, source filtering
- **Relevance tuning**: Field boosting, function scores, minimum should match
- **Caching**: Query cache, request cache, proper invalidation
- **Monitoring**: Track latency, slow queries, heap usage, cache hit rates
- **Autocomplete**: Completion suggester, edge n-grams, term suggesters

The key is measuring, identifying bottlenecks, and applying targeted optimizations while understanding trade-offs.
`,
};

export default searchOptimizationSection;
