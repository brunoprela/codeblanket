import { Quiz } from '@/lib/types';

const searchOptimizationMCQ: Quiz = {
  id: 'search-optimization-mcq',
  title: 'Search Optimization - Multiple Choice Questions',
  questions: [
    {
      id: 'search-opt-mcq-1',
      type: 'multiple-choice',
      question:
        'You have an e-commerce index where users frequently filter by "category" and "in_stock" status. These filters don\'t affect relevance scoring—they simply exclude products. Your query currently uses "must" clauses for these conditions. What is the most effective optimization?',
      options: [
        'Move both conditions to "should" clauses to make them optional',
        'Move both conditions to "filter" clauses because filters are cached and don\'t calculate scores',
        'Create separate indices for each category to avoid filtering',
        'Use "must_not" clauses instead because they\'re faster than "must"',
      ],
      correctAnswer: 1,
      explanation:
        'Filter clauses are optimal for yes/no conditions that don\'t affect relevance scoring. Unlike "must" clauses, filters: (1) don\'t calculate relevance scores (faster), (2) are automatically cached by Elasticsearch (subsequent queries with same filters are extremely fast), and (3) use bitsets which are very efficient for set operations. In this scenario, filtering by category="electronics" and in_stock=true doesn\'t need scoring—you either match or you don\'t. The query cache can store these filter results and reuse them across many queries. "Should" clauses would make conditions optional (wrong), separate indices add complexity (overkill), and "must_not" is for exclusion, not filtering by specific values.',
    },
    {
      id: 'search-opt-mcq-2',
      type: 'multiple-choice',
      question:
        'Your application allows users to paginate through search results. Users are complaining that page 500 (offset 50,000) takes 10 seconds to load, while page 1 is instant. The query is: GET /products/_search?from=50000&size=100. What is causing the slowness and what is the best solution?',
      options: [
        'The index is too large; implement sharding to distribute the load',
        'Deep pagination requires the coordinating node to sort results from all shards (e.g., 50,100 results per shard) just to return 100 results. Use the search_after parameter instead',
        'Increase the index.max_result_window setting to handle deeper pagination',
        'Add more replica shards to distribute the query load across more nodes',
      ],
      correctAnswer: 1,
      explanation:
        "Deep pagination with from/size is expensive because Elasticsearch must: (1) fetch top (from + size) results from EACH shard, (2) send all results to coordinating node, (3) sort ALL results globally, (4) return only the requested page. For from=50,000&size=100 across 10 shards, the coordinating node must sort 501,000 results (50,100 × 10 shards) to return just 100 results! This is wasteful. The search_after parameter solves this by using a cursor-based approach: it only fetches results after a specific sort value, providing constant performance regardless of page depth. Increasing index.max_result_window just allows the problem query without fixing it. More shards/replicas don't address the fundamental algorithmic issue of deep pagination.",
    },
    {
      id: 'search-opt-mcq-3',
      type: 'multiple-choice',
      question:
        'You implement autocomplete using edge n-grams (min_gram=2, max_gram=15) on a "product_name" field. Your index size has increased from 100GB to 280GB. Query performance is good (15ms), but the storage cost is becoming problematic. Which approach would reduce index size while maintaining acceptable autocomplete performance?',
      options: [
        'Use completion suggester type instead, which stores suggestions in an in-memory FST structure with ~20-30% storage overhead instead of 180%',
        'Reduce max_gram to 8 to decrease the number of generated tokens',
        'Use the scroll API to paginate autocomplete results',
        'Enable best_compression codec to reduce index size by half',
      ],
      correctAnswer: 0,
      explanation:
        'The completion suggester is purpose-built for autocomplete and uses a Finite State Transducer (FST) stored in memory, which is much more space-efficient than edge n-grams. Edge n-grams create many tokens: "elasticsearch" with min=2, max=15 creates ["el", "ela", "elas", "elast", "elasti", "elastic", "elastics", "elasticse", "elasticsea", "elasticsear", "elasticsearc", "elasticsearch"] = 12 tokens! For a 5-word product name, this generates 40-60 tokens. The completion suggester stores just the inputs you specify (typically 3-5 variants) with minimal overhead, increasing index size by only 20-30%. While reducing max_gram helps somewhat, it limits autocomplete to 8 characters. Compression helps but doesn\'t solve the fundamental issue. The completion suggester is also faster (1-5ms vs 15ms) because it\'s in-memory. For autocomplete specifically, completion suggester is the better choice.',
    },
    {
      id: 'search-opt-mcq-4',
      type: 'multiple-choice',
      question:
        'Your query performs aggregations on a "user_id" field that has 10 million unique values. Queries are slow (5-10 seconds) and you\'re seeing high heap usage. The field is correctly mapped as "keyword". What is the most effective optimization for this specific scenario?',
      options: [
        'Enable eager_global_ordinals on the user_id field to preload ordinals into memory',
        'Use a cardinality aggregation instead of terms aggregation if you only need unique count, or use composite aggregation for paginating through buckets',
        'Change user_id field type from keyword to integer for better performance',
        'Increase the size parameter in the terms aggregation to return more results',
      ],
      correctAnswer: 1,
      explanation:
        "Aggregating on a field with 10 million unique values creates 10 million buckets in memory, which is extremely expensive. Terms aggregations must track all buckets, compute counts, and return the top N. This causes: (1) high heap usage (potentially 2-4GB just for this query), (2) slow performance (must process millions of buckets), (3) risk of circuit breaker trips. The solution depends on your use case: If you only need the unique count (how many users?), use cardinality aggregation which uses HyperLogLog approximation and uses ~40KB instead of 4GB! If you need to iterate through all users, use composite aggregation which paginates through buckets without loading all into memory at once. Eager global ordinals helps with frequently aggregated fields but doesn't solve the fundamental issue of high cardinality. Changing to integer doesn't help with cardinality. Increasing size makes it worse (more results to track).",
    },
    {
      id: 'search-opt-mcq-5',
      type: 'multiple-choice',
      question:
        'You\'re optimizing a multi-field query that searches across "title", "description", and "tags". Currently: {"multi_match": {"query": "laptop", "fields": ["title", "description", "tags"]}}. You want title matches to be much more important than description, and tags to be more important than description but less than title. How would you implement this?',
      options: [
        'Use separate match queries for each field and manually multiply scores in your application',
        'Use field boosting: "fields": ["title^3", "tags^2", "description^1"] to make title 3x more important',
        'Create a function_score query that checks which field matched and applies weights',
        'Index the same content into three different indices (title_index, description_index, tags_index) and query with different priorities',
      ],
      correctAnswer: 1,
      explanation:
        'Field boosting with the caret (^) syntax is the standard and most efficient way to weight fields differently in Elasticsearch. The syntax "title^3" multiplies the relevance score from the title field by 3, "tags^2" multiplies by 2, and "description^1" (or just "description") uses the base score. This is built into Elasticsearch\'s scoring algorithm and is very efficient—it happens during the scoring phase without additional overhead. In the query {"multi_match": {"query": "laptop", "fields": ["title^3", "tags^2", "description"]}}, a document with "laptop" in the title will score 3x higher than the same document with "laptop" only in the description (all else being equal). While function_score can achieve similar results, it\'s more complex and has more overhead. Manually manipulating scores in the application is inefficient and bypasses Elasticsearch\'s optimized scoring. Separate indices would be a massive architectural complexity for something that field boosting handles elegantly.',
    },
  ],
};

export default searchOptimizationMCQ;
