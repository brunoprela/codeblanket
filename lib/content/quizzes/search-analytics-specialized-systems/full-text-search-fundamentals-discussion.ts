import { QuizQuestion } from '@/lib/types';

export const fullTextSearchFundamentalsDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'fts-discussion-1',
    question:
      'You\'re building search functionality for a large e-commerce platform with millions of products. Users search for products using natural language queries like "red running shoes size 10" or product codes like "ABC-123-XL". How would you design your text analysis pipeline to handle both types of queries effectively? Discuss the challenges and trade-offs of trying to handle both structured and unstructured queries in a single search interface.',
    sampleAnswer: `To handle both natural language and structured queries effectively, I would implement a **multi-field indexing strategy** with different analyzers:

**For natural language search (product titles, descriptions):**
- Use standard analyzer with English language settings
- Apply stemming to match "running" with "run"
- Remove stop words but carefully consider words like "size" which could be important
- Implement synonym expansion for color variations ("red" → "crimson", "scarlet")
- Use edge n-grams for autocomplete functionality
- Boost title matches higher than description matches (3x-5x multiplier)

**For structured data (product codes, SKUs, exact sizes):**
- Use keyword analyzer for exact matching
- Index product codes with no analysis
- Create separate fields for numeric attributes (size, price) with proper data types
- Index sizes both as keywords ("10") and as numeric values for range queries

**Challenges and trade-offs:**

1. **Query ambiguity**: When a user types "10", do they mean size 10 or $10 or 10-pack? Solution: Multi-match query with different field boosts
2. **Analysis conflicts**: Aggressive stemming can break product codes ("ABC-123-XL" → "abc 123 xl"). Solution: Index to multiple fields with different analysis
3. **Performance**: Multiple analyzers and complex queries increase index size and query latency. Solution: Use filters for exact matches (faster, cacheable)
4. **Relevance tuning**: Natural language and structured matches have different scoring characteristics. Solution: Function score query combining text relevance with business logic (inventory, sales rank)

**Implementation approach:**
\`\`\`
Product document:
{
  "title": "Nike Red Running Shoes",           // analyzed: standard
  "title.exact": "Nike Red Running Shoes",     // keyword: exact match
  "title.autocomplete": "Nike Red Running...", // edge n-grams
  "sku": "ABC-123-XL",                         // keyword: exact only
  "size": "10",                                // keyword + numeric
  "description": "...",                        // analyzed: standard
  "category": "athletic/running",              // keyword: filters
  "in_stock": true                             // boolean: filters
}
\`\`\`

Query strategy:
- Parse query to detect product codes (regex patterns)
- If detected, boost SKU field heavily (10x)
- Otherwise, use multi-match across title and description
- Apply filters for category, stock, size as needed
- Use function score to incorporate business metrics

This approach provides flexibility to handle diverse query types while maintaining relevance and performance.`,
    keyPoints: [
      'Multi-field indexing with different analyzers for different data types',
      'Use keyword analyzers for exact matching of structured data (SKUs, codes)',
      'Apply standard analyzers with stemming for natural language fields',
      'Query ambiguity requires careful field boosting and business logic',
      'Performance trade-offs between index size, query complexity, and relevance',
    ],
  },
  {
    id: 'fts-discussion-2',
    question:
      'A user searches for "machine learning engineer" and you have two documents: Document A contains "machine learning engineer" as an exact phrase 10 times, while Document B contains "machine" 20 times, "learning" 15 times, and "engineer" 25 times scattered throughout a much longer document, but never together as a phrase. Using your knowledge of TF-IDF and BM25 scoring, explain which document should rank higher and why. What modifications to the scoring algorithm would you implement to improve relevance for this scenario?',
    sampleAnswer: `**Which should rank higher?** Document A should rank higher because it contains the exact phrase "machine learning engineer", which is a much stronger signal of relevance than scattered individual terms.

**TF-IDF Analysis:**

For Document A (exact phrase 10 times):
- Each term appears 10 times in close proximity
- TF is moderate: sqrt(10) ≈ 3.16 per term
- Phrase bonus would apply
- Proximity score is high (terms adjacent)
- Document is likely focused and concise

For Document B (scattered, many occurrences):
- "machine": tf = sqrt(20) ≈ 4.47
- "learning": tf = sqrt(15) ≈ 3.87
- "engineer": tf = sqrt(25) = 5.0
- BUT terms are not near each other
- Document is longer, which dilutes relevance
- No phrase matching bonus

**Why Document A should rank higher:**

1. **Phrase matching**: The exact phrase appearing together is much more relevant than individual terms scattered throughout
2. **Document length normalization (BM25)**: Document B is longer, so BM25 will penalize it. The "b" parameter (typically 0.75) adjusts for document length
3. **Intent matching**: User\'s query is a specific phrase; Document A matches intent better
4. **Semantic coherence**: Adjacent terms in Document A suggest focused content about the topic

**Potential scoring issues with basic TF-IDF:**

Basic TF-IDF might actually rank Document B higher because:
- Raw term frequency is higher in Document B
- TF-IDF alone doesn't consider proximity or phrase matching
- This would be a poor user experience

**Modifications to improve relevance:**

1. **Phrase matching boost (critical)**
   - Exact phrase match gets significant multiplier (2x-5x)
   - Use positional data from inverted index
   - Slop queries for near-matches: "machine learning engineer"~2 (within 2 positions)

2. **Proximity scoring**
   - Terms closer together score higher
   - Calculate span (distance between first and last query term)
   - Shorter spans get exponential boost

3. **Field-length normalization (BM25's key feature)**
   - BM25's "b" parameter (0.75) penalizes longer documents
   - Formula considers avgFieldLength vs actual field length
   - Prevents length bias

4. **Coordinated query term bonus**
   - Documents containing ALL query terms score higher
   - More important than high frequency of individual terms

5. **Semantic scoring**
   - Consider context windows around terms
   - Use position-aware features
   - Machine learning models (Learning to Rank) can learn phrase importance

**Implementation in Elasticsearch:**
\`\`\`json
{
  "query": {
    "bool": {
      "should": [
        {
          "match_phrase": {
            "content": {
              "query": "machine learning engineer",
              "boost": 5.0          // Heavy boost for exact phrase
            }
          }
        },
        {
          "match_phrase": {
            "content": {
              "query": "machine learning engineer",
              "slop": 2,
              "boost": 3.0          // Moderate boost for near-match
            }
          }
        },
        {
          "match": {
            "content": {
              "query": "machine learning engineer",
              "boost": 1.0          // Baseline for scattered terms
            }
          }
        }
      ]
    }
  }
}
\`\`\`

**Result**: Document A would score much higher due to phrase match boost, while Document B only gets credit for term matches. This aligns with user intent and provides better search experience.

Modern search engines like Elasticsearch (using BM25 + phrase boosting) would correctly rank Document A higher by default, but explicit phrase query configuration ensures optimal results.`,
    keyPoints: [
      'Exact phrase matching is more relevant than scattered term occurrences',
      'BM25 document length normalization prevents longer documents from dominating',
      'Proximity scoring ensures terms appearing together rank higher',
      'Phrase match boost (2x-5x) is critical for multi-word queries',
      'Modern relevance requires combining term frequency, proximity, and phrase matching',
    ],
  },
  {
    id: 'fts-discussion-3',
    question:
      'Your search application is experiencing slow query performance for wildcard queries like "elast*" and especially "e*rch". Users need this functionality for partial word matching. At the same time, you\'re seeing your index size has grown to 3x your original document size due to various optimizations. Discuss the engineering trade-offs between query flexibility, performance, and storage costs. What specific technical solutions would you propose to address both issues, and what would be the implementation considerations?',
    sampleAnswer: `This scenario involves a classic **three-way trade-off** between **query flexibility**, **performance**, and **storage costs**. Let\'s analyze the problem and solutions:

**Problem Analysis:**

1. **Why wildcards are slow:**
   - \`elast*\` requires scanning all terms starting with "elast" in the inverted index
   - \`*rch\` or \`e*rch\` requires scanning virtually ALL terms (no usable prefix)
   - No way to jump directly to matches like with exact term lookup
   - Each matching term requires fetching its postings list
   - Query time grows with number of matching terms

2. **Why index size ballooned:**
   - Edge n-grams for autocomplete (can be 5-10x original text)
   - Multiple analyzers (standard, keyword, stemmed) on same field
   - Positional data for phrase queries
   - Stored fields for highlighting
   - Doc values for sorting/aggregations

**Technical Solutions:**

**Solution 1: Edge N-Grams (Recommended for prefix matching)**

Instead of wildcard queries at query time, pre-compute prefixes at index time:

\`\`\`
Index "elasticsearch" with edge n-grams (min=3, max=10):
["ela", "elas", "elast", "elasti", "elastic", "elastics", "elasticse", "elasticsea"]
\`\`\`

Benefits:
- Query "elast" becomes exact term lookup (fast!)
- Query time: O(1) term lookup vs O(n) term scanning
- Works great for autocomplete and prefix matching

Cost:
- Index size increases 2-3x for this field
- Only works for prefix (start of word), not suffix

**Solution 2: N-Grams for Partial Matching**

For true substring matching (including \`*rch\` or \`e*rch\`):

\`\`\`
Index "search" with trigrams (n=3):
["sea", "ear", "arc", "rch"]

Query "*rch" finds all terms containing "rch"
\`\`\`

Benefits:
- True substring matching anywhere in word
- Fast query time (term lookups)

Cost:
- Index size increases 3-5x
- Can produce many false positives (need post-filtering)

**Solution 3: Hybrid Approach (Recommended)**

Balance flexibility, performance, and cost:

\`\`\`
Product document fields:
{
  "name": "Elasticsearch Guide",              // Standard analysis (baseline)
  "name.prefix": "Elasticsearch Guide",       // Edge n-grams (min=3, max=15)
  "name.keyword": "Elasticsearch Guide",      // Exact match, no n-grams
  "name.ngram": "Elasticsearch Guide"         // Full n-grams (optional, last resort)
}
\`\`\`

Query strategy:
1. If query starts with \`*\` → Use ngram field (expensive, last resort)
2. If query ends with \`*\` (prefix) → Use prefix field (fast)
3. Exact match → Use keyword field (fastest)
4. Regular search → Use standard field (analyzed)

**Solution 4: Query-Time Restrictions**

Practical limits to prevent abuse:
\`\`\`
- Minimum prefix length: "el*" OK, "e*" rejected
- Reject leading wildcards: "*search" rejected
- Maximum wildcard expansion: Stop after 1000 matching terms
- Query timeouts: 5 seconds hard limit
- Rate limiting: Heavy queries count more
\`\`\`

**Solution 5: Optimize Storage**

Address the 3x index size:

1. **Remove unnecessary data:**
   - Disable _source storage for large fields if not needed for highlighting
   - Use doc_values: false for fields not used in sorting/aggregations
   - Set store: false for fields not needed individually

2. **Consolidate analyzers:**
   - Review if you really need 3-4 analyzers per field
   - Combine similar use cases

3. **Compression:**
   - Enable best_compression codec (slower index/search, 30-50% smaller)
   - Use shorter field names (JSON overhead)

4. **Field-specific decisions:**
   - Long descriptions: Standard analysis only
   - Product names: Standard + prefix (autocomplete)
   - SKUs/codes: Keyword only (exact match)
   - Rarely searched fields: Don't index

**Implementation Considerations:**

1. **Migration strategy:**
   - Create new index with optimized mappings
   - Reindex with zero downtime using aliases
   - A/B test query performance
   - Monitor index size, query latency

2. **Cost-benefit analysis:**
   - Edge n-grams: +100% storage, 10x faster prefix queries → **Worth it for autocomplete**
   - Full n-grams: +300% storage, 10x faster substring queries → **Only if critical**
   - Positional data: +30% storage, required for phrase queries → **Depends on usage**

3. **Monitoring:**
   - Track query latency by query type
   - Monitor index size growth over time
   - Measure storage costs (S3, EBS)
   - Analyze slow query logs

4. **User experience:**
   - Minimum 3 characters before search triggers
   - Show "refine your search" for too many results
   - Autocomplete with edge n-grams
   - Full text for main search

**My Recommendation:**

For most applications:
- Use edge n-grams for prefix/autocomplete on key fields (product names)
- Disable or restrict true wildcard queries (*middle* or *end)
- If wildcard required, enforce minimum prefix length (3-4 chars)
- Remove unused analyzers and stored data
- Set query timeouts and monitoring

This provides 90% of desired functionality while keeping index size at ~1.5x (vs current 3x) and dramatically improving query performance for common patterns.

The key insight: **Move work from query time to index time** for common patterns (prefixes), but don't over-optimize for rare patterns (wildcards with no prefix).`,
    keyPoints: [
      'Wildcard queries are slow because they require term enumeration, not direct lookup',
      'Edge n-grams move work from query time to index time for prefix matching',
      'Trade-off: 2-3x index size increase for 10x query performance improvement',
      'Hybrid approach: Different field configurations for different query types',
      'Query restrictions (minimum length, timeouts) protect system from expensive queries',
      'Storage optimization: Remove unnecessary analyzers, disable _source when possible',
    ],
  },
];
