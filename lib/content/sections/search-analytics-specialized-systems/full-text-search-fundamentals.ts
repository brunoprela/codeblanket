import { ModuleSection } from '@/lib/types';

const fullTextSearchFundamentalsSection: ModuleSection = {
  id: 'full-text-search-fundamentals',
  title: 'Full-Text Search Fundamentals',
  content: `
# Full-Text Search Fundamentals

## Introduction

Full-text search is a technique for searching and retrieving documents that contain one or more words or phrases from a large collection of text documents. Unlike simple substring matching, full-text search engines understand natural language, rank results by relevance, and can handle complex queries efficiently at scale.

Search is a critical component of modern applications—from e-commerce product searches to enterprise document management, from social media content discovery to log analysis. Understanding how search engines work internally is essential for building scalable, performant search experiences.

## Why Full-Text Search Matters

Traditional database systems excel at exact matches and structured queries, but they struggle with:

- **Relevance Ranking**: Not all matches are equally important
- **Natural Language**: Users search with typos, synonyms, and variations
- **Performance at Scale**: Searching millions of documents in milliseconds
- **Complex Queries**: Boolean logic, phrase matching, proximity searches
- **Linguistic Understanding**: Stemming, stopwords, language-specific analysis

Full-text search engines solve these challenges using specialized data structures and algorithms designed specifically for text retrieval.

## The Inverted Index

The fundamental data structure powering full-text search is the **inverted index** (also called a postings list).

### How It Works

In a traditional "forward index" (like a book), you go from document → words. An inverted index reverses this: you go from word → documents.

**Example Documents:**
\`\`\`
Doc1: "The quick brown fox"
Doc2: "The lazy dog"
Doc3: "Quick brown dogs"
\`\`\`

**Inverted Index:**
\`\`\`
Term        → Document IDs
"quick"     → [1, 3]
"brown"     → [1, 3]
"fox"       → [1]
"lazy"      → [2]
"dog"       → [2]
"dogs"      → [3]
\`\`\`

### Components of an Inverted Index

1. **Dictionary (Lexicon)**: All unique terms in the corpus
2. **Postings Lists**: For each term, a list of documents containing that term
3. **Positional Information**: Optional data about where terms appear
4. **Term Frequency**: How many times a term appears in each document
5. **Document Metadata**: Document IDs, field information, stored fields

### Advanced Postings Format

Modern search engines store rich information in postings:

\`\`\`
Term: "search"
├── Doc 42
│   ├── Frequency: 3
│   ├── Positions: [10, 45, 103]
│   ├── Fields: [title, body]
│   └── Proximity data
├── Doc 89
│   ├── Frequency: 1
│   ├── Positions: [5]
│   └── Fields: [body]
\`\`\`

This enables:
- **Phrase queries**: "machine learning" (must be adjacent)
- **Proximity queries**: "machine NEAR/5 learning" (within 5 words)
- **Field boosting**: Title matches score higher than body matches

## Text Analysis Pipeline

Before text is indexed, it goes through an **analysis pipeline** that transforms raw text into searchable terms.

### Analysis Steps

1. **Character Filtering**
   - Remove HTML tags: \`<p>Hello</p>\` → \`Hello\`
   - Normalize whitespace
   - Handle special characters

2. **Tokenization**
   - Split text into tokens (words)
   - Handle punctuation, hyphens, etc.
   - Example: "user-friendly" → ["user", "friendly"] or ["user-friendly"]

3. **Token Filtering**
   - **Lowercase**: "Search" → "search" (case-insensitive matching)
   - **Stop words**: Remove "the", "a", "is" (language-specific)
   - **Stemming**: "running", "runs" → "run" (reduce to root form)
   - **Synonyms**: "quick" → ["quick", "fast", "rapid"]
   - **ASCII folding**: "café" → "cafe"

### Example Pipeline

\`\`\`
Input: "The QUICK brown foxes are RUNNING!"

Character Filter:
  → "The QUICK brown foxes are RUNNING"

Tokenizer:
  → ["The", "QUICK", "brown", "foxes", "are", "RUNNING"]

Lowercase Filter:
  → ["the", "quick", "brown", "foxes", "are", "running"]

Stop Word Filter:
  → ["quick", "brown", "foxes", "running"]

Stemmer:
  → ["quick", "brown", "fox", "run"]

Final Terms in Index: ["quick", "brown", "fox", "run"]
\`\`\`

Now a search for "run" will match documents containing "running", "runs", "ran", etc.

### Analyzers

Different fields may need different analysis:
- **Standard Analyzer**: General purpose, good for most use cases
- **Whitespace Analyzer**: Split on whitespace only
- **Simple Analyzer**: Lowercase and split on non-letters
- **Language Analyzers**: Language-specific stemming and stopwords
- **Keyword Analyzer**: No analysis, exact matching (for IDs, tags)

## Scoring and Relevance Ranking

When multiple documents match a query, how do we rank them? This is where **relevance scoring** comes in.

### TF-IDF (Term Frequency - Inverse Document Frequency)

The classic relevance algorithm has two components:

**Term Frequency (TF)**: How often does the term appear in this document?
- Intuition: If "machine learning" appears 10 times in a document, it's probably more relevant than if it appears once
- Formula: \`tf(t, d) = frequency of term t in document d\`
- Often normalized: \`tf(t, d) = sqrt(frequency)\` (diminishing returns)

**Inverse Document Frequency (IDF)**: How rare is this term across all documents?
- Intuition: "the" appears in every document (low IDF), but "elasticsearch" is rare (high IDF)
- Rare terms are more discriminative
- Formula: \`idf(t) = log(total_docs / docs_containing_term)\`

**TF-IDF Score**:
\`\`\`
score(t, d) = tf(t, d) × idf(t)
\`\`\`

### Example Calculation

Corpus: 1,000,000 documents

**Query**: "machine learning"

**Document A**:
- "machine" appears 5 times, appears in 100,000 docs
  - tf = sqrt(5) ≈ 2.24
  - idf = log(1,000,000 / 100,000) = log(10) ≈ 2.3
  - score = 2.24 × 2.3 ≈ 5.15

- "learning" appears 3 times, appears in 50,000 docs
  - tf = sqrt(3) ≈ 1.73
  - idf = log(1,000,000 / 50,000) = log(20) ≈ 3.0
  - score = 1.73 × 3.0 ≈ 5.19

**Total Score for Document A**: 5.15 + 5.19 = **10.34**

### BM25: Modern Relevance Algorithm

**BM25** (Best Matching 25) is an improved version of TF-IDF used by modern search engines including Elasticsearch and Lucene.

Key improvements:
1. **Term saturation**: After a point, more occurrences don't help much
2. **Document length normalization**: Penalize long documents
3. **Tunable parameters**: k1 (term saturation), b (length normalization)

BM25 is now the default in most search engines and generally outperforms TF-IDF.

### Additional Scoring Factors

- **Field boosting**: Title matches score higher than body
- **Phrase matching**: Exact phrase gets bonus
- **Proximity**: Closer terms score higher
- **Recency**: Newer documents score higher
- **Custom scoring**: Click-through rates, user preferences
- **Function score**: Combine text relevance with business logic

## Query Types

Full-text search engines support various query types:

### 1. Term Query
Match documents containing exact term (after analysis):
\`\`\`
term: "elasticsearch"
\`\`\`

### 2. Match Query
Analyze query text and match any/all terms:
\`\`\`
match: "distributed search engine"
  → Find docs with "distributed" OR "search" OR "engine"
\`\`\`

### 3. Phrase Query
Match exact phrase (terms in order):
\`\`\`
phrase: "distributed search engine"
  → Terms must be adjacent in this order
\`\`\`

### 4. Boolean Query
Combine queries with logic:
\`\`\`
must: ["elasticsearch", "distributed"]     // AND
should: ["fast", "scalable"]               // OR (boosts score)
must_not: ["slow"]                         // NOT
filter: [category: "databases"]            // Must match, no scoring
\`\`\`

### 5. Wildcard and Regex
Pattern matching:
\`\`\`
wildcard: "elast*"        → matches "elastic", "elasticsearch"
regex: "el.*ch"           → matches "elasticsearch", "elastic search"
\`\`\`

### 6. Fuzzy Query
Tolerate typos (Levenshtein distance):
\`\`\`
fuzzy: "elasticseerch" with distance=2
  → matches "elasticsearch"
\`\`\`

## Fuzzy Matching and Typo Tolerance

Users make typos. Search engines handle this with **fuzzy matching**.

### Edit Distance

**Levenshtein distance** measures how many single-character edits (insertions, deletions, substitutions) are needed to transform one word into another:

\`\`\`
"elasticseerch" → "elasticsearch"
  - Insert 'a' after 'e'
  - Distance: 1
\`\`\`

Most search engines support:
- Distance 0: Exact match
- Distance 1: 1 typo allowed (good for short words)
- Distance 2: 2 typos allowed (good for longer words)

### N-Grams for Fuzzy Search

Split terms into overlapping sequences:

\`\`\`
"search" with trigrams (n=3):
  → ["sea", "ear", "arc", "rch"]

"seerch" with trigrams:
  → ["see", "eer", "erc", "rch"]

Overlap: ["rch"] (1/4 trigrams match)
\`\`\`

N-grams enable:
- Fast fuzzy matching without edit distance calculation
- Autocomplete and "search-as-you-type"
- Partial word matching

### Phonetic Matching

Sound-based matching using algorithms like **Soundex** or **Metaphone**:
\`\`\`
"Smith" and "Smyth" → Same phonetic code
"Knight" and "Night" → Same phonetic code
\`\`\`

Useful for name searches and queries where spelling is uncertain.

## Search Quality Metrics

How do we measure if our search is good?

### Precision and Recall

**Precision**: Of the results returned, how many are relevant?
\`\`\`
Precision = Relevant Results Returned / Total Results Returned
\`\`\`

**Recall**: Of all relevant documents, how many did we find?
\`\`\`
Recall = Relevant Results Returned / Total Relevant Documents
\`\`\`

**Example**: Search returns 10 results, 7 are relevant, but there are 20 relevant docs total:
- Precision = 7/10 = 70%
- Recall = 7/20 = 35%

There's often a trade-off: Increasing recall (showing more results) may decrease precision (include less relevant results).

### Mean Average Precision (MAP)

Precision at different recall levels, averaged across queries. Industry standard metric.

### Mean Reciprocal Rank (MRR)

Average of 1/(rank of first relevant result):
\`\`\`
Query 1: First relevant result at position 1 → 1/1 = 1.0
Query 2: First relevant result at position 3 → 1/3 = 0.33
Query 3: First relevant result at position 2 → 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
\`\`\`

### Normalized Discounted Cumulative Gain (NDCG)

Measures ranking quality with graded relevance (not just binary relevant/not relevant):
- Highly relevant result at position 1: Great!
- Highly relevant result at position 10: Not as good
- Slightly relevant result at position 1: Okay

Industry standard for search quality.

### Business Metrics

- **Click-Through Rate (CTR)**: % of users who click a result
- **Session Success Rate**: % of sessions ending with engagement
- **Zero Results Rate**: % of queries returning no results
- **Average Results per Query**: Too few? Too many?
- **Query Refinement Rate**: % of users modifying their query

## Performance Considerations

### Index Size

Inverted indexes are typically:
- **30-50% of original document size** with minimal metadata
- **80-100% of original size** with full positional data
- Can be larger with ngrams, synonyms, multiple analyzers

### Query Performance

- **Simple term queries**: Microseconds to milliseconds
- **Complex boolean queries**: Can be slower with many clauses
- **Wildcard queries**: Can be expensive if prefix is short (e.g., \`*search\`)
- **Regex queries**: Most expensive, scan many terms

**Optimization strategies**:
- Use filters (cached, no scoring) when possible
- Limit wildcard prefix length
- Use edge n-grams instead of wildcards for autocomplete
- Cache common queries
- Use query-time boosting judiciously

### Index Updates

- **Real-time indexing**: Documents searchable in seconds
- **Near-real-time**: Small delay (1-5 seconds) for better performance
- **Batch indexing**: Update index periodically (minutes/hours)

Trade-off: Faster updates = more overhead, slower queries

## Common Mistakes

### 1. Over-Analysis
Using aggressive stemming can cause false matches:
\`\`\`
Stemming "news" → "new"
Query for "new features" matches "breaking news" ❌
\`\`\`

### 2. Ignoring Relevance Tuning
Default relevance scoring may not match business needs:
- E-commerce: Boost in-stock products, promoted items
- Content: Consider freshness, author authority
- Enterprise: Consider permissions, department

### 3. One-Size-Fits-All Analysis
Different fields need different analysis:
- **Product codes**: No analysis (exact match)
- **Descriptions**: Standard analysis
- **Autocomplete**: Edge n-grams
- **Names**: Phonetic analysis

### 4. Not Handling Zero Results
13-15% of searches return no results. Implement:
- Fuzzy fallback
- Synonym expansion
- Related searches
- "Did you mean?" suggestions

### 5. Ignoring Performance
Queries like \`*search*\` or complex nested queries can be slow:
- Test query performance with realistic data volumes
- Monitor slow queries
- Optimize or restrict expensive query types

## Best Practices

### 1. Design Your Analysis Pipeline
- Understand your data and queries
- Test different analyzers on sample data
- Use different analysis for indexing vs searching when needed

### 2. Implement Relevance Tuning
- Collect user feedback (clicks, conversions)
- A/B test ranking changes
- Boost important fields and recent documents
- Use business logic (inventory, margins)

### 3. Build Observability
- Monitor query latency (p50, p95, p99)
- Track zero results rate
- Log slow queries
- Measure relevance metrics (NDCG, CTR)

### 4. Plan for Scale
- Estimate index size and growth
- Shard appropriately
- Cache common queries
- Consider read replicas for query load

### 5. Provide Great UX
- Fast responses (<100ms ideal, <500ms acceptable)
- Autocomplete and suggestions
- Faceted navigation (filters)
- Spell correction
- Highlight matching terms
- "Did you mean?" for typos

## Interview Tips

When discussing search in interviews:

1. **Start with inverted indexes**: Show you understand the core data structure
2. **Discuss analysis**: Explain tokenization, stemming, how they affect results
3. **Cover relevance**: Mention TF-IDF or BM25, explain the intuition
4. **Mention scaling**: Sharding, replication, caching
5. **Consider the use case**: E-commerce vs logs vs documents have different needs
6. **Think about tradeoffs**: Precision vs recall, real-time vs performance
7. **Cover the full query lifecycle**: Analysis → query inverted index → score → rank → return

**Example question**: "How would you implement search for a e-commerce site?"

**Strong answer**: "I'd use a search engine like Elasticsearch. First, I'd index product data with fields like title, description, category, price. The title would get higher boost than description. I'd use standard analysis for text fields but keyword analysis for SKUs and categories. For relevance, beyond text matching, I'd boost in-stock items and consider factors like sales velocity and margins. I'd implement autocomplete using edge n-grams on the product title. For scale, I'd shard by product category and use replicas for query load. I'd track metrics like CTR and zero-results rate to continuously improve relevance."

## Summary

Full-text search is built on:
- **Inverted indexes** for efficient term lookup
- **Text analysis** to normalize and process text
- **Relevance scoring (TF-IDF, BM25)** to rank results
- **Fuzzy matching** for typo tolerance
- **Various query types** for different search needs
- **Performance optimizations** for scale

Modern search engines like Elasticsearch, Solr, and Algolia handle these complexities, but understanding the fundamentals helps you design better search experiences, debug issues, and make informed architecture decisions.
`,
};

export default fullTextSearchFundamentalsSection;
