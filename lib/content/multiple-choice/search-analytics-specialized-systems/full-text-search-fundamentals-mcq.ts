import { Quiz } from '@/lib/types';

const fullTextSearchFundamentalsMCQ: Quiz = {
  id: 'full-text-search-fundamentals-mcq',
  title: 'Full-Text Search Fundamentals - Multiple Choice Questions',
  questions: [
    {
      id: 'fts-mcq-1',
      type: 'multiple-choice',
      question:
        'You have a search index with 1 million documents. A term "elasticsearch" appears in 100 documents, while "the" appears in 900,000 documents. According to TF-IDF principles, why would a search for "elasticsearch" likely return more relevant results than a search for "the"?',
      options: [
        'Because "elasticsearch" is a longer word and gets higher term frequency scores',
        'Because "elasticsearch" has higher inverse document frequency (IDF), making it more discriminative',
        'Because "the" is automatically removed as a stop word in all search engines',
        'Because "elasticsearch" has more characters, which increases its TF-IDF score',
      ],
      correctAnswer: 1,
      explanation:
        'The inverse document frequency (IDF) component of TF-IDF gives higher scores to terms that appear in fewer documents. Since "elasticsearch" appears in only 100 documents (IDF = log(1,000,000/100) ≈ 9.2) versus "the" appearing in 900,000 documents (IDF = log(1,000,000/900,000) ≈ 0.1), "elasticsearch" has a much higher IDF and is more discriminative. This is the core principle of IDF: rare terms are more valuable for distinguishing relevant documents. While "the" is often a stop word, the fundamental reason for the difference is the IDF calculation, not stop word removal.',
    },
    {
      id: 'fts-mcq-2',
      type: 'multiple-choice',
      question:
        'Your e-commerce search application uses stemming in its analysis pipeline. A user searches for "running shoes" and the query returns products with "run", "runs", "runner", and "running" in their descriptions. However, users are now complaining that searches for "bass guitar" also return results for "base guitar" and "basic guitar techniques". What is the root cause and best solution?',
      options: [
        'The stemming algorithm is working correctly; the issue is that users need to use exact phrase matching with quotes',
        'Over-aggressive stemming is reducing "bass" and "base" to the same stem "bas". Use a less aggressive stemmer or exclude certain terms from stemming',
        'The inverted index is corrupted and needs to be rebuilt with proper analysis',
        'The TF-IDF scoring is too low; increase the term frequency weighting to prioritize exact matches',
      ],
      correctAnswer: 1,
      explanation:
        'This is a classic problem with over-aggressive stemming. Many stemming algorithms reduce "bass", "base", and "basic" to the same stem ("bas"), causing false matches. The solution is to either: (1) use a less aggressive stemmer (like Porter instead of Snowball), (2) use lemmatization instead of stemming (which understands context), or (3) maintain a protected word list that excludes certain domain-specific terms from stemming. Simply using phrase matching wouldn\'t solve the underlying issue, and TF-IDF weighting wouldn\'t prevent the false matches since the terms have already been reduced to the same stem during analysis.',
    },
    {
      id: 'fts-mcq-3',
      type: 'multiple-choice',
      question:
        'You need to implement autocomplete functionality that works as users type. Which indexing strategy would provide the best balance of query performance and storage for prefix matching queries like "elast" matching "elasticsearch"?',
      options: [
        'Use wildcard queries (elast*) at query time on standard analyzed text',
        'Index using edge n-grams with min_gram=3 and max_gram=15, enabling O(1) term lookup for prefixes',
        'Use full n-grams (trigrams/bigrams) to enable any substring matching',
        'Store all possible permutations of each word in the inverted index',
      ],
      correctAnswer: 1,
      explanation:
        'Edge n-grams are the optimal solution for autocomplete/prefix matching. They pre-compute all prefixes at index time (e.g., "ela", "elas", "elast", "elastic", "elastics"...), converting expensive wildcard queries into fast exact term lookups in the inverted index. This provides O(1) query time complexity instead of O(n) for wildcard scanning. While this increases index size (typically 2-3x for the field), the query performance improvement (10-100x faster) makes it worthwhile for user-facing autocomplete. Full n-grams provide substring matching but are more expensive (3-5x storage) and unnecessary for prefix-only matching. Wildcard queries are slow because they require scanning the term dictionary. Storing all permutations would be extremely wasteful.',
    },
    {
      id: 'fts-mcq-4',
      type: 'multiple-choice',
      question:
        'An inverted index for a document collection contains the following entry: Term "distributed" → [Doc3: freq=5, positions=[2,15,42,89,103], Doc7: freq=2, positions=[5,67], Doc12: freq=1, position=[34]]. A user queries for the exact phrase "distributed systems". Which documents could potentially match this phrase query, and what additional information would be needed to verify?',
      options: [
        'Only Doc3 matches because it has the highest frequency of the term "distributed"',
        'All three documents (3, 7, 12) could potentially match, but we need to check if "systems" appears immediately after each "distributed" position (position+1)',
        'Doc7 and Doc12 match because shorter documents are more relevant in phrase queries',
        'None can match because phrase queries require both terms to have the same frequency',
      ],
      correctAnswer: 1,
      explanation:
        'For a phrase query like "distributed systems", the search engine needs to verify that "systems" appears immediately after "distributed" (at position+1). All three documents could potentially match, but we need to check the positions. For Doc3, we\'d check if "systems" appears at positions [3, 16, 43, 90, 104]. For Doc7, we\'d check positions [6, 68]. For Doc12, we\'d check position [35]. The positional data in the inverted index enables this efficient verification. Simply having high frequency doesn\'t guarantee a phrase match—the terms must be adjacent. This is why modern inverted indexes store positional information: it enables phrase queries, proximity queries, and better relevance scoring while only modestly increasing index size.',
    },
    {
      id: 'fts-mcq-5',
      type: 'multiple-choice',
      question:
        'Your search application is tracking quality metrics and shows: Precision = 85%, Recall = 45%, and users are frequently refining their searches. What does this indicate, and what would be the most effective strategy to improve the search experience?',
      options: [
        'Search quality is excellent (85% precision is high); the low recall is acceptable because users only look at top results anyway',
        'The search is too restrictive and missing many relevant results (low recall). Implement query expansion using synonyms, fuzzy matching, or relaxing filters to increase recall',
        'High precision with low recall indicates the index is corrupted. Rebuild the index with proper analysis',
        'The metrics indicate perfect search quality. Focus optimization efforts elsewhere',
      ],
      correctAnswer: 1,
      explanation:
        'Precision = 85% means 85% of returned results are relevant (good), but Recall = 45% means the search is only finding 45% of all relevant documents (problematic). This pattern—high precision, low recall—indicates the search is too restrictive. Users refining their searches supports this: they aren\'t finding what they want in initial results. Solutions to increase recall include: (1) query expansion with synonyms ("laptop" → "notebook computer"), (2) fuzzy matching to handle typos, (3) stemming to match variations, (4) relaxing boolean logic (OR instead of AND), or (5) removing overly restrictive filters. The challenge is increasing recall without significantly hurting precision. While it\'s true users mainly look at top results, they should be finding what they need without multiple refinements. The 85% precision leaves room to trade off some precision for better recall.',
    },
  ],
};

export default fullTextSearchFundamentalsMCQ;
