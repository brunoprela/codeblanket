export const reRankingRelevance = {
  title: 'Re-ranking & Relevance',
  content: `
# Re-ranking & Relevance

## Introduction

Initial retrieval with embeddings is just the first step. Re-ranking applies more sophisticated, computationally expensive models to reorder results and maximize relevance. This dramatically improves the quality of documents passed to your LLM.

In this comprehensive section, we'll explore cross-encoder models, re-ranking APIs, hybrid scoring strategies, and production-grade re-ranking systems that transform good search into great search.

## Why Re-ranking Matters

Vector similarity search is fast but has limitations:

### Initial Retrieval Problems

\`\`\`python
# Problem: Vector similarity alone misses nuances
query = "best practices for python async programming"
results = semantic_search(query, top_k=100)

# Results might include:
# 1. "Python async basics" (similar embedding)
# 2. "Python threading vs async" (mentions async)
# 3. "Best practices for Java async" (wrong language!)
# 4. "Advanced async patterns" (what you really want!)
\`\`\`

### Re-ranking Solution

\`\`\`python
# Solution: Use cross-encoder to re-rank
initial_results = semantic_search(query, top_k=100)
reranked = cross_encoder_rerank(query, initial_results, top_k=10)

# Now top result: "Advanced async patterns in Python" ✓
\`\`\`

### Benefits of Re-ranking

1. **Higher Precision**: More accurate top results
2. **Better Context**: LLM gets the most relevant documents
3. **Query Understanding**: Cross-encoders understand query-document relationships
4. **Faster Than Brute Force**: Only re-rank top candidates
5. **Significant Quality Boost**: Often 10-20% improvement in relevance

## Cross-Encoder Models

Cross-encoders jointly encode query and document together, capturing their relationship.

### Bi-Encoder vs Cross-Encoder

\`\`\`python
# Bi-Encoder (for initial retrieval)
query_emb = encode(query)      # Encode separately
doc_emb = encode(document)
similarity = cosine(query_emb, doc_emb)  # Compare vectors

# Cross-Encoder (for re-ranking)
combined = f"{query} [SEP] {document}"  # Encode together
relevance_score = model(combined)  # Direct relevance score
\`\`\`

**Trade-offs:**
- Bi-encoders: Fast, scalable, can pre-compute embeddings
- Cross-encoders: Slow, more accurate, must encode each pair

### Implementing Cross-Encoder Re-ranking

\`\`\`python
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class CrossEncoderReranker:
    """
    Re-rank search results using cross-encoder model.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder.
        
        Args:
            model_name: HuggingFace model name
                       Popular options:
                       - ms-marco-MiniLM-L-6-v2 (fast, good)
                       - ms-marco-MiniLM-L-12-v2 (slower, better)
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: Initial retrieved documents
            top_k: Number of top results to return
        
        Returns:
            Re-ranked documents with scores
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]


# Example usage
reranker = CrossEncoderReranker()

query = "How to use async/await in Python?"
initial_docs = [
    "Python async/await tutorial for beginners",
    "JavaScript async patterns",
    "Advanced Python asyncio patterns",
    "Python threading vs multiprocessing",
    "Async best practices in Python"
]

reranked = reranker.rerank(query, initial_docs, top_k=3)

print("Re-ranked Results:")
for doc, score in reranked:
    print(f"{score:.3f}: {doc}")
\`\`\`

**Output:**
\`\`\`
Re-ranked Results:
0.945: Advanced Python asyncio patterns
0.912: Async best practices in Python
0.887: Python async/await tutorial for beginners
\`\`\`

### Batch Re-ranking for Performance

\`\`\`python
import numpy as np

class BatchedCrossEncoderReranker:
    """
    Batched re-ranking for better performance.
    """
    
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Re-rank with batching for efficiency.
        """
        pairs = [[query, doc] for doc in documents]
        
        # Predict in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores)
        
        # Sort and return top K
        doc_score_pairs = list(zip(documents, all_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
\`\`\`

## Cohere Re-rank API

Cohere provides a managed re-ranking API that's production-ready:

\`\`\`python
import cohere
from typing import List, Dict

class CohereReranker:
    """
    Re-rank using Cohere's managed API.
    """
    
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        model: str = "rerank-english-v2.0"
    ) -> List[Dict]:
        """
        Re-rank documents using Cohere.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of results
            model: Cohere model (rerank-english-v2.0, rerank-multilingual-v2.0)
        
        Returns:
            Re-ranked results with scores
        """
        response = self.client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=top_k
        )
        
        results = []
        for result in response.results:
            results.append({
                "document": documents[result.index],
                "score": result.relevance_score,
                "index": result.index
            })
        
        return results


# Example usage
reranker = CohereReranker(api_key="your-api-key")

query = "machine learning best practices"
docs = [
    "ML best practices for production",
    "Deep learning tutorial",
    "ML model optimization techniques",
    "Python programming basics"
]

reranked = reranker.rerank(query, docs, top_k=3)

for result in reranked:
    print(f"{result['score']:.3f}: {result['document']}")
\`\`\`

**Cohere Benefits:**
- ✅ No model hosting required
- ✅ Excellent quality
- ✅ Multilingual support
- ✅ Production-ready scale

**Pricing:** ~$2 per 1,000 searches (check current pricing)

## Hybrid Scoring Strategies

Combine multiple signals for optimal relevance:

\`\`\`python
from typing import List, Dict, Callable
import numpy as np

class HybridScorer:
    """
    Combine multiple relevance signals.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize hybrid scorer.
        
        Args:
            weights: Weight for each scoring method
        """
        self.weights = weights or {
            "semantic": 0.4,
            "cross_encoder": 0.4,
            "bm25": 0.2
        }
    
    def score(
        self,
        query: str,
        document: str,
        semantic_score: float,
        cross_encoder_score: float,
        bm25_score: float
    ) -> float:
        """
        Calculate hybrid score.
        
        Args:
            query: Search query
            document: Document text
            semantic_score: Embedding similarity (0-1)
            cross_encoder_score: Cross-encoder score (0-1)
            bm25_score: BM25 score (normalized to 0-1)
        
        Returns:
            Combined score
        """
        return (
            self.weights["semantic"] * semantic_score +
            self.weights["cross_encoder"] * cross_encoder_score +
            self.weights["bm25"] * bm25_score
        )
    
    def rank_documents(
        self,
        query: str,
        documents: List[str],
        semantic_scores: List[float],
        cross_encoder_scores: List[float],
        bm25_scores: List[float],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Rank documents using hybrid scoring.
        """
        combined_scores = [
            self.score(
                query, doc,
                sem_score, ce_score, bm25_score
            )
            for doc, sem_score, ce_score, bm25_score in zip(
                documents,
                semantic_scores,
                cross_encoder_scores,
                bm25_scores
            )
        ]
        
        # Sort by combined score
        doc_score_pairs = list(zip(documents, combined_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
\`\`\`

## BM25 + Dense Re-ranking

Combine sparse (BM25) and dense (embedding) retrieval:

\`\`\`python
from rank_bm25 import BM25Okapi
import numpy as np

class BM25DenseReranker:
    """
    Combine BM25 keyword search with dense retrieval.
    """
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
    
    def index_documents(self, documents: List[str]):
        """
        Index documents for BM25.
        
        Args:
            documents: Document collection
        """
        self.documents = documents
        
        # Tokenize documents
        self.tokenized_docs = [
            doc.lower().split() for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def get_bm25_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for query.
        
        Args:
            query: Search query
        
        Returns:
            BM25 scores (normalized)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize to 0-1
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def rerank(
        self,
        query: str,
        dense_scores: np.ndarray,
        alpha: float = 0.5,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Combine BM25 and dense scores.
        
        Args:
            query: Search query
            dense_scores: Semantic similarity scores
            alpha: Weight for dense scores (1-alpha for BM25)
            top_k: Number of results
        
        Returns:
            Re-ranked results
        """
        # Get BM25 scores
        bm25_scores = self.get_bm25_scores(query)
        
        # Normalize dense scores to 0-1
        dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        
        # Combine scores
        combined_scores = alpha * dense_scores_norm + (1 - alpha) * bm25_scores
        
        # Sort and return top K
        doc_score_pairs = list(zip(self.documents, combined_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]


# Example usage
reranker = BM25DenseReranker()

documents = [
    "Python async programming tutorial",
    "Async/await patterns in JavaScript",
    "Python threading and multiprocessing guide",
    "Advanced asyncio in Python"
]

reranker.index_documents(documents)

query = "python async patterns"
dense_scores = np.array([0.8, 0.6, 0.4, 0.9])  # From embedding search

results = reranker.rerank(query, dense_scores, alpha=0.5, top_k=3)

for doc, score in results:
    print(f"{score:.3f}: {doc}")
\`\`\`

## Reciprocal Rank Fusion (RRF)

Combine rankings from multiple sources:

\`\`\`python
from typing import List, Dict
from collections import defaultdict

class ReciprocalRankFusion:
    """
    Combine multiple ranking results using RRF.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: Constant for RRF formula (default 60)
        """
        self.k = k
    
    def fuse(
        self,
        rankings: List[List[str]],
        top_k: int = 10
    ) -> List[str]:
        """
        Fuse multiple rankings using RRF.
        
        Args:
            rankings: List of ranked document lists
                     Each list is ordered by relevance
            top_k: Number of results to return
        
        Returns:
            Fused ranking
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                rrf_scores[doc] += 1.0 / (self.k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, score in sorted_docs[:top_k]]


# Example: Combine results from different retrievers
rrf = ReciprocalRankFusion(k=60)

# Rankings from different methods
semantic_ranking = ["doc1", "doc3", "doc5", "doc2"]
bm25_ranking = ["doc2", "doc1", "doc4", "doc3"]
cross_encoder_ranking = ["doc3", "doc1", "doc2", "doc5"]

# Fuse rankings
fused = rrf.fuse(
    [semantic_ranking, bm25_ranking, cross_encoder_ranking],
    top_k=3
)

print("Fused ranking:", fused)
\`\`\`

## Learning to Rank (LTR)

Train custom ranking models on your data:

\`\`\`python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class LearningToRankModel:
    """
    Custom learning-to-rank model.
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
    
    def extract_features(
        self,
        query: str,
        document: str,
        semantic_score: float,
        bm25_score: float,
        doc_length: int
    ) -> np.ndarray:
        """
        Extract features for ranking.
        
        Args:
            query: Search query
            document: Document text
            semantic_score: Embedding similarity
            bm25_score: BM25 score
            doc_length: Document length in tokens
        
        Returns:
            Feature vector
        """
        features = [
            semantic_score,
            bm25_score,
            doc_length,
            len(query.split()),  # Query length
            len(set(query.lower().split()) & set(document.lower().split())),  # Term overlap
        ]
        return np.array(features)
    
    def train(
        self,
        training_data: List[Dict],
        relevance_scores: List[float]
    ):
        """
        Train the ranking model.
        
        Args:
            training_data: List of feature dicts
            relevance_scores: Ground truth relevance (0-1)
        """
        X = np.array([
            self.extract_features(
                data['query'],
                data['document'],
                data['semantic_score'],
                data['bm25_score'],
                data['doc_length']
            )
            for data in training_data
        ])
        
        y = np.array(relevance_scores)
        
        self.model.fit(X, y)
    
    def rank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Rank documents using trained model.
        
        Args:
            query: Search query
            documents: Documents with features
            top_k: Number of results
        
        Returns:
            Ranked documents with scores
        """
        X = np.array([
            self.extract_features(
                query,
                doc['text'],
                doc['semantic_score'],
                doc['bm25_score'],
                doc['length']
            )
            for doc in documents
        ])
        
        scores = self.model.predict(X)
        
        # Sort by predicted score
        doc_score_pairs = list(zip([d['text'] for d in documents], scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]
\`\`\`

## Production Re-ranking System

Complete production-ready re-ranking:

\`\`\`python
from typing import List, Dict, Optional
import time

class ProductionReranker:
    """
    Production-grade re-ranking system.
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cache: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize production reranker.
        
        Args:
            cross_encoder_model: Model to use
            use_cache: Enable result caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache = {}
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10,
        method: str = "cross_encoder"
    ) -> List[Dict]:
        """
        Re-rank documents with multiple strategies.
        
        Args:
            query: Search query
            documents: Documents with metadata
            top_k: Number of results
            method: Reranking method (cross_encoder, hybrid, rrf)
        
        Returns:
            Re-ranked documents with scores
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query, method)
        if self.use_cache and cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # Rerank based on method
        if method == "cross_encoder":
            results = self._cross_encoder_rerank(query, documents, top_k)
        elif method == "hybrid":
            results = self._hybrid_rerank(query, documents, top_k)
        elif method == "rrf":
            results = self._rrf_rerank(query, documents, top_k)
        else:
            results = documents[:top_k]
        
        # Add timing metadata
        elapsed = time.time() - start_time
        for result in results:
            result['rerank_time_ms'] = elapsed * 1000
        
        # Cache results
        if self.use_cache:
            self._cache[cache_key] = (results, time.time())
        
        return results
    
    def _cross_encoder_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Cross-encoder reranking."""
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        return documents[:top_k]
    
    def _hybrid_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Hybrid reranking combining multiple signals."""
        # Combine semantic, cross-encoder, and keyword scores
        for doc in documents:
            doc['rerank_score'] = (
                0.4 * doc.get('semantic_score', 0) +
                0.4 * doc.get('cross_encoder_score', 0) +
                0.2 * doc.get('bm25_score', 0)
            )
        
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        return documents[:top_k]
    
    def _rrf_rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Reciprocal rank fusion."""
        # Implement RRF logic
        # Simplified version
        return documents[:top_k]
    
    def _get_cache_key(self, query: str, method: str) -> str:
        """Generate cache key."""
        import hashlib
        content = f"{query}:{method}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = {}
\`\`\`

## Best Practices

### When to Re-rank

✅ **Do re-rank when:**
- Initial retrieval gets 50-100+ candidates
- Quality is more important than speed
- You have budget for compute/API calls
- User expects highly relevant top results

❌ **Skip re-ranking when:**
- Real-time constraints (< 100ms)
- Initial results are already excellent
- Cost is primary concern
- Small candidate set (< 20 documents)

### Re-ranking Strategy Selection

| Use Case | Recommended Strategy |
|----------|---------------------|
| **High Quality, Cost OK** | Cohere Re-rank API |
| **Self-Hosted** | Cross-encoder (HuggingFace) |
| **Balanced** | Hybrid (BM25 + Dense + Cross-encoder) |
| **Multiple Signals** | Reciprocal Rank Fusion |
| **Custom Domain** | Learning to Rank |

### Performance Optimization

\`\`\`python
# Optimize re-ranking performance
class OptimizedReranker:
    """Performance-optimized reranker."""
    
    def rerank(self, query: str, docs: List[str], top_k: int = 10):
        # 1. Filter by threshold first
        filtered = [d for d in docs if d['initial_score'] > 0.5]
        
        # 2. Re-rank only top N candidates
        candidates = filtered[:50]  # Don't re-rank all 100
        
        # 3. Batch processing
        reranked = self._batch_rerank(query, candidates)
        
        # 4. Return top K
        return reranked[:top_k]
\`\`\`

## Summary

Re-ranking dramatically improves RAG quality:

- **Cross-Encoders**: Most accurate, understand query-document relationships
- **Cohere API**: Production-ready managed solution
- **Hybrid Scoring**: Combine multiple signals for best results
- **BM25 + Dense**: Balance keyword and semantic matching
- **RRF**: Fuse rankings from multiple sources
- **LTR**: Custom models for specific domains

**Key Takeaway:** Re-ranking is essential for production RAG systems. It's the difference between "good enough" and "excellent" search results.

**Production Pattern:**
1. Initial retrieval: 50-100 candidates (fast)
2. Re-ranking: Top 20-30 candidates (accurate)
3. Final selection: Top 5-10 for LLM context (optimal)
`,
};
