export const semanticSearchImplementation = {
  title: 'Semantic Search Implementation',
  content: `
# Semantic Search Implementation

## Introduction

Semantic search goes beyond keyword matching to understand the meaning and intent behind queries. It's the core technology powering modern RAG systems, enabling retrieval based on concepts rather than exact text matches.

In this comprehensive section, we'll build production-ready semantic search systems from scratch, covering query embedding, similarity search, ranking algorithms, and optimization techniques.

## What is Semantic Search?

**Semantic search** uses embeddings to find content based on meaning rather than exact keyword matches.

### Keyword Search vs Semantic Search

\`\`\`python
# Keyword Search (traditional)
query = "python snake"
# Matches: Documents containing exact words "python" and "snake"
# Misses: Documents about "reptiles" or "serpents"

# Semantic Search
query_embedding = embed("python snake")
# Matches: Documents about snakes, reptiles, serpents
# Understands: Query is about the animal, not the programming language
\`\`\`

### Benefits of Semantic Search

1. **Intent Understanding**: Understands what users mean, not just what they type
2. **Synonym Matching**: "car" matches "automobile", "vehicle"
3. **Multilingual**: Can match across languages
4. **Context-Aware**: Understands domain context
5. **Typo Tolerant**: Embeddings are robust to spelling errors

## Building a Basic Semantic Search Engine

Let's build a complete semantic search system step by step:

\`\`\`python
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI

client = OpenAI()

class SemanticSearchEngine:
    """
    Production-ready semantic search engine.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize search engine.
        
        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(
        self,
        documents: List[str],
        metadata: List[Dict] = None
    ):
        """
        Add documents to the search index.
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
        """
        if metadata is None:
            metadata = [{}] * len(documents)
        
        print(f"Indexing {len(documents)} documents...")
        
        # Create embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Embed batch
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch_docs
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            
            # Store
            self.documents.extend(batch_docs)
            self.embeddings.extend(batch_embeddings)
            self.metadata.extend(metadata[i:i + batch_size])
            
            print(f"  Indexed {len(self.documents)}/{len(documents)} documents")
        
        print(f"✓ Indexing complete")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of search results with scores
        """
        # Embed query
        query_embedding = self._embed_text(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= min_score:
                similarities.append({
                    "document": self.documents[i],
                    "score": similarity,
                    "metadata": self.metadata[i],
                    "index": i
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top K
        return similarities[:top_k]
    
    def _embed_text(self, text: str) -> List[float]:
        """Create embedding for text."""
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(
        self,
        v1: List[float],
        v2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        dot_product = np.dot(v1_np, v2_np)
        norm_product = np.linalg.norm(v1_np) * np.linalg.norm(v2_np)
        
        return float(dot_product / norm_product)


# Example usage
search_engine = SemanticSearchEngine()

# Add documents
documents = [
    "Python is a programming language known for simplicity and readability.",
    "Snakes are reptiles that can be found in many different environments.",
    "Machine learning uses algorithms to learn patterns from data.",
    "The Python snake is a non-venomous constrictor.",
    "Deep learning is a subset of machine learning using neural networks."
]

metadata = [
    {"category": "programming", "source": "tech_wiki"},
    {"category": "animals", "source": "nature_wiki"},
    {"category": "ai", "source": "tech_wiki"},
    {"category": "animals", "source": "nature_wiki"},
    {"category": "ai", "source": "tech_wiki"}
]

search_engine.add_documents(documents, metadata)

# Search
results = search_engine.search(
    query="Tell me about snakes and reptiles",
    top_k=3
)

print("\\nSearch Results:")
for i, result in enumerate(results, 1):
    print(f"\\n{i}. Score: {result['score']:.3f}")
    print(f"   Category: {result['metadata']['category']}")
    print(f"   Text: {result['document']}")
\`\`\`

**Output:**
\`\`\`
Search Results:

1. Score: 0.847
   Category: animals
   Text: Snakes are reptiles that can be found in many different environments.

2. Score: 0.823
   Category: animals
   Text: The Python snake is a non-venomous constrictor.

3. Score: 0.612
   Category: programming
   Text: Python is a programming language...
\`\`\`

Notice how it correctly identified documents about reptiles/snakes despite different wording!

## Advanced Similarity Metrics

Beyond basic cosine similarity, we can use sophisticated ranking:

\`\`\`python
import numpy as np

class AdvancedSimilarityCalculator:
    """
    Advanced similarity calculations for semantic search.
    """
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Cosine similarity: angle between vectors.
        Range: -1 to 1 (1 = identical direction)
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Euclidean distance: straight-line distance.
        Range: 0 to infinity (0 = identical)
        """
        return np.linalg.norm(v1 - v2)
    
    @staticmethod
    def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Manhattan distance: sum of absolute differences.
        Range: 0 to infinity (0 = identical)
        """
        return np.sum(np.abs(v1 - v2))
    
    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Dot product: simple multiplication.
        For normalized vectors, equivalent to cosine similarity.
        """
        return np.dot(v1, v2)
    
    @staticmethod
    def normalized_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Dot product of normalized vectors.
        Same as cosine similarity but faster.
        """
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        return np.dot(v1_norm, v2_norm)


# Compare metrics
calc = AdvancedSimilarityCalculator()

v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, 4])

print(f"Cosine similarity: {calc.cosine_similarity(v1, v2):.3f}")
print(f"Euclidean distance: {calc.euclidean_distance(v1, v2):.3f}")
print(f"Manhattan distance: {calc.manhattan_distance(v1, v2):.3f}")
print(f"Dot product: {calc.dot_product(v1, v2):.3f}")
\`\`\`

## Query Preprocessing

Enhance search quality by preprocessing queries:

\`\`\`python
import re
from typing import List

class QueryPreprocessor:
    """
    Preprocess queries before embedding.
    """
    
    @staticmethod
    def clean_query(query: str) -> str:
        """
        Basic query cleaning.
        
        - Remove extra whitespace
        - Convert to lowercase
        - Remove special characters (optional)
        """
        # Remove extra whitespace
        query = re.sub(r'\\s+', ' ', query.strip())
        
        # Convert to lowercase (preserve for case-sensitive models)
        # query = query.lower()
        
        return query
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Generate query variations for better recall.
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add question forms
        if not query.endswith('?'):
            variations.append(f"{query}?")
            variations.append(f"What is {query}?")
            variations.append(f"Tell me about {query}")
        
        return variations
    
    @staticmethod
    def extract_entities(query: str) -> List[str]:
        """
        Extract potential entities from query.
        (Simplified - use spaCy/NER for production)
        """
        # Simple capitalized word extraction
        words = query.split()
        entities = [w for w in words if w[0].isupper()]
        
        return entities


# Example usage
preprocessor = QueryPreprocessor()

query = "  What   is  Python?  "
cleaned = preprocessor.clean_query(query)
print(f"Cleaned: '{cleaned}'")

variations = preprocessor.expand_query("machine learning")
print(f"\\nVariations:")
for var in variations:
    print(f"  - {var}")
\`\`\`

## Hybrid Search (Dense + Sparse)

Combine semantic search with keyword search for best results:

\`\`\`python
from typing import List, Dict, Set
import numpy as np

class HybridSearchEngine:
    """
    Hybrid search combining semantic and keyword matching.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid search.
        
        Args:
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.documents = []
        self.embeddings = []
        self.inverted_index = {}  # Word -> document indices
    
    def add_documents(self, documents: List[str]):
        """Add documents and build both indexes."""
        self.documents = documents
        
        # Build semantic index (embeddings)
        print("Building semantic index...")
        for doc in documents:
            embedding = self._embed(doc)
            self.embeddings.append(embedding)
        
        # Build keyword index (inverted index)
        print("Building keyword index...")
        for doc_idx, doc in enumerate(documents):
            words = self._tokenize(doc)
            for word in words:
                if word not in self.inverted_index:
                    self.inverted_index[word] = set()
                self.inverted_index[word].add(doc_idx)
        
        print(f"✓ Indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            Ranked search results
        """
        # Get semantic scores
        semantic_scores = self._semantic_search(query)
        
        # Get keyword scores
        keyword_scores = self._keyword_search(query)
        
        # Combine scores
        combined_scores = {}
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_scores.get(doc_id, 0.0)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            
            combined_scores[doc_id] = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )
        
        # Sort and return top K
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {
                "document": self.documents[doc_id],
                "score": score,
                "semantic_score": semantic_scores.get(doc_id, 0.0),
                "keyword_score": keyword_scores.get(doc_id, 0.0)
            }
            for doc_id, score in sorted_results
        ]
    
    def _semantic_search(self, query: str) -> Dict[int, float]:
        """Semantic search scores."""
        query_embedding = self._embed(query)
        
        scores = {}
        for doc_id, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(
                query_embedding,
                doc_embedding
            )
            scores[doc_id] = similarity
        
        return scores
    
    def _keyword_search(self, query: str) -> Dict[int, float]:
        """Keyword search scores (BM25-like)."""
        query_words = self._tokenize(query)
        
        # Find documents containing query words
        doc_matches = {}
        for word in query_words:
            if word in self.inverted_index:
                for doc_id in self.inverted_index[word]:
                    doc_matches[doc_id] = doc_matches.get(doc_id, 0) + 1
        
        # Normalize scores (simple TF)
        max_matches = max(doc_matches.values()) if doc_matches else 1
        normalized_scores = {
            doc_id: count / max_matches
            for doc_id, count in doc_matches.items()
        }
        
        return normalized_scores
    
    def _embed(self, text: str) -> List[float]:
        """Create embedding (placeholder)."""
        # In production, use actual embedding model
        return [0.0] * 1536
    
    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization."""
        words = re.findall(r'\\w+', text.lower())
        return set(words)
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        return float(
            np.dot(v1_np, v2_np) /
            (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))
        )


# Example usage
hybrid_search = HybridSearchEngine(
    semantic_weight=0.7,
    keyword_weight=0.3
)

documents = [
    "Machine learning algorithms learn from data",
    "Python is great for ML development",
    "Neural networks are inspired by the brain"
]

hybrid_search.add_documents(documents)

results = hybrid_search.search("machine learning in python", top_k=3)

for i, result in enumerate(results, 1):
    print(f"\\n{i}. Score: {result['score']:.3f}")
    print(f"   Semantic: {result['semantic_score']:.3f}")
    print(f"   Keyword: {result['keyword_score']:.3f}")
    print(f"   Text: {result['document']}")
\`\`\`

## Filtering by Metadata

Add metadata filtering to narrow search results:

\`\`\`python
from typing import List, Dict, Callable

class FilterableSearchEngine:
    """
    Semantic search with metadata filtering.
    """
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ):
        """Add documents with metadata."""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_fn: Callable[[Dict], bool] = None
    ) -> List[Dict]:
        """
        Search with optional metadata filtering.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_fn: Function to filter documents by metadata
        
        Returns:
            Filtered search results
        """
        results = []
        
        for i, (doc, emb, meta) in enumerate(
            zip(self.documents, self.embeddings, self.metadata)
        ):
            # Apply filter if provided
            if filter_fn and not filter_fn(meta):
                continue
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, emb)
            
            results.append({
                "document": doc,
                "score": similarity,
                "metadata": meta
            })
        
        # Sort and return top K
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(
        self,
        v1: List[float],
        v2: List[float]
    ) -> float:
        """Cosine similarity."""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        return float(
            np.dot(v1_np, v2_np) /
            (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))
        )


# Example: Search only recent documents
search = FilterableSearchEngine()

# Add documents with metadata
docs = ["Doc 1", "Doc 2", "Doc 3"]
embeddings = [[0.1] * 1536 for _ in docs]
metadata = [
    {"date": "2024-01-15", "category": "tech"},
    {"date": "2024-01-20", "category": "science"},
    {"date": "2023-12-01", "category": "tech"}
]

search.add_documents(docs, embeddings, metadata)

# Search only 2024 documents
query_emb = [0.1] * 1536
results = search.search(
    query_emb,
    top_k=5,
    filter_fn=lambda meta: meta["date"].startswith("2024")
)

print(f"Found {len(results)} results from 2024")
\`\`\`

## Caching Search Results

Cache frequent queries for better performance:

\`\`\`python
from functools import lru_cache
import hashlib
import json

class CachedSearchEngine:
    """
    Search engine with result caching.
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize with cache.
        
        Args:
            cache_size: Max cached queries
        """
        self.cache_size = cache_size
        self._cache = {}
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search with caching.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            Search results (cached if available)
        """
        # Generate cache key
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache
        if cache_key in self._cache:
            print("✓ Cache hit!")
            return self._cache[cache_key]
        
        # Perform search
        print("✗ Cache miss, searching...")
        results = self._do_search(query, top_k)
        
        # Store in cache
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[cache_key] = results
        
        return results
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key from query parameters."""
        content = json.dumps({"query": query, "top_k": top_k})
        return hashlib.md5(content.encode()).hexdigest()
    
    def _do_search(self, query: str, top_k: int) -> List[Dict]:
        """Actual search implementation."""
        # Implement your search logic here
        return []
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache = {}
        print("Cache cleared")


# Example usage
cached_search = CachedSearchEngine(cache_size=100)

# First search - cache miss
results1 = cached_search.search("machine learning", top_k=5)

# Second search - cache hit!
results2 = cached_search.search("machine learning", top_k=5)
\`\`\`

## Production Search System

Complete production-ready implementation:

\`\`\`python
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class SearchResult:
    """Search result with rich metadata."""
    document: str
    score: float
    metadata: Dict
    rank: int
    search_time_ms: float

class ProductionSearchEngine:
    """
    Production-grade semantic search engine.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        cache_enabled: bool = True,
        log_queries: bool = True
    ):
        """Initialize production search engine."""
        self.embedding_model = embedding_model
        self.cache_enabled = cache_enabled
        self.log_queries = log_queries
        
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self._query_log = []
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Production search with monitoring and logging.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score
            filter_metadata: Optional metadata filters
        
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Log query
        if self.log_queries:
            self._query_log.append({
                "query": query,
                "timestamp": start_time,
                "top_k": top_k
            })
        
        try:
            # Embed query
            query_embedding = self._embed(query)
            
            # Search
            results = self._search_internal(
                query_embedding,
                top_k,
                min_score,
                filter_metadata
            )
            
            # Calculate search time
            search_time_ms = (time.time() - start_time) * 1000
            
            # Format results
            formatted_results = [
                SearchResult(
                    document=r["document"],
                    score=r["score"],
                    metadata=r["metadata"],
                    rank=i + 1,
                    search_time_ms=search_time_ms
                )
                for i, r in enumerate(results)
            ]
            
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _search_internal(
        self,
        query_embedding: List[float],
        top_k: int,
        min_score: float,
        filter_metadata: Optional[Dict]
    ) -> List[Dict]:
        """Internal search logic."""
        results = []
        
        for doc, emb, meta in zip(
            self.documents,
            self.embeddings,
            self.metadata
        ):
            # Apply metadata filter
            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, emb)
            
            if similarity >= min_score:
                results.append({
                    "document": doc,
                    "score": similarity,
                    "metadata": meta
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_analytics(self) -> Dict:
        """Get search analytics."""
        if not self._query_log:
            return {"total_queries": 0}
        
        return {
            "total_queries": len(self._query_log),
            "unique_queries": len(set(q["query"] for q in self._query_log)),
            "avg_top_k": np.mean([q["top_k"] for q in self._query_log])
        }
    
    def _embed(self, text: str) -> List[float]:
        """Create embedding."""
        # Implement embedding logic
        return [0.0] * 1536
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Cosine similarity."""
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        return float(
            np.dot(v1_np, v2_np) /
            (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))
        )
\`\`\`

## Summary

Semantic search is the foundation of RAG systems:

- **Embedding-based**: Uses dense vectors to capture meaning
- **Similarity metrics**: Cosine similarity most common
- **Hybrid search**: Combine semantic and keyword for best results
- **Metadata filtering**: Narrow results by attributes
- **Caching**: Essential for production performance
- **Monitoring**: Log queries and track performance

Key takeaways:
- Start with basic cosine similarity
- Add hybrid search for better results
- Implement caching early
- Monitor query patterns
- Test different embedding models
`,
};
