export const textEmbeddingsDeepDive = {
  title: 'Text Embeddings Deep Dive',
  content: `
# Text Embeddings Deep Dive

## Introduction

Text embeddings are the foundation of modern semantic search and RAG systems. They transform human-readable text into dense vector representations that capture meaning, enabling machines to understand semantic similarity and relationships between pieces of text.

In this comprehensive section, we'll explore what embeddings are, how they work, the different embedding models available, and how to use them effectively in production RAG systems.

## What Are Embeddings?

**Embeddings** are dense vector representations of text that capture semantic meaning in a high-dimensional space. Instead of treating text as a sequence of characters or words, embeddings represent text as points in a vector space where semantically similar texts are close together.

### Visual Intuition

Imagine a 3D space where:
- "cat" and "kitten" are close together
- "dog" and "puppy" are close together  
- "cat" and "dog" are somewhat close (both animals)
- "cat" and "car" are far apart (different concepts)

In reality, embeddings use hundreds or thousands of dimensions, but the principle is the same: semantic similarity translates to spatial proximity.

### Example: Semantic Similarity

\`\`\`python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding (text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity (v1: list[float], v2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1_np = np.array (v1)
    v2_np = np.array (v2)
    return float (np.dot (v1_np, v2_np) / (np.linalg.norm (v1_np) * np.linalg.norm (v2_np)))

# Get embeddings for similar and dissimilar texts
cat_emb = get_embedding("A small domestic feline animal")
kitten_emb = get_embedding("A baby cat")
car_emb = get_embedding("A motorized vehicle for transportation")

# Compare similarities
print(f"cat <-> kitten: {cosine_similarity (cat_emb, kitten_emb):.3f}")  # High ~0.85
print(f"cat <-> car: {cosine_similarity (cat_emb, car_emb):.3f}")        # Low ~0.65
\`\`\`

**Output:**
\`\`\`
cat <-> kitten: 0.847
cat <-> car: 0.623
\`\`\`

The higher similarity score (0.847) between "cat" and "kitten" shows they're semantically related, while the lower score (0.623) between "cat" and "car" shows they're less related.

## How Embeddings Work

Embeddings are created by neural networks trained on massive text corpora. These models learn to encode text into vectors such that similar meanings produce similar vectors.

### The Embedding Process

1. **Input**: Raw text string
2. **Tokenization**: Split text into tokens
3. **Neural Network**: Process tokens through transformer model
4. **Pooling**: Combine token representations
5. **Output**: Fixed-length vector (e.g., 1536 dimensions)

\`\`\`python
# The process (simplified conceptual view)
text = "RAG systems use embeddings"
↓
tokens = ["RAG", "systems", "use", "embeddings"]
↓
neural_network_processing (tokens)
↓
embedding_vector = [0.123, -0.456, 0.789, ..., 0.234]  # 1536 dims
\`\`\`

### Key Properties of Good Embeddings

1. **Semantic Preservation**: Similar meanings → similar vectors
2. **Dimensionality**: Fixed size regardless of input length
3. **Density**: Every dimension contains information
4. **Efficiency**: Fast to compute and compare
5. **Generalization**: Work across different domains

## Popular Embedding Models

Let\'s explore the major embedding models available for production use:

### 1. OpenAI Embeddings

OpenAI offers state-of-the-art embedding models optimized for search and RAG:

#### text-embedding-3-small

\`\`\`python
from openai import OpenAI

client = OpenAI()

def create_embedding_small (text: str) -> list[float]:
    """
    Create embedding using text-embedding-3-small.
    
    Specs:
    - Dimensions: 1536
    - Cost: $0.02 per 1M tokens
    - Performance: Excellent
    - Speed: Very fast
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Example usage
text = "RAG combines retrieval with generation"
embedding = create_embedding_small (text)
print(f"Dimensions: {len (embedding)}")  # 1536
print(f"First 5 values: {embedding[:5]}")
\`\`\`

**Best For:**
- Most production RAG systems
- Cost-sensitive applications
- High-volume search

#### text-embedding-3-large

\`\`\`python
def create_embedding_large (text: str) -> list[float]:
    """
    Create embedding using text-embedding-3-large.
    
    Specs:
    - Dimensions: 3072
    - Cost: $0.13 per 1M tokens
    - Performance: Best-in-class
    - Speed: Fast
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
\`\`\`

**Best For:**
- Maximum accuracy requirements
- Complex domain-specific content
- When cost is less constrained

### 2. Sentence Transformers (Open Source)

Powerful open-source embeddings you can run locally:

\`\`\`python
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbeddings:
    """
    Local embedding generation using sentence-transformers.
    No API costs, runs on your hardware.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dims, good quality
        - all-mpnet-base-v2: Slower, 768 dims, better quality
        - multi-qa-mpnet-base-dot-v1: Optimized for Q&A
        """
        self.model = SentenceTransformer (model_name)
        self.model_name = model_name
    
    def encode (self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            2D array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single (self, text: str) -> np.ndarray:
        """Encode single text."""
        return self.encode([text])[0]

# Example usage
local_model = LocalEmbeddings("all-MiniLM-L6-v2")

texts = [
    "RAG systems retrieve relevant documents",
    "Vector databases store embeddings",
    "Python is a programming language"
]

embeddings = local_model.encode (texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)

# Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity (embeddings)
print("Similarity matrix:")
print(similarities)
\`\`\`

**Best For:**
- No API costs
- Privacy-sensitive data
- High-volume applications
- Edge deployment

### 3. Cohere Embeddings

Cohere offers multilingual embeddings optimized for search:

\`\`\`python
import cohere

class CohereEmbeddings:
    """
    Cohere embedding generation.
    """
    
    def __init__(self, api_key: str):
        self.client = cohere.Client (api_key)
    
    def embed_documents(
        self, 
        texts: list[str], 
        model: str = "embed-english-v3.0"
    ) -> list[list[float]]:
        """
        Embed documents for indexing.
        
        Args:
            texts: List of documents
            model: Cohere model name
        
        Returns:
            List of embedding vectors
        """
        response = self.client.embed(
            texts=texts,
            model=model,
            input_type="search_document"  # For indexing
        )
        return response.embeddings
    
    def embed_query(
        self, 
        query: str, 
        model: str = "embed-english-v3.0"
    ) -> list[float]:
        """
        Embed query for search.
        
        Args:
            query: Search query
            model: Cohere model name
        
        Returns:
            Query embedding vector
        """
        response = self.client.embed(
            texts=[query],
            model=model,
            input_type="search_query"  # For searching
        )
        return response.embeddings[0]

# Example usage
cohere_emb = CohereEmbeddings (api_key="your-api-key")

# Embed documents
docs = ["Document 1 text", "Document 2 text"]
doc_embeddings = cohere_emb.embed_documents (docs)

# Embed query
query = "Search query"
query_embedding = cohere_emb.embed_query (query)
\`\`\`

**Best For:**
- Multilingual applications
- Asymmetric search (query ≠ document style)
- Fine-grained control over input types

### 4. Google's E5 Embeddings (Open Source)

State-of-the-art open-source embeddings:

\`\`\`python
from sentence_transformers import SentenceTransformer

class E5Embeddings:
    """
    E5 embedding models from Microsoft/Google.
    """
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize E5 model.
        
        Args:
            model_size: 'small', 'base', or 'large'
        """
        model_map = {
            "small": "intfloat/e5-small-v2",  # 384 dims
            "base": "intfloat/e5-base-v2",    # 768 dims
            "large": "intfloat/e5-large-v2"   # 1024 dims
        }
        self.model = SentenceTransformer (model_map[model_size])
    
    def encode_documents (self, texts: list[str]) -> list[list[float]]:
        """Encode documents with 'passage:' prefix."""
        prefixed = [f"passage: {text}" for text in texts]
        return self.model.encode (prefixed).tolist()
    
    def encode_query (self, query: str) -> list[float]:
        """Encode query with 'query:' prefix."""
        prefixed = f"query: {query}"
        return self.model.encode([prefixed])[0].tolist()

# Example usage
e5 = E5Embeddings("base")

docs = ["RAG systems use embeddings", "Vector search is fast"]
doc_embs = e5.encode_documents (docs)

query = "What is RAG?"
query_emb = e5.encode_query (query)
\`\`\`

**Best For:**
- High-quality open-source option
- Research and experimentation
- Custom fine-tuning

## Embedding Dimensions

Embeddings come in different dimensionalities. More dimensions can capture more nuance but require more storage and computation.

### Dimension Comparison

\`\`\`python
def analyze_embedding_dimensions():
    """Compare different embedding dimensions."""
    
    models = {
        "OpenAI small": {"dims": 1536, "cost": "$0.02/1M tokens"},
        "OpenAI large": {"dims": 3072, "cost": "$0.13/1M tokens"},
        "MiniLM": {"dims": 384, "cost": "Free (local)"},
        "MPNet": {"dims": 768, "cost": "Free (local)"},
        "E5-large": {"dims": 1024, "cost": "Free (local)"}
    }
    
    for name, info in models.items():
        # Storage per 1M vectors
        storage_gb = (info["dims"] * 4 * 1_000_000) / (1024**3)
        print(f"{name}:")
        print(f"  Dimensions: {info['dims']}")
        print(f"  Cost: {info['cost']}")
        print(f"  Storage (1M vectors): {storage_gb:.2f} GB")
        print()

analyze_embedding_dimensions()
\`\`\`

**Output:**
\`\`\`
OpenAI small:
  Dimensions: 1536
  Cost: $0.02/1M tokens
  Storage (1M vectors): 5.72 GB

OpenAI large:
  Dimensions: 3072
  Cost: $0.13/1M tokens
  Storage (1M vectors): 11.44 GB

MiniLM:
  Dimensions: 384
  Cost: Free (local)
  Storage (1M vectors): 1.43 GB
...
\`\`\`

### Reducing Dimensions

You can reduce embedding dimensions to save storage:

\`\`\`python
from sklearn.decomposition import PCA
import numpy as np

def reduce_dimensions(
    embeddings: np.ndarray, 
    target_dims: int = 512
) -> np.ndarray:
    """
    Reduce embedding dimensions using PCA.
    
    Args:
        embeddings: Original embeddings (n_samples, n_features)
        target_dims: Target number of dimensions
    
    Returns:
        Reduced embeddings
    """
    pca = PCA(n_components=target_dims)
    reduced = pca.fit_transform (embeddings)
    
    # Show variance retained
    variance_retained = sum (pca.explained_variance_ratio_)
    print(f"Variance retained: {variance_retained:.2%}")
    
    return reduced

# Example: Reduce 1536D to 512D
original = np.random.randn(1000, 1536)  # 1000 embeddings
reduced = reduce_dimensions (original, target_dims=512)

print(f"Original shape: {original.shape}")
print(f"Reduced shape: {reduced.shape}")
print(f"Storage savings: {(1 - 512/1536):.1%}")
\`\`\`

## Semantic Similarity Metrics

Different ways to measure similarity between embeddings:

### Cosine Similarity

Most common metric for embeddings:

\`\`\`python
import numpy as np

def cosine_similarity (v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Range: -1 to 1
    - 1.0: Identical direction
    - 0.0: Orthogonal (unrelated)
    - -1.0: Opposite direction
    """
    return np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Same direction, scaled
v3 = np.array([1, 0, 0])  # Different direction

print(f"v1 <-> v2: {cosine_similarity (v1, v2):.3f}")  # 1.0 (identical)
print(f"v1 <-> v3: {cosine_similarity (v1, v3):.3f}")  # Different
\`\`\`

### Dot Product

Faster than cosine similarity for normalized vectors:

\`\`\`python
def dot_product_similarity (v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Dot product similarity.
    Works well when vectors are normalized.
    """
    return np.dot (v1, v2)

def normalize_vector (v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    return v / np.linalg.norm (v)

# With normalized vectors, dot product ≈ cosine similarity
v1_norm = normalize_vector (v1)
v2_norm = normalize_vector (v2)

print(f"Dot product: {dot_product_similarity (v1_norm, v2_norm):.3f}")
print(f"Cosine sim: {cosine_similarity (v1, v2):.3f}")
\`\`\`

### Euclidean Distance

Measures straight-line distance in vector space:

\`\`\`python
def euclidean_distance (v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate Euclidean distance.
    
    Lower values = more similar
    """
    return np.linalg.norm (v1 - v2)

# Example
distance = euclidean_distance (v1, v2)
print(f"Distance: {distance:.3f}")
\`\`\`

## Batching Embeddings

For production systems, batch embedding creation for efficiency:

\`\`\`python
from typing import List
from openai import OpenAI
import time

client = OpenAI()

def create_embeddings_batch(
    texts: List[str],
    batch_size: int = 100,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Create embeddings in batches for efficiency.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call
        model: Embedding model to use
    
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len (texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            
            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend (batch_embeddings)
            
            print(f"Processed {len (all_embeddings)}/{len (texts)} texts")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Retry or handle error
            continue
    
    return all_embeddings

# Example usage
texts = [f"Document {i}" for i in range(250)]
embeddings = create_embeddings_batch (texts, batch_size=100)
print(f"Created {len (embeddings)} embeddings")
\`\`\`

## Caching Embeddings

Cache embeddings to reduce costs and improve performance:

\`\`\`python
import hashlib
import json
from pathlib import Path
from typing import Optional

class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputing.
    """
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path (cache_dir)
        self.cache_dir.mkdir (exist_ok=True)
    
    def _get_cache_key (self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get (self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        cache_key = self._get_cache_key (text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open (cache_file) as f:
                data = json.load (f)
                return data["embedding"]
        return None
    
    def set (self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        cache_key = self._get_cache_key (text, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open (cache_file, 'w') as f:
            json.dump({
                "text": text,
                "model": model,
                "embedding": embedding
            }, f)
    
    def get_or_create(
        self, 
        text: str, 
        model: str,
        embedding_fn
    ) -> List[float]:
        """Get from cache or create new embedding."""
        # Try cache first
        cached = self.get (text, model)
        if cached is not None:
            return cached
        
        # Create new embedding
        embedding = embedding_fn (text)
        self.set (text, model, embedding)
        
        return embedding

# Example usage
cache = EmbeddingCache()

def embed_with_cache (text: str) -> List[float]:
    """Embed text with caching."""
    return cache.get_or_create(
        text=text,
        model="text-embedding-3-small",
        embedding_fn=lambda t: get_embedding (t)
    )

# First call: creates embedding
emb1 = embed_with_cache("RAG systems use embeddings")  # API call

# Second call: uses cache
emb2 = embed_with_cache("RAG systems use embeddings")  # From cache!
\`\`\`

## Choosing the Right Embedding Model

Consider these factors when selecting an embedding model:

\`\`\`python
def evaluate_embedding_model(
    model_name: str,
    test_queries: List[str],
    test_docs: List[str]
) -> dict:
    """
    Evaluate embedding model performance.
    
    Returns metrics for decision making.
    """
    import time
    
    # Test encoding speed
    start = time.time()
    query_embs = [get_embedding (q, model_name) for q in test_queries]
    doc_embs = [get_embedding (d, model_name) for d in test_docs]
    elapsed = time.time() - start
    
    # Calculate metrics
    return {
        "model": model_name,
        "dimensions": len (query_embs[0]),
        "speed": f"{elapsed:.2f}s for {len (test_queries + test_docs)} embeddings",
        "throughput": f"{len (test_queries + test_docs) / elapsed:.1f} emb/sec"
    }
\`\`\`

### Decision Matrix

| Criterion | Best Choice |
|-----------|-------------|
| **Highest Quality** | OpenAI text-embedding-3-large |
| **Best Cost/Performance** | OpenAI text-embedding-3-small |
| **No API Costs** | Sentence-BERT (local) |
| **Multilingual** | Cohere or multilingual-e5 |
| **Privacy Critical** | Local models (Sentence-BERT, E5) |
| **High Volume** | Local models + batch processing |

## Production Best Practices

### 1. Normalize Before Storing

\`\`\`python
def normalize_embedding (embedding: List[float]) -> List[float]:
    """Normalize embedding to unit length."""
    emb_array = np.array (embedding)
    norm = np.linalg.norm (emb_array)
    return (emb_array / norm).tolist()
\`\`\`

### 2. Handle Long Texts

\`\`\`python
def embed_long_text (text: str, max_tokens: int = 8000) -> List[float]:
    """Handle texts longer than model limit."""
    import tiktoken
    
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode (text)
    
    if len (tokens) <= max_tokens:
        return get_embedding (text)
    
    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode (truncated_tokens)
    
    return get_embedding (truncated_text)
\`\`\`

### 3. Monitor Quality

\`\`\`python
def monitor_embedding_quality(
    query: str,
    retrieved_docs: List[str],
    threshold: float = 0.3
) -> dict:
    """Monitor if retrieved docs are relevant."""
    query_emb = get_embedding (query)
    doc_embs = [get_embedding (doc) for doc in retrieved_docs]
    
    similarities = [
        cosine_similarity (query_emb, doc_emb) 
        for doc_emb in doc_embs
    ]
    
    return {
        "avg_similarity": np.mean (similarities),
        "min_similarity": np.min (similarities),
        "below_threshold": sum (s < threshold for s in similarities)
    }
\`\`\`

## Summary

Text embeddings are the foundation of semantic search and RAG systems. Key takeaways:

- **Embeddings transform text into semantic vectors** for similarity search
- **Multiple models available** - choose based on cost, quality, and deployment needs
- **OpenAI embeddings** offer excellent quality and ease of use
- **Local models** provide zero-cost alternatives with privacy benefits
- **Caching is essential** for production systems
- **Batch processing** improves efficiency
- **Cosine similarity** is the standard metric for comparing embeddings

Understanding embeddings deeply enables you to build high-quality RAG systems that accurately retrieve relevant information.
`,
};
