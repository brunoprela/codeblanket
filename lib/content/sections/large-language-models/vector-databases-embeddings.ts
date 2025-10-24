export const vectorDatabasesEmbeddings = {
  title: 'Vector Databases & Embeddings',
  id: 'vector-databases-embeddings',
  content: `
# Vector Databases & Embeddings

## Introduction

Embeddings transform text into numerical vectors that capture semantic meaning, enabling computers to understand similarity and relationships between concepts. Vector databases store and efficiently search these embeddings at scale, powering semantic search, recommendation systems, and RAG applications. This section covers embedding models, vector databases, indexing strategies, and building production search systems.

### Why Vector Databases Matter

**Semantic Search**: Find by meaning, not just keywords
**Similarity**: Measure how related concepts are
**Scale**: Search millions of vectors in milliseconds
**RAG Foundation**: Retrieve relevant context for LLMs
**Personalization**: Power recommendations and matching

---

## Understanding Embeddings

### From Text to Vectors

\`\`\`python
"""
Creating and using text embeddings
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextEmbeddings:
    """
    Generate and work with text embeddings
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Load embedding model
        
        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dims, good quality
        - all-mpnet-base-v2: Best quality, 768 dims, slower
        - multi-qa-mpnet-base-dot-v1: Optimized for QA
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text):
        """
        Convert text to embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts):
        """
        Efficiently embed multiple texts
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_similarity(self, text1, text2):
        """
        Measure semantic similarity
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        return similarity
    
    def find_most_similar(self, query, candidates):
        """
        Find most similar texts to query
        """
        query_emb = self.embed_text(query)
        candidate_embs = self.embed_batch(candidates)
        
        # Compute similarities
        similarities = cosine_similarity([query_emb], candidate_embs)[0]
        
        # Sort by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        
        results = [
            {
                'text': candidates[i],
                'similarity': similarities[i]
            }
            for i in ranked_indices
        ]
        
        return results

# Example usage
embedder = TextEmbeddings()

# Embed texts
text1 = "Machine learning is a subset of artificial intelligence"
text2 = "AI includes machine learning and deep learning"
text3 = "The weather is nice today"

emb1 = embedder.embed_text(text1)
print(f"Embedding shape: {emb1.shape}")  # (384,)
print(f"First 5 values: {emb1[:5]}")

# Similarity
sim_12 = embedder.compute_similarity(text1, text2)
sim_13 = embedder.compute_similarity(text1, text3)

print(f"Similarity (text1, text2): {sim_12:.3f}")  # ~0.75 (high)
print(f"Similarity (text1, text3): {sim_13:.3f}")  # ~0.15 (low)

# Search
query = "What is machine learning?"
documents = [
    "Machine learning is a method of data analysis",
    "The sky is blue",
    "AI and ML are transforming industries",
    "I like pizza"
]

results = embedder.find_most_similar(query, documents)
print("\\nTop results:")
for r in results[:2]:
    print(f"  {r['similarity']:.3f}: {r['text']}")

# Output shows semantically similar texts rank higher!
\`\`\`

### Embedding Models Comparison

\`\`\`python
"""
Different embedding models and their use cases
"""

class EmbeddingModels:
    """
    Overview of embedding models
    """
    
    def __init__(self):
        self.models = {
            # General purpose
            "all-MiniLM-L6-v2": {
                "dims": 384,
                "speed": "Very fast",
                "quality": "Good",
                "use_case": "General semantic search",
                "size": "80MB"
            },
            "all-mpnet-base-v2": {
                "dims": 768,
                "speed": "Medium",
                "quality": "Best",
                "use_case": "When quality > speed",
                "size": "420MB"
            },
            
            # Specialized
            "multi-qa-mpnet-base-dot-v1": {
                "dims": 768,
                "speed": "Medium",
                "quality": "Excellent for QA",
                "use_case": "Question answering",
                "size": "420MB"
            },
            "msmarco-distilbert-base-v4": {
                "dims": 768,
                "speed": "Fast",
                "quality": "Good for search",
                "use_case": "Document retrieval",
                "size": "250MB"
            },
            
            # Multilingual
            "paraphrase-multilingual-mpnet-base-v2": {
                "dims": 768,
                "speed": "Medium",
                "quality": "Good",
                "use_case": "50+ languages",
                "size": "970MB"
            },
            
            # Code
            "code-search-net": {
                "dims": 768,
                "speed": "Medium",
                "quality": "Good for code",
                "use_case": "Code search",
                "size": "420MB"
            },
            
            # OpenAI
            "text-embedding-ada-002": {
                "dims": 1536,
                "speed": "API call",
                "quality": "Excellent",
                "use_case": "Production, API-based",
                "cost": "$0.0001/1k tokens"
            },
            
            # Voyage AI
            "voyage-2": {
                "dims": 1024,
                "speed": "API call",
                "quality": "Excellent",
                "use_case": "Production, optimized",
                "cost": "$0.0001/1k tokens"
            }
        }
    
    def recommend_model(self, requirements):
        """
        Recommend model based on requirements
        """
        if requirements['budget'] == 'free':
            if requirements['quality'] == 'best':
                return "all-mpnet-base-v2"
            elif requirements['speed'] == 'fast':
                return "all-MiniLM-L6-v2"
        
        elif requirements['task'] == 'code':
            return "code-search-net"
        
        elif requirements['languages'] > 1:
            return "paraphrase-multilingual-mpnet-base-v2"
        
        elif requirements['scale'] == 'production':
            return "text-embedding-ada-002"  # API, no hosting

# Using OpenAI embeddings
import openai

def openai_embeddings(texts):
    """
    OpenAI ada-002 embeddings
    """
    client = openai.OpenAI()
    
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

# Example
texts = ["Hello world", "Machine learning"]
embeddings = openai_embeddings(texts)
print(f"Shape: {embeddings.shape}")  # (2, 1536)

# Cost: 2 texts * ~2 tokens each * $0.0001/1k = $0.0004
\`\`\`

---

## Vector Databases

### FAISS: Facebook AI Similarity Search

\`\`\`python
"""
FAISS for fast similarity search
"""

import faiss
import numpy as np

class FAISSIndex:
    """
    Build and search FAISS index
    """
    
    def __init__(self, dimension=384):
        """
        Initialize FAISS index
        
        Index types:
        - IndexFlatL2: Exact search (brute force)
        - IndexIVFFlat: Inverted file index (faster)
        - IndexHNSW: Hierarchical graph (fast + accurate)
        """
        self.dimension = dimension
        self.index = None
        self.texts = []  # Store original texts
    
    def build_flat_index(self, embeddings):
        """
        Flat index: Exact search, best quality
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"Added {self.index.ntotal} vectors")
    
    def build_ivf_index(self, embeddings, nlist=100):
        """
        IVF index: Faster search for large datasets
        
        nlist: Number of clusters
        - Small dataset (<10k): 100
        - Medium (10k-100k): 1000
        - Large (>100k): 4096+
        """
        # Quantizer
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # IVF index
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Train on data
        self.index.train(embeddings.astype('float32'))
        
        # Add vectors
        self.index.add(embeddings.astype('float32'))
        
        print(f"Trained and added {self.index.ntotal} vectors")
    
    def build_hnsw_index(self, embeddings, M=32):
        """
        HNSW: Hierarchical Navigable Small World
        
        Best for: Speed + accuracy balance
        M: Number of connections (16-64)
        - Higher M = better recall, more memory
        """
        self.index = faiss.IndexHNSWFlat(self.dimension, M)
        self.index.add(embeddings.astype('float32'))
        print(f"Built HNSW index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding, k=5, nprobe=10):
        """
        Search for k nearest neighbors
        
        nprobe: Number of clusters to search (IVF only)
        - Higher = more accurate, slower
        """
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = nprobe
        
        query = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query, k)
        
        return distances[0], indices[0]
    
    def save(self, path):
        """
        Save index to disk
        """
        faiss.write_index(self.index, path)
    
    def load(self, path):
        """
        Load index from disk
        """
        self.index = faiss.read_index(path)

# Example usage
from sentence_transformers import SentenceTransformer

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language",
    "The weather is nice today",
    "Natural language processing analyzes text"
]

embeddings = model.encode(documents)

# Build index
index = FAISSIndex(dimension=384)
index.build_hnsw_index(embeddings)  # Fast + accurate

# Search
query = "What is artificial intelligence?"
query_emb = model.encode(query)

distances, indices = index.search(query_emb, k=3)

print("Top 3 results:")
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"{i+1}. Distance: {dist:.3f} - {documents[idx]}")

# Performance comparison
performance = {
    "IndexFlatL2": {
        "search_time": "O(n)",
        "accuracy": "100% (exact)",
        "memory": "n * d * 4 bytes",
        "use": "Small datasets (<10k)"
    },
    "IndexIVFFlat": {
        "search_time": "O(n/nlist)",
        "accuracy": "95-99%",
        "memory": "Similar to flat",
        "use": "Medium datasets (10k-1M)"
    },
    "IndexHNSW": {
        "search_time": "O(log n)",
        "accuracy": "98-99.5%",
        "memory": "Higher (graph structure)",
        "use": "Best all-around"
    }
}
\`\`\`

### Pinecone: Managed Vector Database

\`\`\`python
"""
Pinecone for production vector search
"""

import pinecone

class PineconeVectorDB:
    """
    Pinecone managed vector database
    
    Benefits:
    - Fully managed (no infrastructure)
    - Auto-scaling
    - Real-time updates
    - Metadata filtering
    - Hybrid search
    """
    
    def __init__(self, api_key, environment="us-west1-gcp"):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = None
    
    def create_index(self, index_name, dimension=384, metric="cosine"):
        """
        Create Pinecone index
        
        Metrics:
        - cosine: Best for semantic similarity
        - euclidean: L2 distance
        - dotproduct: For normalized vectors
        """
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                pods=1,  # Scale up for production
                pod_type="p1.x1"  # Different tiers available
            )
        
        self.index = pinecone.Index(index_name)
    
    def upsert_vectors(self, vectors, metadata=None, ids=None):
        """
        Insert or update vectors
        
        vectors: List of embedding vectors
        metadata: List of dicts with additional info
        ids: List of unique IDs
        """
        if ids is None:
            ids = [str(i) for i in range(len(vectors))]
        
        if metadata is None:
            metadata = [{} for _ in vectors]
        
        # Prepare data
        to_upsert = [
            (id, vector.tolist(), meta)
            for id, vector, meta in zip(ids, vectors, metadata)
        ]
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(to_upsert), batch_size):
            batch = to_upsert[i:i+batch_size]
            self.index.upsert(vectors=batch)
    
    def query(self, query_vector, top_k=5, filter=None, include_metadata=True):
        """
        Query for similar vectors
        
        filter: Metadata filter (e.g., {"category": "tech"})
        """
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )
        
        return results
    
    def delete(self, ids):
        """
        Delete vectors by ID
        """
        self.index.delete(ids=ids)
    
    def update_metadata(self, id, metadata):
        """
        Update vector metadata
        """
        self.index.update(id=id, set_metadata=metadata)

# Example usage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize
db = PineconeVectorDB(api_key="your-key")
db.create_index("semantic-search", dimension=384)

# Prepare data
documents = [
    "Machine learning tutorial for beginners",
    "Advanced deep learning techniques",
    "How to cook pasta",
    "Python programming guide"
]

metadata = [
    {"category": "ML", "difficulty": "beginner"},
    {"category": "ML", "difficulty": "advanced"},
    {"category": "cooking", "difficulty": "beginner"},
    {"category": "programming", "difficulty": "beginner"}
]

# Embed and insert
embeddings = model.encode(documents)
db.upsert_vectors(
    vectors=embeddings,
    metadata=metadata,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Query
query = "machine learning basics"
query_emb = model.encode(query)

results = db.query(query_emb, top_k=3)

print("Results:")
for match in results['matches']:
    print(f"  Score: {match['score']:.3f}")
    print(f"  Text: {match['metadata']}")

# Query with filter
ml_results = db.query(
    query_emb,
    top_k=3,
    filter={"category": "ML"}  # Only ML documents
)

# Pricing (as of 2024)
pricing = {
    "p1.x1": "$70/month (1M vectors, 100 QPS)",
    "p1.x2": "$140/month (2M vectors, 200 QPS)",
    "p1.x4": "$280/month (4M vectors, 400 QPS)",
    "storage": "$0.25/GB/month"
}
\`\`\`

### Chroma: Open-Source Vector DB

\`\`\`python
"""
ChromaDB for local development
"""

import chromadb
from chromadb.utils import embedding_functions

class ChromaVectorDB:
    """
    Chroma: Open-source, embedded vector database
    
    Perfect for:
    - Local development
    - Small to medium datasets
    - No infrastructure needed
    """
    
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initialize ChromaDB
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
    
    def create_collection(self, name, embedding_function=None):
        """
        Create or get collection
        """
        if embedding_function is None:
            # Use default sentence transformers
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )
    
    def add_documents(self, documents, metadata=None, ids=None):
        """
        Add documents (auto-embeds them)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    def query(self, query_text, n_results=5, where=None):
        """
        Query with natural language
        
        where: Metadata filter (e.g., {"category": "tech"})
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_by_id(self, ids):
        """
        Get documents by ID
        """
        return self.collection.get(ids=ids)
    
    def delete(self, ids):
        """
        Delete documents
        """
        self.collection.delete(ids=ids)

# Example usage
db = ChromaVectorDB()
db.create_collection("documents")

# Add documents
documents = [
    "Python is a programming language",
    "Machine learning is a subset of AI",
    "The weather is sunny today",
    "Neural networks power deep learning"
]

metadata = [
    {"category": "programming"},
    {"category": "AI"},
    {"category": "weather"},
    {"category": "AI"}
]

db.add_documents(documents, metadata=metadata)

# Query
results = db.query("artificial intelligence", n_results=2)

print("Results:")
for doc, meta, dist in zip(results['documents'][0], 
                            results['metadatas'][0],
                            results['distances'][0]):
    print(f"  Distance: {dist:.3f}")
    print(f"  Category: {meta['category']}")
    print(f"  Text: {doc}\\n")

# Query with filter
ai_results = db.query(
    "machine learning",
    n_results=5,
    where={"category": "AI"}
)
\`\`\`

---

## Advanced Indexing Strategies

### Hybrid Search

\`\`\`python
"""
Combine semantic search with keyword search
"""

from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    """
    Combine dense (semantic) and sparse (keyword) search
    
    Why hybrid?
    - Semantic search: Understands meaning, may miss exact matches
    - Keyword search: Exact matches, misses synonyms
    - Hybrid: Best of both worlds
    """
    
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        
        # BM25 for keyword search
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def semantic_search(self, query_embedding, k=10):
        """
        Dense vector search
        """
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = [
            {
                'index': idx,
                'score': similarities[idx],
                'text': self.documents[idx]
            }
            for idx in top_indices
        ]
        
        return results
    
    def keyword_search(self, query, k=10):
        """
        Sparse BM25 search
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = [
            {
                'index': idx,
                'score': scores[idx],
                'text': self.documents[idx]
            }
            for idx in top_indices
        ]
        
        return results
    
    def hybrid_search(self, query, query_embedding, k=10, alpha=0.5):
        """
        Combine semantic and keyword search
        
        alpha: Weight for semantic search (0-1)
        - 0: Pure keyword
        - 1: Pure semantic
        - 0.5: Equal weight
        """
        # Get results from both
        semantic_results = self.semantic_search(query_embedding, k=k*2)
        keyword_results = self.keyword_search(query, k=k*2)
        
        # Normalize scores to [0, 1]
        semantic_scores = self._normalize_scores([r['score'] for r in semantic_results])
        keyword_scores = self._normalize_scores([r['score'] for r in keyword_results])
        
        # Combine scores
        combined_scores = {}
        
        for i, result in enumerate(semantic_results):
            idx = result['index']
            combined_scores[idx] = alpha * semantic_scores[i]
        
        for i, result in enumerate(keyword_results):
            idx = result['index']
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * keyword_scores[i]
            else:
                combined_scores[idx] = (1 - alpha) * keyword_scores[i]
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = [
            {
                'index': idx,
                'score': score,
                'text': self.documents[idx]
            }
            for idx, score in sorted_indices[:k]
        ]
        
        return results
    
    def _normalize_scores(self, scores):
        """
        Min-max normalization
        """
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score == 0:
            return scores
        
        return (scores - min_score) / (max_score - min_score)

# Example
documents = [
    "Python is a high-level programming language",
    "Machine learning uses algorithms to learn from data",
    "The Python programming language is easy to learn",
    "Deep learning is a subset of machine learning"
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

search = HybridSearch(documents, embeddings)

query = "Python language"
query_emb = model.encode(query)

# Compare search methods
print("Semantic search:")
for r in search.semantic_search(query_emb, k=2):
    print(f"  {r['score']:.3f}: {r['text']}")

print("\\nKeyword search:")
for r in search.keyword_search(query, k=2):
    print(f"  {r['score']:.3f}: {r['text']}")

print("\\nHybrid search:")
for r in search.hybrid_search(query, query_emb, k=2):
    print(f"  {r['score']:.3f}: {r['text']}")
\`\`\`

### Reranking

\`\`\`python
"""
Rerank search results for better relevance
"""

from sentence_transformers import CrossEncoder

class Reranker:
    """
    Two-stage retrieval:
    1. Fast vector search (get top 100)
    2. Slow reranking (rerank to top 10)
    
    Why?
    - Vector search: Fast, approximate
    - Reranking: Slow, accurate
    - Hybrid: Best of both
    """
    
    def __init__(self):
        # Reranking model (cross-encoder)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query, documents, top_k=10):
        """
        Rerank documents for query
        """
        # Create pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            {
                'score': scores[idx],
                'text': documents[idx]
            }
            for idx in ranked_indices
        ]
        
        return results

# Two-stage retrieval pipeline
class TwoStageRetrieval:
    """
    Complete retrieval pipeline
    """
    
    def __init__(self, vector_db, reranker):
        self.vector_db = vector_db
        self.reranker = reranker
    
    def retrieve(self, query, initial_k=100, final_k=10):
        """
        1. Vector search: Get top 100 (fast)
        2. Rerank: Get top 10 (accurate)
        """
        # Stage 1: Fast vector search
        query_emb = model.encode(query)
        candidates = self.vector_db.search(query_emb, k=initial_k)
        
        # Stage 2: Rerank
        final_results = self.reranker.rerank(
            query,
            [c['text'] for c in candidates],
            top_k=final_k
        )
        
        return final_results

# Example
documents = [...]  # Large dataset
vector_db = FAISSIndex(...)
reranker = Reranker()

pipeline = TwoStageRetrieval(vector_db, reranker)

query = "How does machine learning work?"
results = pipeline.retrieve(query, initial_k=100, final_k=10)

# Performance
performance = {
    "Vector search only": "Fast, ~95% accuracy",
    "Reranking all": "Slow, 99% accuracy, impractical",
    "Two-stage": "Fast + accurate, best approach"
}
\`\`\`

---

## Production Considerations

### Scaling and Optimization

\`\`\`python
"""
Building production vector search systems
"""

class ProductionVectorSearch:
    """
    Production best practices
    """
    
    def chunking_strategy(self, documents):
        """
        Break documents into chunks for embedding
        
        Why chunk?
        - Embeddings work best on ~512 tokens
        - Large docs need splitting
        - Better retrieval granularity
        """
        chunks = []
        
        for doc in documents:
            # Chunk by paragraphs
            paragraphs = doc.split('\\n\\n')
            
            for para in paragraphs:
                # Further split if too long
                if len(para.split()) > 512:
                    sentences = para.split('. ')
                    current_chunk = ""
                    
                    for sent in sentences:
                        if len(current_chunk.split()) + len(sent.split()) < 512:
                            current_chunk += sent + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent + ". "
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                else:
                    chunks.append(para)
        
        return chunks
    
    def embedding_cache(self):
        """
        Cache embeddings to avoid recomputation
        """
        import hashlib
        import pickle
        
        cache = {}
        
        def get_embedding(text):
            # Hash text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check cache
            if text_hash in cache:
                return cache[text_hash]
            
            # Compute embedding
            embedding = model.encode(text)
            
            # Cache
            cache[text_hash] = embedding
            
            return embedding
        
        return get_embedding
    
    def batch_processing(self, texts, batch_size=32):
        """
        Process in batches for efficiency
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = model.encode(batch)
            embeddings.extend(batch_embs)
        
        return np.array(embeddings)
    
    def monitoring(self):
        """
        Monitor vector search performance
        """
        metrics = {
            "search_latency": "p50, p95, p99",
            "cache_hit_rate": "% of cached embeddings used",
            "index_size": "GB of memory/disk",
            "qps": "Queries per second",
            "recall": "% of relevant docs retrieved"
        }
        
        return metrics

# Cost optimization
cost_optimization = {
    "Embedding generation": {
        "problem": "Expensive API calls",
        "solution": "Cache embeddings, batch processing"
    },
    "Storage": {
        "problem": "Large vector indexes",
        "solution": "Quantization, compression"
    },
    "Search latency": {
        "problem": "Slow queries",
        "solution": "HNSW index, GPU acceleration"
    },
    "Updates": {
        "problem": "Frequent index rebuilds",
        "solution": "Incremental updates, versioning"
    }
}
\`\`\`

---

## Conclusion

Vector databases and embeddings enable:

1. **Semantic Search**: Find by meaning, not just keywords
2. **Similarity**: Measure relatedness between concepts
3. **Scale**: Search millions of vectors in milliseconds
4. **RAG**: Power retrieval for LLMs

**Key Technologies**:
- **Embeddings**: Sentence Transformers, OpenAI ada-002
- **Vector DBs**: FAISS (local), Pinecone (managed), Chroma (open-source)
- **Indexing**: HNSW for speed, IVF for scale
- **Hybrid Search**: Combine semantic + keyword
- **Reranking**: Two-stage for accuracy

**Best Practices**:
- Use all-MiniLM-L6-v2 for development
- Use ada-002 for production (if budget allows)
- Chunk documents to ~512 tokens
- Cache embeddings aggressively
- Use hybrid search + reranking
- Monitor latency and recall

**Costs** (1M vectors):
- FAISS: Free (self-hosted)
- Chroma: Free (open-source)
- Pinecone: $70/month
- OpenAI embeddings: ~$100 one-time

Vector search is the foundation for RAG, semantic search, and intelligent information retrieval.
`,
};
