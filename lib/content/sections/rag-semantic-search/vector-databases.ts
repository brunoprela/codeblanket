export const vectorDatabases = {
  title: 'Vector Databases',
  content: `
# Vector Databases

## Introduction

Vector databases are specialized storage systems designed for high-dimensional vector data. They're the backbone of modern RAG systems, enabling fast similarity search across millions of embeddings in milliseconds.

In this comprehensive section, we'll explore what vector databases are, how they work, compare popular options, and learn to build production-grade vector storage systems.

## What Are Vector Databases?

**Vector databases** are purpose-built systems for storing and querying vector embeddings. Unlike traditional databases that index text or numbers, vector databases index high-dimensional vectors and support operations like:

- **Similarity search**: Find vectors most similar to a query vector
- **Nearest neighbor search**: Locate K nearest neighbors
- **Filtered search**: Combine vector similarity with metadata filters
- **Hybrid search**: Combine dense vectors with traditional keyword search

### Why Not Use Regular Databases?

Traditional databases struggle with vector operations:

\`\`\`python
# Naive approach in PostgreSQL (slow!)
SELECT text, embedding
FROM documents
ORDER BY embedding <-> query_embedding  # Cosine distance
LIMIT 5;
# Problem: Must calculate distance for EVERY row!
# With 1M vectors: ~1M distance calculations per query
\`\`\`

Vector databases use specialized indexes (HNSW, IVF) to search efficiently:

\`\`\`python
# Vector database approach (fast!)
results = vector_db.search(query_embedding, top_k=5)
# Uses index to search ~log(N) vectors instead of N
# 1M vectors: ~20 comparisons instead of 1M!
\`\`\`

## Core Concepts

### Similarity Metrics

Vector databases support different similarity metrics:

#### 1. Cosine Similarity

Measures angle between vectors (most common for text embeddings):

\`\`\`python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: dot product of normalized vectors.
    Range: -1 to 1 (higher is more similar)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Same direction, different magnitude
print(f"Cosine similarity: {cosine_similarity(v1, v2):.3f}")  # 1.0
\`\`\`

#### 2. Euclidean Distance (L2)

Straight-line distance in vector space:

\`\`\`python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance (L2).
    Range: 0 to infinity (lower is more similar)
    """
    return np.linalg.norm(a - b)
\`\`\`

#### 3. Dot Product

Simple multiplication (fast for normalized vectors):

\`\`\`python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product similarity.
    For normalized vectors, equivalent to cosine similarity.
    """
    return np.dot(a, b)
\`\`\`

### Indexing Algorithms

Vector databases use specialized indexes for fast search:

#### HNSW (Hierarchical Navigable Small World)

Most popular algorithm for approximate nearest neighbor search:

- **Builds**: Multi-layer graph structure
- **Search**: Navigates from coarse to fine layers
- **Speed**: Very fast queries
- **Accuracy**: High recall
- **Memory**: Higher memory usage

#### IVF (Inverted File Index)

Partitions vectors into clusters:

- **Builds**: Clusters vectors using k-means
- **Search**: Searches only relevant clusters
- **Speed**: Fast for large datasets
- **Accuracy**: Good with proper tuning
- **Memory**: More memory efficient

## Popular Vector Databases

### 1. FAISS (Local/Self-Hosted)

Facebook's library for efficient similarity search (great for getting started):

\`\`\`python
import faiss
import numpy as np
from typing import List, Tuple

class FAISSVectorStore:
    """
    Local vector store using FAISS.
    Fast, free, runs locally.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension (e.g., 1536 for OpenAI)
            index_type: 'flat' (exact) or 'hnsw' (approximate)
        """
        self.dimension = dimension
        self.documents = []
        self.metadata = []
        
        # Create index
        if index_type == "flat":
            # Exact search (slower, 100% accurate)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "hnsw":
            # Approximate search (faster, ~99% accurate)
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatL2(dimension)
    
    def add(
        self,
        vectors: np.ndarray,
        documents: List[str],
        metadata: List[dict]
    ):
        """
        Add vectors to index.
        
        Args:
            vectors: Array of shape (n, dimension)
            documents: List of document texts
            metadata: List of metadata dicts
        """
        # Ensure vectors are float32 (FAISS requirement)
        vectors = vectors.astype('float32')
        
        # Add to index
        self.index.add(vectors)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        print(f"Added {len(documents)} vectors. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            k: Number of results to return
        
        Returns:
            List of (document, distance, metadata) tuples
        """
        # Ensure correct shape and type
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, filepath: str):
        """Save index to disk."""
        faiss.write_index(self.index, filepath)
        print(f"Saved index to {filepath}")
    
    def load(self, filepath: str):
        """Load index from disk."""
        self.index = faiss.read_index(filepath)
        print(f"Loaded index from {filepath}")


# Example usage
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    """Get OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

# Create FAISS store
store = FAISSVectorStore(dimension=1536, index_type="hnsw")

# Add documents
docs = [
    "RAG combines retrieval with generation",
    "Vector databases enable semantic search",
    "FAISS is a library for similarity search",
    "Embeddings capture semantic meaning"
]

# Embed and add
vectors = np.array([get_embedding(doc) for doc in docs])
metadata = [{"source": f"doc_{i}"} for i in range(len(docs))]
store.add(vectors, docs, metadata)

# Search
query = "What is semantic search?"
query_vec = get_embedding(query)
results = store.search(query_vec, k=2)

for doc, dist, meta in results:
    print(f"Distance: {dist:.3f} - {doc}")
\`\`\`

**Best For:**
- ✅ Local development
- ✅ Small to medium datasets (< 1M vectors)
- ✅ No API costs
- ✅ Full control

**Limitations:**
- ❌ No built-in cloud hosting
- ❌ Limited metadata filtering
- ❌ Requires manual scaling

### 2. Pinecone (Cloud)

Fully managed vector database with excellent developer experience:

\`\`\`python
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict

class PineconeVectorStore:
    """
    Pinecone vector database wrapper.
    Managed cloud service with excellent scaling.
    """
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """
        Initialize Pinecone.
        
        Args:
            api_key: Pinecone API key
            index_name: Name for your index
            dimension: Vector dimension
            metric: Similarity metric (cosine, euclidean, dotproduct)
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
            print(f"Created index: {index_name}")
        
        self.index = self.pc.Index(index_name)
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict]
    ):
        """
        Insert or update vectors.
        
        Args:
            vectors: List of embedding vectors
            ids: Unique IDs for each vector
            metadata: Metadata for each vector
        """
        # Format for Pinecone
        vectors_with_metadata = [
            {
                "id": id_,
                "values": vector,
                "metadata": meta
            }
            for id_, vector, meta in zip(ids, vectors, metadata)
        ]
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_with_metadata), batch_size):
            batch = vectors_with_metadata[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Upserted {len(vectors)} vectors")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filter_dict: Optional metadata filter
        
        Returns:
            List of matches with scores and metadata
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        return results.matches
    
    def delete(self, ids: List[str]):
        """Delete vectors by ID."""
        self.index.delete(ids=ids)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return self.index.describe_index_stats()


# Example usage
store = PineconeVectorStore(
    api_key="your-api-key",
    index_name="rag-demo",
    dimension=1536
)

# Add documents
documents = [
    "Pinecone is a managed vector database",
    "It scales automatically with your needs",
    "Supports metadata filtering and hybrid search"
]

vectors = [get_embedding(doc).tolist() for doc in documents]
ids = [f"doc_{i}" for i in range(len(documents))]
metadata = [{"text": doc, "source": "demo"} for doc in documents]

store.upsert(vectors, ids, metadata)

# Search
query = "What is Pinecone?"
query_vec = get_embedding(query).tolist()
results = store.search(query_vec, top_k=2)

for match in results:
    print(f"Score: {match.score:.3f}")
    print(f"Text: {match.metadata['text']}")
    print()
\`\`\`

**Best For:**
- ✅ Production applications
- ✅ Automatic scaling
- ✅ High performance
- ✅ Metadata filtering

**Pricing:**
- Free tier: 100K vectors
- Paid: ~$70/month for 1M vectors

### 3. Weaviate (Self-Hosted/Cloud)

Open-source vector database with rich features:

\`\`\`python
import weaviate
from typing import List, Dict

class WeaviateVectorStore:
    """
    Weaviate vector database wrapper.
    Open-source with both cloud and self-hosted options.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        class_name: str = "Document"
    ):
        """
        Initialize Weaviate client.
        
        Args:
            url: Weaviate instance URL
            class_name: Name of your data class
        """
        self.client = weaviate.Client(url=url)
        self.class_name = class_name
        
        # Create class if it doesn't exist
        self._create_class_if_needed()
    
    def _create_class_if_needed(self):
        """Create Weaviate class schema."""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We provide vectors
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"]
                },
                {
                    "name": "source",
                    "dataType": ["string"]
                }
            ]
        }
        
        try:
            self.client.schema.create_class(schema)
            print(f"Created class: {self.class_name}")
        except Exception as e:
            # Class might already exist
            pass
    
    def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[Dict]
    ):
        """
        Add documents with vectors.
        
        Args:
            texts: Document texts
            vectors: Embedding vectors
            metadata: Additional metadata
        """
        # Batch import
        with self.client.batch as batch:
            for text, vector, meta in zip(texts, vectors, metadata):
                properties = {
                    "text": text,
                    **meta
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=vector
                )
        
        print(f"Added {len(texts)} documents")
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            limit: Max results
        
        Returns:
            List of results with similarity scores
        """
        result = (
            self.client.query
            .get(self.class_name, ["text", "source"])
            .with_near_vector({"vector": query_vector})
            .with_limit(limit)
            .with_additional(["distance"])
            .do()
        )
        
        return result.get("data", {}).get("Get", {}).get(self.class_name, [])


# Example usage (requires Weaviate instance running)
# docker run -d -p 8080:8080 semitechnologies/weaviate:latest

store = WeaviateVectorStore(
    url="http://localhost:8080",
    class_name="RAGDocument"
)

# Add documents
docs = ["Weaviate is an open-source vector database"]
vectors = [get_embedding(doc).tolist() for doc in docs]
metadata = [{"source": "demo"}]

store.add(docs, vectors, metadata)

# Search
query_vec = get_embedding("vector database").tolist()
results = store.search(query_vec, limit=3)
\`\`\`

**Best For:**
- ✅ Open-source preference
- ✅ Self-hosting needs
- ✅ GraphQL queries
- ✅ Rich filtering

### 4. Chroma (Local Development)

Simple, lightweight vector database perfect for prototyping:

\`\`\`python
import chromadb
from typing import List, Dict

class ChromaVectorStore:
    """
    Chroma vector database wrapper.
    Simple, embeddable database for development.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize Chroma.
        
        Args:
            collection_name: Name of collection
            persist_directory: Where to store data
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: List[Dict] = None
    ):
        """
        Add documents to collection.
        
        Args:
            documents: Text documents
            embeddings: Vector embeddings
            ids: Unique IDs
            metadatas: Optional metadata
        """
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Dict = None
    ) -> Dict:
        """
        Query collection.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results
            where: Optional metadata filter
        
        Returns:
            Query results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection.name)


# Example usage
store = ChromaVectorStore(
    collection_name="my_documents",
    persist_directory="./my_vectordb"
)

# Add documents
docs = [
    "Chroma is easy to use",
    "It stores vectors locally",
    "Perfect for prototyping"
]

embeddings = [get_embedding(doc).tolist() for doc in docs]
ids = [f"id_{i}" for i in range(len(docs))]
metadata = [{"source": "tutorial"} for _ in docs]

store.add(docs, embeddings, ids, metadata)

# Query
query_emb = get_embedding("local storage").tolist()
results = store.query(query_emb, n_results=2)

print(results["documents"])
\`\`\`

**Best For:**
- ✅ Local development
- ✅ Simple setup
- ✅ Prototyping
- ✅ Embedded applications

### 5. Qdrant (Self-Hosted/Cloud)

Modern vector database with excellent features:

\`\`\`python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict

class QdrantVectorStore:
    """
    Qdrant vector database wrapper.
    Modern, performant, great for production.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        vector_size: int = 1536,
        host: str = "localhost",
        port: int = 6333
    ):
        """Initialize Qdrant client."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
        # Create collection if needed
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {collection_name}")
        except Exception:
            pass  # Collection exists
    
    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict]
    ):
        """
        Add vectors to collection.
        
        Args:
            ids: Unique IDs
            vectors: Embeddings
            payloads: Metadata/documents
        """
        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload=payload
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = None
    ) -> List[Dict]:
        """Search for similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
\`\`\`

## Comparison Matrix

| Feature | FAISS | Pinecone | Weaviate | Chroma | Qdrant |
|---------|-------|----------|----------|--------|--------|
| **Hosting** | Local | Cloud | Both | Local | Both |
| **Cost** | Free | $$$ | Free/$ | Free | Free/$ |
| **Scale** | Medium | Large | Large | Small | Large |
| **Setup** | Easy | Easiest | Medium | Easy | Easy |
| **Metadata Filter** | Limited | Excellent | Excellent | Good | Excellent |
| **Best For** | Prototyping | Production | Self-host | Dev | Production |

## Production Best Practices

\`\`\`python
class ProductionVectorDB:
    """Production-ready vector database patterns."""
    
    def __init__(self):
        self.store = None  # Your chosen DB
    
    def batch_upsert(
        self,
        vectors: List,
        batch_size: int = 100
    ):
        """Upsert in batches to avoid timeouts."""
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.store.upsert(batch)
    
    def search_with_fallback(
        self,
        query_vector: List[float],
        min_results: int = 3
    ):
        """Search with threshold fallback."""
        results = self.store.search(
            query_vector,
            score_threshold=0.7
        )
        
        if len(results) < min_results:
            # Lower threshold if too few results
            results = self.store.search(
                query_vector,
                score_threshold=0.5
            )
        
        return results
\`\`\`

## Summary

Vector databases enable fast semantic search in RAG systems. Choose based on:

- **Development**: FAISS or Chroma
- **Production**: Pinecone, Qdrant, or Weaviate
- **Self-hosted**: Weaviate or Qdrant
- **Cloud**: Pinecone

Key considerations:
- Scale requirements
- Budget
- Metadata filtering needs
- Hosting preferences
`,
};
