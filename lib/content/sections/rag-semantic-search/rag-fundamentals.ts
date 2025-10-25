export const ragFundamentals = {
  title: 'RAG Fundamentals',
  content: `
# RAG Fundamentals

## Introduction

Retrieval-Augmented Generation (RAG) represents one of the most transformative patterns in modern AI applications. It bridges the gap between the vast knowledge encoded in Large Language Models and the specific, up-to-date information stored in your documents, databases, and knowledge bases.

In this comprehensive section, we'll explore the foundational concepts of RAG, understand why it matters for production applications, and learn how to implement effective RAG systems that can search and understand large document collections.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI pattern that combines information retrieval with text generation. Instead of relying solely on an LLM's pre-trained knowledge, RAG systems:

1. **Retrieve** relevant information from external sources based on a user query
2. **Augment** the LLM prompt with the retrieved context
3. **Generate** a response that incorporates the retrieved information

This simple but powerful pattern enables LLMs to:
- Access information beyond their training cutoff date
- Reference specific documents, databases, or knowledge bases
- Provide more accurate, grounded responses with citations
- Reduce hallucinations by relying on factual retrieved content

### The RAG Workflow

\`\`\`
User Query → Retrieval System → Relevant Documents → LLM + Context → Response
\`\`\`

Here\'s a concrete example:

**Without RAG:**
- User: "What were our Q3 2024 sales figures?"
- LLM: "I don't have access to your specific sales data..."

**With RAG:**
- User: "What were our Q3 2024 sales figures?"
- System: Retrieves Q3 2024 sales report from document store
- LLM (with context): "Based on your Q3 2024 sales report, total sales were $4.2M, up 18% from Q2..."

## Why RAG Matters in Production

RAG has become essential for production AI applications because it solves several critical challenges:

### 1. **Knowledge Freshness**

LLMs have a training cutoff date. GPT-4's knowledge stops at a certain point, meaning it can't answer questions about recent events, new products, or updated documentation. RAG bridges this gap by retrieving current information.

**Example Use Case:** A customer support chatbot needs to reference the latest product documentation, which is updated weekly. RAG ensures the bot always has access to current information.

### 2. **Domain-Specific Knowledge**

While LLMs have broad general knowledge, they often lack deep expertise in niche domains or proprietary information. RAG enables you to augment the LLM with:
- Internal company documents
- Specialized technical manuals
- Proprietary research
- Confidential customer data

### 3. **Cost Efficiency**

Fine-tuning models is expensive and time-consuming. RAG provides similar benefits (access to specific knowledge) without the need to retrain models. You can update your knowledge base instantly without retraining.

### 4. **Hallucination Reduction**

LLMs sometimes "hallucinate" - generating plausible-sounding but incorrect information. RAG significantly reduces hallucinations by grounding responses in retrieved factual content.

### 5. **Transparency and Citations**

RAG systems can provide citations, showing users where information came from. This builds trust and allows users to verify information.

**Example:**
\`\`\`
Response: "The Python implementation uses AsyncIO for concurrency [1]."
Source [1]: docs/architecture.md, lines 45-67
\`\`\`

### 6. **Scalability**

RAG scales better than fine-tuning for growing knowledge bases. Adding new information is as simple as indexing new documents, rather than retraining models.

## Core Components of RAG Systems

A production RAG system consists of several key components:

### 1. **Document Store**

The repository where your source documents are stored. This could be:
- File storage (S3, local filesystem)
- Databases (PostgreSQL, MongoDB)
- Document management systems
- Content management systems (CMS)

### 2. **Chunking System**

Documents must be broken into smaller chunks for effective retrieval. The chunking strategy significantly impacts RAG performance:

\`\`\`python
# Simple chunking example
def chunk_text (text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Source text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len (text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append (chunk)
        start = end - overlap  # Create overlap
    
    return chunks

# Example usage
document = "Large document text here..."
chunks = chunk_text (document, chunk_size=500, overlap=100)
print(f"Created {len (chunks)} chunks from document")
\`\`\`

### 3. **Embedding Model**

Transforms text chunks into dense vector representations (embeddings) that capture semantic meaning:

\`\`\`python
from openai import OpenAI

client = OpenAI()

def create_embedding (text: str) -> list[float]:
    """
    Create embedding vector for text using OpenAI's model.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector (1536 dimensions for text-embedding-3-small)
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Example usage
chunk = "RAG combines retrieval with generation..."
embedding = create_embedding (chunk)
print(f"Embedding dimension: {len (embedding)}")  # 1536
\`\`\`

### 4. **Vector Database**

Stores embeddings and enables fast similarity search:

\`\`\`python
import numpy as np
from typing import List, Tuple

class SimpleVectorStore:
    """
    Simple in-memory vector store for demonstration.
    Production systems should use Pinecone, Weaviate, or similar.
    """
    
    def __init__(self):
        self.vectors = []
        self.documents = []
        self.metadata = []
    
    def add (self, vector: List[float], document: str, metadata: dict = None):
        """Add a vector to the store."""
        self.vectors.append (np.array (vector))
        self.documents.append (document)
        self.metadata.append (metadata or {})
    
    def search (self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
        
        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        query_np = np.array (query_vector)
        
        # Calculate cosine similarity with all vectors
        similarities = []
        for i, vec in enumerate (self.vectors):
            similarity = np.dot (query_np, vec) / (np.linalg.norm (query_np) * np.linalg.norm (vec))
            similarities.append((self.documents[i], float (similarity), self.metadata[i]))
        
        # Sort by similarity and return top k
        similarities.sort (key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Example usage
vector_store = SimpleVectorStore()

# Add some documents
docs = [
    "RAG combines retrieval with generation",
    "Vector databases enable semantic search",
    "Embeddings capture semantic meaning"
]

for doc in docs:
    embedding = create_embedding (doc)
    vector_store.add (embedding, doc, {"source": "tutorial"})

# Search
query = "What is semantic search?"
query_embedding = create_embedding (query)
results = vector_store.search (query_embedding, top_k=2)

for doc, score, metadata in results:
    print(f"Score: {score:.3f} - {doc}")
\`\`\`

### 5. **Retrieval System**

Handles querying the vector database and retrieving relevant chunks:

\`\`\`python
from typing import List

class Retriever:
    """
    Handles document retrieval for RAG.
    """
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def retrieve (self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            List of relevant document chunks
        """
        # Embed the query
        query_embedding = self.embedding_model (query)
        
        # Search vector store
        results = self.vector_store.search (query_embedding, top_k=top_k)
        
        # Extract documents
        documents = [doc for doc, score, metadata in results]
        
        return documents
\`\`\`

### 6. **Generation System**

Combines retrieved context with the query to generate a response:

\`\`\`python
from openai import OpenAI

client = OpenAI()

class Generator:
    """
    Handles response generation with retrieved context.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def generate (self, query: str, context_docs: List[str]) -> str:
        """
        Generate response using query and retrieved context.
        
        Args:
            query: User query
            context_docs: Retrieved relevant documents
        
        Returns:
            Generated response
        """
        # Format context
        context = "\\n\\n".join([
            f"[Document {i+1}]\\n{doc}" 
            for i, doc in enumerate (context_docs)
        ])
        
        # Create prompt with context
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
\`\`\`

## Building a Complete RAG System

Let\'s put it all together into a complete RAG system:

\`\`\`python
from typing import List
from openai import OpenAI

client = OpenAI()

class RAGSystem:
    """
    Complete RAG system combining retrieval and generation.
    """
    
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.client = OpenAI()
    
    def index_documents (self, documents: List[str], metadata: List[dict] = None):
        """
        Index documents into the RAG system.
        
        Args:
            documents: List of text documents to index
            metadata: Optional list of metadata dicts for each document
        """
        if metadata is None:
            metadata = [{}] * len (documents)
        
        print(f"Indexing {len (documents)} documents...")
        
        for i, (doc, meta) in enumerate (zip (documents, metadata)):
            # Create chunks
            chunks = chunk_text (doc, chunk_size=500, overlap=100)
            
            # Embed and store each chunk
            for j, chunk in enumerate (chunks):
                embedding = self._create_embedding (chunk)
                chunk_meta = {
                    **meta,
                    "doc_id": i,
                    "chunk_id": j,
                    "chunk_text": chunk
                }
                self.vector_store.add (embedding, chunk, chunk_meta)
        
        print(f"Indexed {len (self.vector_store.documents)} chunks")
    
    def _create_embedding (self, text: str) -> List[float]:
        """Create embedding for text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def query (self, question: str, top_k: int = 5) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
        
        Returns:
            Dict with answer and sources
        """
        # Retrieve relevant documents
        query_embedding = self._create_embedding (question)
        results = self.vector_store.search (query_embedding, top_k=top_k)
        
        # Extract documents and sources
        context_docs = [doc for doc, score, meta in results]
        sources = [meta for doc, score, meta in results]
        
        # Generate answer
        answer = self._generate_answer (question, context_docs)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context_docs
        }
    
    def _generate_answer (self, question: str, context_docs: List[str]) -> str:
        """Generate answer using retrieved context."""
        context = "\\n\\n".join([
            f"[Document {i+1}]\\n{doc}" 
            for i, doc in enumerate (context_docs)
        ])
        
        prompt = f"""Answer the question based on the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite which document you're referencing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Example usage
rag = RAGSystem()

# Index some documents
documents = [
    "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating responses.",
    "Vector databases store embeddings and enable fast similarity search. Popular options include Pinecone, Weaviate, and Chroma.",
    "Chunking strategies affect RAG performance. Common approaches include fixed-size chunking, sentence-based chunking, and semantic chunking.",
    "Embeddings are dense vector representations that capture semantic meaning. OpenAI's text-embedding-3-small produces 1536-dimensional vectors."
]

metadata = [
    {"source": "rag_intro.md", "date": "2024-01-15"},
    {"source": "vector_dbs.md", "date": "2024-01-20"},
    {"source": "chunking.md", "date": "2024-01-22"},
    {"source": "embeddings.md", "date": "2024-01-25"}
]

rag.index_documents (documents, metadata)

# Query the system
result = rag.query("What is RAG and how does it work?")
print("Answer:", result["answer"])
print("\\nSources:")
for source in result["sources"]:
    print(f"  - {source.get('source', 'unknown')}")
\`\`\`

## Chunking Strategies

The way you chunk documents significantly impacts RAG performance. Let\'s explore common strategies:

### Fixed-Size Chunking

Simple and predictable, but may split semantic units:

\`\`\`python
def fixed_size_chunking (text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    
    while start < len (text):
        end = min (start + chunk_size, len (text))
        chunks.append (text[start:end])
        start = end - overlap
    
    return chunks
\`\`\`

### Sentence-Based Chunking

Respects sentence boundaries for better coherence:

\`\`\`python
import re

def sentence_chunking (text: str, target_size: int = 1000) -> List[str]:
    """
    Chunk text by sentences, targeting a specific size.
    """
    # Split into sentences
    sentences = re.split (r'(?<=[.!?])\\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len (sentence)
        
        if current_size + sentence_size > target_size and current_chunk:
            # Save current chunk and start new one
            chunks.append(' '.join (current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append (sentence)
            current_size += sentence_size
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join (current_chunk))
    
    return chunks
\`\`\`

## Retrieval Strategies

Different retrieval strategies suit different use cases:

### Top-K Retrieval

Simplest approach - retrieve the top K most similar documents:

\`\`\`python
def top_k_retrieval (query_embedding: List[float], vector_store, k: int = 5) -> List[str]:
    """Retrieve top K most similar documents."""
    results = vector_store.search (query_embedding, top_k=k)
    return [doc for doc, score, meta in results]
\`\`\`

### Threshold-Based Retrieval

Only retrieve documents above a similarity threshold:

\`\`\`python
def threshold_retrieval(
    query_embedding: List[float], 
    vector_store, 
    threshold: float = 0.7,
    max_results: int = 10
) -> List[str]:
    """Retrieve documents above similarity threshold."""
    results = vector_store.search (query_embedding, top_k=max_results)
    
    # Filter by threshold
    filtered = [
        doc for doc, score, meta in results 
        if score >= threshold
    ]
    
    return filtered
\`\`\`

## When to Use RAG

RAG is ideal for:

✅ **Q&A over documents** - Customer support, documentation search
✅ **Knowledge management** - Internal wikis, research databases
✅ **Personalized assistants** - Chatbots with access to user data
✅ **Research tools** - Academic paper search and summarization
✅ **Code search** - Finding relevant code in large codebases
✅ **Legal/compliance** - Searching regulations and contracts

RAG may not be necessary for:

❌ **Simple factual questions** - LLM knowledge may suffice
❌ **Small knowledge bases** - Direct context in prompt may work
❌ **Real-time data needs** - May need direct API integration
❌ **Deterministic lookups** - Traditional database queries better

## RAG vs Fine-Tuning

A common question: when should you use RAG vs fine-tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge Updates** | Instant - just update docs | Slow - requires retraining |
| **Cost** | Low - no training needed | High - training costs |
| **Accuracy** | High for factual Q&A | High for style/format |
| **Transparency** | High - can show sources | Low - knowledge in weights |
| **Domain Adaptation** | Good for knowledge | Better for style/behavior |
| **Setup Time** | Fast - hours | Slow - days/weeks |

**Best Practice:** Use RAG for knowledge access, fine-tuning for behavior/style adaptation. They're complementary!

## Common RAG Patterns

### Pattern 1: Simple Q&A

User asks question → Retrieve docs → Generate answer

\`\`\`python
answer = rag.query("What is RAG?")
\`\`\`

### Pattern 2: Conversational RAG

Multi-turn conversation with context retrieval:

\`\`\`python
conversation_history = []

def conversational_query (question: str) -> str:
    # Retrieve relevant docs
    docs = retrieve (question)
    
    # Include conversation history in prompt
    messages = conversation_history + [
        {"role": "user", "content": question}
    ]
    
    # Generate with context
    answer = generate (messages, docs)
    
    # Update history
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})
    
    return answer
\`\`\`

### Pattern 3: Multi-Index RAG

Search across multiple document collections:

\`\`\`python
def multi_index_query (question: str) -> str:
    # Retrieve from multiple indexes
    docs_code = code_index.search (question)
    docs_wiki = wiki_index.search (question)
    docs_support = support_index.search (question)
    
    # Combine and generate
    all_docs = docs_code + docs_wiki + docs_support
    answer = generate (question, all_docs)
    
    return answer
\`\`\`

## Performance Considerations

### Token Limits

Be mindful of context window limits:

\`\`\`python
import tiktoken

def count_tokens (text: str, model: str = "gpt-4") -> int:
    """Count tokens in text."""
    encoding = tiktoken.encoding_for_model (model)
    return len (encoding.encode (text))

def fit_context_window (docs: List[str], max_tokens: int = 8000) -> List[str]:
    """
    Select documents that fit in context window.
    """
    selected = []
    total_tokens = 0
    
    for doc in docs:
        doc_tokens = count_tokens (doc)
        if total_tokens + doc_tokens <= max_tokens:
            selected.append (doc)
            total_tokens += doc_tokens
        else:
            break
    
    return selected
\`\`\`

### Caching

Cache embeddings and results to reduce costs:

\`\`\`python
from functools import lru_cache

@lru_cache (maxsize=1000)
def cached_embedding (text: str) -> tuple:
    """Cache embeddings for repeated queries."""
    embedding = create_embedding (text)
    return tuple (embedding)  # Lists aren't hashable, use tuple
\`\`\`

## Production Best Practices

1. **Chunk Size**: Start with 500-1000 tokens, experiment
2. **Overlap**: Use 10-20% overlap between chunks
3. **Top-K**: Retrieve 3-5 documents for most queries
4. **Metadata**: Include source, date, and other context
5. **Monitoring**: Track retrieval quality and response accuracy
6. **Citations**: Always provide sources for transparency
7. **Error Handling**: Handle cases where no relevant docs are found
8. **Caching**: Cache embeddings and frequent queries

## Summary

RAG is a foundational pattern for production AI applications. It enables:
- Access to current, domain-specific information
- Reduced hallucinations through grounded generation
- Cost-effective knowledge integration without fine-tuning
- Transparent, cited responses

The key components - chunking, embedding, vector search, and generation - work together to create powerful question-answering systems that scale to large knowledge bases.

In the following sections, we'll dive deeper into each component, exploring advanced techniques for embeddings, chunking strategies, vector databases, and production optimization.
`,
};
