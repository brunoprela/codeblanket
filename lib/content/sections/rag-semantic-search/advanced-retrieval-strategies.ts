export const advancedRetrievalStrategies = {
  title: 'Advanced Retrieval Strategies',
  content: `
# Advanced Retrieval Strategies

## Introduction

While basic top-K similarity search is a good starting point, production RAG systems require sophisticated retrieval strategies to maximize relevance and diversity. This section explores advanced techniques that significantly improve retrieval quality.

## Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity to avoid returning redundant documents.

### The Problem with Pure Similarity

Top-K similarity can return very similar documents:

\`\`\`python
query = "What is machine learning?"
results = search (query, top_k=5)

# Problem: All 5 results might be nearly identical!
# 1. "Machine learning is..."
# 2. "ML is..." (same content, different wording)
# 3. "Machine learning involves..." (redundant)
# 4. "Understanding machine learning..." (redundant)
# 5. "Machine learning explained..." (redundant)
\`\`\`

### MMR Solution

MMR selects documents that are:
1. **Relevant** to the query
2. **Diverse** from already-selected documents

\`\`\`python
from typing import List, Tuple
import numpy as np

class MMRRetriever:
    """
    Maximal Marginal Relevance retriever.
    Balances relevance and diversity.
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR retriever.
        
        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
                         0.5 is a good default
        """
        self.lambda_param = lambda_param
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        documents: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents using MMR.
        
        Args:
            query_embedding: Query vector
            doc_embeddings: Document vectors
            documents: Document texts
            k: Number of documents to retrieve
        
        Returns:
            List of (document, score) tuples
        """
        # Calculate relevance scores (similarity to query)
        relevance_scores = [
            self._cosine_sim (query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]
        
        selected_indices = []
        selected_docs = []
        remaining_indices = list (range (len (documents)))
        
        for _ in range (min (k, len (documents))):
            if not remaining_indices:
                break
            
            # Calculate MMR score for each remaining document
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component (max similarity to selected docs)
                if selected_indices:
                    max_sim_to_selected = max(
                        self._cosine_sim(
                            doc_embeddings[idx],
                            doc_embeddings[sel_idx]
                        )
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim_to_selected = 0
                
                # MMR formula
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_sim_to_selected
                )
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx, best_score = max (mmr_scores, key=lambda x: x[1])
            selected_indices.append (best_idx)
            selected_docs.append((documents[best_idx], best_score))
            remaining_indices.remove (best_idx)
        
        return selected_docs
    
    def _cosine_sim (self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(
            np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))
        )


# Example usage
retriever = MMRRetriever (lambda_param=0.7)  # Favor relevance

query_emb = np.random.randn(384)
doc_embs = [np.random.randn(384) for _ in range(10)]
docs = [f"Document {i}" for i in range(10)]

results = retriever.retrieve (query_emb, doc_embs, docs, k=5)

for doc, score in results:
    print(f"{score:.3f}: {doc}")
\`\`\`

**Lambda Parameter Tuning:**
- \`lambda=1.0\`: Pure relevance (no diversity)
- \`lambda=0.5\`: Balanced (default)
- \`lambda=0.0\`: Pure diversity (may lose relevance)

## Hypothetical Document Embeddings (HyDE)

Generate hypothetical answer, then search for documents similar to it.

### The Concept

Instead of embedding the query directly, HyDE:
1. Uses LLM to generate a hypothetical answer
2. Embeds the hypothetical answer
3. Searches for documents similar to the answer

\`\`\`python
from openai import OpenAI

client = OpenAI()

class HyDERetriever:
    """
    Hypothetical Document Embeddings retriever.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
    
    def retrieve(
        self,
        query: str,
        documents: List[str],
        doc_embeddings: List[np.ndarray],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using HyDE.
        
        Args:
            query: User query
            documents: Document texts
            doc_embeddings: Document embeddings
            k: Number of results
        
        Returns:
            Retrieved documents with scores
        """
        # Step 1: Generate hypothetical answer
        hypothetical_answer = self._generate_hypothetical_answer (query)
        
        print(f"Hypothetical answer: {hypothetical_answer[:100]}...")
        
        # Step 2: Embed hypothetical answer
        hyp_embedding = self._embed (hypothetical_answer)
        
        # Step 3: Search using hypothetical embedding
        similarities = [
            (doc, self._cosine_sim (hyp_embedding, doc_emb))
            for doc, doc_emb in zip (documents, doc_embeddings)
        ]
        
        # Sort and return top K
        similarities.sort (key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _generate_hypothetical_answer (self, query: str) -> str:
        """
        Generate hypothetical answer using LLM.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a detailed answer to the question. Be specific and comprehensive."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\\n\\nAnswer:"
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def _embed (self, text: str) -> np.ndarray:
        """Create embedding."""
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array (response.data[0].embedding)
    
    def _cosine_sim (self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity."""
        return float(
            np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))
        )


# Example
hyde = HyDERetriever()

query = "How does gradient descent work in neural networks?"
# HyDE generates hypothetical answer, then searches for similar docs
# More effective than searching with just the question!
\`\`\`

**When to Use HyDE:**
- ✅ Complex questions requiring detailed answers
- ✅ When queries are very different from document style
- ✅ Technical/academic domains

**When NOT to Use:**
- ❌ Simple factual lookups
- ❌ When query and documents have similar style
- ❌ Cost-sensitive applications (requires extra LLM call)

## Parent-Child Retrieval

Retrieve small chunks for precision, return large chunks for context.

\`\`\`python
from typing import Dict, List, Tuple

class ParentChildRetriever:
    """
    Retrieve child chunks but return parent documents.
    Best of both worlds: precision + context.
    """
    
    def __init__(self):
        self.child_chunks = []  # Small chunks for search
        self.child_embeddings = []
        self.parent_docs = {}  # parent_id -> full document
        self.child_to_parent = {}  # child_id -> parent_id
    
    def add_documents(
        self,
        documents: List[str],
        chunk_size: int = 200,
        parent_chunk_size: int = 1000
    ):
        """
        Add documents with parent-child relationships.
        
        Args:
            documents: Full documents
            chunk_size: Size of child chunks (for retrieval)
            parent_chunk_size: Size of parent chunks (for context)
        """
        for doc_id, doc in enumerate (documents):
            # Create parent chunks (larger)
            parent_chunks = self._chunk (doc, parent_chunk_size)
            
            for parent_idx, parent_chunk in enumerate (parent_chunks):
                parent_id = f"doc_{doc_id}_parent_{parent_idx}"
                self.parent_docs[parent_id] = parent_chunk
                
                # Create child chunks (smaller) within each parent
                child_chunks = self._chunk (parent_chunk, chunk_size)
                
                for child_idx, child_chunk in enumerate (child_chunks):
                    child_id = f"{parent_id}_child_{child_idx}"
                    self.child_chunks.append (child_chunk)
                    self.child_to_parent[child_id] = parent_id
                    
                    # Embed child chunk
                    embedding = self._embed (child_chunk)
                    self.child_embeddings.append (embedding)
    
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Search child chunks, return parent documents.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of (parent_doc, child_chunk, score)
        """
        # Embed query
        query_emb = self._embed (query)
        
        # Search child chunks for precision
        child_results = []
        for child_chunk, child_emb in zip(
            self.child_chunks,
            self.child_embeddings
        ):
            similarity = self._cosine_sim (query_emb, child_emb)
            child_results.append((child_chunk, similarity))
        
        # Sort by similarity
        child_results.sort (key=lambda x: x[1], reverse=True)
        
        # Get parent documents
        results = []
        seen_parents = set()
        
        for child_chunk, score in child_results:
            # Find parent for this child
            child_id = f"{child_chunk}"  # Simplified
            parent_id = self.child_to_parent.get (child_id)
            
            if parent_id and parent_id not in seen_parents:
                parent_doc = self.parent_docs[parent_id]
                results.append((parent_doc, child_chunk, score))
                seen_parents.add (parent_id)
                
                if len (results) >= k:
                    break
        
        return results
    
    def _chunk (self, text: str, size: int) -> List[str]:
        """Simple chunking."""
        return [text[i:i+size] for i in range(0, len (text), size)]
    
    def _embed (self, text: str) -> np.ndarray:
        """Embed text (placeholder)."""
        return np.random.randn(384)
    
    def _cosine_sim (self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity."""
        return float(
            np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))
        )
\`\`\`

**Benefits:**
- 🎯 **Precision**: Small chunks for accurate retrieval
- 📄 **Context**: Large chunks for complete information
- ⚡ **Efficiency**: Only embed small chunks

## Multi-Query Retrieval

Generate multiple query variations to improve recall.

\`\`\`python
class MultiQueryRetriever:
    """
    Generate multiple query variations for better recall.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def retrieve(
        self,
        query: str,
        documents: List[str],
        doc_embeddings: List[np.ndarray],
        k: int = 5,
        num_queries: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using multiple query variations.
        
        Args:
            query: Original query
            documents: Document texts
            doc_embeddings: Document embeddings
            k: Number of results
            num_queries: Number of query variations
        
        Returns:
            Merged results from all queries
        """
        # Generate query variations
        query_variations = self._generate_query_variations(
            query,
            num_queries
        )
        
        print(f"Query variations:")
        for qv in query_variations:
            print(f"  - {qv}")
        
        # Search with each query
        all_results = {}  # doc -> max score
        
        for query_var in query_variations:
            query_emb = self._embed (query_var)
            
            for doc, doc_emb in zip (documents, doc_embeddings):
                similarity = self._cosine_sim (query_emb, doc_emb)
                
                # Keep maximum score across all queries
                if doc not in all_results or similarity > all_results[doc]:
                    all_results[doc] = similarity
        
        # Sort and return top K
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def _generate_query_variations(
        self,
        query: str,
        num_variations: int
    ) -> List[str]:
        """
        Generate query variations using LLM.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {num_variations} different ways to ask the following question. Vary the wording and perspective."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.8
        )
        
        # Parse variations (simplified)
        variations = [query]  # Include original
        content = response.choices[0].message.content
        lines = [line.strip() for line in content.split('\\n') if line.strip()]
        variations.extend (lines[:num_variations])
        
        return variations[:num_variations + 1]
    
    def _embed (self, text: str) -> np.ndarray:
        """Embed text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array (response.data[0].embedding)
    
    def _cosine_sim (self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity."""
        return float(
            np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))
        )
\`\`\`

## Ensemble Retrieval

Combine multiple retrieval methods for best results.

\`\`\`python
from typing import List, Tuple, Dict

class EnsembleRetriever:
    """
    Combine multiple retrieval strategies.
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        """
        Initialize ensemble retriever.
        
        Args:
            weights: Weight for each retrieval method
        """
        self.weights = weights or {
            "semantic": 0.5,
            "keyword": 0.3,
            "mmr": 0.2
        }
    
    def retrieve(
        self,
        query: str,
        documents: List[str],
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Ensemble retrieval combining multiple methods.
        
        Args:
            query: Search query
            documents: Document collection
            k: Number of results
        
        Returns:
            Merged and ranked results
        """
        # Get results from each retriever
        semantic_results = self._semantic_retrieve (query, documents)
        keyword_results = self._keyword_retrieve (query, documents)
        mmr_results = self._mmr_retrieve (query, documents)
        
        # Combine scores
        combined_scores = {}
        
        for doc, score in semantic_results:
            combined_scores[doc] = score * self.weights["semantic"]
        
        for doc, score in keyword_results:
            combined_scores[doc] = (
                combined_scores.get (doc, 0) +
                score * self.weights["keyword"]
            )
        
        for doc, score in mmr_results:
            combined_scores[doc] = (
                combined_scores.get (doc, 0) +
                score * self.weights["mmr"]
            )
        
        # Sort and return top K
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def _semantic_retrieve(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[str, float]]:
        """Semantic search."""
        # Implement semantic search
        return []
    
    def _keyword_retrieve(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[str, float]]:
        """Keyword search."""
        # Implement keyword search
        return []
    
    def _mmr_retrieve(
        self,
        query: str,
        documents: List[str]
    ) -> List[Tuple[str, float]]:
        """MMR search."""
        # Implement MMR
        return []
\`\`\`

## Summary

Advanced retrieval strategies significantly improve RAG quality:

- **MMR**: Balances relevance with diversity
- **HyDE**: Generates hypothetical answers for better matching
- **Parent-Child**: Precision in search, context in results
- **Multi-Query**: Multiple perspectives improve recall
- **Ensemble**: Combines methods for best performance

Choose based on your use case:
- Use **MMR** to avoid redundant results
- Use **HyDE** for complex, technical queries
- Use **Parent-Child** when you need full context
- Use **Multi-Query** to improve recall
- Use **Ensemble** for maximum quality
`,
};
