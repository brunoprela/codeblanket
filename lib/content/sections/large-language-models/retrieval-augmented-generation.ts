export const retrievalAugmentedGeneration = {
  title: 'Retrieval-Augmented Generation (RAG)',
  id: 'retrieval-augmented-generation',
  content: `
# Retrieval-Augmented Generation (RAG)

## Introduction

Retrieval-Augmented Generation (RAG) combines the knowledge retrieval of search systems with the generation capabilities of LLMs. Instead of relying solely on the model's training data, RAG retrieves relevant documents and uses them as context for generation. This reduces hallucinations, enables up-to-date information, and allows LLMs to answer questions about private documents without fine-tuning.

### Why RAG

**Reduce Hallucinations**: Ground responses in retrieved facts
**Fresh Information**: Access current data beyond training cutoff
**Private Data**: Query proprietary documents without training
**Cost-Effective**: No fine-tuning needed
**Explainable**: Can cite sources

---

## Basic RAG Pipeline

### End-to-End Implementation

\`\`\`python
"""
Complete RAG system from scratch
"""

from sentence_transformers import SentenceTransformer
import faiss
import anthropic
import numpy as np

class BasicRAG:
    """
    Simple RAG implementation
    
    Steps:
    1. Index documents (embed and store)
    2. Retrieve relevant docs for query
    3. Generate answer using retrieved context
    """
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = anthropic.Anthropic()
        self.index = None
        self.documents = []
    
    def index_documents (self, documents):
        """
        Step 1: Create searchable index
        """
        self.documents = documents
        
        # Embed documents
        print(f"Embedding {len (documents)} documents...")
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add (embeddings.astype('float32'))
        
        print(f"Indexed {self.index.ntotal} documents")
    
    def retrieve (self, query, k=3):
        """
        Step 2: Retrieve relevant documents
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )
        
        # Get documents
        retrieved_docs = [self.documents[i] for i in indices[0]]
        
        return retrieved_docs
    
    def generate (self, query, context_docs):
        """
        Step 3: Generate answer with context
        """
        # Build context
        context = "\\n\\n".join([
            f"Document {i+1}:\\n{doc}"
            for i, doc in enumerate (context_docs)
        ])
        
        # Create prompt
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer based only on the context provided. If the answer is not in the context, say so.

Answer:"""
        
        # Generate
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def query (self, question, k=3):
        """
        Complete RAG pipeline
        """
        # Retrieve
        relevant_docs = self.retrieve (question, k=k)
        
        # Generate
        answer = self.generate (question, relevant_docs)
        
        return {
            'answer': answer,
            'sources': relevant_docs
        }

# Example usage
rag = BasicRAG()

# Index knowledge base
documents = [
    "Paris is the capital of France. It is known for the Eiffel Tower.",
    "The Eiffel Tower was built in 1889 for the World\'s Fair.",
    "France is a country in Western Europe.",
    "The population of Paris is about 2.2 million people."
]

rag.index_documents (documents)

# Query
question = "When was the Eiffel Tower built?"
result = rag.query (question)

print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"\\nSources:")
for i, source in enumerate (result['sources']):
    print(f"{i+1}. {source}")
\`\`\`

---

## Document Processing

### Chunking Strategies

\`\`\`python
"""
Effective document chunking for RAG
"""

class DocumentChunker:
    """
    Break documents into optimal chunks
    
    Why chunk?
    - Embeddings work best on ~512 tokens
    - Precise retrieval (specific passages vs whole docs)
    - Better context window utilization
    """
    
    def fixed_size_chunks (self, text, chunk_size=512, overlap=50):
        """
        Fixed-size chunks with overlap
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len (words), chunk_size - overlap):
            chunk = ' '.join (words[i:i + chunk_size])
            chunks.append (chunk)
            
            if i + chunk_size >= len (words):
                break
        
        return chunks
    
    def semantic_chunks (self, text):
        """
        Chunk by semantic boundaries (paragraphs, sections)
        """
        # Split by double newline (paragraphs)
        chunks = text.split('\\n\\n')
        
        # Filter empty
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # Merge small chunks
        merged = []
        current = ""
        
        for chunk in chunks:
            if len (current.split()) + len (chunk.split()) < 512:
                current += "\\n\\n" + chunk if current else chunk
            else:
                if current:
                    merged.append (current)
                current = chunk
        
        if current:
            merged.append (current)
        
        return merged
    
    def sentence_window (self, text, window_size=3):
        """
        Chunks with sentence-level windows
        
        Each chunk = 3 sentences, overlapping
        Good for: Q&A, precise retrieval
        """
        import nltk
        nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize (text)
        chunks = []
        
        for i in range(0, len (sentences), window_size - 1):
            chunk = ' '.join (sentences[i:i + window_size])
            chunks.append (chunk)
        
        return chunks
    
    def hierarchical_chunks (self, text):
        """
        Multiple granularities (sections, paragraphs, sentences)
        
        Benefits:
        - Coarse retrieval: Find relevant section
        - Fine retrieval: Get specific passage
        """
        # Split by headers (##, ###, etc.)
        sections = self._split_by_headers (text)
        
        hierarchical = []
        for section in sections:
            hierarchical.append({
                'level': 'section',
                'text': section,
                'children': []
            })
            
            # Split section into paragraphs
            paragraphs = section.split('\\n\\n')
            for para in paragraphs:
                hierarchical[-1]['children'].append({
                    'level': 'paragraph',
                    'text': para
                })
        
        return hierarchical
    
    def _split_by_headers (self, text):
        """Helper to split markdown by headers"""
        import re
        sections = re.split (r'\\n#{1,6} ', text)
        return [s.strip() for s in sections if s.strip()]

# Chunking comparison
chunking_strategies = {
    "Fixed Size": {
        "pros": "Simple, consistent size",
        "cons": "May split mid-sentence",
        "use": "General purpose"
    },
    "Semantic": {
        "pros": "Preserves meaning",
        "cons": "Variable size",
        "use": "Documents with clear structure"
    },
    "Sentence Window": {
        "pros": "Precise, contextual",
        "cons": "More chunks = slower",
        "use": "Q&A, fact extraction"
    },
    "Hierarchical": {
        "pros": "Multi-level retrieval",
        "cons": "Complex implementation",
        "use": "Large documents, technical docs"
    }
}

# Example: Process document
text = """
# Introduction
This is an introduction paragraph.

# Main Content
First paragraph of main content.

Second paragraph with more details.

# Conclusion
Final thoughts here.
"""

chunker = DocumentChunker()

# Different strategies
fixed = chunker.fixed_size_chunks (text)
semantic = chunker.semantic_chunks (text)
sentences = chunker.sentence_window (text)

print(f"Fixed: {len (fixed)} chunks")
print(f"Semantic: {len (semantic)} chunks")
print(f"Sentences: {len (sentences)} chunks")
\`\`\`

---

## Advanced RAG Techniques

### Query Transformation

\`\`\`python
"""
Improve queries before retrieval
"""

class QueryTransformer:
    """
    Transform user queries for better retrieval
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def expand_query (self, query):
        """
        Generate multiple query variations
        
        Why? Different phrasings may match different docs
        """
        prompt = f"""Generate 3 variations of this query with different phrasings:

Original: {query}

Variations:
1."""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        variations = [query] + response.content[0].text.strip().split('\\n')
        return variations
    
    def decompose_query (self, query):
        """
        Break complex query into sub-queries
        
        Example:
        "Compare Python and Java" →
        1. "What is Python?"
        2. "What is Java?"
        3. "Python vs Java comparison"
        """
        prompt = f"""Break this complex question into simpler sub-questions:

Question: {query}

Sub-questions:
1."""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        sub_queries = response.content[0].text.strip().split('\\n')
        return [q.strip() for q in sub_queries if q.strip()]
    
    def hypothetical_document (self, query):
        """
        HyDE: Generate hypothetical answer, embed that
        
        Idea: Model\'s answer may be closer to real docs than query
        """
        prompt = f"""Write a detailed answer to this question:

{query}

Answer:"""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        hypothetical_doc = response.content[0].text
        return hypothetical_doc

# Multi-query RAG
class MultiQueryRAG(BasicRAG):
    """
    Use multiple query formulations
    """
    
    def __init__(self):
        super().__init__()
        self.query_transformer = QueryTransformer (self.llm)
    
    def query (self, question, k=3):
        """
        1. Generate query variations
        2. Retrieve for each
        3. Merge and deduplicate results
        4. Generate answer
        """
        # Generate variations
        queries = self.query_transformer.expand_query (question)
        
        # Retrieve for each
        all_docs = []
        seen = set()
        
        for q in queries:
            docs = self.retrieve (q, k=k)
            for doc in docs:
                if doc not in seen:
                    all_docs.append (doc)
                    seen.add (doc)
        
        # Generate answer
        answer = self.generate (question, all_docs[:k])  # Top k unique
        
        return {
            'answer': answer,
            'sources': all_docs[:k],
            'queries_used': queries
        }
\`\`\`

### Reranking

\`\`\`python
"""
Rerank retrieved documents for better relevance
"""

from sentence_transformers import CrossEncoder

class RerankedRAG(BasicRAG):
    """
    RAG with reranking
    
    Pipeline:
    1. Vector search: Get top 20 (fast, approximate)
    2. Rerank: Score with cross-encoder (slow, accurate)
    3. Use top 3 for generation
    """
    
    def __init__(self):
        super().__init__()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def retrieve_and_rerank (self, query, retrieve_k=20, final_k=3):
        """
        Two-stage retrieval
        """
        # Stage 1: Fast vector search
        candidates = self.retrieve (query, k=retrieve_k)
        
        # Stage 2: Rerank with cross-encoder
        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict (pairs)
        
        # Sort by reranked scores
        ranked_indices = np.argsort (scores)[::-1]
        ranked_docs = [candidates[i] for i in ranked_indices[:final_k]]
        
        return ranked_docs
    
    def query (self, question, retrieve_k=20, final_k=3):
        """
        RAG with reranking
        """
        # Retrieve and rerank
        relevant_docs = self.retrieve_and_rerank(
            question,
            retrieve_k=retrieve_k,
            final_k=final_k
        )
        
        # Generate
        answer = self.generate (question, relevant_docs)
        
        return {
            'answer': answer,
            'sources': relevant_docs
        }

# Performance improvement
improvements = {
    "Basic RAG": "Retrieves top-k, may include irrelevant docs",
    "Multi-query RAG": "+15-25% recall (finds more relevant docs)",
    "Reranked RAG": "+10-20% precision (better relevance of top docs)",
    "Combined": "+30-40% overall quality"
}
\`\`\`

---

## Evaluation

### Measuring RAG Quality

\`\`\`python
"""
Evaluate RAG system performance
"""

class RAGEvaluator:
    """
    Comprehensive RAG evaluation
    """
    
    def retrieval_metrics (self, retrieved_docs, relevant_docs):
        """
        Measure retrieval quality
        
        Metrics:
        - Precision: % of retrieved docs that are relevant
        - Recall: % of relevant docs that are retrieved
        - MRR: Mean Reciprocal Rank
        - NDCG: Normalized Discounted Cumulative Gain
        """
        retrieved_set = set (retrieved_docs)
        relevant_set = set (relevant_docs)
        
        # Precision
        precision = len (retrieved_set & relevant_set) / len (retrieved_set)
        
        # Recall
        recall = len (retrieved_set & relevant_set) / len (relevant_set)
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc in enumerate (retrieved_docs):
            if doc in relevant_set:
                mrr = 1 / (i + 1)
                break
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mrr': mrr
        }
    
    def generation_metrics (self, generated_answer, reference_answer, context):
        """
        Measure generation quality
        
        Metrics:
        - Faithfulness: Does answer stay true to context?
        - Answer relevance: Does answer address the question?
        - Context relevance: Is retrieved context relevant?
        """
        # Use LLM as judge
        faithfulness = self.check_faithfulness (generated_answer, context)
        relevance = self.check_relevance (generated_answer, reference_answer)
        
        return {
            'faithfulness': faithfulness,
            'relevance': relevance
        }
    
    def check_faithfulness (self, answer, context):
        """
        Verify answer is grounded in context
        """
        prompt = f"""Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context? Answer yes or no.
If no, what claims are unsupported?

Evaluation:"""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        evaluation = response.content[0].text.lower()
        is_faithful = "yes" in evaluation
        
        return is_faithful
    
    def end_to_end_evaluation (self, test_cases):
        """
        Evaluate complete RAG pipeline
        
        test_cases: List of {question, expected_answer, relevant_docs}
        """
        results = {
            'retrieval_precision': [],
            'retrieval_recall': [],
            'faithfulness': [],
            'answer_quality': []
        }
        
        for test in test_cases:
            # Run RAG
            rag_result = self.rag.query (test['question'])
            
            # Retrieval metrics
            ret_metrics = self.retrieval_metrics(
                rag_result['sources'],
                test['relevant_docs']
            )
            results['retrieval_precision'].append (ret_metrics['precision'])
            results['retrieval_recall'].append (ret_metrics['recall'])
            
            # Generation metrics
            gen_metrics = self.generation_metrics(
                rag_result['answer'],
                test['expected_answer'],
                '\\n'.join (rag_result['sources'])
            )
            results['faithfulness'].append (gen_metrics['faithfulness'])
        
        # Aggregate
        summary = {
            'avg_precision': np.mean (results['retrieval_precision']),
            'avg_recall': np.mean (results['retrieval_recall']),
            'faithfulness_rate': np.mean (results['faithfulness'])
        }
        
        return summary

# Example evaluation
test_cases = [
    {
        'question': "When was the Eiffel Tower built?",
        'expected_answer': "1889",
        'relevant_docs': ["The Eiffel Tower was built in 1889..."]
    },
    # More test cases...
]

evaluator = RAGEvaluator (rag_system)
results = evaluator.end_to_end_evaluation (test_cases)

print(f"Precision: {results['avg_precision']:.2%}")
print(f"Recall: {results['avg_recall']:.2%}")
print(f"Faithfulness: {results['faithfulness_rate']:.2%}")
\`\`\`

---

## Production RAG Systems

### Complete Implementation

\`\`\`python
"""
Production-ready RAG system
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import anthropic
from typing import List, Dict
import logging

class ProductionRAG:
    """
    Enterprise RAG system with all bells and whistles
    """
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        self.setup_components()
    
    def setup_logging (self):
        """
        Logging for debugging and monitoring
        """
        logging.basicConfig (level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_components (self):
        """
        Initialize all components
        """
        # Embedding model
        self.embedder = SentenceTransformer(
            self.config['embedding_model']
        )
        
        # Vector database
        self.db = chromadb.PersistentClient(
            path=self.config['db_path']
        )
        self.collection = self.db.get_or_create_collection(
            name=self.config['collection_name']
        )
        
        # Reranker
        self.reranker = CrossEncoder(
            self.config['reranker_model']
        )
        
        # LLM
        self.llm = anthropic.Anthropic(
            api_key=self.config['anthropic_key']
        )
    
    def ingest_documents (self, documents: List[str], metadata: List[Dict] = None):
        """
        Ingest and index documents
        
        Features:
        - Chunking
        - Deduplication
        - Metadata tracking
        - Progress monitoring
        """
        self.logger.info (f"Ingesting {len (documents)} documents...")
        
        # Chunk documents
        chunker = DocumentChunker()
        all_chunks = []
        all_metadata = []
        
        for i, doc in enumerate (documents):
            chunks = chunker.semantic_chunks (doc)
            all_chunks.extend (chunks)
            
            # Add metadata
            for chunk in chunks:
                meta = metadata[i] if metadata else {}
                meta['doc_id'] = i
                meta['chunk_index'] = len (all_chunks) - 1
                all_metadata.append (meta)
        
        # Deduplicate
        unique_chunks, unique_metadata = self._deduplicate(
            all_chunks,
            all_metadata
        )
        
        # Add to vector DB
        self.collection.add(
            documents=unique_chunks,
            metadatas=unique_metadata,
            ids=[f"chunk_{i}" for i in range (len (unique_chunks))]
        )
        
        self.logger.info (f"Indexed {len (unique_chunks)} unique chunks")
    
    def _deduplicate (self, chunks, metadata):
        """
        Remove duplicate chunks
        """
        seen = set()
        unique_chunks = []
        unique_metadata = []
        
        for chunk, meta in zip (chunks, metadata):
            # Simple hash-based dedup
            chunk_hash = hash (chunk)
            if chunk_hash not in seen:
                seen.add (chunk_hash)
                unique_chunks.append (chunk)
                unique_metadata.append (meta)
        
        return unique_chunks, unique_metadata
    
    def query(
        self,
        question: str,
        retrieve_k: int = 20,
        final_k: int = 3,
        filters: Dict = None
    ) -> Dict:
        """
        Complete RAG query
        """
        self.logger.info (f"Query: {question}")
        
        # Step 1: Retrieve candidates
        candidates = self.collection.query(
            query_texts=[question],
            n_results=retrieve_k,
            where=filters
        )
        
        retrieved_docs = candidates['documents'][0]
        retrieved_metadata = candidates['metadatas'][0]
        
        self.logger.info (f"Retrieved {len (retrieved_docs)} candidates")
        
        # Step 2: Rerank
        pairs = [[question, doc] for doc in retrieved_docs]
        scores = self.reranker.predict (pairs)
        
        # Sort by score
        ranked_indices = np.argsort (scores)[::-1]
        final_docs = [retrieved_docs[i] for i in ranked_indices[:final_k]]
        final_metadata = [retrieved_metadata[i] for i in ranked_indices[:final_k]]
        
        self.logger.info (f"Reranked to top {final_k}")
        
        # Step 3: Generate answer
        answer = self._generate_answer (question, final_docs)
        
        # Step 4: Verify faithfulness
        is_faithful = self._check_faithfulness (answer, final_docs)
        
        if not is_faithful:
            self.logger.warning("Answer may not be fully faithful to sources")
        
        return {
            'answer': answer,
            'sources': final_docs,
            'metadata': final_metadata,
            'faithful': is_faithful
        }
    
    def _generate_answer (self, question, context_docs):
        """
        Generate answer with citations
        """
        context = "\\n\\n".join([
            f"[{i+1}] {doc}"
            for i, doc in enumerate (context_docs)
        ])
        
        prompt = f"""Answer the question based on the provided sources. Include citations [1], [2], etc.

Sources:
{context}

Question: {question}

Requirements:
- Answer based only on the sources
- Include citations for each claim
- If the answer is not in the sources, say so
- Be concise but complete

Answer:"""
        
        response = self.llm.messages.create(
            model=self.config['llm_model'],
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _check_faithfulness (self, answer, sources):
        """
        Verify answer is grounded in sources
        """
        context = "\\n\\n".join (sources)
        
        prompt = f"""Verify if this answer is fully supported by the sources.

Sources:
{context}

Answer:
{answer}

Is every claim in the answer supported by the sources? Answer yes or no.

Verification:"""
        
        response = self.llm.messages.create(
            model="claude-3-haiku-20240307",  # Fast model for verification
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return "yes" in response.content[0].text.lower()

# Configuration
config = {
    'embedding_model': 'all-MiniLM-L6-v2',
    'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'llm_model': 'claude-3-5-sonnet-20241022',
    'db_path': './rag_db',
    'collection_name': 'documents',
    'anthropic_key': 'your-key'
}

# Initialize
rag = ProductionRAG(config)

# Ingest documents
documents = [...]  # Your documents
rag.ingest_documents (documents)

# Query
result = rag.query("What is machine learning?")
print(result['answer'])
\`\`\`

---

## Conclusion

RAG enables LLMs to:

1. **Access External Knowledge**: Query documents beyond training data
2. **Reduce Hallucinations**: Ground responses in retrieved facts
3. **Stay Current**: Use up-to-date information
4. **Private Data**: Query proprietary documents without fine-tuning

**Key Components**:
- Document chunking (semantic, 512 tokens)
- Vector search (FAISS, Pinecone, Chroma)
- Reranking (cross-encoder for precision)
- Generation (with citations)
- Evaluation (retrieval + generation metrics)

**Best Practices**:
- Chunk semantically, ~512 tokens
- Use hybrid search (semantic + keyword)
- Rerank with cross-encoder
- Generate with citations
- Verify faithfulness
- Monitor metrics

**Performance**:
- Basic RAG: 60-70% answer quality
- + Multi-query: 70-80%
- + Reranking: 75-85%
- + Verification: 80-90%

**Costs** (1M queries):
- Embeddings: $100 (one-time)
- Vector DB: $70/month (Pinecone)
- LLM calls: $5,000-$15,000
- Total: ~$0.005-$0.015 per query

RAG is the most practical way to give LLMs access to your private data without expensive fine-tuning.
`,
};
