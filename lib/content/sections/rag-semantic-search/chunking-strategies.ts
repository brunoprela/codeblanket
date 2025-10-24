export const chunkingStrategies = {
  title: 'Chunking Strategies',
  content: `
# Chunking Strategies

## Introduction

Chunking is one of the most critical yet often overlooked aspects of building effective RAG systems. The way you split documents into smaller pieces directly impacts retrieval quality, context relevance, and ultimately the accuracy of your AI system's responses.

In this comprehensive section, we'll explore why chunking matters, dive into various chunking strategies, implement production-ready chunking systems, and learn how to optimize chunk size and overlap for your specific use case.

## Why Chunking Matters

Documents are often too large to process in a single operation. Chunking solves several critical problems:

### 1. **Token Limit Constraints**

LLMs have maximum context windows (4K, 8K, 32K, 128K tokens). Even with large context windows, retrieving entire documents is inefficient:

\`\`\`python
# Problem: Can't embed a 50,000-word document at once
large_doc = load_document("enterprise_report.pdf")  # 50K words
# embedding = get_embedding(large_doc)  # ERROR: Token limit exceeded!

# Solution: Chunk into manageable pieces
chunks = chunk_document(large_doc, chunk_size=500)
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
\`\`\`

### 2. **Retrieval Precision**

Smaller chunks enable more precise retrieval. Instead of retrieving an entire 50-page document, you retrieve the specific 2-3 paragraphs that answer the question:

**Without Chunking:**
- Query: "What is the revenue for Q3?"
- Retrieved: Entire 50-page annual report (mostly irrelevant)

**With Chunking:**
- Query: "What is the revenue for Q3?"
- Retrieved: 2 paragraphs specifically about Q3 revenue

### 3. **Context Efficiency**

Smaller, relevant chunks maximize the use of limited context windows:

\`\`\`python
# With good chunking
context = retrieve_top_chunks(query, k=5)  # 5 relevant paragraphs
# Efficient use of context window with highly relevant info

# Without chunking
context = retrieve_documents(query, k=2)  # 2 entire documents
# Context window filled with mostly irrelevant information
\`\`\`

### 4. **Semantic Cohesion**

Good chunks maintain semantic coherence - each chunk discusses a single topic or concept, making embeddings more meaningful.

## Fixed-Size Chunking

The simplest approach: split text into chunks of a fixed character or token count.

### Character-Based Fixed Chunking

\`\`\`python
from typing import List

def fixed_size_chunking(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: Source text to chunk
        chunk_size: Target size for each chunk (characters)
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Define end of chunk
        end = start + chunk_size
        
        # Extract chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap
        
        # Avoid infinite loop if overlap >= chunk_size
        if overlap >= chunk_size:
            start = end
    
    return chunks


# Example usage
document = """
Retrieval-Augmented Generation (RAG) is a powerful pattern for building AI applications.
It combines information retrieval with text generation to create systems that can access
external knowledge bases and provide accurate, grounded responses.

RAG systems work by first retrieving relevant documents from a vector database, then using
those documents as context for an LLM to generate a response. This approach reduces
hallucinations and enables the model to reference specific, up-to-date information.
""" * 10  # Repeat for demonstration

chunks = fixed_size_chunking(document, chunk_size=200, overlap=50)

print(f"Created {len(chunks)} chunks")
print(f"\\nFirst chunk:\\n{chunks[0]}")
print(f"\\nSecond chunk:\\n{chunks[1]}")
print(f"\\nOverlap: '{chunks[0][-50:]}' == '{chunks[1][:50]}'")
\`\`\`

**Advantages:**
- ✅ Simple to implement
- ✅ Predictable chunk sizes
- ✅ Fast processing

**Disadvantages:**
- ❌ May split sentences mid-word
- ❌ Ignores document structure
- ❌ No semantic awareness

### Token-Based Fixed Chunking

More accurate than character-based chunking because it counts actual tokens:

\`\`\`python
import tiktoken
from typing import List

def token_based_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    model: str = "gpt-4"
) -> List[str]:
    """
    Split text into chunks based on token count.
    
    Args:
        text: Source text
        chunk_size: Target tokens per chunk
        overlap: Overlap in tokens
        model: Model to use for tokenization
    
    Returns:
        List of text chunks
    """
    # Get tokenizer for model
    encoding = tiktoken.encoding_for_model(model)
    
    # Encode entire text
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk tokens
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move with overlap
        start = end - overlap
    
    return chunks


# Example usage
encoding = tiktoken.encoding_for_model("gpt-4")

doc = "Your document text here..." * 100
chunks = token_based_chunking(doc, chunk_size=500, overlap=50)

for i, chunk in enumerate(chunks[:3]):
    tokens = encoding.encode(chunk)
    print(f"Chunk {i+1}: {len(tokens)} tokens")
\`\`\`

## Sentence-Based Chunking

Respects sentence boundaries for better semantic coherence:

\`\`\`python
import re
from typing import List

def sentence_chunking(
    text: str,
    target_chunk_size: int = 1000,
    max_chunk_size: int = 1500
) -> List[str]:
    """
    Chunk text by sentences while targeting a specific size.
    
    Args:
        text: Source text
        target_chunk_size: Target characters per chunk
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        List of chunks at sentence boundaries
    """
    # Split into sentences using regex
    # Handles common sentence endings: . ! ? followed by space/newline
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence exceeds max, save current chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        
        # If at or above target and have content, save chunk
        elif current_size >= target_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        
        # Otherwise, add to current chunk
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# Example usage
document = """
RAG systems retrieve relevant documents. They use vector similarity search. 
The retrieved documents provide context. This context helps the LLM generate 
accurate responses. The approach reduces hallucinations significantly.

Vector databases store embeddings efficiently. They enable fast similarity search.
Popular options include Pinecone and Weaviate. Each has different trade-offs.
The choice depends on your specific requirements.
"""

chunks = sentence_chunking(document, target_chunk_size=100, max_chunk_size=150)

for i, chunk in enumerate(chunks):
    print(f"\\nChunk {i+1} ({len(chunk)} chars):\\n{chunk}")
\`\`\`

**Advantages:**
- ✅ Maintains sentence integrity
- ✅ More semantically coherent
- ✅ Better for readability

**Disadvantages:**
- ❌ Variable chunk sizes
- ❌ May still split semantic units (paragraphs, sections)

## Recursive Character Text Splitting

Advanced chunking that tries multiple separators in order of preference:

\`\`\`python
from typing import List, Optional

class RecursiveTextSplitter:
    """
    Recursively split text using hierarchy of separators.
    Inspired by LangChain's implementation.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize recursive splitter.
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators in priority order
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators in order of preference
        self.separators = separators or [
            "\\n\\n",  # Paragraph breaks
            "\\n",     # Line breaks
            ". ",     # Sentences
            "! ",
            "? ",
            ", ",     # Clauses
            " ",      # Words
            ""        # Characters
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text recursively."""
        return self._split_text(text, self.separators)
    
    def _split_text(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text.
        
        Args:
            text: Text to split
            separators: Current list of separators to try
        
        Returns:
            List of chunks
        """
        final_chunks = []
        
        # Choose the first separator that exists in text
        separator = separators[-1]  # Default to last (finest granularity)
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
        
        # Split by the chosen separator
        splits = text.split(separator) if separator else [text]
        
        # Process each split
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, go to next separator
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                # Recursively split this piece with finer separators
                if len(separators) > 1:
                    other_chunks = self._split_text(split, separators[1:])
                    final_chunks.extend(other_chunks)
                else:
                    # Last resort: force split by character
                    final_chunks.append(split[:self.chunk_size])
        
        # Merge remaining splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(
        self, 
        splits: List[str], 
        separator: str
    ) -> List[str]:
        """
        Merge small splits into chunks of appropriate size.
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few items for overlap
                overlap_items = []
                overlap_length = 0
                for item in reversed(current_chunk):
                    if overlap_length + len(item) <= self.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_length += len(item)
                    else:
                        break
                
                current_chunk = overlap_items + [split]
                current_length = overlap_length + split_len
            else:
                current_chunk.append(split)
                current_length += split_len
        
        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks


# Example usage
splitter = RecursiveTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

document = """
# RAG Systems

RAG systems combine retrieval with generation. They work in several steps.

## Step 1: Retrieval
First, relevant documents are retrieved from a vector database. The retrieval uses semantic similarity.

## Step 2: Augmentation
Retrieved documents are added as context to the prompt. This provides grounding.

## Step 3: Generation
The LLM generates a response using the provided context. This reduces hallucinations.
"""

chunks = splitter.split_text(document)

print(f"Created {len(chunks)} chunks\\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):")
    print(chunk)
    print("-" * 50)
\`\`\`

**Advantages:**
- ✅ Respects document structure
- ✅ Tries to keep semantic units together
- ✅ Hierarchical splitting
- ✅ Configurable separators

**Disadvantages:**
- ❌ More complex implementation
- ❌ Slower than simple splitting

## Semantic Chunking

Split text based on semantic similarity between sentences:

\`\`\`python
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticChunker:
    """
    Chunk text based on semantic similarity.
    Groups similar sentences together.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        max_chunk_size: int = 1000
    ):
        """
        Initialize semantic chunker.
        
        Args:
            model_name: Sentence transformer model
            similarity_threshold: Min similarity to stay in same chunk
            max_chunk_size: Maximum characters per chunk
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text semantically.
        
        Args:
            text: Source text
        
        Returns:
            List of semantic chunks
        """
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\\s+', text)
        
        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.model.encode(sentences)
        
        # Group similar sentences
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_size = len(sentence)
            
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(
                embeddings[i-1],
                embeddings[i]
            )
            
            # Decide whether to add to current chunk or start new one
            should_split = (
                similarity < self.similarity_threshold or
                current_size + sentence_size > self.max_chunk_size
            )
            
            if should_split and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _cosine_similarity(
        self, 
        v1: np.ndarray, 
        v2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between vectors."""
        return float(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )


# Example usage
chunker = SemanticChunker(
    similarity_threshold=0.6,
    max_chunk_size=500
)

document = """
Dogs are popular pets. They are loyal and friendly. Many people have dogs at home.

Cats are also common pets. They are independent animals. Cats require less attention than dogs.

Fish make great pets for small spaces. They are quiet and relaxing to watch. 
Aquariums can be beautiful decorations.

Regular exercise is important for health. Walking helps cardiovascular fitness.
Exercise reduces stress and improves mood.
"""

chunks = chunker.chunk_text(document)

print(f"Created {len(chunks)} semantic chunks\\n")
for i, chunk in enumerate(chunks):
    print(f"Semantic Chunk {i+1}:")
    print(chunk)
    print("-" * 60)
\`\`\`

**Advantages:**
- ✅ Semantically coherent chunks
- ✅ Groups related content
- ✅ Better embeddings

**Disadvantages:**
- ❌ Computationally expensive
- ❌ Requires embedding model
- ❌ Slower processing

## Markdown-Aware Chunking

Respect markdown structure (headers, lists, code blocks):

\`\`\`python
import re
from typing import List, Tuple

class MarkdownChunker:
    """
    Chunk markdown documents while preserving structure.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_markdown(self, markdown: str) -> List[dict]:
        """
        Chunk markdown document.
        
        Args:
            markdown: Markdown text
        
        Returns:
            List of chunks with metadata
        """
        # Parse markdown into sections
        sections = self._parse_sections(markdown)
        
        # Create chunks respecting section boundaries
        chunks = []
        current_chunk = ""
        current_metadata = {}
        
        for section in sections:
            section_text = section["text"]
            section_size = len(section_text)
            
            # If section fits in current chunk
            if len(current_chunk) + section_size <= self.chunk_size:
                current_chunk += section_text
                current_metadata = section["metadata"]
            else:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "metadata": current_metadata
                    })
                
                # Start new chunk with this section
                current_chunk = section_text
                current_metadata = section["metadata"]
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": current_metadata
            })
        
        return chunks
    
    def _parse_sections(self, markdown: str) -> List[dict]:
        """
        Parse markdown into sections by headers.
        
        Returns sections with their hierarchy.
        """
        lines = markdown.split('\\n')
        sections = []
        current_section = []
        current_headers = {}
        
        for line in lines:
            # Check for header
            header_match = re.match(r'^(#{1,6})\\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_section:
                    sections.append({
                        "text": '\\n'.join(current_section),
                        "metadata": current_headers.copy()
                    })
                    current_section = []
                
                # Update headers
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_headers[f"h{level}"] = title
                
                # Clear lower-level headers
                for i in range(level + 1, 7):
                    current_headers.pop(f"h{i}", None)
                
                current_section.append(line)
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                "text": '\\n'.join(current_section),
                "metadata": current_headers
            })
        
        return sections


# Example usage
markdown_doc = """
# RAG Systems

RAG combines retrieval with generation.

## Components

### Vector Database
Stores embeddings for fast search.

### Embedding Model
Converts text to vectors.

## Implementation

### Python Setup
Install required libraries.

### Code Example
Here's a basic implementation.
"""

chunker = MarkdownChunker(chunk_size=150)
chunks = chunker.chunk_markdown(markdown_doc)

for i, chunk in enumerate(chunks):
    print(f"\\nChunk {i+1}:")
    print(f"Headers: {chunk['metadata']}")
    print(f"Text: {chunk['text'][:100]}...")
\`\`\`

## Chunk Overlap Strategies

Overlap prevents context loss at chunk boundaries:

\`\`\`python
def chunking_with_smart_overlap(
    text: str,
    chunk_size: int = 1000,
    overlap_size: int = 200
) -> List[dict]:
    """
    Create chunks with overlapping context.
    
    Returns chunks with metadata about overlaps.
    """
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        chunk = {
            "id": chunk_id,
            "text": text[start:end],
            "start_pos": start,
            "end_pos": end,
            "prev_overlap": text[max(0, start-overlap_size):start] if start > 0 else None,
            "next_overlap": text[end:min(end+overlap_size, len(text))] if end < len(text) else None
        }
        
        chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap_size
        chunk_id += 1
    
    return chunks
\`\`\`

## Optimizing Chunk Size

Finding the optimal chunk size for your use case:

\`\`\`python
from typing import List, Dict
import numpy as np

def evaluate_chunk_sizes(
    documents: List[str],
    queries: List[str],
    chunk_sizes: List[int] = [200, 500, 1000, 1500]
) -> Dict[int, float]:
    """
    Evaluate different chunk sizes on your data.
    
    Returns retrieval quality scores for each size.
    """
    results = {}
    
    for size in chunk_sizes:
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = fixed_size_chunking(doc, chunk_size=size)
            all_chunks.extend(chunks)
        
        # Embed chunks (simplified)
        # chunk_embeddings = embed_all(all_chunks)
        
        # Test retrieval for each query
        # scores = test_retrieval(queries, chunk_embeddings)
        
        # For demonstration, using random scores
        avg_score = np.random.random()
        results[size] = avg_score
        
        print(f"Chunk size {size}: score = {avg_score:.3f}")
    
    return results

# Find optimal size
# optimal_size = max(results, key=results.get)
# print(f"\\nOptimal chunk size: {optimal_size}")
\`\`\`

## Production Chunking System

Complete production-ready chunking system:

\`\`\`python
from typing import List, Dict, Optional
from enum import Enum

class ChunkingStrategy(Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"

class ProductionChunker:
    """
    Production-ready chunking system with multiple strategies.
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs
    
    def chunk_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Chunk multiple documents with metadata.
        
        Args:
            documents: List of documents to chunk
            metadata: Optional metadata for each document
        
        Returns:
            List of chunks with metadata
        """
        if metadata is None:
            metadata = [{}] * len(documents)
        
        all_chunks = []
        
        for doc_id, (doc, meta) in enumerate(zip(documents, metadata)):
            # Chunk document
            chunks = self._chunk_single(doc)
            
            # Add metadata
            for chunk_id, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "metadata": meta
                }
                all_chunks.append(chunk_data)
        
        return all_chunks
    
    def _chunk_single(self, text: str) -> List[str]:
        """Chunk single document using selected strategy."""
        if self.strategy == ChunkingStrategy.FIXED:
            return fixed_size_chunking(
                text,
                self.chunk_size,
                self.chunk_overlap
            )
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return sentence_chunking(
                text,
                self.chunk_size
            )
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            splitter = RecursiveTextSplitter(
                self.chunk_size,
                self.chunk_overlap
            )
            return splitter.split_text(text)
        else:
            # Default to fixed
            return fixed_size_chunking(
                text,
                self.chunk_size,
                self.chunk_overlap
            )
\`\`\`

## Summary

Chunking strategies significantly impact RAG system performance:

- **Fixed-size chunking**: Simple, predictable, but may split semantic units
- **Sentence-based**: Maintains coherence, variable sizes
- **Recursive**: Respects document structure, hierarchical
- **Semantic**: Groups related content, computationally expensive
- **Markdown-aware**: Preserves document structure

**Best Practices:**
- Start with 500-1000 character chunks
- Use 10-20% overlap
- Choose strategy based on document type
- Test different approaches on your data
- Monitor retrieval quality metrics

In the next section, we'll explore vector databases where these chunks are stored and retrieved.
`,
};
