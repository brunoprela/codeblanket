export const contextWindowManagement = {
  title: 'Context Window Management',
  id: 'context-window-management',
  content: `
# Context Window Management

## Introduction

LLMs have limited context windows—the maximum amount of text they can process at once. While modern models support 100k-200k tokens, efficiently managing this context is critical for cost, latency, and accuracy. This section covers chunking strategies, context compression, sliding windows, and techniques to work with documents longer than the context limit.

### The Context Problem

**Token Limits**: Models have hard limits (8k, 100k, 200k tokens)
**Cost**: Longer context = higher API costs
**Latency**: More tokens = slower inference
**Quality**: Too much irrelevant context hurts performance
**Solution**: Smart context management

---

## Understanding Context Windows

### Token Limits by Model

\`\`\`python
"""
Context window sizes across models
"""

context_limits = {
    # OpenAI
    "gpt-3.5-turbo": {
        "context": "16k tokens",
        "cost": "$0.50/$1.50 per 1M (in/out)",
        "use": "Chat, moderate documents"
    },
    "gpt-4": {
        "context": "8k tokens",
        "cost": "$30/$60 per 1M",
        "use": "Complex reasoning, short context"
    },
    "gpt-4-32k": {
        "context": "32k tokens",
        "cost": "$60/$120 per 1M",
        "use": "Longer documents"
    },
    "gpt-4-turbo": {
        "context": "128k tokens",
        "cost": "$10/$30 per 1M",
        "use": "Long documents, best value"
    },
    
    # Anthropic
    "claude-3-haiku": {
        "context": "200k tokens",
        "cost": "$0.25/$1.25 per 1M",
        "use": "Fast, long context"
    },
    "claude-3-sonnet": {
        "context": "200k tokens",
        "cost": "$3/$15 per 1M",
        "use": "Best quality + long context"
    },
    "claude-3-opus": {
        "context": "200k tokens",
        "cost": "$15/$75 per 1M",
        "use": "Most capable"
    },
    
    # Open Source
    "llama-2-7b": {
        "context": "4k tokens",
        "cost": "Free (self-hosted)",
        "use": "Local, short context"
    },
    "mistral-7b": {
        "context": "8k tokens",
        "cost": "Free",
        "use": "Local, moderate context"
    }
}

def estimate_tokens (text):
    """
    Rough token estimation
    Rule of thumb: ~4 characters = 1 token
    """
    return len (text) // 4

# Example
document = "..." * 10000  # Long document
tokens = estimate_tokens (document)
print(f"Estimated tokens: {tokens}")

# Choose appropriate model
if tokens < 8000:
    model = "gpt-4"
elif tokens < 128000:
    model = "gpt-4-turbo"
else:
    model = "claude-3-sonnet"  # 200k context
\`\`\`

---

## Document Chunking

### Strategies for Long Documents

\`\`\`python
"""
Chunking strategies for documents exceeding context limits
"""

import tiktoken

class DocumentChunker:
    """
    Intelligent document chunking
    """
    
    def __init__(self, model="gpt-4", chunk_size=2000, overlap=200):
        self.encoding = tiktoken.encoding_for_model (model)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def count_tokens (self, text):
        """Accurate token counting"""
        return len (self.encoding.encode (text))
    
    def fixed_size_chunks (self, text):
        """
        Split into fixed-size chunks with overlap
        
        Overlap ensures context isn't lost at boundaries
        """
        tokens = self.encoding.encode (text)
        chunks = []
        
        start = 0
        while start < len (tokens):
            # Get chunk
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode (chunk_tokens)
            chunks.append (chunk_text)
            
            # Move forward (with overlap)
            start = end - self.overlap
        
        return chunks
    
    def semantic_chunks (self, text):
        """
        Chunk by semantic boundaries (better quality)
        """
        # Split by paragraphs/sections
        sections = text.split('\\n\\n')
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for section in sections:
            section_tokens = self.count_tokens (section)
            
            if current_tokens + section_tokens < self.chunk_size:
                # Add to current chunk
                current_chunk += section + "\\n\\n"
                current_tokens += section_tokens
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append (current_chunk)
                current_chunk = section + "\\n\\n"
                current_tokens = section_tokens
        
        if current_chunk:
            chunks.append (current_chunk)
        
        return chunks
    
    def hierarchical_chunks (self, text):
        """
        Multiple granularities for better retrieval
        
        Levels:
        1. Document (full text if fits)
        2. Sections (h1, h2 headers)
        3. Paragraphs
        4. Sentences (if needed)
        """
        import re
        
        hierarchy = {
            'document': text,
            'sections': [],
            'paragraphs': [],
            'sentences': []
        }
        
        # Split by headers
        sections = re.split (r'\\n#+\\s', text)
        for section in sections:
            if section.strip():
                hierarchy['sections'].append (section.strip())
                
                # Split into paragraphs
                paragraphs = section.split('\\n\\n')
                hierarchy['paragraphs'].extend([p.strip() for p in paragraphs if p.strip()])
        
        return hierarchy

# Example: Process long document
chunker = DocumentChunker (model="gpt-4-turbo", chunk_size=100000)

long_document = """..."""  # 500k token document

# Strategy 1: Fixed chunks
fixed_chunks = chunker.fixed_size_chunks (long_document)
print(f"Fixed chunking: {len (fixed_chunks)} chunks")

# Strategy 2: Semantic chunks
semantic_chunks = chunker.semantic_chunks (long_document)
print(f"Semantic chunking: {len (semantic_chunks)} chunks")

# Strategy 3: Hierarchical
hierarchy = chunker.hierarchical_chunks (long_document)
print(f"Sections: {len (hierarchy['sections'])}")
print(f"Paragraphs: {len (hierarchy['paragraphs'])}")
\`\`\`

---

## Sliding Window Approach

### Process Very Long Documents

\`\`\`python
"""
Sliding window for processing long documents
"""

class SlidingWindowProcessor:
    """
    Process document in overlapping windows
    
    Use case: Summarization, analysis of very long documents
    """
    
    def __init__(self, llm, window_size=100000, stride=90000):
        self.llm = llm
        self.window_size = window_size
        self.stride = stride  # Move forward by stride (overlap = window_size - stride)
    
    def process_with_sliding_window (self, document, task):
        """
        Process document in windows, aggregate results
        """
        chunker = DocumentChunker (chunk_size=self.window_size)
        tokens = chunker.encoding.encode (document)
        
        results = []
        start = 0
        
        while start < len (tokens):
            # Get window
            end = min (start + self.window_size, len (tokens))
            window_tokens = tokens[start:end]
            window_text = chunker.encoding.decode (window_tokens)
            
            # Process window
            result = self.process_window (window_text, task)
            results.append (result)
            
            # Move window
            start += self.stride
            
            # Stop if we've covered the document
            if end >= len (tokens):
                break
        
        # Aggregate results
        final_result = self.aggregate_results (results, task)
        return final_result
    
    def process_window (self, window_text, task):
        """
        Process single window
        """
        prompt = f"""{task}

Text:
{window_text}

Result:"""
        
        response = self.llm.generate (prompt)
        return response
    
    def aggregate_results (self, results, task):
        """
        Combine results from all windows
        """
        # For summarization: summarize the summaries
        if "summarize" in task.lower():
            combined = "\\n\\n".join (results)
            
            final_prompt = f"""Combine these summaries into one coherent summary:

{combined}

Final summary:"""
            
            return self.llm.generate (final_prompt)
        
        # For extraction: merge extracted items
        elif "extract" in task.lower():
            # Deduplicate and merge
            all_items = "\\n".join (results)
            return all_items
        
        # Default: concatenate
        return "\\n\\n".join (results)

# Example: Summarize 1M token document
processor = SlidingWindowProcessor(
    llm=llm,
    window_size=100000,  # 100k per window
    stride=90000  # 10k overlap
)

huge_document = """..."""  # 1M tokens

summary = processor.process_with_sliding_window(
    huge_document,
    task="Summarize the main points"
)

print(summary)
\`\`\`

---

## Context Compression

### Reduce Context Size

\`\`\`python
"""
Compress context to fit more information
"""

class ContextCompressor:
    """
    Techniques to compress context
    
    Methods:
    1. Summarization
    2. Extraction (keep only relevant)
    3. Reranking (remove least relevant)
    4. Prompt compression (LongLLMLingua)
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def compress_by_summarization (self, text, target_ratio=0.3):
        """
        Summarize to reduce size
        
        target_ratio: Keep 30% of original length
        """
        original_length = len (text.split())
        target_length = int (original_length * target_ratio)
        
        prompt = f"""Summarize this text to approximately {target_length} words, keeping key information:

{text}

Summary:"""
        
        summary = self.llm.generate (prompt)
        return summary
    
    def compress_by_extraction (self, text, query, keep_ratio=0.5):
        """
        Keep only relevant sentences
        """
        sentences = text.split('. ')
        
        # Score each sentence by relevance to query
        scores = []
        for sent in sentences:
            score = self.relevance_score (sent, query)
            scores.append((score, sent))
        
        # Keep top sentences
        scores.sort (reverse=True)
        keep_count = int (len (sentences) * keep_ratio)
        relevant_sentences = [sent for _, sent in scores[:keep_count]]
        
        # Reorder by original position
        compressed = '. '.join (relevant_sentences)
        return compressed
    
    def relevance_score (self, sentence, query):
        """
        Score sentence relevance to query
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query, sentence])
        similarity = cosine_similarity (vectors[0:1], vectors[1:2])[0][0]
        
        return similarity
    
    def compress_with_reranking (self, chunks, query, top_k=5):
        """
        Use reranker to keep most relevant chunks
        """
        from sentence_transformers import CrossEncoder
        
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Score each chunk
        pairs = [[query, chunk] for chunk in chunks]
        scores = reranker.predict (pairs)
        
        # Keep top-k
        ranked_indices = np.argsort (scores)[::-1][:top_k]
        compressed_chunks = [chunks[i] for i in ranked_indices]
        
        return compressed_chunks
    
    def adaptive_compression (self, text, query, max_tokens=100000):
        """
        Compress until fits in context
        """
        chunker = DocumentChunker()
        current_tokens = chunker.count_tokens (text)
        
        if current_tokens <= max_tokens:
            return text  # Already fits
        
        # Try extraction first
        compressed = self.compress_by_extraction (text, query, keep_ratio=0.7)
        current_tokens = chunker.count_tokens (compressed)
        
        if current_tokens <= max_tokens:
            return compressed
        
        # If still too large, summarize
        compressed = self.compress_by_summarization (compressed, target_ratio=0.5)
        return compressed

# Example
compressor = ContextCompressor (llm)

long_text = """..."""  # 200k tokens
query = "What are the main findings?"

# Compress to fit in 100k context
compressed = compressor.adaptive_compression(
    long_text,
    query,
    max_tokens=100000
)

print(f"Original: {chunker.count_tokens (long_text)} tokens")
print(f"Compressed: {chunker.count_tokens (compressed)} tokens")
\`\`\`

---

## Hierarchical Summarization

### Map-Reduce Pattern

\`\`\`python
"""
Hierarchical summarization for very long documents
"""

class HierarchicalSummarizer:
    """
    Map-Reduce pattern for summarization
    
    1. Split document into chunks
    2. Summarize each chunk (Map)
    3. Summarize the summaries (Reduce)
    4. Repeat until single summary
    """
    
    def __init__(self, llm, chunk_size=100000):
        self.llm = llm
        self.chunk_size = chunk_size
    
    def summarize (self, document):
        """
        Hierarchical summarization
        """
        # Base case: document fits in context
        chunker = DocumentChunker (chunk_size=self.chunk_size)
        tokens = chunker.count_tokens (document)
        
        if tokens <= self.chunk_size:
            return self.summarize_chunk (document)
        
        # Recursive case: split and summarize
        chunks = chunker.semantic_chunks (document)
        
        # Map: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate (chunks):
            print(f"Summarizing chunk {i+1}/{len (chunks)}")
            summary = self.summarize_chunk (chunk)
            chunk_summaries.append (summary)
        
        # Reduce: Combine summaries
        combined_summaries = "\\n\\n".join (chunk_summaries)
        
        # Recursively summarize (if still too long)
        if chunker.count_tokens (combined_summaries) > self.chunk_size:
            return self.summarize (combined_summaries)
        else:
            return self.final_summary (chunk_summaries)
    
    def summarize_chunk (self, chunk):
        """
        Summarize single chunk
        """
        prompt = f"""Summarize the key points from this text:

{chunk}

Summary:"""
        
        return self.llm.generate (prompt, max_tokens=500)
    
    def final_summary (self, summaries):
        """
        Create final comprehensive summary
        """
        combined = "\\n\\n".join([
            f"Section {i+1}:\\n{summary}"
            for i, summary in enumerate (summaries)
        ])
        
        prompt = f"""Create a comprehensive summary from these section summaries:

{combined}

Final Summary:"""
        
        return self.llm.generate (prompt, max_tokens=1000)

# Example: Summarize 1M token research paper
summarizer = HierarchicalSummarizer (llm, chunk_size=100000)

research_paper = """..."""  # 1M tokens

summary = summarizer.summarize (research_paper)
print(summary)

# Process:
# 1. Split into 10 chunks (100k each)
# 2. Summarize each → 10 summaries (5k each)
# 3. Combine summaries → 50k tokens
# 4. Final summary → 1k tokens
\`\`\`

---

## Conversation History Management

### Managing Chat Context

\`\`\`python
"""
Efficient conversation history management
"""

class ConversationManager:
    """
    Manage conversation history within context limits
    
    Strategies:
    1. Sliding window (keep last N messages)
    2. Summarization (summarize old messages)
    3. Importance-based (keep most important)
    """
    
    def __init__(self, max_context_tokens=100000):
        self.max_context_tokens = max_context_tokens
        self.messages = []
        self.summary = None
    
    def add_message (self, role, content):
        """
        Add message to conversation
        """
        self.messages.append({
            'role': role,
            'content': content,
            'tokens': self.count_tokens (content)
        })
        
        # Check if context exceeds limit
        if self.total_tokens() > self.max_context_tokens:
            self.compress_history()
    
    def total_tokens (self):
        """
        Count total tokens in conversation
        """
        total = sum (msg['tokens'] for msg in self.messages)
        if self.summary:
            total += self.count_tokens (self.summary)
        return total
    
    def compress_history (self):
        """
        Compress old messages when context full
        """
        # Keep last 10 messages (recent context)
        keep_count = 10
        
        if len (self.messages) <= keep_count:
            return
        
        # Summarize old messages
        old_messages = self.messages[:-keep_count]
        self.summary = self.summarize_messages (old_messages)
        
        # Keep only recent messages
        self.messages = self.messages[-keep_count:]
    
    def summarize_messages (self, messages):
        """
        Summarize conversation history
        """
        conversation = "\\n\\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        prompt = f"""Summarize this conversation, keeping key information and context:

{conversation}

Summary:"""
        
        return llm.generate (prompt, max_tokens=500)
    
    def get_context_for_llm (self, system_prompt):
        """
        Build context for LLM call
        """
        context = [
            {'role': 'system', 'content': system_prompt}
        ]
        
        # Add summary if exists
        if self.summary:
            context.append({
                'role': 'system',
                'content': f"Previous conversation summary: {self.summary}"
            })
        
        # Add recent messages
        for msg in self.messages:
            context.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        return context
    
    def count_tokens (self, text):
        """Token counting"""
        return len (text) // 4  # Rough estimate

# Example usage
conversation = ConversationManager (max_context_tokens=100000)

# Long conversation
for i in range(100):
    conversation.add_message('user', f"Message {i}")
    conversation.add_message('assistant', f"Response {i}")

# Get context (automatically compressed)
context = conversation.get_context_for_llm("You are a helpful assistant")

print(f"Total messages: 200")
print(f"Messages in context: {len (context)}")
print(f"Total tokens: {conversation.total_tokens()}")
\`\`\`

---

## Production Best Practices

### Optimization Strategies

\`\`\`python
"""
Production context management
"""

class ProductionContextManager:
    """
    Enterprise-grade context management
    """
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def optimize_context (self, context, query, model):
        """
        Optimize context for cost and quality
        """
        # 1. Count tokens
        total_tokens = self.count_tokens_accurate (context)
        model_limit = self.get_model_limit (model)
        
        # 2. If fits, return as-is
        if total_tokens <= model_limit * 0.8:  # Leave 20% margin
            return context
        
        # 3. Compress
        compressed = self.compress_intelligently (context, query)
        
        # 4. Verify still fits
        compressed_tokens = self.count_tokens_accurate (compressed)
        if compressed_tokens > model_limit:
            # Use smaller chunks with map-reduce
            return self.use_map_reduce (context, query)
        
        return compressed
    
    def count_tokens_accurate (self, text):
        """
        Accurate token counting (cached)
        """
        text_hash = hash (text)
        
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        import tiktoken
        encoding = tiktoken.encoding_for_model (self.config['model'])
        count = len (encoding.encode (text))
        
        self.cache[text_hash] = count
        return count
    
    def get_model_limit (self, model):
        """
        Get context limit for model
        """
        limits = {
            'gpt-4': 8192,
            'gpt-4-turbo': 128000,
            'claude-3-sonnet': 200000
        }
        return limits.get (model, 8192)
    
    def compress_intelligently (self, context, query):
        """
        Smart compression based on query
        """
        # Use reranking + extraction
        compressor = ContextCompressor (llm)
        return compressor.adaptive_compression (context, query)
    
    def monitor_costs (self):
        """
        Track context-related costs
        """
        metrics = {
            'avg_input_tokens': self.calculate_avg_input(),
            'total_cost': self.calculate_total_cost(),
            'compression_ratio': self.calculate_compression_ratio()
        }
        
        return metrics

# Cost optimization
cost_savings = {
    "No compression": {
        "avg_tokens": 150000,
        "cost_per_call": "$1.50",
        "monthly_cost": "$45,000"
    },
    "With compression": {
        "avg_tokens": 50000,
        "cost_per_call": "$0.50",
        "monthly_cost": "$15,000",
        "savings": "$30,000/month (67%)"
    }
}
\`\`\`

---

## Conclusion

Context window management enables:

1. **Handle Long Documents**: Process beyond context limits
2. **Reduce Costs**: Smaller context = lower API costs
3. **Improve Quality**: Relevant context > more context
4. **Faster Response**: Less tokens = faster inference

**Key Techniques**:
- Chunking (fixed, semantic, hierarchical)
- Sliding windows (overlapping processing)
- Compression (summarization, extraction, reranking)
- Map-Reduce (hierarchical aggregation)
- Conversation management (sliding window, summarization)

**Best Practices**:
- Use semantic chunking for quality
- Overlap chunks by 10-20%
- Compress irrelevant context
- Cache token counts
- Monitor costs closely

**Performance**:
- Compression: 50-70% size reduction
- Cost savings: 40-60% with smart management
- Quality: Minimal degradation with good compression

**Model Selection**:
- <10k tokens: GPT-4
- 10k-100k: GPT-4 Turbo
- 100k+: Claude 3 (200k context)
- >200k: Map-reduce approach

Effective context management is the difference between a $45k/month bill and a $15k/month bill while maintaining quality.
`,
};
