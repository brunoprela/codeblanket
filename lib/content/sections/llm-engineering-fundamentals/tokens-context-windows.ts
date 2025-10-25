/**
 * Tokens, Context Windows & Limitations Section
 * Module 1: LLM Engineering Fundamentals
 */

export const tokenscontextwindowsSection = {
  id: 'tokens-context-windows',
  title: 'Tokens, Context Windows & Limitations',
  content: `# Tokens, Context Windows & Limitations

Master token management and context windows to build efficient, cost-effective LLM applications.

## What Are Tokens?

Tokens are the fundamental units that LLMs process. Understanding tokens is critical for:
- **Cost** - You pay per token
- **Context limits** - Models have maximum token limits
- **Performance** - More tokens = slower processing

### Tokenization Basics

\`\`\`python
"""
Tokenization examples:

"Hello world" → ["Hello", " world"] → 2 tokens
"ChatGPT" → ["Chat", "GPT"] → 2 tokens  
"indivisible" → ["ind", "iv", "is", "ible"] → 4 tokens
"   " → ["   "] → 1 token (whitespace)
"🎉" → ["🎉"] → 1-2 tokens (emoji)

Rough rule: 1 token ≈ 4 characters ≈ 0.75 words
"""

# But this is just an approximation!
# Real tokenization is more complex
\`\`\`

### Tokenization Algorithms

Modern LLMs use **Byte Pair Encoding (BPE)** or similar:

\`\`\`python
"""
BPE Process:
1. Start with character vocabulary
2. Find most frequent character pairs
3. Merge them into single tokens
4. Repeat until vocabulary size reached

Example:
"aaabdaaabac" 
→ "aa" is frequent, merge to "Z"
→ "ZabdZabac"
→ "Za" is frequent, merge to "Y"
→ "YbdYbac"
... and so on

This creates efficient subword units!
"""
\`\`\`

## Accurate Token Counting with tiktoken

Never estimate in production - use tiktoken for accurate counts.

### Setup and Basic Usage

\`\`\`python
# pip install tiktoken

import tiktoken

# Get encoding for specific model
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Count tokens in text
text = "Hello, how are you doing today?"
tokens = encoding.encode (text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len (tokens)}")
# Output: Token count: 8

# Decode back to text
decoded = encoding.decode (tokens)
print(f"Decoded: {decoded}")

# See individual tokens
for token in tokens:
    print(f"{token}: {encoding.decode([token])}")
\`\`\`

### Counting Tokens in Messages

\`\`\`python
import tiktoken
from typing import List, Dict

def count_tokens_in_messages(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo"
) -> int:
    """
    Count tokens in a list of messages.
    
    Based on OpenAI's cookbook:
    https://github.com/openai/openai-cookbook
    """
    try:
        encoding = tiktoken.encoding_for_model (model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Different models have different token overhead
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        tokens_per_message = 4  # every message has metadata
        tokens_per_name = -1    # if name field used
    else:
        tokens_per_message = 3
        tokens_per_name = 1
    
    num_tokens = 0
    
    for message in messages:
        num_tokens += tokens_per_message
        
        for key, value in message.items():
            num_tokens += len (encoding.encode (value))
            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3  # every reply is primed with assistant
    
    return num_tokens

# Example
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you explain it more simply?"}
]

token_count = count_tokens_in_messages (messages)
print(f"Total tokens in conversation: {token_count}")
\`\`\`

### Token Counter Utility Class

\`\`\`python
import tiktoken
from typing import List, Dict, Optional

class TokenCounter:
    """
    Utility class for token counting across models.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model (model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count (self, text: str) -> int:
        """Count tokens in text."""
        return len (self.encoding.encode (text))
    
    def count_messages (self, messages: List[Dict[str, str]]) -> Dict:
        """
        Count tokens in messages with detailed breakdown.
        """
        tokens_per_message = 4 if "gpt" in self.model else 3
        
        total = 0
        breakdown = []
        
        for msg in messages:
            msg_tokens = tokens_per_message
            content_tokens = len (self.encoding.encode (msg['content']))
            msg_tokens += content_tokens
            
            total += msg_tokens
            breakdown.append({
                'role': msg['role'],
                'content_preview': msg['content'][:50] + "...",
                'tokens': msg_tokens
            })
        
        total += 3  # assistant priming
        
        return {
            'total_tokens': total,
            'message_breakdown': breakdown,
            'average_per_message': total / len (messages) if messages else 0
        }
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost based on token counts."""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            'gpt-4-turbo-preview': {'prompt': 10.00, 'completion': 30.00},
            'gpt-4': {'prompt': 30.00, 'completion': 60.00},
            'gpt-3.5-turbo': {'prompt': 0.50, 'completion': 1.50},
        }
        
        rates = pricing.get (self.model, pricing['gpt-3.5-turbo'])
        
        prompt_cost = (prompt_tokens / 1_000_000) * rates['prompt']
        completion_cost = (completion_tokens / 1_000_000) * rates['completion']
        
        return prompt_cost + completion_cost
    
    def fits_in_context(
        self,
        messages: List[Dict[str, str]],
        max_context: int,
        response_buffer: int = 1000
    ) -> Dict:
        """
        Check if messages fit in context window.
        
        Args:
            messages: Conversation messages
            max_context: Model\'s context window size
            response_buffer: Tokens to reserve for response
        """
        used_tokens = self.count_messages (messages)['total_tokens']
        available_for_response = max_context - used_tokens
        
        return {
            'fits': available_for_response >= response_buffer,
            'used_tokens': used_tokens,
            'available_tokens': available_for_response,
            'max_context': max_context,
            'usage_percent': (used_tokens / max_context) * 100,
            'can_respond': available_for_response > 0
        }

# Usage
counter = TokenCounter("gpt-3.5-turbo")

# Count single text
text = "This is a test message to count tokens."
print(f"Tokens: {counter.count (text)}")

# Count messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain Python in detail."}
]

result = counter.count_messages (messages)
print(f"Total: {result['total_tokens']}")
for msg in result['message_breakdown']:
    print(f"  {msg['role']}: {msg['tokens']} tokens")

# Check if fits in context
fit_check = counter.fits_in_context (messages, max_context=4096)
print(f"Fits in context: {fit_check['fits']}")
print(f"Usage: {fit_check['usage_percent']:.1f}%")
\`\`\`

## Context Window Limits

Different models have different context windows. Know your limits!

### Model Context Windows (2024)

\`\`\`python
CONTEXT_WINDOWS = {
    # OpenAI
    "gpt-4-turbo-preview": 128_000,    # 128K tokens
    "gpt-4": 8_192,                     # 8K tokens
    "gpt-4-32k": 32_768,                # 32K tokens
    "gpt-3.5-turbo": 16_385,            # 16K tokens
    "gpt-3.5-turbo-16k": 16_385,        # 16K tokens
    
    # Anthropic
    "claude-3-opus-20240229": 200_000,  # 200K tokens!
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    
    # Google
    "gemini-pro": 32_768,               # 32K tokens
    "gemini-pro-vision": 16_384,        # 16K tokens
    
    # Open Source
    "llama-3-70b": 8_192,               # 8K tokens
    "mixtral-8x7b": 32_768,             # 32K tokens
}

def get_context_window (model: str) -> int:
    """Get context window for model."""
    return CONTEXT_WINDOWS.get (model, 4096)  # default 4K

def recommend_model_for_context (required_tokens: int) -> List[str]:
    """Recommend models that can handle required tokens."""
    suitable = []
    
    for model, context_size in CONTEXT_WINDOWS.items():
        if context_size >= required_tokens:
            suitable.append({
                'model': model,
                'context_size': context_size,
                'overhead': context_size - required_tokens
            })
    
    # Sort by overhead (prefer tighter fit for cost)
    suitable.sort (key=lambda x: x['overhead'])
    
    return suitable

# Usage
required = 50_000  # Need 50K tokens
recommendations = recommend_model_for_context (required)

print(f"Models that can handle {required:,} tokens:")
for rec in recommendations[:5]:
    print(f"  {rec['model']}: {rec['context_size']:,} tokens")
\`\`\`

### The Context Window Problem

\`\`\`python
"""
Why Context Windows Matter:

1. COST
   - Input tokens cost money
   - Larger context = higher cost per request
   - Repeated information is expensive

2. LATENCY
   - More tokens = slower processing
   - Linear relationship typically
   - Can significantly impact UX

3. QUALITY
   - Very long contexts can confuse models
   - "Lost in the middle" problem
   - Important info can be overlooked

4. LIMITS
   - Hard limits on context size
   - Exceeding causes errors
   - Must truncate or summarize
"""

def analyze_context_implications(
    token_count: int,
    model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Analyze the implications of a given context size.
    """
    context_window = get_context_window (model)
    
    # Estimate latency (rough)
    base_latency = 0.5  # seconds
    token_latency = token_count * 0.0001  # 0.1ms per token
    estimated_latency = base_latency + token_latency
    
    # Estimate cost
    counter = TokenCounter (model)
    prompt_cost = counter.estimate_cost (token_count, 0)
    
    return {
        'token_count': token_count,
        'context_window': context_window,
        'usage_percent': (token_count / context_window) * 100,
        'fits': token_count < context_window,
        'estimated_latency_seconds': estimated_latency,
        'estimated_prompt_cost': prompt_cost,
        'recommendation': 'OK' if token_count < context_window * 0.8 else 'TRUNCATE'
    }

# Example
analysis = analyze_context_implications(10_000, "gpt-3.5-turbo")
print(f"Token count: {analysis['token_count']:,}")
print(f"Context usage: {analysis['usage_percent']:.1f}%")
print(f"Estimated latency: {analysis['estimated_latency_seconds']:.2f}s")
print(f"Recommendation: {analysis['recommendation']}")
\`\`\`

## Strategies for Long Documents

When your content exceeds context limits, use these strategies.

### Strategy 1: Chunking

\`\`\`python
import tiktoken
from typing import List

def chunk_text(
    text: str,
    max_tokens: int = 1000,
    overlap: int = 100,
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Split text into chunks that fit in context.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Token overlap between chunks for context
        model: Model to tokenize for
    """
    encoding = tiktoken.encoding_for_model (model)
    
    # Tokenize entire text
    tokens = encoding.encode (text)
    
    chunks = []
    start = 0
    
    while start < len (tokens):
        # Get chunk
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = encoding.decode (chunk_tokens)
        chunks.append (chunk_text)
        
        # Move start forward (with overlap)
        start = end - overlap
    
    return chunks

# Usage
long_text = "Your very long document text here... " * 1000

chunks = chunk_text (long_text, max_tokens=500, overlap=50)
print(f"Split into {len (chunks)} chunks")
print(f"First chunk length: {len (chunks[0])} chars")
\`\`\`

### Strategy 2: Summarization

\`\`\`python
from openai import OpenAI
from typing import List

def summarize_long_text(
    text: str,
    model: str = "gpt-3.5-turbo",
    max_context: int = 4000
) -> str:
    """
    Summarize text that's too long for context.
    Uses chunking + summarization + final summary.
    """
    client = OpenAI()
    counter = TokenCounter (model)
    
    # Check if summarization needed
    token_count = counter.count (text)
    
    if token_count < max_context * 0.5:
        return text  # No summarization needed
    
    # Chunk the text
    chunks = chunk_text (text, max_tokens=2000)
    
    # Summarize each chunk
    chunk_summaries = []
    
    for i, chunk in enumerate (chunks):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Summarize the following text concisely, preserving key information."},
                {"role": "user", "content": chunk}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        chunk_summaries.append (summary)
        print(f"Summarized chunk {i+1}/{len (chunks)}")
    
    # Combine summaries
    combined = "\\n\\n".join (chunk_summaries)
    
    # If still too long, summarize the summaries!
    combined_tokens = counter.count (combined)
    
    if combined_tokens > max_context * 0.5:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Create a comprehensive summary from these section summaries."},
                {"role": "user", "content": combined}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        final_summary = response.choices[0].message.content
        print(f"Created final summary")
        return final_summary
    
    return combined

# Usage
long_document = "..." * 10000  # Very long text

summary = summarize_long_text (long_document)
print(f"Original: {len (long_document)} chars")
print(f"Summary: {len (summary)} chars")
\`\`\`

### Strategy 3: Retrieval (RAG)

\`\`\`python
"""
Retrieval-Augmented Generation (RAG):
Instead of putting entire document in context,
only retrieve relevant sections.

Steps:
1. Split document into chunks
2. Create embeddings for chunks
3. Store in vector database
4. When user asks question:
   - Get question embedding
   - Find most similar chunks
   - Only send those chunks to LLM

Benefits:
- Handles unlimited document size
- Only relevant context used
- More cost-effective
- Better quality (focused context)

We'll cover RAG in detail in Module 11!
"""

def simple_retrieval_example(
    document: str,
    question: str,
    top_k: int = 3
) -> str:
    """
    Simplified retrieval example (no embeddings).
    In production, use embeddings + vector DB.
    """
    # Chunk document
    chunks = chunk_text (document, max_tokens=500)
    
    # Simple keyword matching (not ideal, but illustrative)
    question_words = set (question.lower().split())
    
    # Score chunks by overlap with question
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set (chunk.lower().split())
        overlap = len (question_words & chunk_words)
        scored_chunks.append((chunk, overlap))
    
    # Get top chunks
    scored_chunks.sort (key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
    
    # Combine relevant chunks
    context = "\\n\\n---\\n\\n".join (relevant_chunks)
    
    return context

# Usage
doc = "Long document about Python programming..." * 100
question = "How do Python decorators work?"

relevant_context = simple_retrieval_example (doc, question, top_k=3)
print(f"Retrieved {len (relevant_context)} characters of relevant context")

# Now send only relevant context to LLM
# messages = [
#     {"role": "system", "content": "Answer based on context provided."},
#     {"role": "user", "content": f"Context:\\n{relevant_context}\\n\\nQuestion: {question}"}
# ]
\`\`\`

### Strategy 4: Hierarchical Summarization

\`\`\`python
from typing import List
from openai import OpenAI

def hierarchical_summarize(
    text: str,
    model: str = "gpt-3.5-turbo",
    target_tokens: int = 1000
) -> str:
    """
    Summarize using hierarchical approach:
    - Level 1: Summarize small chunks
    - Level 2: Summarize those summaries
    - Repeat until target length
    """
    client = OpenAI()
    counter = TokenCounter (model)
    
    current_text = text
    level = 1
    
    while counter.count (current_text) > target_tokens:
        print(f"Level {level}: {counter.count (current_text)} tokens")
        
        # Chunk current text
        chunks = chunk_text (current_text, max_tokens=2000)
        
        if len (chunks) == 1:
            # Can't chunk further, just summarize directly
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"Summarize to approximately {target_tokens} tokens."},
                    {"role": "user", "content": current_text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Summarize concisely, preserving key information."},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3,
                max_tokens=500
            )
            summaries.append (response.choices[0].message.content)
        
        # Combine summaries for next level
        current_text = "\\n\\n".join (summaries)
        level += 1
    
    print(f"Final: {counter.count (current_text)} tokens")
    return current_text

# Usage
huge_document = "..." * 50000
compressed = hierarchical_summarize (huge_document, target_tokens=1000)
print(f"Compressed from ~{len (huge_document)} to {len (compressed)} chars")
\`\`\`

## Token Optimization Techniques

Reduce costs without sacrificing quality.

### Remove Unnecessary Tokens

\`\`\`python
import re
from typing import str

def optimize_prompt (prompt: str) -> str:
    """
    Remove unnecessary tokens from prompt.
    """
    # Remove extra whitespace
    prompt = re.sub (r'\\s+', ' ', prompt)
    
    # Remove comments if code
    # (be careful - might remove important context)
    
    # Remove redundant phrases
    redundant = [
        "Please ",
        "Could you please ",
        "I would like you to ",
        "Can you ",
    ]
    
    for phrase in redundant:
        prompt = prompt.replace (phrase, "")
    
    # Trim
    prompt = prompt.strip()
    
    return prompt

# Example
verbose = """
Please could you help me understand what Python decorators are?
I would like you to explain them in simple terms with examples.
Can you also tell me when to use them?
"""

optimized = optimize_prompt (verbose)

print(f"Original: {len (verbose)} chars")
print(f"Optimized: {len (optimized)} chars")
print(f"Savings: {(1 - len (optimized)/len (verbose)) * 100:.1f}%")
\`\`\`

### Use Shorter Models When Possible

\`\`\`python
def recommend_model_for_task(
    task_complexity: str,
    input_size: int,
    budget: str
) -> str:
    """
    Recommend most cost-effective model for task.
    """
    # Simple tasks -> cheaper models
    if task_complexity == "simple":
        if input_size < 1000:
            return "gpt-3.5-turbo"  # Cheapest
        else:
            return "claude-3-haiku"  # Cheap + fast
    
    # Medium complexity
    if task_complexity == "medium":
        if budget == "low":
            return "gpt-3.5-turbo"
        return "claude-3-sonnet"
    
    # Complex tasks
    if input_size > 50_000:
        return "claude-3-opus"  # Large context
    
    return "gpt-4-turbo"  # Best quality

# Usage
model = recommend_model_for_task(
    task_complexity="simple",
    input_size=500,
    budget="low"
)
print(f"Recommended: {model}")
\`\`\`

## Production Token Management

\`\`\`python
from dataclasses import dataclass
from typing import Optional
import tiktoken

@dataclass
class TokenBudget:
    """Manage token budget for a request."""
    max_total: int
    used_for_prompt: int = 0
    reserved_for_response: int = 1000
    
    @property
    def available_for_context (self) -> int:
        """Tokens available for context."""
        return self.max_total - self.reserved_for_response
    
    @property
    def remaining (self) -> int:
        """Tokens remaining for response."""
        return self.max_total - self.used_for_prompt
    
    def can_fit (self, additional_tokens: int) -> bool:
        """Check if additional tokens fit in budget."""
        return self.used_for_prompt + additional_tokens < self.available_for_context
    
    def add_tokens (self, count: int) -> bool:
        """Add tokens to budget. Returns False if exceeds."""
        if not self.can_fit (count):
            return False
        self.used_for_prompt += count
        return True

class TokenManager:
    """Production token manager with budgeting."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.counter = TokenCounter (model)
        self.context_window = get_context_window (model)
    
    def create_budget(
        self,
        expected_response_tokens: int = 1000
    ) -> TokenBudget:
        """Create token budget for request."""
        return TokenBudget(
            max_total=self.context_window,
            reserved_for_response=expected_response_tokens
        )
    
    def fit_messages_to_budget(
        self,
        messages: List[Dict[str, str]],
        budget: TokenBudget
    ) -> List[Dict[str, str]]:
        """
        Fit messages into token budget, truncating if needed.
        """
        # Always keep system message
        result = []
        if messages and messages[0]['role'] == 'system':
            system_msg = messages[0]
            system_tokens = self.counter.count (system_msg['content'])
            budget.add_tokens (system_tokens)
            result.append (system_msg)
            messages = messages[1:]
        
        # Add messages from most recent until budget full
        for msg in reversed (messages):
            msg_tokens = self.counter.count (msg['content']) + 4
            if budget.can_fit (msg_tokens):
                budget.add_tokens (msg_tokens)
                result.insert(1 if result else 0, msg)
            else:
                break
        
        return result

# Usage
manager = TokenManager("gpt-3.5-turbo")

# Create budget
budget = manager.create_budget (expected_response_tokens=500)
print(f"Available for context: {budget.available_for_context:,} tokens")

# Fit messages
long_conversation = [
    {"role": "system", "content": "You are helpful."}
] + [
    {"role": "user", "content": f"Question {i}?"},
    {"role": "assistant", "content": "Answer..." * 100}
] * 100  # Way too long!

fitted = manager.fit_messages_to_budget (long_conversation, budget)
print(f"Original messages: {len (long_conversation)}")
print(f"Fitted messages: {len (fitted)}")
print(f"Tokens used: {budget.used_for_prompt:,}")
print(f"Remaining for response: {budget.remaining:,}")
\`\`\`

## Key Takeaways

1. **Tokens are not words** - 1 token ≈ 4 chars ≈ 0.75 words
2. **Use tiktoken** for accurate counting in production
3. **Context windows vary widely** - 8K to 200K tokens
4. **Input tokens cost money** - more context = higher cost
5. **Manage token budgets** - reserve space for responses
6. **Chunk long documents** with overlap for context
7. **Summarize when needed** - hierarchical approach works well
8. **Use RAG** for very long documents (covered in Module 11)
9. **Optimize prompts** - remove unnecessary tokens
10. **Monitor token usage** - it adds up quickly!

## Next Steps

Now you understand tokens and context limits. Next: **Temperature, Top-P & Sampling Parameters** - learning to control the randomness and creativity of model outputs.`,
};
