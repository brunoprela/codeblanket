/**
 * Caching & Performance Section
 * Module 1: LLM Engineering Fundamentals
 */

export const cachingperformanceSection = {
    id: 'caching-performance',
    title: 'Caching & Performance',
    content: `# Caching & Performance

Master caching strategies to dramatically reduce costs and improve performance in production LLM applications.

## Why Caching Matters

Caching can reduce costs by 80-90% and latency by 95%+.

### The Impact of Caching

\`\`\`python
"""
WITHOUT CACHING:
Request: "What is Python?"
- API call: 500ms
- Cost: $0.001
- Every time!

WITH CACHING:
Request 1: "What is Python?"
- API call: 500ms
- Cost: $0.001
- Cache result

Request 2-100: "What is Python?"
- Cache hit: 5ms (100x faster!)
- Cost: $0.000 (FREE!)
- 99% cost reduction

Real Example:
- 1000 requests/day
- 30% cache hit rate
- Saves: $9/day = $270/month
"""
\`\`\`

## Exact Match Caching

Simplest form: cache identical requests.

### Basic In-Memory Cache

\`\`\`python
from typing import Dict, Optional
import hashlib
import json

class SimpleCache:
    """
    Simple in-memory cache for LLM responses.
    """
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate cache key from request parameters."""
        # Create deterministic string from all parameters
        cache_input = json.dumps({
            'messages': messages,
            'model': model,
            **kwargs
        }, sort_keys=True)
        
        # Hash it for compact key
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, messages: list, model: str, **kwargs) -> Optional[str]:
        """Get cached response if exists."""
        key = self.get_key(messages, model, **kwargs)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, messages: list, model: str, response: str, **kwargs):
        """Cache a response."""
        key = self.get_key(messages, model, **kwargs)
        self.cache[key] = response
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# Usage
from openai import OpenAI

cache = SimpleCache()
client = OpenAI()

def chat_with_cache(messages: list, **kwargs) -> str:
    """Chat with caching."""
    
    # Check cache
    cached = cache.get(messages, model="gpt-3.5-turbo", **kwargs)
    if cached:
        print("[Cache HIT]")
        return cached
    
    print("[Cache MISS - calling API]")
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        **kwargs
    )
    
    result = response.choices[0].message.content
    
    # Cache it
    cache.set(messages, "gpt-3.5-turbo", result, **kwargs)
    
    return result

# Test
messages = [{"role": "user", "content": "What is 2+2?"}]

# First call - cache miss
result1 = chat_with_cache(messages)
print(result1)

# Second call - cache hit!
result2 = chat_with_cache(messages)
print(result2)

# Check stats
stats = cache.get_stats()
print(f"\\nCache stats: {stats}")
\`\`\`

## Redis Caching

For production, use Redis for distributed caching.

### Redis Cache Implementation

\`\`\`python
# pip install redis

import redis
import json
import hashlib
from typing import Optional

class RedisCache:
    """
    Redis-based cache for LLM responses.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600  # Time to live in seconds
    ):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get_key(self, messages: list, model: str, **kwargs) -> str:
        """Generate cache key."""
        cache_input = json.dumps({
            'messages': messages,
            'model': model,
            **kwargs
        }, sort_keys=True)
        
        return f"llm:{hashlib.md5(cache_input.encode()).hexdigest()}"
    
    def get(self, messages: list, model: str, **kwargs) -> Optional[str]:
        """Get cached response."""
        key = self.get_key(messages, model, **kwargs)
        
        cached = self.redis_client.get(key)
        
        if cached:
            self.hits += 1
            return cached
        
        self.misses += 1
        return None
    
    def set(
        self,
        messages: list,
        model: str,
        response: str,
        **kwargs
    ):
        """Cache a response with TTL."""
        key = self.get_key(messages, model, **kwargs)
        self.redis_client.setex(key, self.ttl, response)
    
    def clear(self):
        """Clear all cached items."""
        for key in self.redis_client.scan_iter("llm:*"):
            self.redis_client.delete(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        # Count keys
        cache_size = sum(1 for _ in self.redis_client.scan_iter("llm:*"))
        
        return {
            'cache_size': cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# Usage
cache = RedisCache(ttl=3600)  # Cache for 1 hour

# Use same interface as SimpleCache
messages = [{"role": "user", "content": "Explain Python"}]

# Check cache
cached = cache.get(messages, "gpt-3.5-turbo")

if not cached:
    # Call API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    result = response.choices[0].message.content
    
    # Cache it
    cache.set(messages, "gpt-3.5-turbo", result)
else:
    result = cached

print(result)
\`\`\`

## Semantic Caching

Cache similar (not just identical) requests.

### Semantic Cache with Embeddings

\`\`\`python
# pip install openai numpy scikit-learn

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional

class SemanticCache:
    """
    Cache that matches semantically similar queries.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.client = OpenAI()
        self.similarity_threshold = similarity_threshold
        
        # Storage: list of (embedding, prompt, response) tuples
        self.cache: List[Tuple[np.ndarray, str, str]] = []
        
        self.hits = 0
        self.misses = 0
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def get(self, prompt: str) -> Optional[str]:
        """
        Get cached response if similar prompt exists.
        """
        if not self.cache:
            self.misses += 1
            return None
        
        # Get embedding for query
        query_embedding = self.get_embedding(prompt)
        
        # Find most similar cached prompt
        best_similarity = 0
        best_response = None
        
        for cached_embedding, cached_prompt, cached_response in self.cache:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                cached_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = cached_response
        
        # Return if above threshold
        if best_similarity >= self.similarity_threshold:
            self.hits += 1
            print(f"[Semantic cache HIT - similarity: {best_similarity:.3f}]")
            return best_response
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str):
        """Cache a prompt-response pair."""
        embedding = self.get_embedding(prompt)
        self.cache.append((embedding, prompt, response))
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# Usage
semantic_cache = SemanticCache(similarity_threshold=0.95)

def chat_with_semantic_cache(prompt: str) -> str:
    """Chat with semantic caching."""
    
    # Check semantic cache
    cached = semantic_cache.get(prompt)
    if cached:
        return cached
    
    print("[Cache MISS - calling API]")
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    
    # Cache it
    semantic_cache.set(prompt, result)
    
    return result

# Test - similar queries hit cache!
result1 = chat_with_semantic_cache("What is Python?")
print(result1)

# Very similar - should hit cache
result2 = chat_with_semantic_cache("What is the Python programming language?")
print(result2)

# Different - should miss
result3 = chat_with_semantic_cache("What is Java?")
print(result3)

# Check stats
stats = semantic_cache.get_stats()
print(f"\\nSemantic cache stats: {stats}")
\`\`\`

## Claude Prompt Caching

Claude offers native prompt caching to reduce costs.

### Using Claude's Prompt Caching

\`\`\`python
# pip install anthropic

from anthropic import Anthropic

client = Anthropic()

# Large context that we want to cache
large_context = """
[... 10,000 words of documentation ...]
"""

# First request - caches the context
response1 = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}  # ← Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "What is the main topic?"}
    ]
)

print(f"Tokens used: {response1.usage.input_tokens}")
print(f"Cache write: {response1.usage.cache_creation_input_tokens}")

# Second request - uses cached context!
response2 = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": large_context,  # Same context
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "What are the key points?"}
    ]
)

print(f"\\nSecond request:")
print(f"Tokens used: {response2.usage.input_tokens}")
print(f"Cache read: {response2.usage.cache_read_input_tokens}")
print(f"Cost savings: ~90%!")  # Cached tokens cost 10% of normal
\`\`\`

## Performance Optimization

Beyond caching, optimize for speed.

### Parallel Requests

\`\`\`python
import asyncio
from openai import AsyncOpenAI
from typing import List

client = AsyncOpenAI()

async def call_llm_async(prompt: str) -> str:
    """Async LLM call."""
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def batch_calls(prompts: List[str]) -> List[str]:
    """Make multiple LLM calls in parallel."""
    tasks = [call_llm_async(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Usage
prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Java?",
    "What is C++?",
    "What is Rust?"
]

import time

# Sequential (slow)
start = time.time()
sequential_results = []
for prompt in prompts:
    result = chat_with_cache(messages=[{"role": "user", "content": prompt}])
    sequential_results.append(result)
sequential_time = time.time() - start

# Parallel (fast!)
start = time.time()
parallel_results = asyncio.run(batch_calls(prompts))
parallel_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.1f}x")
\`\`\`

### Request Batching

\`\`\`python
from typing import List
import time

class RequestBatcher:
    """
    Batch multiple requests into one call.
    """
    
    def __init__(self, max_batch_size: int = 10, max_wait_ms: float = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self.pending_requests: List[str] = []
        self.pending_start: Optional[float] = None
    
    def add_request(self, prompt: str) -> Optional[List[str]]:
        """
        Add request to batch. Returns results if batch is ready.
        """
        self.pending_requests.append(prompt)
        
        if self.pending_start is None:
            self.pending_start = time.time()
        
        # Check if should flush
        should_flush = (
            len(self.pending_requests) >= self.max_batch_size or
            time.time() - self.pending_start >= self.max_wait_ms
        )
        
        if should_flush:
            return self.flush()
        
        return None
    
    def flush(self) -> List[str]:
        """Process batched requests."""
        if not self.pending_requests:
            return []
        
        # Create batch prompt
        batch_prompt = "Answer each question separately:\\n\\n"
        for i, prompt in enumerate(self.pending_requests, 1):
            batch_prompt += f"{i}. {prompt}\\n"
        
        # Call LLM once for all requests
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": batch_prompt}]
        )
        
        # Parse responses
        result_text = response.choices[0].message.content
        results = [line.strip() for line in result_text.split('\\n') if line.strip()]
        
        # Reset batch
        self.pending_requests = []
        self.pending_start = None
        
        return results

# Usage
batcher = RequestBatcher(max_batch_size=5, max_wait_ms=100)

# Add requests
batcher.add_request("What is 2+2?")
batcher.add_request("What is 3+3?")
batcher.add_request("What is 4+4?")
batcher.add_request("What is 5+5?")

# This one triggers batch processing
results = batcher.add_request("What is 6+6?")

if results:
    print("Batch results:")
    for result in results:
        print(f"  - {result}")
\`\`\`

## Production Caching System

Combine strategies for production.

### Complete Caching Solution

\`\`\`python
from typing import Optional, Dict, Tuple
from enum import Enum
import time

class CacheStrategy(Enum):
    """Cache strategy types."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    NONE = "none"

class ProductionCache:
    """
    Production-ready caching system.
    
    Features:
    - Exact match caching
    - Semantic caching
    - TTL support
    - Statistics tracking
    - Cost estimation
    """
    
    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.EXACT,
        ttl: int = 3600,
        semantic_threshold: float = 0.95
    ):
        self.strategy = strategy
        self.ttl = ttl
        
        # Caches
        self.exact_cache = SimpleCache()
        
        if strategy == CacheStrategy.SEMANTIC:
            self.semantic_cache = SemanticCache(semantic_threshold)
        
        # Metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_api_cost = 0.0
        self.saved_cost = 0.0
    
    def get(
        self,
        messages: list,
        model: str,
        estimated_cost: float,
        **kwargs
    ) -> Tuple[Optional[str], bool]:
        """
        Get from cache.
        
        Returns: (response, from_cache)
        """
        self.total_requests += 1
        
        # Try exact cache first
        if self.strategy in [CacheStrategy.EXACT, CacheStrategy.SEMANTIC]:
            result = self.exact_cache.get(messages, model, **kwargs)
            if result:
                self.cache_hits += 1
                self.saved_cost += estimated_cost
                return result, True
        
        # Try semantic cache
        if self.strategy == CacheStrategy.SEMANTIC:
            prompt = messages[-1]['content']  # Last message
            result = self.semantic_cache.get(prompt)
            if result:
                self.cache_hits += 1
                self.saved_cost += estimated_cost
                return result, True
        
        # Cache miss
        self.cache_misses += 1
        self.total_api_cost += estimated_cost
        return None, False
    
    def set(
        self,
        messages: list,
        model: str,
        response: str,
        **kwargs
    ):
        """Store in cache."""
        # Store in exact cache
        if self.strategy in [CacheStrategy.EXACT, CacheStrategy.SEMANTIC]:
            self.exact_cache.set(messages, model, response, **kwargs)
        
        # Store in semantic cache
        if self.strategy == CacheStrategy.SEMANTIC:
            prompt = messages[-1]['content']
            self.semantic_cache.set(prompt, response)
    
    def get_metrics(self) -> Dict:
        """Get comprehensive metrics."""
        hit_rate = (
            (self.cache_hits / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )
        
        return {
            'strategy': self.strategy.value,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_api_cost': self.total_api_cost,
            'saved_cost': self.saved_cost,
            'total_cost': self.total_api_cost + self.saved_cost,
            'cost_reduction': (
                (self.saved_cost / (self.total_api_cost + self.saved_cost) * 100)
                if (self.total_api_cost + self.saved_cost) > 0
                else 0
            )
        }

# Usage
cache = ProductionCache(
    strategy=CacheStrategy.SEMANTIC,
    ttl=3600,
    semantic_threshold=0.95
)

def chat_with_production_cache(prompt: str) -> str:
    """Chat with production caching."""
    messages = [{"role": "user", "content": prompt}]
    estimated_cost = 0.001  # Estimate
    
    # Try cache
    cached, from_cache = cache.get(messages, "gpt-3.5-turbo", estimated_cost)
    
    if from_cache:
        print(f"[CACHE HIT]")
        return cached
    
    print(f"[CACHE MISS]")
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    result = response.choices[0].message.content
    
    # Cache it
    cache.set(messages, "gpt-3.5-turbo", result)
    
    return result

# Make requests
prompts = [
    "What is Python?",
    "What's Python?",  # Similar - semantic cache hit
    "Explain Python",  # Similar - semantic cache hit
    "What is Java?",   # Different - cache miss
]

for prompt in prompts:
    result = chat_with_production_cache(prompt)
    print(f"Prompt: {prompt}")
    print(f"Result: {result[:50]}...\\n")

# View metrics
metrics = cache.get_metrics()
print("\\nCache Metrics:")
print(f"  Hit rate: {metrics['hit_rate']:.1f}%")
print(f"  Cost reduction: {metrics['cost_reduction']:.1f}%")
print(f"  Saved: \${metrics['saved_cost']: .4f
}")
print(f"  Spent: \${metrics['total_api_cost']:.4f}")
\`\`\`

## Key Takeaways

1. **Caching saves 80-90%** on costs for repeated queries
2. **Exact match caching** for identical requests
3. **Semantic caching** for similar requests
4. **Use Redis** for distributed caching in production
5. **Set appropriate TTL** - balance freshness vs savings
6. **Claude prompt caching** reduces costs by 90% for large contexts
7. **Parallel requests** for multiple independent calls
8. **Batch requests** when possible
9. **Track cache hit rate** - aim for 30%+
10. **Measure savings** - prove ROI of caching

## Next Steps

Now you can optimize performance and costs. Next: **Local LLM Deployment** - learning to run models locally for privacy and cost control.`,
};

