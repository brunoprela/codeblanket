export const cachingStrategiesContent = `
# Caching Strategies

## Introduction

Caching is one of the most effective optimizations for LLM applications. A well-designed cache can:
- Reduce costs by 50-90%
- Improve response times from seconds to milliseconds
- Reduce load on LLM APIs
- Improve reliability by serving cached responses when APIs are down

In this section, we'll explore caching strategies specifically designed for LLM applications, from simple exact-match caching to sophisticated semantic caching that matches similar prompts.

## Why Caching is Critical for LLM Apps

**Cost Reduction**: Every cache hit saves an API call. At $0.002 per 1K tokens, processing 1M requests could cost $2000 without caching, but only $200 with a 90% cache hit rate.

**Latency Improvement**: Cache hits return in <10ms instead of 2-30 seconds for LLM API calls.

**Reliability**: Serve responses even when LLM providers are experiencing outages.

**Rate Limit Management**: Reduce API calls to stay within provider limits.

**Consistency**: Identical prompts always return the same response (important for deterministic behavior).

However, LLM caching has unique challenges:
- Prompts are rarely character-for-character identical
- Context windows include variable conversation history
- Responses may need to appear "fresh" even when cached
- Balancing staleness vs cost savings

## Redis Basics for Caching

Redis is the go-to choice for LLM caching due to its speed, simplicity, and features.

\`\`\`python
import redis
import json
import hashlib
from typing import Optional
import openai

class SimpleCache:
    """Simple Redis cache for LLM responses."""
    
    def __init__(self, host='localhost', port=6379, ttl=3600):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=0,
            decode_responses=True
        )
        self.ttl = ttl  # Time to live in seconds
    
    def _make_key (self, prompt: str, model: str) -> str:
        """Create cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return f"llm:cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get (self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if it exists."""
        key = self._make_key (prompt, model)
        cached = self.client.get (key)
        
        if cached:
            return json.loads (cached)
        
        return None
    
    def set (self, prompt: str, model: str, response: str):
        """Cache a response."""
        key = self._make_key (prompt, model)
        value = json.dumps (response)
        self.client.setex (key, self.ttl, value)
    
    def exists (self, prompt: str, model: str) -> bool:
        """Check if response is cached."""
        key = self._make_key (prompt, model)
        return self.client.exists (key) > 0


# Usage with caching
cache = SimpleCache (ttl=3600)  # 1 hour TTL

def generate_with_cache (prompt: str, model: str = "gpt-3.5-turbo"):
    """Generate with automatic caching."""
    # Check cache first
    cached_response = cache.get (prompt, model)
    if cached_response:
        print("Cache hit!")
        return cached_response
    
    print("Cache miss, calling API...")
    
    # Call LLM API
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    
    # Store in cache
    cache.set (prompt, model, result)
    
    return result


# Example: Second call is instant
result1 = generate_with_cache("What is Python?")  # Calls API
result2 = generate_with_cache("What is Python?")  # Cache hit!
\`\`\`

## Semantic Caching

Exact-match caching only works for identical prompts. Semantic caching matches prompts that are similar in meaning.

\`\`\`python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import faiss

class SemanticCache:
    """
    Semantic cache using embeddings and vector search.
    
    Matches prompts that are semantically similar even if
    the exact wording differs.
    """
    
    def __init__(
        self,
        redis_client,
        similarity_threshold: float = 0.95,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.redis = redis_client
        self.threshold = similarity_threshold
        
        # Load embedding model
        self.encoder = SentenceTransformer (embedding_model)
        
        # Vector index for similarity search
        self.dimension = 384  # MiniLM output dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine sim)
        
        # Track mappings
        self.id_to_key = {}
        self.next_id = 0
    
    def _embed (self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        embedding = self.encoder.encode (text, convert_to_numpy=True)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm (embedding)
        return embedding
    
    def _make_key (self, prompt: str, model: str) -> str:
        """Create unique key for storing response."""
        return f"semantic:{hashlib.md5(f'{model}:{prompt}'.encode()).hexdigest()}"
    
    def get (self, prompt: str, model: str) -> Optional[Tuple[str, float]]:
        """
        Get cached response for semantically similar prompt.
        
        Returns:
            Tuple of (response, similarity_score) if found, None otherwise
        """
        # Get prompt embedding
        query_embedding = self._embed (prompt)
        
        # Search for similar prompts
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k=1  # Get top 1 match
        )
        
        if len (indices[0]) == 0:
            return None
        
        similarity = similarities[0][0]
        
        # Check if similarity exceeds threshold
        if similarity < self.threshold:
            return None
        
        # Get cache key from index
        vector_id = indices[0][0]
        if vector_id not in self.id_to_key:
            return None
        
        cache_key = self.id_to_key[vector_id]
        
        # Get cached response from Redis
        cached = self.redis.get (cache_key)
        if cached:
            response = json.loads (cached)
            return response, float (similarity)
        
        return None
    
    def set (self, prompt: str, model: str, response: str, ttl: int = 3600):
        """
        Cache a response with semantic indexing.
        """
        # Create cache key
        cache_key = self._make_key (prompt, model)
        
        # Store response in Redis
        self.redis.setex (cache_key, ttl, json.dumps (response))
        
        # Add to vector index
        embedding = self._embed (prompt)
        self.index.add (embedding.reshape(1, -1))
        
        # Track mapping
        self.id_to_key[self.next_id] = cache_key
        self.next_id += 1


# Usage
semantic_cache = SemanticCache (redis_client)

def generate_with_semantic_cache (prompt: str, model: str = "gpt-3.5-turbo"):
    """Generate with semantic caching."""
    # Check semantic cache
    cached = semantic_cache.get (prompt, model)
    
    if cached:
        response, similarity = cached
        print(f"Semantic cache hit! Similarity: {similarity:.3f}")
        return response
    
    print("Cache miss, calling API...")
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content
    
    # Cache with semantic indexing
    semantic_cache.set (prompt, model, result)
    
    return result


# These will match semantically:
r1 = generate_with_semantic_cache("What is Python programming?")  # API call
r2 = generate_with_semantic_cache("Tell me about Python")  # Cache hit!
r3 = generate_with_semantic_cache("Explain Python language")  # Cache hit!
\`\`\`

## Conversation History Caching

For conversational apps, cache based on conversation state:

\`\`\`python
class ConversationCache:
    """Cache for conversational contexts."""
    
    def __init__(self, redis_client, ttl=300):
        self.redis = redis_client
        self.ttl = ttl
    
    def _make_history_hash (self, messages: List[dict]) -> str:
        """Create hash of conversation history."""
        # Serialize messages
        history_str = json.dumps (messages, sort_keys=True)
        return hashlib.sha256(history_str.encode()).hexdigest()
    
    def get (self, messages: List[dict], model: str) -> Optional[str]:
        """Get cached response for conversation state."""
        history_hash = self._make_history_hash (messages)
        key = f"conv:{model}:{history_hash}"
        
        cached = self.redis.get (key)
        if cached:
            return json.loads (cached)
        
        return None
    
    def set (self, messages: List[dict], model: str, response: str):
        """Cache response for conversation state."""
        history_hash = self._make_history_hash (messages)
        key = f"conv:{model}:{history_hash}"
        
        self.redis.setex (key, self.ttl, json.dumps (response))


# Usage with conversation
conv_cache = ConversationCache (redis_client, ttl=300)  # 5 minute TTL

def chat_with_cache (messages: List[dict], model: str = "gpt-3.5-turbo"):
    """Chat with conversation caching."""
    # Check if this exact conversation state is cached
    cached = conv_cache.get (messages, model)
    if cached:
        print("Conversation cache hit!")
        return cached
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    
    result = response.choices[0].message.content
    
    # Cache for this conversation state
    conv_cache.set (messages, model, result)
    
    return result
\`\`\`

## Claude Prompt Caching

Claude offers native prompt caching that caches the KV cache of common prefixes:

\`\`\`python
import anthropic

client = anthropic.Anthropic()

def use_claude_prompt_caching (document: str, query: str):
    """
    Use Claude\'s built-in prompt caching.
    
    The document is marked for caching and will be cached
    on the Claude side for 5 minutes.
    """
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": "You are a helpful assistant that answers questions about documents."
            },
            {
                "type": "text",
                "text": f"Document:\\n\\n{document}",
                "cache_control": {"type": "ephemeral"}  # Cache this part!
            }
        ],
        messages=[
            {"role": "user", "content": query}
        ]
    )
    
    # Check cache performance
    usage = response.usage
    print(f"Cache creation tokens: {usage.cache_creation_input_tokens}")
    print(f"Cache read tokens: {usage.cache_read_input_tokens}")
    print(f"Regular input tokens: {usage.input_tokens}")
    
    return response.content[0].text


# First call creates cache
result1 = use_claude_prompt_caching (long_document, "What is the main topic?")

# Second call reads from cache (much cheaper!)
result2 = use_claude_prompt_caching (long_document, "Summarize the key points")
\`\`\`

## Cache Invalidation Strategies

Decide when cached responses should be refreshed:

\`\`\`python
from datetime import datetime, timedelta
from typing import Optional

class SmartCache:
    """Cache with intelligent invalidation."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def set_with_metadata(
        self,
        key: str,
        value: str,
        ttl: int,
        metadata: dict
    ):
        """Store value with metadata for invalidation."""
        data = {
            'value': value,
            'cached_at': datetime.utcnow().isoformat(),
            'metadata': metadata
        }
        self.redis.setex (key, ttl, json.dumps (data))
    
    def get_if_valid(
        self,
        key: str,
        max_age_seconds: Optional[int] = None
    ) -> Optional[str]:
        """
        Get cached value only if it meets freshness criteria.
        """
        cached = self.redis.get (key)
        if not cached:
            return None
        
        data = json.loads (cached)
        
        # Check age if max_age specified
        if max_age_seconds:
            cached_at = datetime.fromisoformat (data['cached_at'])
            age = (datetime.utcnow() - cached_at).total_seconds()
            
            if age > max_age_seconds:
                # Too old, invalidate
                self.redis.delete (key)
                return None
        
        return data['value']
    
    def invalidate_by_pattern (self, pattern: str):
        """Invalidate all keys matching pattern."""
        keys = self.redis.keys (pattern)
        if keys:
            self.redis.delete(*keys)


# Usage
smart_cache = SmartCache (redis_client)

# Cache with metadata
smart_cache.set_with_metadata(
    "model:gpt-4:version",
    "response",
    ttl=86400,  # 24 hours
    metadata={'model_version': '0314'}
)

# Get only if fresh enough (within 1 hour)
response = smart_cache.get_if_valid(
    "model:gpt-4:version",
    max_age_seconds=3600
)

# Invalidate all caches for a specific model
smart_cache.invalidate_by_pattern("model:gpt-4:*")
\`\`\`

## Distributed Caching

For multi-instance deployments, use distributed caching:

\`\`\`python
import redis
from redis.cluster import RedisCluster

class DistributedCache:
    """Distributed cache using Redis Cluster."""
    
    def __init__(self, nodes: List[dict]):
        """
        Args:
            nodes: List of {'host': 'host', 'port': port} dicts
        """
        self.client = RedisCluster(
            startup_nodes=nodes,
            decode_responses=True
        )
    
    def get_with_stats (self, key: str) -> Tuple[Optional[str], dict]:
        """
        Get value and track cache statistics.
        """
        # Increment total requests
        self.client.incr (f"stats:requests")
        
        value = self.client.get (key)
        
        if value:
            # Increment hits
            self.client.incr (f"stats:hits")
            return value, {'hit': True}
        else:
            # Increment misses
            self.client.incr (f"stats:misses")
            return None, {'hit': False}
    
    def get_stats (self) -> dict:
        """Get cache statistics."""
        requests = int (self.client.get("stats:requests") or 0)
        hits = int (self.client.get("stats:hits") or 0)
        misses = int (self.client.get("stats:misses") or 0)
        
        hit_rate = hits / requests if requests > 0 else 0
        
        return {
            'requests': requests,
            'hits': hits,
            'misses': misses,
            'hit_rate': hit_rate
        }


# Multi-node setup
nodes = [
    {'host': 'redis-1', 'port': 6379},
    {'host': 'redis-2', 'port': 6379},
    {'host': 'redis-3', 'port': 6379}
]

cache = DistributedCache (nodes)
\`\`\`

## Cache Warming

Pre-populate cache with common queries:

\`\`\`python
async def warm_cache (prompts: List[str], model: str = "gpt-3.5-turbo"):
    """
    Warm cache by pre-generating responses for common prompts.
    """
    print(f"Warming cache with {len (prompts)} prompts...")
    
    for i, prompt in enumerate (prompts):
        # Check if already cached
        if not cache.exists (prompt, model):
            # Generate and cache
            response = await generate_completion (prompt, model)
            cache.set (prompt, model, response)
            print(f"Cached {i+1}/{len (prompts)}: {prompt[:50]}...")
        else:
            print(f"Skipped {i+1}/{len (prompts)}: already cached")
    
    print("Cache warming complete!")


# Common queries to warm cache with
common_prompts = [
    "What is Python?",
    "Explain machine learning",
    "How does async work?",
    # ... add more common queries
]

# Warm cache during deployment
asyncio.run (warm_cache (common_prompts))
\`\`\`

## Multi-Tier Caching

Use multiple cache layers for optimal performance:

\`\`\`python
from cachetools import LRUCache
from typing import Optional

class MultiTierCache:
    """
    Three-tier cache:
    1. In-memory (fastest, smallest)
    2. Redis (fast, larger)
    3. Database (slow, largest)
    """
    
    def __init__(
        self,
        redis_client,
        db_client,
        memory_size: int = 1000
    ):
        self.memory_cache = LRUCache (maxsize=memory_size)
        self.redis = redis_client
        self.db = db_client
    
    def get (self, key: str) -> Optional[str]:
        """Get from cache tiers in order."""
        # Tier 1: Memory (< 1ms)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Tier 2: Redis (< 10ms)
        redis_value = self.redis.get (key)
        if redis_value:
            # Promote to memory cache
            self.memory_cache[key] = redis_value
            return redis_value
        
        # Tier 3: Database (< 100ms)
        db_value = self.db.get_cached_response (key)
        if db_value:
            # Promote to both caches
            self.memory_cache[key] = db_value
            self.redis.setex (key, 3600, db_value)
            return db_value
        
        return None
    
    def set (self, key: str, value: str, ttl: int = 3600):
        """Set in all cache tiers."""
        # Set in all tiers
        self.memory_cache[key] = value
        self.redis.setex (key, ttl, value)
        self.db.store_cached_response (key, value, ttl)


# Usage
multi_cache = MultiTierCache (redis_client, db_client)

# First get might hit database
response1 = multi_cache.get("prompt:123")  # ~100ms from DB

# Subsequent gets hit memory
response2 = multi_cache.get("prompt:123")  # <1ms from memory!
\`\`\`

## Best Practices

1. **Start with simple exact-match caching** before implementing semantic caching

2. **Set appropriate TTLs** based on how fresh responses need to be

3. **Use semantic caching** for user-facing applications where prompts vary

4. **Leverage native caching** (like Claude\'s prompt caching) when available

5. **Monitor cache hit rates** and optimize based on data

6. **Warm caches** with common queries during deployment

7. **Use distributed caching** for multi-instance deployments

8. **Implement cache invalidation** for when models or prompts change

9. **Track cost savings** from caching in your monitoring

10. **Consider multi-tier caching** for high-traffic applications

Effective caching can reduce your LLM costs by 50-90% while dramatically improving response times. It's often the single highest-ROI optimization you can make.
`;
