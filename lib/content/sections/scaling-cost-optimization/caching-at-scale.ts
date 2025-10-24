export const cachingAtScale = {
  title: 'Caching at Scale',
  content: `

# Caching at Scale for LLM Applications

## Introduction

Caching is one of the most effective cost optimization strategies for LLM applications. Since LLM API calls are expensive ($0.001-$0.03 per request), caching identical or similar requests can reduce costs by 60-95% while dramatically improving response times. A well-designed caching strategy can:

- Reduce API costs by 60-95%
- Improve response latency from seconds to milliseconds
- Handle traffic spikes without hitting rate limits
- Reduce load on LLM providers
- Improve user experience with instant responses

This section covers:
- Exact match caching strategies
- Semantic caching for similar queries
- Distributed caching architectures
- Cache invalidation patterns
- Caching at different layers
- Production cache implementation

---

## Cache Fundamentals

### Cache Hit vs Miss

\`\`\`python
import time
from typing import Optional
import hashlib

class SimpleCache:
    """Basic in-memory cache"""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache with TTL"""
        self.cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total_requests,
            "hit_rate": hit_rate
        }

# Usage
cache = SimpleCache()

def get_llm_response(prompt: str) -> str:
    """Get LLM response with caching"""
    
    # Generate cache key
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    # Check cache
    cached_response = cache.get(cache_key)
    if cached_response:
        print("✅ Cache HIT - returning cached response")
        return cached_response["value"]
    
    # Cache miss - call LLM
    print("❌ Cache MISS - calling LLM API")
    response = call_openai_api(prompt)  # Expensive!
    
    # Store in cache
    cache.set(cache_key, response)
    
    return response

# Simulate requests
prompts = [
    "What is Python?",
    "What is Python?",  # Duplicate - cache hit!
    "What is Java?",
    "What is Python?",  # Cache hit again!
]

for prompt in prompts:
    response = get_llm_response(prompt)

stats = cache.get_stats()
print(f"\nCache Stats:")
print(f"  Hit Rate: {stats['hit_rate']:.1%}")
print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

# Output:
# Cache Miss - calling LLM API
# Cache HIT - returning cached response
# Cache Miss - calling LLM API
# Cache HIT - returning cached response
#
# Cache Stats:
#   Hit Rate: 50.0%
#   Hits: 2, Misses: 2
\`\`\`

**Cost Savings**: 50% hit rate = 50% cost reduction!

---

## Exact Match Caching with Redis

Production-grade caching with Redis:

\`\`\`python
import redis.asyncio as redis
import hashlib
import json
from typing import Optional
import asyncio

class RedisLLMCache:
    """Production Redis cache for LLM responses"""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis = None
        self.redis_url = redis_url
        self.default_ttl = ttl
    
    async def connect(self):
        """Initialize Redis connection"""
        self.redis = await redis.from_url(self.redis_url)
    
    def generate_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate unique cache key from request parameters"""
        
        # Include all parameters that affect the response
        key_components = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
        }
        
        # Sort to ensure consistent ordering
        key_str = json.dumps(key_components, sort_keys=True)
        
        # Hash for fixed-size key
        return f"llm_cache:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    async def get(self, cache_key: str) -> Optional[dict]:
        """Get cached response"""
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        return None
    
    async def set(
        self,
        cache_key: str,
        response: dict,
        ttl: Optional[int] = None
    ):
        """Cache LLM response"""
        ttl = ttl or self.default_ttl
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(response)
        )
    
    async def get_cached_or_generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        **kwargs
    ):
        """Get from cache or generate new response"""
        
        # Generate cache key
        cache_key = self.generate_cache_key(
            prompt=prompt,
            model=model,
            temperature=temperature,
            **kwargs
        )
        
        # Check cache
        cached_response = await self.get(cache_key)
        if cached_response:
            print("✅ Cache HIT")
            return {
                **cached_response,
                "from_cache": True
            }
        
        # Cache miss - call LLM
        print("❌ Cache MISS - calling LLM")
        
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs
        )
        
        response_dict = {
            "content": response.choices[0].message.content,
            "model": model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        # Cache the response
        await self.set(cache_key, response_dict)
        
        return {
            **response_dict,
            "from_cache": False
        }
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        info = await self.redis.info("stats")
        
        return {
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_connections": info.get("total_connections_received", 0)
        }

# Usage
cache = RedisLLMCache(redis_url="redis://localhost:6379")
await cache.connect()

# First request - cache miss
response1 = await cache.get_cached_or_generate(
    prompt="Explain quantum computing",
    model="gpt-3.5-turbo",
    temperature=0.7
)
print(f"From cache: {response1['from_cache']}")  # False
print(f"Cost: ~$0.0005")

# Second identical request - cache hit!
response2 = await cache.get_cached_or_generate(
    prompt="Explain quantum computing",
    model="gpt-3.5-turbo",
    temperature=0.7
)
print(f"From cache: {response2['from_cache']}")  # True
print(f"Cost: $0.0000 (cached!)")
print(f"Latency: ~5ms (vs ~2000ms for API call)")
\`\`\`

---

## Semantic Caching

Cache similar (not just identical) queries:

\`\`\`python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

class SemanticCache:
    """Cache based on semantic similarity, not exact match"""
    
    def __init__(
        self,
        redis_url: str,
        similarity_threshold: float = 0.95,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.redis = redis.from_url(redis_url)
        self.encoder = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding"""
        return self.encoder.encode(query, convert_to_tensor=False)
    
    async def find_similar_cached_query(
        self,
        query: str,
        top_k: int = 10
    ) -> Optional[Tuple[str, dict, float]]:
        """Find most similar cached query"""
        
        # Get query embedding
        query_embedding = self.encode_query(query)
        
        # Get recent cache keys (in production, use a separate index)
        cache_keys = await self.redis.keys("semantic_cache:*")
        
        if not cache_keys:
            return None
        
        # Get cached queries and embeddings
        best_match = None
        best_similarity = 0.0
        
        for cache_key in cache_keys[:top_k]:  # Limit search
            cached_data = await self.redis.get(cache_key)
            if not cached_data:
                continue
            
            cached = json.loads(cached_data)
            cached_embedding = np.array(cached["embedding"])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = (cached["query"], cached["response"], similarity)
        
        return best_match
    
    async def get_or_generate(self, query: str, generator_func):
        """Get semantically similar cached response or generate new"""
        
        # Check for similar cached query
        similar = await self.find_similar_cached_query(query)
        
        if similar:
            original_query, response, similarity = similar
            print(f"✅ Semantic cache HIT (similarity: {similarity:.2%})")
            print(f"   Original: '{original_query}'")
            print(f"   Current:  '{query}'")
            return {
                "response": response,
                "from_cache": True,
                "similarity": similarity
            }
        
        # Cache miss - generate new response
        print("❌ Semantic cache MISS")
        response = await generator_func(query)
        
        # Cache with embedding
        query_embedding = self.encode_query(query)
        cache_key = f"semantic_cache:{hashlib.sha256(query.encode()).hexdigest()}"
        
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps({
                "query": query,
                "response": response,
                "embedding": query_embedding.tolist()
            })
        )
        
        return {
            "response": response,
            "from_cache": False
        }

# Usage
semantic_cache = SemanticCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.95
)

async def generate_response(query: str) -> str:
    """Generate LLM response"""
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# First query
result1 = await semantic_cache.get_or_generate(
    "What is machine learning?",
    generate_response
)
# Output: Semantic cache MISS (generates response, costs $0.0005)

# Similar query - cache hit!
result2 = await semantic_cache.get_or_generate(
    "Can you explain machine learning?",  # Similar but not identical
    generate_response
)
# Output: Semantic cache HIT (similarity: 96%)
#         Returns cached response, costs $0!

# Very similar query - also hits cache
result3 = await semantic_cache.get_or_generate(
    "What does machine learning mean?",
    generate_response
)
# Output: Semantic cache HIT (similarity: 97%)
\`\`\`

**Powerful**: Can achieve 70-90% hit rates (vs 20-40% with exact matching)

---

## Multi-Layer Caching

Cache at multiple levels for maximum efficiency:

\`\`\`python
from functools import lru_cache
import asyncio

class MultiLayerCache:
    """
    Multi-layer caching architecture:
    L1: In-memory (Python dict/LRU)
    L2: Redis (shared across servers)
    L3: Database (persistent, for audit)
    """
    
    def __init__(
        self,
        redis_url: str,
        postgres_url: str,
        l1_size: int = 1000
    ):
        # L1: In-memory LRU cache (fastest)
        self.l1_cache = {}
        self.l1_max_size = l1_size
        self.l1_access_count = {}
        
        # L2: Redis (fast, shared)
        self.redis = redis.from_url(redis_url)
        
        # L3: PostgreSQL (persistent)
        self.db = create_postgres_connection(postgres_url)
        
        # Stats
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[dict]:
        """Get from cache, checking layers in order"""
        
        # L1: Check in-memory cache (microseconds)
        if key in self.l1_cache:
            self.l1_hits += 1
            print("⚡ L1 hit (in-memory)")
            return self.l1_cache[key]
        
        # L2: Check Redis (milliseconds)
        redis_value = await self.redis.get(f"cache:{key}")
        if redis_value:
            self.l2_hits += 1
            print("🚀 L2 hit (Redis)")
            
            # Promote to L1
            value = json.loads(redis_value)
            self._set_l1(key, value)
            return value
        
        # L3: Check database (tens of milliseconds)
        db_value = await self.db.fetch_one(
            "SELECT response FROM cache WHERE key = $1",
            key
        )
        if db_value:
            self.l3_hits += 1
            print("💾 L3 hit (Database)")
            
            # Promote to L2 and L1
            value = db_value["response"]
            await self.redis.setex(f"cache:{key}", 3600, json.dumps(value))
            self._set_l1(key, value)
            return value
        
        # Complete miss
        self.misses += 1
        print("❌ Complete miss")
        return None
    
    async def set(self, key: str, value: dict):
        """Set in all cache layers"""
        
        # L1: In-memory
        self._set_l1(key, value)
        
        # L2: Redis (async, don't wait)
        asyncio.create_task(
            self.redis.setex(f"cache:{key}", 3600, json.dumps(value))
        )
        
        # L3: Database (async, don't wait)
        asyncio.create_task(
            self.db.execute(
                "INSERT INTO cache (key, response, created_at) VALUES ($1, $2, NOW()) ON CONFLICT (key) DO UPDATE SET response = $2",
                key,
                json.dumps(value)
            )
        )
    
    def _set_l1(self, key: str, value: dict):
        """Set in L1 cache with LRU eviction"""
        
        if len(self.l1_cache) >= self.l1_max_size:
            # Evict least recently used
            lru_key = min(self.l1_access_count, key=self.l1_access_count.get)
            del self.l1_cache[lru_key]
            del self.l1_access_count[lru_key]
        
        self.l1_cache[key] = value
        self.l1_access_count[key] = time.time()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.l1_hits + self.l2_hits + self.l3_hits + self.misses
        
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "l3_hits": self.l3_hits,
            "misses": self.misses,
            "total_requests": total,
            "overall_hit_rate": (self.l1_hits + self.l2_hits + self.l3_hits) / total if total > 0 else 0,
            "l1_hit_rate": self.l1_hits / total if total > 0 else 0,
            "avg_latency": {
                "l1": "< 1ms",
                "l2": "~5ms",
                "l3": "~20ms",
                "miss": "~2000ms"
            }
        }

# Usage
cache = MultiLayerCache(
    redis_url="redis://localhost:6379",
    postgres_url="postgresql://localhost/mydb",
    l1_size=1000
)

# First request - complete miss
response1 = await cache.get("query_123")  # None
# Generate response and cache
await cache.set("query_123", {"response": "..."})

# Second request - L1 hit (microseconds)
response2 = await cache.get("query_123")  # From L1

# After server restart, L1 empty
# Third request - L2 hit (milliseconds)
response3 = await cache.get("query_123")  # From Redis

print(cache.get_stats())
# Output:
# {
#   "l1_hits": 1,
#   "l2_hits": 1, 
#   "l3_hits": 0,
#   "misses": 1,
#   "overall_hit_rate": 0.67
# }
\`\`\`

**Performance**:
- L1 (memory): < 1ms
- L2 (Redis): ~5ms
- L3 (DB): ~20ms
- Miss (LLM API): ~2000ms

---

## Cache Invalidation Strategies

"There are only two hard things in Computer Science: cache invalidation and naming things."

\`\`\`python
from datetime import datetime, timedelta
from typing import List

class CacheInvalidationManager:
    """Manage cache invalidation strategies"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        
        print(f"🗑️  Invalidating keys matching: {pattern}")
        
        cursor = 0
        invalidated_count = 0
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            if keys:
                await self.redis.delete(*keys)
                invalidated_count += len(keys)
            
            if cursor == 0:
                break
        
        print(f"✅ Invalidated {invalidated_count} keys")
        return invalidated_count
    
    async def invalidate_by_tag(self, tag: str):
        """Invalidate all cached responses with specific tag"""
        
        # Store tags in a set
        tag_key = f"cache_tag:{tag}"
        cache_keys = await self.redis.smembers(tag_key)
        
        if cache_keys:
            await self.redis.delete(*cache_keys)
            await self.redis.delete(tag_key)
            print(f"✅ Invalidated {len(cache_keys)} keys with tag '{tag}'")
        
        return len(cache_keys)
    
    async def time_based_invalidation(self, max_age_hours: int = 24):
        """Invalidate cache entries older than specified age"""
        
        # Store timestamps in sorted set
        cutoff_timestamp = time.time() - (max_age_hours * 3600)
        
        # Get expired keys
        expired_keys = await self.redis.zrangebyscore(
            "cache_timestamps",
            0,
            cutoff_timestamp
        )
        
        if expired_keys:
            # Delete expired keys
            await self.redis.delete(*expired_keys)
            
            # Remove from timestamp index
            await self.redis.zremrangebyscore(
                "cache_timestamps",
                0,
                cutoff_timestamp
            )
            
            print(f"✅ Invalidated {len(expired_keys)} expired keys")
        
        return len(expired_keys)
    
    async def smart_invalidation_on_model_update(self, model: str):
        """Invalidate caches when model is updated"""
        
        # Model version changed - invalidate all caches for that model
        pattern = f"cache:*:model={model}:*"
        return await self.invalidate_by_pattern(pattern)
    
    async def conditional_invalidation(
        self,
        condition_func,
        sample_size: int = 1000
    ):
        """Invalidate based on custom condition"""
        
        invalidated = 0
        
        # Sample cache keys
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                count=100
            )
            
            for key in keys:
                cached_data = await self.redis.get(key)
                if cached_data and condition_func(json.loads(cached_data)):
                    await self.redis.delete(key)
                    invalidated += 1
            
            if cursor == 0 or invalidated >= sample_size:
                break
        
        print(f"✅ Conditionally invalidated {invalidated} keys")
        return invalidated

# Usage examples
invalidator = CacheInvalidationManager(redis_client)

# 1. Invalidate by pattern (e.g., all GPT-4 caches)
await invalidator.invalidate_by_pattern("cache:*:model=gpt-4:*")

# 2. Invalidate by tag (e.g., all "product_info" caches)
await invalidator.invalidate_by_tag("product_info")

# 3. Invalidate old entries
await invalidator.time_based_invalidation(max_age_hours=24)

# 4. Invalidate when model changes
await invalidator.smart_invalidation_on_model_update("gpt-4-turbo")

# 5. Conditional invalidation (e.g., responses containing outdated info)
await invalidator.conditional_invalidation(
    condition_func=lambda data: "2023" in data.get("response", "")
)
\`\`\`

---

## Production Cache Implementation

Complete production system with all features:

\`\`\`python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import asyncio
import json

@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: str
    enable_l1: bool = True
    enable_semantic: bool = False
    l1_size: int = 1000
    default_ttl: int = 3600
    semantic_threshold: float = 0.95
    enable_metrics: bool = True

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    total_requests: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0
    total_cost_saved: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits + self.semantic_hits) / self.total_requests
    
    @property
    def avg_latency_saved_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_saved_ms / self.total_requests

class ProductionLLMCache:
    """Production-grade LLM caching system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis = None
        self.l1_cache = {} if config.enable_l1 else None
        self.semantic_cache = None
        self.metrics = CacheMetrics()
        
    async def initialize(self):
        """Initialize cache connections"""
        self.redis = await redis.from_url(self.config.redis_url)
        
        if self.config.enable_semantic:
            self.semantic_cache = SemanticCache(
                redis_url=self.config.redis_url,
                similarity_threshold=self.config.semantic_threshold
            )
        
        print("✅ Cache system initialized")
    
    async def get_or_generate(
        self,
        prompt: str,
        generator_func,
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict:
        """Get cached response or generate new one"""
        
        start_time = time.time()
        self.metrics.total_requests += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        
        # L1 Cache check (in-memory)
        if self.l1_cache is not None:
            if cache_key in self.l1_cache:
                self.metrics.l1_hits += 1
                self.metrics.total_latency_saved_ms += (time.time() - start_time) * 1000 - 1
                self.metrics.total_cost_saved += 0.0005  # Estimated cost per request
                
                print("⚡ L1 Cache HIT")
                return {
                    **self.l1_cache[cache_key],
                    "cache_layer": "L1",
                    "from_cache": True
                }
        
        # L2 Cache check (Redis)
        redis_value = await self.redis.get(cache_key)
        if redis_value:
            self.metrics.l2_hits += 1
            self.metrics.total_latency_saved_ms += (time.time() - start_time) * 1000 - 5
            self.metrics.total_cost_saved += 0.0005
            
            value = json.loads(redis_value)
            
            # Promote to L1
            if self.l1_cache is not None:
                self.l1_cache[cache_key] = value
            
            print("🚀 L2 Cache HIT")
            return {
                **value,
                "cache_layer": "L2",
                "from_cache": True
            }
        
        # Semantic cache check (if enabled)
        if self.semantic_cache:
            similar = await self.semantic_cache.find_similar_cached_query(prompt)
            if similar:
                self.metrics.semantic_hits += 1
                self.metrics.total_latency_saved_ms += (time.time() - start_time) * 1000 - 10
                self.metrics.total_cost_saved += 0.0005
                
                _, response, similarity = similar
                print(f"🎯 Semantic Cache HIT (similarity: {similarity:.2%})")
                return {
                    "response": response,
                    "cache_layer": "Semantic",
                    "similarity": similarity,
                    "from_cache": True
                }
        
        # Cache miss - generate new response
        self.metrics.misses += 1
        print("❌ Cache MISS - generating response")
        
        response = await generator_func(prompt, model=model, **kwargs)
        
        # Cache the response
        await self._cache_response(cache_key, response, prompt)
        
        return {
            **response,
            "from_cache": False
        }
    
    def _generate_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key"""
        key_data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(key_str.encode()).hexdigest()}"
    
    async def _cache_response(self, cache_key: str, response: Dict, prompt: str):
        """Cache response in all enabled layers"""
        
        # L1 Cache
        if self.l1_cache is not None:
            # Simple LRU eviction
            if len(self.l1_cache) >= self.config.l1_size:
                # Remove oldest entry
                oldest_key = next(iter(self.l1_cache))
                del self.l1_cache[oldest_key]
            
            self.l1_cache[cache_key] = response
        
        # L2 Cache (Redis)
        await self.redis.setex(
            cache_key,
            self.config.default_ttl,
            json.dumps(response)
        )
        
        # Semantic cache
        if self.semantic_cache:
            query_embedding = self.semantic_cache.encode_query(prompt)
            semantic_key = f"semantic_cache:{cache_key}"
            await self.redis.setex(
                semantic_key,
                self.config.default_ttl,
                json.dumps({
                    "query": prompt,
                    "response": response,
                    "embedding": query_embedding.tolist()
                })
            )
    
    def get_metrics_report(self) -> str:
        """Generate comprehensive metrics report"""
        
        m = self.metrics
        
        return f"""
📊 Cache Performance Report
{'=' * 50}

Requests:
  Total: {m.total_requests:,}
  L1 Hits: {m.l1_hits:,} ({m.l1_hits/max(1,m.total_requests)*100:.1f}%)
  L2 Hits: {m.l2_hits:,} ({m.l2_hits/max(1,m.total_requests)*100:.1f}%)
  Semantic Hits: {m.semantic_hits:,} ({m.semantic_hits/max(1,m.total_requests)*100:.1f}%)
  Misses: {m.misses:,} ({m.misses/max(1,m.total_requests)*100:.1f}%)
  
Overall Hit Rate: {m.hit_rate*100:.1f}%

Performance Impact:
  Avg Latency Saved: {m.avg_latency_saved_ms:.0f}ms per request
  Total Latency Saved: {m.total_latency_saved_ms/1000:.1f}s
  
Cost Impact:
  Total Cost Saved: \${m.total_cost_saved:.2f}
  Avg Cost Saved per Request: \${m.total_cost_saved/max(1,m.total_requests):.4f}

💰 Monthly Projections (at current rate):
  Requests: {m.total_requests * 30:,}
  Cost Saved: \${m.total_cost_saved * 30:.2f}
"""

# Usage
config = CacheConfig(
    redis_url="redis://localhost:6379",
    enable_l1=True,
    enable_semantic=True,
    l1_size=1000,
    default_ttl=3600,
    semantic_threshold=0.95
)

cache = ProductionLLMCache(config)
await cache.initialize()

async def generate_llm_response(prompt: str, model: str, **kwargs):
    """Generate LLM response"""
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    return {
        "content": response.choices[0].message.content,
        "usage": response.usage._asdict()
    }

# Use cache
response = await cache.get_or_generate(
    prompt="What is Python?",
    generator_func=generate_llm_response,
    model="gpt-3.5-turbo"
)

# After 10,000 requests...
print(cache.get_metrics_report())

# Example output:
# 📊 Cache Performance Report
# Total: 10,000
# L1 Hits: 3,500 (35.0%)
# L2 Hits: 4,000 (40.0%)  
# Semantic Hits: 1,500 (15.0%)
# Misses: 1,000 (10.0%)
#
# Overall Hit Rate: 90.0%
# Total Cost Saved: $4.50
# Monthly Cost Saved: $135
\`\`\`

---

## Best Practices

### 1. Choose TTL Wisely
- Short-lived data (news): 5-15 minutes
- Relatively stable (docs): 1-24 hours
- Static content: Days or weeks
- Invalidate manually when data changes

### 2. Monitor Hit Rates
- Track hit rates per cache layer
- Aim for 60%+ overall hit rate
- Investigate if hit rate drops

### 3. Cache Warm-Up
- Pre-populate common queries
- Refresh before expiration
- Avoid thundering herd

### 4. Size Appropriately
- L1: 100-10,000 items (depends on memory)
- L2: Millions of items (Redis)
- Monitor memory usage

### 5. Measure ROI
- Track cost saved vs cache infrastructure cost
- Cache is almost always worth it for LLMs

---

## Summary

Caching at scale can reduce LLM costs by 60-95%:

- **Exact Match**: Simple, effective for identical queries
- **Semantic Caching**: Match similar queries (70-90% hit rate)
- **Multi-Layer**: Memory + Redis + DB for maximum performance
- **Smart Invalidation**: Keep cache fresh without over-invalidating
- **Monitor Metrics**: Hit rate, latency saved, cost saved

A well-implemented cache is the single most effective cost optimization for production LLM applications.

`,
  exercises: [
    {
      prompt:
        'Implement a production Redis cache with TTL, metrics tracking, and hit rate monitoring. Deploy and measure actual cost savings.',
      solution: `Use RedisLLMCache implementation, deploy with your application, monitor for 1 week, calculate ROI.`,
    },
    {
      prompt:
        'Build a semantic cache that achieves 80%+ hit rate on your production queries. Measure quality degradation if any.',
      solution: `Use SemanticCache with appropriate similarity threshold, test on production-like data.`,
    },
    {
      prompt:
        'Design and implement a multi-layer cache (L1 + L2) and measure latency improvements at each layer.',
      solution: `Implement MultiLayerCache, instrument with timing, compare: L1 (~1ms), L2 (~5ms), Miss (~2000ms).`,
    },
  ],
};
