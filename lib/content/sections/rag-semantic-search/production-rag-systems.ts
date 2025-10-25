export const productionRagSystems = {
  title: 'Production RAG Systems',
  content: `
# Production RAG Systems

## Introduction

Building a production RAG system requires more than just retrieval and generation. You need caching, streaming, monitoring, error handling, cost optimization, and performance tuning. This section covers everything needed to run RAG at scale.

## Production Architecture

Complete production RAG architecture:

\`\`\`python
from typing import List, Dict, Optional, AsyncIterator
import asyncio
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionRAGSystem:
    """
    Production-ready RAG system with all necessary features.
    """
    
    def __init__(
        self,
        vector_store,
        llm_client,
        cache=None,
        monitor=None
    ):
        """
        Initialize production RAG.
        
        Args:
            vector_store: Vector database client
            llm_client: LLM client
            cache: Cache client (optional)
            monitor: Monitoring client (optional)
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.cache = cache or SimpleCache()
        self.monitor = monitor or SimpleMonitor()
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "errors": 0,
            "avg_latency": 0
        }
    
    async def query(
        self,
        user_query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> Dict:
        """
        Query RAG system with production features.
        
        Args:
            user_query: User\'s question
            top_k: Number of documents to retrieve
            use_cache: Whether to use cache
        
        Returns:
            Response with answer and metadata
        """
        start_time = datetime.now()
        self.metrics["total_queries"] += 1
        
        try:
            # Check cache first
            if use_cache:
                cached_result = await self._check_cache (user_query)
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    self.monitor.log_event("cache_hit", {"query": user_query})
                    return cached_result
            
            # Retrieve documents
            retrieved_docs = await self._retrieve_with_retry(
                user_query,
                top_k
            )
            
            # Generate answer
            answer = await self._generate_with_timeout(
                user_query,
                retrieved_docs,
                timeout=30
            )
            
            # Build response
            response = {
                "answer": answer,
                "sources": retrieved_docs,
                "query": user_query,
                "cached": False,
                "latency_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            # Cache result
            if use_cache:
                await self._cache_result (user_query, response)
            
            # Log metrics
            await self._log_metrics (response)
            
            return response
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error (f"Error processing query: {e}")
            await self._handle_error (e, user_query)
            raise
    
    async def _check_cache (self, query: str) -> Optional[Dict]:
        """Check if query result is cached."""
        cache_key = self._generate_cache_key (query)
        return await self.cache.get (cache_key)
    
    async def _cache_result (self, query: str, response: Dict):
        """Cache query result."""
        cache_key = self._generate_cache_key (query)
        await self.cache.set(
            cache_key,
            response,
            ttl=3600  # 1 hour TTL
        )
    
    def _generate_cache_key (self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        return f"rag_query:{hashlib.md5(query.encode()).hexdigest()}"
    
    async def _retrieve_with_retry(
        self,
        query: str,
        top_k: int,
        max_retries: int = 3
    ) -> List[Dict]:
        """Retrieve with exponential backoff retry."""
        for attempt in range (max_retries):
            try:
                return await self.vector_store.search (query, top_k=top_k)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning (f"Retrieval attempt {attempt + 1} failed, retrying in {wait_time}s")
                await asyncio.sleep (wait_time)
    
    async def _generate_with_timeout(
        self,
        query: str,
        docs: List[Dict],
        timeout: int = 30
    ) -> str:
        """Generate answer with timeout."""
        try:
            return await asyncio.wait_for(
                self._generate_answer (query, docs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error("Generation timeout")
            return "I'm sorry, but generating an answer took too long. Please try again."
    
    async def _generate_answer(
        self,
        query: str,
        docs: List[Dict]
    ) -> str:
        """Generate answer from retrieved docs."""
        context = "\\n\\n".join([doc["text"] for doc in docs])
        
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\\n{context}\\n\\nQuestion: {query}"
                }
            ]
        )
        
        return response.choices[0].message.content
    
    async def _log_metrics (self, response: Dict):
        """Log performance metrics."""
        await self.monitor.log_metric("query_latency", response["latency_ms"])
        await self.monitor.log_metric("num_sources", len (response["sources"]))
    
    async def _handle_error (self, error: Exception, query: str):
        """Handle and log errors."""
        await self.monitor.log_error({
            "error": str (error),
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_metrics (self) -> Dict:
        """Get system metrics."""
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_queries"]
            if self.metrics["total_queries"] > 0
            else 0
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate
        }
\`\`\`

## Caching Strategy

Implement effective caching:

\`\`\`python
import hashlib
import json
from typing import Optional
import redis

class RAGCache:
    """
    Production caching for RAG results.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize cache.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_client = redis.from_url (redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def get (self, query: str) -> Optional[Dict]:
        """
        Get cached result for query.
        
        Args:
            query: User query
        
        Returns:
            Cached result or None
        """
        cache_key = self._generate_key (query)
        
        cached_data = self.redis_client.get (cache_key)
        if cached_data:
            return json.loads (cached_data)
        
        return None
    
    def set(
        self,
        query: str,
        result: Dict,
        ttl: Optional[int] = None
    ):
        """
        Cache query result.
        
        Args:
            query: User query
            result: Query result to cache
            ttl: Time to live in seconds
        """
        cache_key = self._generate_key (query)
        ttl = ttl or self.default_ttl
        
        # Mark as cached
        result["cached"] = True
        result["cache_time"] = datetime.now().isoformat()
        
        self.redis_client.setex(
            cache_key,
            ttl,
            json.dumps (result)
        )
    
    def _generate_key (self, query: str) -> str:
        """Generate cache key from query."""
        # Normalize query
        normalized = query.lower().strip()
        
        # Hash for consistent key
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        return f"rag:query:{query_hash}"
    
    def invalidate_pattern (self, pattern: str):
        """Invalidate all keys matching pattern."""
        for key in self.redis_client.scan_iter (f"rag:*{pattern}*"):
            self.redis_client.delete (key)
    
    def get_stats (self) -> Dict:
        """Get cache statistics."""
        info = self.redis_client.info("stats")
        
        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / (
                info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)
            )
        }


# Example usage
cache = RAGCache()

# Cache query result
query = "What is machine learning?"
result = {"answer": "ML is...", "sources": [...]}
cache.set (query, result, ttl=3600)

# Retrieve from cache
cached = cache.get (query)
if cached:
    print("Cache hit!")
\`\`\`

## Streaming Responses

Stream RAG responses for better UX:

\`\`\`python
from typing import AsyncIterator

class StreamingRAG:
    """
    RAG system with streaming responses.
    """
    
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
    
    async def stream_query(
        self,
        user_query: str,
        top_k: int = 5
    ) -> AsyncIterator[Dict]:
        """
        Stream RAG response.
        
        Args:
            user_query: User's question
            top_k: Number of documents
        
        Yields:
            Response chunks
        """
        # 1. Yield retrieval status
        yield {
            "type": "status",
            "message": "Retrieving relevant documents..."
        }
        
        # 2. Retrieve documents
        docs = await self.vector_store.search (user_query, top_k=top_k)
        
        # Yield sources
        yield {
            "type": "sources",
            "sources": docs
        }
        
        # 3. Stream generation
        yield {
            "type": "status",
            "message": "Generating answer..."
        }
        
        # Stream LLM response
        context = "\\n\\n".join([doc["text"] for doc in docs])
        
        stream = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on context."
                },
                {
                    "role": "user",
                    "content": f"Context:\\n{context}\\n\\nQuestion: {user_query}"
                }
            ],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }
        
        # Final status
        yield {
            "type": "complete",
            "message": "Done"
        }


# Example usage with FastAPI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
streaming_rag = StreamingRAG(vector_store, llm_client)

@app.get("/stream-query")
async def stream_query (query: str):
    """Stream RAG response."""
    
    async def generate():
        async for chunk in streaming_rag.stream_query (query):
            yield json.dumps (chunk) + "\\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )
\`\`\`

## Monitoring & Observability

Comprehensive monitoring:

\`\`\`python
from dataclasses import dataclass, asdict
from typing import List
import time

@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_docs_retrieved: int
    cache_hit: bool
    error: Optional[str] = None

class RAGMonitor:
    """
    Monitor RAG system performance.
    """
    
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.errors: List[Dict] = []
    
    def track_query(
        self,
        query: str,
        retrieval_time: float,
        generation_time: float,
        num_docs: int,
        cache_hit: bool,
        error: Optional[str] = None
    ):
        """
        Track query metrics.
        
        Args:
            query: User query
            retrieval_time: Time for retrieval (ms)
            generation_time: Time for generation (ms)
            num_docs: Number of documents retrieved
            cache_hit: Whether result was cached
            error: Error message if any
        """
        metrics = QueryMetrics(
            query=query,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=retrieval_time + generation_time,
            num_docs_retrieved=num_docs,
            cache_hit=cache_hit,
            error=error
        )
        
        self.metrics.append (metrics)
        
        if error:
            self.errors.append({
                "query": query,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_summary (self, last_n: int = 100) -> Dict:
        """
        Get performance summary.
        
        Args:
            last_n: Number of recent queries to analyze
        
        Returns:
            Performance summary
        """
        recent = self.metrics[-last_n:]
        
        if not recent:
            return {}
        
        total_queries = len (recent)
        cache_hits = sum(1 for m in recent if m.cache_hit)
        errors = sum(1 for m in recent if m.error)
        
        avg_retrieval = sum (m.retrieval_time_ms for m in recent) / total_queries
        avg_generation = sum (m.generation_time_ms for m in recent) / total_queries
        avg_total = sum (m.total_time_ms for m in recent) / total_queries
        
        return {
            "total_queries": total_queries,
            "cache_hit_rate": cache_hits / total_queries,
            "error_rate": errors / total_queries,
            "avg_retrieval_ms": avg_retrieval,
            "avg_generation_ms": avg_generation,
            "avg_total_ms": avg_total,
            "p95_latency_ms": self._percentile(
                [m.total_time_ms for m in recent],
                0.95
            ),
            "p99_latency_ms": self._percentile(
                [m.total_time_ms for m in recent],
                0.99
            )
        }
    
    def _percentile (self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted (values)
        index = int (len (sorted_values) * percentile)
        return sorted_values[min (index, len (sorted_values) - 1)]
    
    def export_metrics (self) -> List[Dict]:
        """Export metrics for analysis."""
        return [asdict (m) for m in self.metrics]


# Example usage with instrumentation
monitor = RAGMonitor()

async def query_with_monitoring (query: str):
    """Query RAG with full monitoring."""
    
    # Track retrieval
    retrieval_start = time.time()
    docs = await vector_store.search (query)
    retrieval_time = (time.time() - retrieval_start) * 1000
    
    # Track generation
    generation_start = time.time()
    try:
        answer = await generate (query, docs)
        generation_time = (time.time() - generation_start) * 1000
        error = None
    except Exception as e:
        generation_time = (time.time() - generation_start) * 1000
        error = str (e)
        raise
    finally:
        # Log metrics
        monitor.track_query(
            query=query,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_docs=len (docs),
            cache_hit=False,
            error=error
        )
    
    return answer

# Get performance summary
summary = monitor.get_summary (last_n=100)
print(f"Avg latency: {summary['avg_total_ms']:.2f}ms")
print(f"P95 latency: {summary['p95_latency_ms']:.2f}ms")
print(f"Cache hit rate: {summary['cache_hit_rate']:.2%}")
\`\`\`

## Error Handling

Robust error handling:

\`\`\`python
from enum import Enum

class RAGErrorType(Enum):
    """Types of RAG errors."""
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT = "rate_limit"

class RAGError(Exception):
    """Custom RAG error."""
    
    def __init__(
        self,
        error_type: RAGErrorType,
        message: str,
        details: Optional[Dict] = None
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)

class ResilientRAG:
    """
    RAG system with comprehensive error handling.
    """
    
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
    
    async def query(
        self,
        user_query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Query with error handling.
        
        Args:
            user_query: User\'s question
            top_k: Number of documents
        
        Returns:
            Response with answer or error
        """
        try:
            # Validate input
            self._validate_query (user_query)
            
            # Retrieve with fallback
            docs = await self._retrieve_with_fallback (user_query, top_k)
            
            # Generate with fallback
            answer = await self._generate_with_fallback (user_query, docs)
            
            return {
                "success": True,
                "answer": answer,
                "sources": docs
            }
            
        except RAGError as e:
            logger.error (f"RAG error: {e.error_type.value} - {e.message}")
            return {
                "success": False,
                "error_type": e.error_type.value,
                "error_message": e.message,
                "fallback_answer": self._get_fallback_answer (e.error_type)
            }
        
        except Exception as e:
            logger.error (f"Unexpected error: {e}")
            return {
                "success": False,
                "error_type": "unknown",
                "error_message": str (e),
                "fallback_answer": "I'm sorry, but I encountered an unexpected error."
            }
    
    def _validate_query (self, query: str):
        """Validate query input."""
        if not query or not query.strip():
            raise RAGError(
                RAGErrorType.INVALID_INPUT,
                "Query cannot be empty"
            )
        
        if len (query) > 1000:
            raise RAGError(
                RAGErrorType.INVALID_INPUT,
                "Query too long (max 1000 characters)"
            )
    
    async def _retrieve_with_fallback(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:
        """Retrieve with fallback strategies."""
        try:
            docs = await self.vector_store.search (query, top_k=top_k)
            
            if not docs:
                # Fallback: Try broader search
                docs = await self.vector_store.search(
                    query,
                    top_k=top_k * 2,
                    min_score=0.5  # Lower threshold
                )
            
            return docs
            
        except Exception as e:
            raise RAGError(
                RAGErrorType.RETRIEVAL_FAILED,
                f"Document retrieval failed: {str (e)}"
            )
    
    async def _generate_with_fallback(
        self,
        query: str,
        docs: List[Dict]
    ) -> str:
        """Generate with fallback strategies."""
        try:
            # Try primary model
            return await self._generate (query, docs, model="gpt-4")
        
        except Exception as e:
            logger.warning (f"Primary generation failed: {e}, trying fallback")
            
            try:
                # Fallback to cheaper/faster model
                return await self._generate (query, docs, model="gpt-3.5-turbo")
            
            except Exception as e:
                raise RAGError(
                    RAGErrorType.GENERATION_FAILED,
                    f"Answer generation failed: {str (e)}"
                )
    
    async def _generate(
        self,
        query: str,
        docs: List[Dict],
        model: str
    ) -> str:
        """Generate answer with specified model."""
        context = "\\n\\n".join([doc["text"] for doc in docs])
        
        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer based on context."
                },
                {
                    "role": "user",
                    "content": f"Context:\\n{context}\\n\\nQuestion: {query}"
                }
            ]
        )
        
        return response.choices[0].message.content
    
    def _get_fallback_answer (self, error_type: RAGErrorType) -> str:
        """Get fallback answer based on error type."""
        fallbacks = {
            RAGErrorType.RETRIEVAL_FAILED: "I'm having trouble accessing the knowledge base. Please try again.",
            RAGErrorType.GENERATION_FAILED: "I found relevant information but couldn't formulate an answer.",
            RAGErrorType.TIMEOUT: "The request took too long. Please try a simpler question.",
            RAGErrorType.RATE_LIMIT: "Too many requests. Please wait a moment and try again.",
            RAGErrorType.INVALID_INPUT: "Please provide a valid question."
        }
        
        return fallbacks.get(
            error_type,
            "I encountered an error. Please try again."
        )
\`\`\`

## Cost Optimization

Optimize costs for production:

\`\`\`python
class CostOptimizedRAG:
    """
    RAG system optimized for cost.
    """
    
    def __init__(self):
        self.costs = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "embedding": 0.0001  # per 1K tokens
        }
        self.total_cost = 0
    
    async def query(
        self,
        user_query: str,
        budget_mode: str = "balanced"  # 'cheap', 'balanced', 'premium'
    ) -> Dict:
        """
        Query with cost optimization.
        
        Args:
            user_query: User's question
            budget_mode: Cost optimization strategy
        
        Returns:
            Response with cost info
        """
        # Select model based on budget
        model = self._select_model (budget_mode)
        
        # Optimize retrieval count
        top_k = self._select_top_k (budget_mode)
        
        # Track costs
        query_cost = 0
        
        # Retrieval cost
        embedding_tokens = len (user_query.split()) * 1.3  # Rough estimate
        query_cost += (embedding_tokens / 1000) * self.costs["embedding"]
        
        # Retrieve docs
        docs = await self.vector_store.search (user_query, top_k=top_k)
        
        # Generation cost
        context = "\\n\\n".join([doc["text"] for doc in docs])
        input_tokens = len((context + user_query).split()) * 1.3
        
        response = await self.llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"Context:\\n{context}\\n\\nQ: {user_query}"}
            ]
        )
        
        output_tokens = len (response.choices[0].message.content.split()) * 1.3
        
        # Calculate generation cost
        query_cost += (input_tokens / 1000) * self.costs[model]["input"]
        query_cost += (output_tokens / 1000) * self.costs[model]["output"]
        
        self.total_cost += query_cost
        
        return {
            "answer": response.choices[0].message.content,
            "model_used": model,
            "query_cost_usd": query_cost,
            "total_cost_usd": self.total_cost
        }
    
    def _select_model (self, budget_mode: str) -> str:
        """Select model based on budget."""
        models = {
            "cheap": "gpt-3.5-turbo",
            "balanced": "gpt-3.5-turbo",
            "premium": "gpt-4"
        }
        return models.get (budget_mode, "gpt-3.5-turbo")
    
    def _select_top_k (self, budget_mode: str) -> int:
        """Select top_k based on budget."""
        top_k_map = {
            "cheap": 3,
            "balanced": 5,
            "premium": 10
        }
        return top_k_map.get (budget_mode, 5)
\`\`\`

## Best Practices

### Production Checklist

✅ **Performance**
- Implement caching (Redis/Memcached)
- Stream responses for better UX
- Use connection pooling
- Optimize retrieval count

✅ **Reliability**
- Add retry logic with exponential backoff
- Implement timeouts
- Use fallback strategies
- Handle all error cases

✅ **Monitoring**
- Track latency (P50, P95, P99)
- Monitor cache hit rates
- Log errors with context
- Alert on anomalies

✅ **Cost**
- Cache expensive operations
- Use appropriate models
- Optimize context size
- Monitor spending

### Performance Targets

| Metric | Target | Action if Exceeded |
|--------|--------|-------------------|
| **P95 Latency** | < 2s | Optimize retrieval/caching |
| **Error Rate** | < 1% | Improve error handling |
| **Cache Hit Rate** | > 30% | Tune cache strategy |
| **Cost per Query** | < $0.05 | Use cheaper models |

## Summary

Production RAG systems require comprehensive infrastructure:

- **Caching**: Redis caching for performance
- **Streaming**: Stream responses for better UX
- **Monitoring**: Track latency, errors, costs
- **Error Handling**: Graceful degradation with fallbacks
- **Cost Optimization**: Smart model selection and caching

**Key Takeaway:** Production RAG is 20% retrieval+generation, 80% infrastructure.

**Production Pattern:**
1. Start with basic RAG
2. Add caching layer
3. Implement monitoring
4. Add error handling
5. Optimize costs
6. Scale horizontally
`,
};
