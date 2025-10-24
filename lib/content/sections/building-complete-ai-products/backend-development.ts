export const backendDevelopment = {
  title: 'Backend Development',
  id: 'backend-development',
  content: `
# Backend Development for AI Applications

## Introduction

The backend for an AI application must handle unique challenges: managing expensive GPU compute, implementing token streaming, handling long-running async jobs, managing rate limits across multiple providers, and tracking costs in real-time.

This section covers building production-grade backends for AI applications using FastAPI/Python with patterns from real systems processing millions of AI requests.

### Architecture Overview

\`\`\`
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼───────────────────────────────────┐
│         API Layer (FastAPI)              │
│  - Auth & Rate Limiting                  │
│  - Request Validation                    │
│  - Response Streaming                    │
└──────┬───────────────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│      Application Layer                   │
│  - Business Logic                        │
│  - Provider Routing                      │
│  - Cost Tracking                         │
└──────┬────────────────┬──────────────────┘
       │                │
┌──────▼─────┐   ┌──────▼─────────────────┐
│   Redis    │   │   Postgres             │
│  - Cache   │   │  - Users, Jobs, Logs   │
│  - Queue   │   │  - Analytics           │
└────────────┘   └────────────────────────┘
       │
┌──────▼───────────────────────────────────┐
│      Workers (Celery/RQ)                 │
│  - Image Generation                      │
│  - Video Processing                      │
│  - Batch Jobs                            │
└──────────────────────────────────────────┘
\`\`\`

---

## Streaming API Endpoint

### Server-Sent Events (SSE)

\`\`\`python
"""
Streaming LLM Response with FastAPI
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import asyncio
from typing import AsyncGenerator

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4000

async def stream_llm_response(
    prompt: str,
    model: str,
    max_tokens: int
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from Claude API
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                # Format as SSE
                yield f"data: {json.dumps({'token': text})}\\n\\n"
                
                # Small delay for backpressure
                await asyncio.sleep(0.001)
        
        # Signal completion
        yield f"data: [DONE]\\n\\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\\n\\n"

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint
    """
    return StreamingResponse(
        stream_llm_response(
            request.prompt,
            request.model,
            request.max_tokens
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
\`\`\`

### Token Counting & Cost Tracking

\`\`\`python
"""
Real-time token counting and cost tracking
"""

import tiktoken
from typing import Dict

class TokenTracker:
    """Track tokens and costs across providers"""
    
    # Pricing per 1M tokens
    PRICING = {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00,
            "output": 15.00
        },
        "gpt-4o": {
            "input": 5.00,
            "output": 15.00
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60
        }
    }
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost in USD"""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost

# Usage in streaming endpoint
async def stream_with_tracking(
    prompt: str,
    model: str,
    user_id: str
) -> AsyncGenerator[str, None]:
    tracker = TokenTracker()
    input_tokens = tracker.count_tokens(prompt)
    output_tokens = 0
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    with client.messages.stream(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            output_tokens += tracker.count_tokens(text)
            
            yield f"data: {json.dumps({'token': text})}\\n\\n"
    
    # Calculate final cost
    cost = tracker.calculate_cost(model, input_tokens, output_tokens)
    
    # Log to database
    await log_usage(
        user_id=user_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost
    )
    
    # Send final stats
    yield f"data: {json.dumps({
        'tokens': output_tokens,
        'cost': round(cost, 4)
    })}\\n\\n"
\`\`\`

---

## Background Job System

### Celery Worker Setup

\`\`\`python
"""
Async job processing with Celery
"""

from celery import Celery
from redis import Redis
import os

# Initialize Celery
celery_app = Celery(
    'tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540  # 9 minutes soft limit
)

@celery_app.task(bind=True, max_retries=3)
def generate_image(self, prompt: str, user_id: str):
    """
    Background image generation task
    """
    from app.services.image_gen import ImageGenerator
    
    try:
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'progress': 10, 'status': 'Starting...'}
        )
        
        # Initialize generator
        generator = ImageGenerator()
        
        # Generate
        self.update_state(
            state='PROGRESS',
            meta={'progress': 50, 'status': 'Generating...'}
        )
        
        result = generator.generate(prompt)
        
        # Upload to S3
        self.update_state(
            state='PROGRESS',
            meta={'progress': 90, 'status': 'Uploading...'}
        )
        
        url = upload_to_s3(result['image'], user_id)
        
        # Log to database
        log_generation(
            user_id=user_id,
            prompt=prompt,
            url=url,
            cost=result['cost']
        )
        
        return {
            'url': url,
            'cost': result['cost']
        }
        
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

# FastAPI endpoint to trigger job
@app.post("/api/images/generate")
async def create_image_job(
    prompt: str,
    user: User = Depends(get_current_user)
):
    """
    Create async image generation job
    """
    # Check user has credits
    if user.credits < 10:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    # Create job
    task = generate_image.delay(prompt, user.id)
    
    # Store job metadata
    await db.jobs.insert_one({
        "id": task.id,
        "user_id": user.id,
        "type": "image",
        "prompt": prompt,
        "status": "queued",
        "created_at": datetime.utcnow()
    })
    
    return {"job_id": task.id}

# Job status endpoint
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Poll job status
    """
    task = celery_app.AsyncResult(job_id)
    
    if task.state == 'PENDING':
        response = {
            'status': 'queued',
            'progress': 0
        }
    elif task.state == 'PROGRESS':
        response = {
            'status': 'processing',
            'progress': task.info.get('progress', 0),
            'message': task.info.get('status', ')
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': 'completed',
            'progress': 100,
            'result': task.result
        }
    elif task.state == 'FAILURE':
        response = {
            'status': 'failed',
            'error': str(task.info)
        }
    else:
        response = {
            'status': task.state.lower(),
            'progress': 0
        }
    
    return response
\`\`\`

---

## Rate Limiting & Quotas

### Multi-Layer Rate Limiting

\`\`\`python
"""
Rate limiting with Redis
"""

from fastapi import HTTPException, Request
from redis import Redis
import time

class RateLimiter:
    """
    Multi-tier rate limiter
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_limit(
        self,
        user_id: str,
        tier: str = "free"
    ) -> dict:
        """
        Check if user is within rate limits
        
        Tiers:
        - free: 10 requests/minute
        - pro: 100 requests/minute
        - enterprise: unlimited
        """
        
        if tier == "enterprise":
            return {"allowed": True}
        
        limits = {
            "free": {"minute": 10, "hour": 100, "day": 1000},
            "pro": {"minute": 100, "hour": 1000, "day": 10000}
        }
        
        tier_limits = limits.get(tier, limits["free"])
        
        now = int(time.time())
        
        # Check minute limit
        minute_key = f"rate_limit:{user_id}:minute:{now // 60}"
        minute_count = self.redis.incr(minute_key)
        self.redis.expire(minute_key, 60)
        
        if minute_count > tier_limits["minute"]:
            return {
                "allowed": False,
                "retry_after": 60 - (now % 60),
                "limit": tier_limits["minute"],
                "window": "minute"
            }
        
        # Check hourly limit
        hour_key = f"rate_limit:{user_id}:hour:{now // 3600}"
        hour_count = self.redis.incr(hour_key)
        self.redis.expire(hour_key, 3600)
        
        if hour_count > tier_limits["hour"]:
            return {
                "allowed": False,
                "retry_after": 3600 - (now % 3600),
                "limit": tier_limits["hour"],
                "window": "hour"
            }
        
        # Check daily limit
        day_key = f"rate_limit:{user_id}:day:{now // 86400}"
        day_count = self.redis.incr(day_key)
        self.redis.expire(day_key, 86400)
        
        if day_count > tier_limits["day"]:
            return {
                "allowed": False,
                "retry_after": 86400 - (now % 86400),
                "limit": tier_limits["day"],
                "window": "day"
            }
        
        return {
            "allowed": True,
            "remaining": {
                "minute": tier_limits["minute"] - minute_count,
                "hour": tier_limits["hour"] - hour_count,
                "day": tier_limits["day"] - day_count
            }
        }

# Dependency for FastAPI
redis_client = Redis.from_url(os.getenv("REDIS_URL"))
rate_limiter = RateLimiter(redis_client)

async def check_rate_limit(
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Rate limit middleware
    """
    result = await rate_limiter.check_limit(user.id, user.tier)
    
    if not result["allowed"]:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {result['retry_after']}s",
            headers={
                "Retry-After": str(result["retry_after"]),
                "X-RateLimit-Limit": str(result["limit"]),
                "X-RateLimit-Window": result["window"]
            }
        )
    
    # Add headers
    request.state.rate_limit = result["remaining"]

# Apply to endpoints
@app.post("/api/chat", dependencies=[Depends(check_rate_limit)])
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    ...
\`\`\`

---

## Provider Management

### Multi-Provider Routing

\`\`\`python
"""
Intelligent routing across multiple LLM providers
"""

from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    COHERE = "cohere"
    REPLICATE = "replicate"

@dataclass
class ProviderConfig:
    name: Provider
    models: List[str]
    rate_limit: int  # requests per minute
    cost_per_1k_tokens: float
    avg_latency_ms: float
    success_rate: float

class ProviderRouter:
    """
    Route requests to optimal provider
    """
    
    def __init__(self):
        self.providers = {
            Provider.ANTHROPIC: ProviderConfig(
                name=Provider.ANTHROPIC,
                models=["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                rate_limit=1000,
                cost_per_1k_tokens=0.003,
                avg_latency_ms=800,
                success_rate=0.99
            ),
            Provider.OPENAI: ProviderConfig(
                name=Provider.OPENAI,
                models=["gpt-4o", "gpt-4o-mini"],
                rate_limit=3000,
                cost_per_1k_tokens=0.005,
                avg_latency_ms=600,
                success_rate=0.98
            )
        }
        
        self.redis = Redis.from_url(os.getenv("REDIS_URL"))
    
    async def get_provider_usage(self, provider: Provider) -> int:
        """Get current minute's request count"""
        minute = int(time.time()) // 60
        key = f"provider_usage:{provider.value}:{minute}"
        return int(self.redis.get(key) or 0)
    
    async def select_provider(
        self,
        model_type: str,
        optimize_for: str = "cost"  # "cost", "speed", "reliability"
    ) -> Optional[Provider]:
        """
        Select best provider based on criteria
        """
        
        candidates = []
        
        for provider, config in self.providers.items():
            # Check if provider supports model type
            if not any(model_type in model for model in config.models):
                continue
            
            # Check rate limit
            usage = await self.get_provider_usage(provider)
            if usage >= config.rate_limit:
                continue
            
            # Calculate score based on optimization goal
            if optimize_for == "cost":
                score = 1 / config.cost_per_1k_tokens
            elif optimize_for == "speed":
                score = 1 / config.avg_latency_ms
            elif optimize_for == "reliability":
                score = config.success_rate
            else:
                score = (
                    (1 / config.cost_per_1k_tokens) * 0.4 +
                    (1 / config.avg_latency_ms) * 0.3 +
                    config.success_rate * 0.3
                )
            
            candidates.append((provider, score))
        
        if not candidates:
            return None
        
        # Return best provider
        return max(candidates, key=lambda x: x[1])[0]
    
    async def execute_with_fallback(
        self,
        func,
        *args,
        **kwargs
    ):
        """
        Execute with automatic provider fallback
        """
        providers = [Provider.ANTHROPIC, Provider.OPENAI, Provider.COHERE]
        
        last_error = None
        
        for provider in providers:
            try:
                # Track usage
                minute = int(time.time()) // 60
                key = f"provider_usage:{provider.value}:{minute}"
                self.redis.incr(key)
                self.redis.expire(key, 60)
                
                # Execute
                return await func(provider, *args, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider} failed: {e}")
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")

# Usage
router = ProviderRouter()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat with automatic provider selection
    """
    
    async def call_provider(provider: Provider, prompt: str):
        if provider == Provider.ANTHROPIC:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif provider == Provider.OPENAI:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    
    # Execute with fallback
    result = await router.execute_with_fallback(
        call_provider,
        request.prompt
    )
    
    return {"response": result}
\`\`\`

---

## Database Schema

### PostgreSQL Schema for AI App

\`\`\`sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    tier VARCHAR(20) DEFAULT 'free',
    credits INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW()
);

-- API usage logs
CREATE TABLE api_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    endpoint VARCHAR(100),
    model VARCHAR(50),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost DECIMAL(10, 6),
    latency_ms INTEGER,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_logs_user_created ON api_logs(user_id, created_at DESC);

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    type VARCHAR(50),
    status VARCHAR(20),
    prompt TEXT,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_jobs_user_status ON jobs(user_id, status);

-- Generations table
CREATE TABLE generations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    type VARCHAR(20),
    prompt TEXT,
    url VARCHAR(500),
    cost DECIMAL(10, 6),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analytics
CREATE MATERIALIZED VIEW daily_usage AS
SELECT 
    user_id,
    DATE(created_at) as date,
    COUNT(*) as request_count,
    SUM(cost) as total_cost,
    SUM(input_tokens + output_tokens) as total_tokens
FROM api_logs
GROUP BY user_id, DATE(created_at);

CREATE INDEX idx_daily_usage_user_date ON daily_usage(user_id, date DESC);
\`\`\`

---

## Conclusion

Backend development for AI applications requires:

1. **Streaming**: SSE/WebSocket for token-by-token delivery
2. **Job System**: Celery/RQ for async processing
3. **Rate Limiting**: Multi-tier, multi-window limits
4. **Provider Management**: Routing, fallback, cost optimization
5. **Cost Tracking**: Real-time token counting
6. **Database**: Efficient schema for logs and analytics

**Tech Stack**:
- **FastAPI**: API framework
- **Celery**: Background jobs
- **Redis**: Cache, queue, rate limiting
- **PostgreSQL**: Persistent storage
- **SQLAlchemy**: ORM

This architecture handles millions of AI requests efficiently and cost-effectively.
`,
};
