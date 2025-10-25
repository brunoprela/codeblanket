export const middlewareCors = {
  title: 'Middleware & CORS',
  id: 'middleware-cors',
  content: `
# Middleware & CORS

## Introduction

Middleware are functions that process every request and response, enabling cross-cutting concerns without duplicating code across endpoints. Essential for production APIs to handle logging, authentication, rate limiting, CORS, security headers, and request/response transformation.

**Why middleware matters:**
- **DRY principle**: Implement once, apply everywhere
- **Separation of concerns**: Keep endpoint code focused on business logic
- **Centralized control**: Security, logging, monitoring in one place
- **Performance**: Efficient request/response processing
- **Scalability**: Consistent behavior across all endpoints

**Common middleware use cases:**
- Request logging and metrics
- Authentication and authorization
- Rate limiting and throttling
- CORS (Cross-Origin Resource Sharing)
- Security headers (CSP, HSTS, X-Frame-Options)
- Request ID generation for tracing
- Response compression
- Error handling and transformation
- Request validation
- Performance monitoring

In this section, you'll master:
- Custom middleware implementation
- Middleware ordering and execution flow
- CORS configuration for frontend integration
- Rate limiting strategies
- Security headers
- Production middleware patterns
- Performance considerations

---

## Middleware Fundamentals

### How Middleware Works

\`\`\`
Request Flow (Onion Model):

Client Request
    ↓
Middleware A (before)
    ↓
Middleware B (before)
    ↓
Middleware C (before)
    ↓
Route Handler (endpoint)
    ↓
Middleware C (after)
    ↓
Middleware B (after)
    ↓
Middleware A (after)
    ↓
Client Response

Key concept: Middleware wraps around your application like layers of an onion.
Each middleware can modify the request before it reaches the endpoint,
and modify the response before it reaches the client.
\`\`\`

### Basic Middleware

\`\`\`python
"""
Basic middleware structure
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response
import time

app = FastAPI()

@app.middleware("http")
async def simple_middleware(request: Request, call_next):
    """
    Basic middleware structure
    
    call_next: Function that calls the next middleware or route handler
    """
    # BEFORE request processing
    print(f"Incoming request: {request.method} {request.url.path}")
    
    # Process request (call next middleware/route)
    response = await call_next(request)
    
    # AFTER request processing
    print(f"Response status: {response.status_code}")
    
    return response

# Multiple middleware example
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """
    Measure request processing time
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response
\`\`\`

---

## Production Middleware Stack

### Complete Middleware Stack

\`\`\`python
"""
Production-ready middleware stack
Ordered correctly for security and functionality
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import logging
import time
from uuid import uuid4
from typing import Callable

logger = logging.getLogger(__name__)

app = FastAPI()

# 1. OUTERMOST: Global Exception Handler
@app.middleware("http")
async def exception_handling_middleware(request: Request, call_next):
    """
    Catch all unhandled exceptions
    Must be outermost to catch errors from all other middleware
    """
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(
            f"Unhandled exception: {exc}",
            exc_info=True,
            extra={
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None
            }
        )
        
        # Send to error tracking (Sentry)
        if hasattr(request.state, "request_id"):
            logger.error(f"Request ID: {request.state.request_id}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown")
            }
        )

# 2. Request ID Generation
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Generate unique request ID for tracing
    Must be early so all logging includes it
    """
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    # Add to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response

# 3. Request Logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Log all requests with timing and context
    After request_id so logs include it
    """
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent")
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    
    logger.info(
        f"Request completed: {response.status_code} in {process_time:.3f}s",
        extra={
            "request_id": request.state.request_id,
            "status_code": response.status_code,
            "process_time": process_time
        }
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    
    return response

# 4. Security Headers
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Add security headers to all responses
    """
    response = await call_next(request)
    
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Enable XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Strict Transport Security (HTTPS only)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    # Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions Policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

# 5. Rate Limiting (before auth to prevent credential discovery)
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """
    Rate limit requests to prevent abuse
    Before authentication to prevent timing attacks
    """
    # Get identifier (IP for now, will be user_id after auth)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    is_limited = await check_rate_limit(client_ip)
    
    if is_limited:
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    
    return await call_next(request)

# Helper function for rate limiting
async def check_rate_limit(identifier: str) -> bool:
    """
    Check if identifier has exceeded rate limit
    Implementation with Redis in production
    """
    # Simplified for example
    # In production: use Redis with sliding window
    return False  # Not rate limited

# 6. CORS (if using add_middleware)
# Applied via app.add_middleware() - see CORS section below

# 7. Authentication (closest to routes)
# Applied per-route via Depends(get_current_user)
# Not global middleware because not all routes need auth
\`\`\`

### Middleware Order Matters

\`\`\`python
"""
Why middleware order is critical
"""

# ❌ WRONG ORDER
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Tries to use request_id...
    logger.info(f"Auth check: {request.state.request_id}")  # ERROR! Not set yet
    return await call_next(request)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = str(uuid4())
    return await call_next(request)

# ✅ CORRECT ORDER
# 1. Request ID first (generate)
# 2. Auth second (use request_id in logs)

# Order guidelines:
# 1. Exception handler (outermost - catches everything)
# 2. Request ID (early - used by all other middleware)
# 3. Logging (after request_id available)
# 4. CORS (early - applies to all requests)
# 5. Security headers
# 6. Rate limiting (before auth to prevent attacks)
# 7. Authentication (closest to routes)
\`\`\`

---

## CORS (Cross-Origin Resource Sharing)

### Understanding CORS

\`\`\`
The Problem: Same-Origin Policy (SOP)

Scenario:
- Your API: https://api.example.com
- Your frontend: https://app.example.com  (different subdomain!)
- Malicious site: https://evil.com

Same-Origin Policy: Browser blocks JavaScript from accessing responses
from different origins (protocol + domain + port).

Without CORS:
- app.example.com can't call api.example.com (different subdomain!)
- This breaks your app

With CORS:
- api.example.com explicitly allows app.example.com
- Browser permits the request
- evil.com still blocked (not in allow list)

Origin = protocol + domain + port:
- https://example.com:443
- https://api.example.com:443 (different! subdomain)
- http://example.com:80 (different! protocol)
- https://example.com:8000 (different! port)
\`\`\`

### CORS Configuration

\`\`\`python
"""
Production CORS configuration
"""

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Basic CORS (development only!)
if settings.DEBUG:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ⚠️ INSECURE! Allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Production CORS (secure)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://example.com",  # Marketing site
        "https://app.example.com",  # Web app
        "https://admin.example.com",  # Admin panel
    ],
    allow_credentials=True,  # Allow cookies and Authorization header
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],  # Specific methods
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],  # Specific headers
    expose_headers=["X-Request-ID", "X-Process-Time"],  # Expose to JavaScript
    max_age=3600,  # Cache preflight for 1 hour
)

# Dynamic CORS (based on environment)
allowed_origins = {
    "development": [
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    "staging": [
        "https://staging.example.com",
    ],
    "production": [
        "https://example.com",
        "https://app.example.com",
    ]
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins.get(settings.ENVIRONMENT, []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
\`\`\`

### CORS Preflight

\`\`\`python
"""
Understanding CORS preflight requests
"""

# Simple requests (no preflight):
# - Methods: GET, HEAD, POST
# - Content-Type: application/x-www-form-urlencoded, multipart/form-data, text/plain
# - No custom headers

# Non-simple requests (preflight required):
# - Methods: PUT, DELETE, PATCH
# - Content-Type: application/json
# - Custom headers: Authorization, X-Custom-Header

# Preflight flow:
"""
1. Browser sends OPTIONS request:
   OPTIONS /api/users/1
   Origin: https://app.example.com
   Access-Control-Request-Method: DELETE
   Access-Control-Request-Headers: Authorization

2. Server responds:
   Access-Control-Allow-Origin: https://app.example.com
   Access-Control-Allow-Methods: DELETE
   Access-Control-Allow-Headers: Authorization
   Access-Control-Max-Age: 3600

3. Browser caches response (1 hour)

4. Browser sends actual request:
   DELETE /api/users/1
   Origin: https://app.example.com
   Authorization: Bearer token...

5. Server responds with data + CORS headers
"""

# Optimize preflight with caching
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour (reduces OPTIONS requests)
)
\`\`\`

### CORS Security

\`\`\`python
"""
CORS security considerations
"""

# ❌ INSECURE: Allow all origins with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Any site can make requests
    allow_credentials=True,  # With your users' cookies!
)
# Result: evil.com can steal user data using their cookies

# ✅ SECURE: Specific origins with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],  # Only YOUR site
    allow_credentials=True,  # Cookies allowed from YOUR site only
)

# ✅ SECURE: Allow all origins WITHOUT credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Any site
    allow_credentials=False,  # No cookies/auth headers
)
# Use case: Public API (no authentication)

# Dynamic origin validation
async def validate_origin(origin: str) -> bool:
    """
    Validate origin against database or regex
    """
    # Check against database
    allowed = await db.query(AllowedOrigin).filter(
        AllowedOrigin.origin == origin
    ).first()
    
    if allowed:
        return True
    
    # Check against pattern (for subdomains)
    import re
    if re.match(r"https://.*\.example\.com$", origin):
        return True
    
    return False
\`\`\`

---

## Rate Limiting

### Redis-Based Rate Limiting

\`\`\`python
"""
Production rate limiting with Redis
Distributed across multiple API servers
"""

import redis.asyncio as redis
from datetime import datetime, timedelta

redis_client = redis.from_url("redis://localhost:6379")

class RateLimiter:
    """
    Token bucket rate limiter with Redis
    """
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int = 60
    ) -> dict:
        """
        Check rate limit using sliding window
        
        Args:
            identifier: Unique identifier (IP, user_id)
            limit: Max requests per window
            window_seconds: Time window in seconds
        
        Returns:
            {
                "allowed": bool,
                "remaining": int,
                "reset_at": int (Unix timestamp)
            }
        """
        key = f"rate_limit:{identifier}"
        now = datetime.utcnow().timestamp()
        window_start = now - window_seconds
        
        # Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove old entries outside window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        pipe.zcard(key)
        
        # Execute pipeline
        results = await pipe.execute()
        request_count = results[1]
        
        # Check if over limit
        if request_count >= limit:
            # Rate limited
            # Get oldest entry to calculate reset time
            oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_at = int(oldest[0][1] + window_seconds)
            else:
                reset_at = int(now + window_seconds)
            
            return {
                "allowed": False,
                "remaining": 0,
                "reset_at": reset_at,
                "limit": limit
            }
        
        # Add current request
        await self.redis.zadd(key, {str(now): now})
        
        # Set expiry
        await self.redis.expire(key, window_seconds)
        
        return {
            "allowed": True,
            "remaining": limit - request_count - 1,
            "reset_at": int(now + window_seconds),
            "limit": limit
        }

rate_limiter = RateLimiter(redis_client)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Apply rate limiting based on user tier
    """
    # Determine identifier and limit
    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        identifier = f"user:{user.id}"
        
        # Tier-based limits
        if user.tier == "premium":
            limit = 10000  # 10,000 requests/minute
        elif user.tier == "pro":
            limit = 1000   # 1,000 requests/minute
        else:
            limit = 100    # 100 requests/minute (free)
    else:
        # Unauthenticated: limit by IP
        identifier = f"ip:{request.client.host}"
        limit = 10  # 10 requests/minute
    
    # Check rate limit
    result = await rate_limiter.check_rate_limit(identifier, limit, window_seconds=60)
    
    # Add rate limit headers (always)
    rate_limit_headers = {
        "X-RateLimit-Limit": str(result["limit"]),
        "X-RateLimit-Remaining": str(result["remaining"]),
        "X-RateLimit-Reset": str(result["reset_at"])
    }
    
    if not result["allowed"]:
        # Rate limited - return 429
        retry_after = result["reset_at"] - int(datetime.utcnow().timestamp())
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Limit: {result['limit']} requests per minute.",
                "limit": result["limit"],
                "reset_at": result["reset_at"],
                "retry_after": retry_after
            },
            headers={
                **rate_limit_headers,
                "Retry-After": str(retry_after)
            }
        )
    
    # Continue request
    response = await call_next(request)
    
    # Add rate limit headers to successful response
    for key, value in rate_limit_headers.items():
        response.headers[key] = value
    
    return response
\`\`\`

### Route-Specific Rate Limiting

\`\`\`python
"""
Apply different rate limits to different endpoints
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Endpoint-specific limits
@app.post("/auth/login")
@limiter.limit("5/minute")  # Only 5 login attempts per minute
async def login(request: Request, credentials: LoginCredentials):
    """Strict limit on login to prevent brute force"""
    return authenticate(credentials)

@app.get("/api/data")
@limiter.limit("100/minute")  # 100 data requests per minute
async def get_data(request: Request):
    """Standard limit for data endpoints"""
    return []

@app.get("/api/search")
@limiter.limit("20/minute")  # Lower limit for expensive searches
async def search(request: Request, query: str):
    """Lower limit for resource-intensive operations"""
    return search_database(query)

# User-specific limits
def get_user_rate_limit(request: Request) -> str:
    """Custom key function based on user tier"""
    if hasattr(request.state, "user"):
        user = request.state.user
        if user.is_premium:
            return "1000/minute"
        else:
            return "100/minute"
    return "10/minute"  # Anonymous

@app.get("/api/premium")
@limiter.limit(get_user_rate_limit)
async def premium_endpoint(request: Request):
    """Dynamic limit based on user tier"""
    return []
\`\`\`

---

## Request/Response Transformation

### Request Preprocessing

\`\`\`python
"""
Middleware for request preprocessing
"""

@app.middleware("http")
async def request_preprocessing_middleware(request: Request, call_next):
    """
    Preprocess requests: normalize, validate, enrich
    """
    # Normalize headers
    # Convert to lowercase for consistent access
    normalized_headers = {
        k.lower(): v for k, v in request.headers.items()
    }
    request.state.normalized_headers = normalized_headers
    
    # Extract and validate API version
    api_version = request.headers.get("API-Version", "v1")
    request.state.api_version = api_version
    
    # Extract client info
    user_agent = request.headers.get("User-Agent", "")
    request.state.client_info = {
        "user_agent": user_agent,
        "is_mobile": "Mobile" in user_agent,
        "is_bot": "bot" in user_agent.lower()
    }
    
    # Block bots from certain endpoints
    if request.state.client_info["is_bot"] and request.url.path.startswith("/api/expensive"):
        return JSONResponse(
            status_code=403,
            content={"error": "Bots not allowed on this endpoint"}
        )
    
    return await call_next(request)
\`\`\`

### Response Postprocessing

\`\`\`python
"""
Middleware for response postprocessing
"""

import gzip

@app.middleware("http")
async def response_compression_middleware(request: Request, call_next):
    """
    Compress responses if client supports gzip
    """
    response = await call_next(request)
    
    # Check if client accepts gzip
    accept_encoding = request.headers.get("Accept-Encoding", "")
    
    if "gzip" in accept_encoding and response.status_code == 200:
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Compress if large enough (> 1KB)
        if len(body) > 1024:
            compressed = gzip.compress(body)
            
            return Response(
                content=compressed,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed))
                },
                media_type=response.media_type
            )
    
    return response
\`\`\`

---

## Production Monitoring

### Metrics Collection

\`\`\`python
"""
Middleware for metrics collection
"""

from prometheus_client import Counter, Histogram

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Collect request metrics for monitoring
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
\`\`\`

---

## Summary

✅ **Middleware fundamentals**: Request/response interception, onion model  
✅ **Middleware ordering**: Critical for security and functionality  
✅ **CORS**: Cross-origin resource sharing for frontend integration  
✅ **Rate limiting**: Redis-based distributed rate limiting  
✅ **Security headers**: Comprehensive security header implementation  
✅ **Request/response transformation**: Preprocessing and postprocessing  
✅ **Monitoring**: Metrics collection for production observability  

### Best Practices

**1. Middleware order**:
- Exception handler outermost
- Request ID early
- Rate limiting before auth
- Authentication innermost

**2. CORS security**:
- Never use * with credentials
- Specify exact origins in production
- Use max_age to cache preflight

**3. Rate limiting**:
- Use Redis for distributed limiting
- Tier-based limits (anonymous < free < premium)
- Add rate limit headers

**4. Security headers**:
- Apply to all responses
- Use HSTS for HTTPS
- Implement CSP

### Next Steps

In the next section, we'll explore **API Documentation**: customizing OpenAPI/Swagger documentation, adding examples, and creating comprehensive API docs.

**Production mindset**: Middleware is your first line of defense. Get the order right, implement rate limiting, configure CORS properly, and add comprehensive security headers!
`,
};
