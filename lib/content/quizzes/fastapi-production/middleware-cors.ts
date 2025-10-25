export const middlewareCorsQuiz = {
  title: 'Middleware & CORS - Discussion Questions',
  id: 'middleware-cors-quiz',
  questions: [
    {
      id: 1,
      question:
        'Design a comprehensive middleware stack for a production FastAPI application that includes: request logging, request ID generation, authentication, rate limiting, CORS, and security headers. Explain the order in which middleware should be applied and why order matters. Implement the complete stack showing how each middleware processes requests and responses.',
      answer: `**Middleware Stack (Order Matters)**:

Order (outer to inner):
1. Exception handling (outermost - catches all)
2. Request ID generation
3. Request logging
4. CORS
5. Security headers
6. Rate limiting
7. Authentication
8. Application routes (innermost)

\`\`\`python
# 1. Exception handling (first - catches everything)
@app.middleware("http")
async def exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled: {e}")
        return JSONResponse({"error": "Internal error"}, 500)

# 2. Request ID (early - used in logging)
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# 3. Logging (after request ID available)
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.time()
    logger.info(f"{request.method} {request.url.path}", extra={"request_id": request.state.request_id})
    response = await call_next(request)
    logger.info(f"Status: {response.status_code} ({time.time()-start:.3f}s)")
    return response

# 4. CORS (before security headers)
app.add_middleware(CORSMiddleware, allow_origins=["https://example.com"])

# 5. Security headers
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    return response

# 6. Rate limiting (before auth to prevent auth spam)
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if await is_rate_limited(request.client.host):
        return JSONResponse({"error": "Rate limited"}, 429)
    return await call_next(request)

# 7. Authentication (innermost - closest to routes)
# Applied per-route via Depends(get_current_user)
\`\`\`

Why order matters: Exception handler must be outermost to catch errors from all other middleware. Request ID must be early so logging can use it. Rate limiting before auth prevents attackers from discovering valid credentials through timing attacks.`,
    },
    {
      id: 2,
      question:
        'Explain CORS in detail: what problem does it solve, how the browser enforces it, and what each CORS header means (Access-Control-Allow-Origin, Allow-Credentials, Allow-Methods, Allow-Headers). Design CORS configuration for a production SaaS application where: (1) the marketing site is at example.com, (2) the web app is at app.example.com, (3) the mobile app needs access, and (4) third-party integrations should be blocked.',
      answer: `**CORS (Cross-Origin Resource Sharing)**:

**Problem**: Same-Origin Policy - browsers block JavaScript from making requests to different origins for security.

Origin = protocol + domain + port
- https://example.com:443
- https://app.example.com:443 (different subdomain = different origin!)

**How browsers enforce**:
1. Browser sends OPTIONS preflight request
2. Server responds with CORS headers
3. Browser checks if origin is allowed
4. If allowed, browser sends actual request
5. If not, browser blocks and shows CORS error

**CORS Headers**:

- **Access-Control-Allow-Origin**: Which origins can access
  - "*": Allow all (insecure!)
  - "https://example.com": Specific origin only

- **Access-Control-Allow-Credentials**: Allow cookies/auth headers
  - "true": Allow credentials
  - Cannot use "*" origin with credentials

- **Access-Control-Allow-Methods**: Which HTTP methods allowed
  - "GET, POST, PUT, DELETE"

- **Access-Control-Allow-Headers**: Which headers allowed
  - "Content-Type, Authorization"

**Production SaaS Configuration**:

\`\`\`python
# Determine allowed origins based on request
def get_allowed_origins(request: Request) -> List[str]:
    origin = request.headers.get("origin")
    
    allowed = [
        "https://example.com",  # Marketing site
        "https://app.example.com",  # Web app
    ]
    
    # Mobile app (capacitor-based)
    if origin and origin.startswith("capacitor://"):
        allowed.append(origin)
    
    return allowed

# Dynamic CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://example.com",
        "https://app.example.com",
        "capacitor://localhost"  # Mobile
    ],
    allow_credentials=True,  # Allow cookies/JWT
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID"],  # Expose to JavaScript
    max_age=3600  # Cache preflight for 1 hour
)
\`\`\`

Third-party integrations: Use API keys (not CORS), server-to-server calls (no browser = no CORS).`,
    },
    {
      id: 3,
      question:
        'Design a rate limiting strategy for a public API that has three tiers: (1) unauthenticated requests (by IP): 10/minute, (2) authenticated free users: 100/minute, (3) premium users: 1000/minute. Implement the rate limiting logic using Redis for distributed rate limiting across multiple API servers. How would you communicate rate limit information to clients (headers)? What happens when limits are exceeded?',
      answer: `**Distributed Rate Limiting with Redis**:

\`\`\`python
import redis.asyncio as redis
from datetime import datetime

redis_client = redis.from_url("redis://localhost:6379")

async def check_rate_limit(
    identifier: str,  # IP or user_id
    limit: int,  # requests per minute
    window: int = 60  # seconds
) -> dict:
    """
    Token bucket algorithm with Redis
    Returns: {allowed: bool, remaining: int, reset_at: timestamp}
    """
    key = f"rate_limit:{identifier}"
    now = datetime.utcnow().timestamp()
    
    # Redis pipeline for atomicity
    pipe = redis_client.pipeline()
    
    # Remove old entries outside window
    pipe.zremrangebyscore(key, 0, now - window)
    
    # Count requests in window
    pipe.zcard(key)
    
    # Add current request
    pipe.zadd(key, {str(now): now})
    
    # Set expiry
    pipe.expire(key, window)
    
    results = await pipe.execute()
    request_count = results[1]
    
    allowed = request_count < limit
    remaining = max(0, limit - request_count - 1)
    reset_at = int(now + window)
    
    return {
        "allowed": allowed,
        "remaining": remaining,
        "reset_at": reset_at,
        "limit": limit
    }

# Rate limit middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Determine identifier and limit
    if hasattr(request.state, "user"):
        # Authenticated
        user = request.state.user
        identifier = f"user:{user.id}"
        
        if user.is_premium:
            limit = 1000  # Premium: 1000/min
        else:
            limit = 100   # Free: 100/min
    else:
        # Unauthenticated
        identifier = f"ip:{request.client.host}"
        limit = 10  # IP: 10/min
    
    # Check rate limit
    rate_limit = await check_rate_limit(identifier, limit)
    
    # Add headers (always, even if not rate limited)
    headers = {
        "X-RateLimit-Limit": str(rate_limit["limit"]),
        "X-RateLimit-Remaining": str(rate_limit["remaining"]),
        "X-RateLimit-Reset": str(rate_limit["reset_at"])
    }
    
    if not rate_limit["allowed"]:
        # Rate limited
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Try again in {rate_limit['reset_at'] - datetime.utcnow().timestamp():.0f}s",
                "limit": rate_limit["limit"],
                "reset_at": rate_limit["reset_at"]
            },
            headers={
                **headers,
                "Retry-After": str(rate_limit["reset_at"] - int(datetime.utcnow().timestamp()))
            }
        )
    
    # Continue request
    response = await call_next(request)
    
    # Add rate limit headers to response
    for key, value in headers.items():
        response.headers[key] = value
    
    return response
\`\`\`

**Rate Limit Headers** (GitHub standard):
- X-RateLimit-Limit: 100 (max requests)
- X-RateLimit-Remaining: 73 (requests left)
- X-RateLimit-Reset: 1640000000 (Unix timestamp)
- Retry-After: 45 (seconds to wait)

**Client handling**:
\`\`\`javascript
async function apiCall() {
    const response = await fetch('/api/data');
    
    // Check rate limit headers
    const remaining = response.headers.get('X-RateLimit-Remaining');
    if (remaining < 10) {
        console.warn('Approaching rate limit!');
    }
    
    if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        console.error(\`Rate limited. Retry in \${retryAfter}s\`);
        // Exponential backoff or wait
    }
}
\`\`\`

**Distributed**: Redis ensures rate limits work across multiple API servers (all check same Redis instance).`,
    },
  ],
};
