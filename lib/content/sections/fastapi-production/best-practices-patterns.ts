export const bestPracticesPatterns = {
  title: 'FastAPI Best Practices & Patterns',
  id: 'best-practices-patterns',
  content: `
# FastAPI Best Practices & Patterns

## Introduction

Production FastAPI applications require proper architecture, security hardening, code quality standards, and performance optimization. This final section consolidates best practices learned throughout the module.

**What makes production-ready code:**
- **Organized structure**: Clear, modular architecture
- **Secure by default**: Security at every layer
- **Well-tested**: Comprehensive test coverage
- **Performant**: Optimized for production load
- **Maintainable**: Clean, documented code
- **Observable**: Logging and monitoring built-in

---

## Project Structure

### Recommended Layout

\`\`\`
my-fastapi-project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # App entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py     # Shared dependencies
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── users.py
│   │       ├── auth.py
│   │       └── posts.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Settings
│   │   ├── security.py        # Auth/crypto
│   │   └── database.py        # DB connection
│   ├── models/                # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── post.py
│   ├── schemas/               # Pydantic models
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── post.py
│   └── services/              # Business logic
│       ├── __init__.py
│       ├── user_service.py
│       └── auth_service.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── alembic/                   # Database migrations
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
\`\`\`

---

## Configuration Management

\`\`\`python
"""
Environment-based configuration with pydantic-settings
"""

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "My API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: list[str] = []
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
\`\`\`

---

## Security Hardening

### Complete Security Checklist

\`\`\`python
"""
Security best practices
"""

# ✅ 1. Use HTTPS in production
# ✅ 2. Set security headers
@app.middleware("http")
async def security_headers (request: Request, call_next):
    response = await call_next (request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response

# ✅ 3. Validate ALL inputs with Pydantic
class UserCreate(BaseModel):
    email: EmailStr  # Validates email format
    password: str = Field (min_length=8, max_length=100)
    username: str = Field (min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$")

# ✅ 4. Hash passwords properly
from passlib.context import CryptContext
pwd_context = CryptContext (schemes=["bcrypt"], deprecated="auto")

def hash_password (password: str) -> str:
    return pwd_context.hash (password)

# ✅ 5. Use parameterized queries (SQLAlchemy does this)
# ❌ BAD: user = db.execute (f"SELECT * FROM users WHERE id={user_id}")
# ✅ GOOD: user = db.execute (select(User).where(User.id == user_id))

# ✅ 6. Implement rate limiting
from slowapi import Limiter
limiter = Limiter (key_func=get_remote_address)

# ✅ 7. Restrict CORS to known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Not "*"
    allow_credentials=True
)

# ✅ 8. Use environment variables for secrets
SECRET_KEY = os.getenv("SECRET_KEY")  # Never hardcode!

# ✅ 9. Implement proper authentication
@app.get("/protected")
async def protected (user: User = Depends (get_current_user)):
    return {"user": user}

# ✅ 10. Log security events
logger.warning("Failed login attempt", extra={"ip": request.client.host})
\`\`\`

---

## Code Quality

### Formatting & Linting

\`\`\`bash
# Install tools
pip install black isort ruff mypy

# Format code
black app/
isort app/

# Lint
ruff check app/

# Type check
mypy app/

# Pre-commit hooks
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff
\`\`\`

---

## Performance Optimization

\`\`\`python
"""
Performance best practices
"""

# 1. Use connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True
)

# 2. Add database indexes
class User(Base):
    email = Column(String, unique=True, index=True)  # Index for lookups
    created_at = Column(DateTime, index=True)

# 3. Use async for I/O operations
@app.get("/data")
async def get_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
    return response.json()

# 4. Implement caching
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@cache (expire=60)
@app.get("/cached")
async def cached_endpoint():
    return expensive_operation()

# 5. Use response models to limit data
@app.get("/users/{user_id}", response_model=UserPublic)
async def get_user (user_id: int):
    return user  # Only returns fields in UserPublic

# 6. Batch database queries
users = await db.execute(
    select(User).options (selectinload(User.posts))  # Eager load
)
\`\`\`

---

## API Versioning

\`\`\`python
"""
URL-based API versioning
"""

v1_router = APIRouter (prefix="/api/v1", tags=["v1"])
v2_router = APIRouter (prefix="/api/v2", tags=["v2"])

@v1_router.get("/users")
async def get_users_v1():
    return []  # Old format

@v2_router.get("/users")
async def get_users_v2():
    return {"users": [], "total": 0}  # New format

app.include_router (v1_router)
app.include_router (v2_router)
\`\`\`

---

## Logging

\`\`\`python
"""
Structured logging with context
"""

import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware (request: Request, call_next):
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        request_id=request.state.request_id,
        user_id=getattr (request.state, "user_id", None)
    )
    
    response = await call_next (request)
    
    logger.info(
        "request_completed",
        status_code=response.status_code,
        request_id=request.state.request_id
    )
    
    return response
\`\`\`

---

## Common Anti-Patterns

\`\`\`python
"""
What NOT to do
"""

# ❌ DON'T store secrets in code
SECRET_KEY = "hardcoded-secret"  # Bad!

# ❌ DON'T use SELECT *
users = db.execute("SELECT * FROM users")

# ❌ DON'T skip input validation
@app.post("/users")
async def create_user (data: dict):  # No validation!
    pass

# ❌ DON'T ignore errors
try:
    user = get_user (id)
except:
    pass  # Silent failure!

# ❌ DON'T block the event loop
@app.get("/bad")
async def blocking():
    time.sleep(5)  # Blocks everything!
    
# ❌ DON'T expose internal errors
except Exception as e:
    return {"error": str (e)}  # Leaks internals!
\`\`\`

---

## Production Checklist

### Pre-Deployment
- [ ] All tests passing (>80% coverage)
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Security headers enabled
- [ ] Rate limiting configured
- [ ] CORS properly restricted
- [ ] Logging configured
- [ ] Monitoring enabled (Sentry/DataDog)
- [ ] Health checks implemented
- [ ] Documentation updated

### Deployment
- [ ] Zero-downtime strategy (rolling/blue-green)
- [ ] Database backups configured
- [ ] SSL certificates valid
- [ ] Load balancer configured
- [ ] Auto-scaling rules set
- [ ] Rollback plan documented

### Post-Deployment
- [ ] Monitor error rates
- [ ] Check response times
- [ ] Verify health checks
- [ ] Test critical flows
- [ ] Review logs
- [ ] Confirm metrics flowing

---

## Summary

✅ **Project structure**: Organized, modular architecture  
✅ **Configuration**: Environment-based with Pydantic  
✅ **Security**: Comprehensive hardening checklist  
✅ **Code quality**: Black, isort, ruff, mypy  
✅ **Performance**: Connection pooling, caching, async  
✅ **Versioning**: URL-based API versions  
✅ **Logging**: Structured logs with context  
✅ **Anti-patterns**: Know what to avoid  
✅ **Checklist**: Production deployment guide  

---

## Conclusion

🎉 **Congratulations!** You've completed FastAPI Production Mastery!

You now know how to build production-ready APIs with:
- FastAPI architecture and async fundamentals
- Request/response modeling with Pydantic
- Database integration with SQLAlchemy
- JWT authentication and authorization
- Background tasks with Celery
- WebSockets for real-time features
- File uploads and streaming
- Error handling and validation
- Middleware and CORS
- API documentation
- Comprehensive testing
- Async patterns
- Production deployment
- Best practices and security

**Go build amazing APIs!** 🚀
`,
};
