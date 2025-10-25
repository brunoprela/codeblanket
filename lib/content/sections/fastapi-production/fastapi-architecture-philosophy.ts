export const fastapiArchitecturePhilosophy = {
  title: 'FastAPI Architecture & Philosophy',
  id: 'fastapi-architecture-philosophy',
  content: `
# FastAPI Architecture & Philosophy

## Introduction

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. Since its release in 2018, FastAPI has become one of the fastest-growing Python frameworks, rivaling established frameworks like Flask and Django in popularity while delivering performance comparable to NodeJS and Go.

**Why FastAPI matters for production:**
- **Speed**: One of the fastest Python frameworks available (thanks to Starlette and Pydantic)
- **Type Safety**: Full IDE support with autocomplete and error checking
- **Automatic Documentation**: Interactive API docs (Swagger UI) generated automatically
- **Modern Python**: Built on async/await, type hints, and Python 3.7+ features
- **Production Ready**: Used by companies like Microsoft, Netflix, Uber

In this section, you'll understand:
- FastAPI's design philosophy and core principles
- The architecture and component ecosystem
- Why FastAPI is ideal for production APIs
- How FastAPI compares to Flask and Django
- When to use FastAPI

### The FastAPI Ecosystem

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                      FastAPI                             │
│  (High-level framework, routing, DI, validation)         │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌─────────┐          ┌──────────────┐
│ Starlette│          │   Pydantic   │
│ (ASGI    │          │ (Data        │
│  toolkit)│          │  validation) │
└──────────┘          └──────────────┘
    │                       │
    ▼                       │
┌─────────────┐            │
│  Uvicorn    │            │
│  (ASGI      │            │
│   server)   │            │
└─────────────┘            │
                           ▼
                    ┌──────────────┐
                    │  Python Type │
                    │    Hints     │
                    └──────────────┘
\`\`\`

---

## Core Philosophy & Design Principles

### 1. Standards-Based

FastAPI is built on **open standards**:

**ASGI (Asynchronous Server Gateway Interface)**:
\`\`\`python
"""
FastAPI is built on ASGI, not WSGI

WSGI (Flask, Django): Synchronous only
ASGI (FastAPI, Starlette): Both sync and async
"""

from fastapi import FastAPI

app = FastAPI()

# Async endpoint (preferred)
@app.get("/users/{user_id}")
async def get_user (user_id: int):
    # Async database call
    user = await db.fetch_user (user_id)
    return user

# Sync endpoint (also supported)
@app.get("/sync-endpoint")
def sync_endpoint():
    # Regular synchronous code
    return {"message": "This works too"}
\`\`\`

**OpenAPI (formerly Swagger)**:
- Automatic API documentation
- Industry-standard format
- Client code generation
- API testing tools

**JSON Schema**:
- Data validation
- Type checking
- Documentation generation

### 2. Type-Driven Development

FastAPI leverages Python **type hints** as the source of truth:

\`\`\`python
"""
Types Drive Everything in FastAPI
"""

from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

app = FastAPI()

# Request model with validation
class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=120)
    is_active: bool = True
    created_at: datetime = Field (default_factory=datetime.utcnow)
    tags: List[str] = []

# Response model
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

@app.post("/users", response_model=UserResponse)
async def create_user (user: User):
    """
    Type hints provide:
    - Request validation
    - Response serialization
    - API documentation
    - IDE autocomplete
    - Error messages
    """
    # Save to database
    db_user = await save_user (user)
    return db_user

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int = Path(..., description="User ID", ge=1),
    include_inactive: bool = Query(False, description="Include inactive users")
):
    """
    Path and Query parameters with validation
    
    - user_id: Must be >= 1
    - include_inactive: Optional boolean, defaults to False
    """
    user = await fetch_user (user_id, include_inactive)
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    return user
\`\`\`

**What types provide**:
- ✅ Automatic validation
- ✅ Serialization/deserialization
- ✅ API documentation
- ✅ IDE autocomplete
- ✅ Type checking (mypy, pyright)

### 3. Developer Experience First

FastAPI prioritizes **developer productivity**:

\`\`\`python
"""
Minimal Boilerplate, Maximum Features
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# That\'s it! You get:
# - Automatic OpenAPI docs at /docs
# - ReDoc documentation at /redoc
# - JSON Schema at /openapi.json
# - Request validation
# - Response serialization
# - Error handling
# - CORS support (with middleware)
# - Authentication support
# - Dependency injection
\`\`\`

**Developer benefits**:
- ⚡ Fast to write code (300-500% productivity increase vs alternatives)
- 🐛 Fewer bugs (type checking catches errors early)
- 📚 Self-documenting (OpenAPI generated from code)
- 🧪 Easy to test (dependency injection, type hints)
- 🚀 Fast to learn (intuitive API, great docs)

### 4. Performance Without Compromise

FastAPI doesn't sacrifice performance for features:

**Benchmarks (requests per second)**:
\`\`\`
NodeJS (Express):      ~40,000 req/s
Go (Gin):              ~45,000 req/s
FastAPI (Uvicorn):     ~37,000 req/s ⭐
Flask:                 ~3,000 req/s
Django:                ~1,500 req/s
\`\`\`

**Why FastAPI is fast**:
1. **ASGI**: Async I/O, non-blocking operations
2. **Starlette**: High-performance ASGI toolkit
3. **Pydantic**: Fast Rust-based validation
4. **Uvicorn**: Fast ASGI server
5. **Async/await**: Efficient concurrency

\`\`\`python
"""
Performance: Async I/O
"""

from fastapi import FastAPI
import httpx

app = FastAPI()

# Slow: Synchronous (blocks thread)
@app.get("/slow")
def slow_endpoint():
    # Each request blocks for 1 second
    response = requests.get("https://api.example.com/data")
    return response.json()

# Fast: Asynchronous (non-blocking)
@app.get("/fast")
async def fast_endpoint():
    # Can handle 1000s of concurrent requests
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
\`\`\`

---

## FastAPI Architecture

### Core Components

#### 1. Application Instance

\`\`\`python
"""
FastAPI Application
"""

from fastapi import FastAPI

# Create app instance
app = FastAPI(
    title="My API",
    description="Production API with FastAPI",
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc
    openapi_url="/openapi.json" # OpenAPI schema
)

# Global state and lifespan
@app.on_event("startup")
async def startup():
    """Run on application startup"""
    print("Starting up...")
    # Connect to database
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    """Run on application shutdown"""
    print("Shutting down...")
    # Close database connections
    await database.disconnect()
\`\`\`

#### 2. Routing System

\`\`\`python
"""
Routing in FastAPI
"""

from fastapi import FastAPI, APIRouter

app = FastAPI()

# Direct routes
@app.get("/")
async def root():
    return {"message": "Home"}

# Route with path parameters
@app.get("/items/{item_id}")
async def get_item (item_id: int):
    return {"item_id": item_id}

# Using APIRouter for organization
router = APIRouter (prefix="/api/v1", tags=["users"])

@router.get("/users")
async def list_users():
    return []

@router.get("/users/{user_id}")
async def get_user (user_id: int):
    return {"user_id": user_id}

# Include router
app.include_router (router)
\`\`\`

#### 3. Dependency Injection System

\`\`\`python
"""
Dependency Injection - FastAPI's Superpower
"""

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

app = FastAPI()

# Dependency: Database session
def get_db() -> Session:
    """
    Dependency that provides database session
    Automatically closes connection after request
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency: Current user
async def get_current_user(
    token: str = Depends (oauth2_scheme),
    db: Session = Depends (get_db)
):
    """
    Dependency that validates token and returns user
    Can depend on other dependencies (get_db)
    """
    user = await validate_token (token, db)
    if not user:
        raise HTTPException (status_code=401, detail="Invalid token")
    return user

# Use dependencies in endpoints
@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends (get_current_user),
    db: Session = Depends (get_db)
):
    """
    Dependencies injected automatically
    - current_user: Validated from token
    - db: Database session
    """
    return current_user

# Dependencies can be reused, nested, and tested easily
\`\`\`

#### 4. Pydantic Models (Request/Response)

\`\`\`python
"""
Pydantic: Data Validation + Serialization
"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    """Request model for creating users"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: Optional[int] = Field(None, ge=0, le=120)
    
    @validator('password')
    def password_strength (cls, v):
        """Custom validation"""
        if not any (c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any (c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v

class UserResponse(BaseModel):
    """Response model (excludes password)"""
    id: int
    username: str
    email: str
    age: Optional[int]
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True  # Allow from ORM models

@app.post("/users", response_model=UserResponse)
async def create_user (user: UserCreate):
    """
    Automatic validation:
    - username: 3-50 chars
    - email: Valid email format
    - password: 8+ chars, uppercase, digit
    - age: 0-120 if provided
    
    Response automatically filtered (password excluded)
    """
    db_user = await save_user (user)
    return db_user
\`\`\`

---

## FastAPI vs Alternatives

### FastAPI vs Flask

\`\`\`python
"""
Flask: Synchronous, Flexible, Mature
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user (user_id):
    # Manual validation
    if user_id < 1:
        return jsonify({"error": "Invalid user ID"}), 400
    
    # Manual JSON handling
    user = fetch_user (user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Manual serialization
    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email
    })

# Pros: Simple, flexible, mature ecosystem
# Cons: No validation, no async, manual error handling, no auto-docs
\`\`\`

\`\`\`python
"""
FastAPI: Async, Type-Safe, Auto-documented
"""

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user (user_id: int = Path(..., ge=1)):
    # Automatic validation (user_id >= 1)
    # Automatic OpenAPI docs
    # Async by default
    user = await fetch_user (user_id)
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    
    # Automatic serialization (response_model)
    return user

# Pros: Validation, async, auto-docs, type safety, fast
# Cons: Newer (less mature), requires async knowledge
\`\`\`

**When to use Flask**:
- Legacy projects
- Simple synchronous APIs
- Need specific Flask extensions
- Team unfamiliar with async

**When to use FastAPI**:
- New projects
- High-performance requirements
- Need automatic documentation
- Modern Python practices (type hints, async)

### FastAPI vs Django/DRF

\`\`\`python
"""
Django REST Framework: Full-featured, Batteries-included
"""

from django.db import models
from rest_framework import serializers, viewsets

class User (models.Model):
    username = models.CharField (max_length=50)
    email = models.EmailField()

class UserSerializer (serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class UserViewSet (viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

# Pros: Admin panel, ORM, auth, mature ecosystem
# Cons: Heavyweight, no async (until 3.1+), slower performance
\`\`\`

\`\`\`python
"""
FastAPI: API-focused, Lightweight, Async
"""

from fastapi import FastAPI
from sqlalchemy.orm import Session

app = FastAPI()

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends (get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    return user

# Pros: Fast, async, lightweight, auto-docs
# Cons: No admin panel, less batteries-included
\`\`\`

**When to use Django/DRF**:
- Full web application (HTML views + API)
- Need Django admin panel
- Large team, established patterns
- Monolithic architecture

**When to use FastAPI**:
- API-only microservices
- High-performance requirements
- Modern async architecture
- Greenfield projects

### Comparison Matrix

| Feature | FastAPI | Flask | Django/DRF |
|---------|---------|-------|------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Async Support | ✅ Native | ⚠️ Limited | ⚠️ Limited (3.1+) |
| Type Safety | ✅ Full | ❌ None | ⚠️ Partial |
| Auto Documentation | ✅ OpenAPI | ❌ Manual | ✅ Browsable API |
| Validation | ✅ Pydantic | ❌ Manual | ✅ Serializers |
| Learning Curve | Medium | Easy | Steep |
| Ecosystem Maturity | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Production Ready | ✅ | ✅ | ✅ |
| Best For | APIs, microservices | Simple APIs | Full apps |

---

## When to Use FastAPI

### ✅ Ideal Use Cases

**1. RESTful APIs**:
- Microservices architecture
- Backend for mobile/web apps
- Public APIs with documentation
- Internal service communication

**2. High-Performance Applications**:
- Real-time data processing
- WebSocket servers
- Streaming data APIs
- High-concurrency systems

**3. Data-Intensive Applications**:
- Machine learning model serving
- Data pipelines
- Analytics APIs
- IoT data ingestion

**4. Modern Python Projects**:
- Python 3.7+ projects
- Type-annotated codebases
- Async/await architecture
- Microservices ecosystem

### ❌ Not Ideal For

**1. Traditional Web Applications**:
- Server-side rendered HTML
- Session-based authentication
- Form handling
- **Use Django instead**

**2. Legacy Python Systems**:
- Python 2.7 or Python 3.6
- Synchronous-only code
- No type hints
- **Use Flask instead**

**3. Simple Scripts**:
- One-off tasks
- Command-line tools
- Data processing scripts
- **Use plain Python**

---

## Production Considerations

### Why FastAPI is Production-Ready

\`\`\`python
"""
Production Features Built-In
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging

app = FastAPI()

# Middleware: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware: Request timing
@app.middleware("http")
async def add_process_time_header (request: Request, call_next):
    start_time = time.time()
    response = await call_next (request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str (process_time)
    return response

# Global exception handling
@app.exception_handler(Exception)
async def global_exception_handler (request: Request, exc: Exception):
    logging.error (f"Global exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }
\`\`\`

### Real-World FastAPI Adoption

**Companies using FastAPI**:
- **Microsoft**: Azure ML pipelines
- **Netflix**: Machine learning serving
- **Uber**: Real-time data APIs
- **Explosion AI**: spaCy API services
- **Kubernetes**: API scaffolding tools

---

## Summary

### Key Takeaways

✅ **FastAPI is built on standards**: ASGI, OpenAPI, JSON Schema  
✅ **Type-driven development**: Types power validation, documentation, IDE support  
✅ **Developer experience**: Minimal boilerplate, maximum productivity  
✅ **Performance**: One of fastest Python frameworks (NodeJS/Go level)  
✅ **Production ready**: Used by top tech companies, battle-tested  
✅ **Modern Python**: Async/await, type hints, Python 3.7+ features  
✅ **Automatic documentation**: OpenAPI/Swagger generated from code

### Core Architecture

- **FastAPI**: High-level framework
- **Starlette**: ASGI toolkit
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### When to Use FastAPI

**Use FastAPI for**:
- RESTful APIs and microservices
- High-performance applications
- Async/await architecture
- Automatic API documentation
- Type-safe Python projects

**Use alternatives for**:
- Traditional web apps with HTML (Django)
- Simple synchronous APIs (Flask)
- Legacy Python systems (Flask)

### Next Steps

In the next section, we'll dive deep into **Request & Response Models with Pydantic**: the foundation of FastAPI's type-driven development. You'll learn how to leverage Pydantic for validation, serialization, and creating rock-solid API contracts.

**Production mindset**: FastAPI isn't just fast—it's designed for building maintainable, type-safe, well-documented production APIs. Master the fundamentals, and you'll build better APIs faster.
`,
};
