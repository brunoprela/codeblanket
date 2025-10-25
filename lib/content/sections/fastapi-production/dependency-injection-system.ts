export const dependencyInjectionSystem = {
  title: 'Dependency Injection System',
  id: 'dependency-injection-system',
  content: `
# Dependency Injection System

## Introduction

Dependency Injection (DI) is FastAPI's **secret weapon** for writing clean, testable, maintainable code. While other frameworks require complex IoC containers or service locators, FastAPI's DI is elegantly simple yet incredibly powerful—built directly into the framework using Python\'s type hints and the \`Depends()\` function.

**Why DI is transformative:**
- **Reusability**: Write once, use everywhere
- **Testability**: Easily mock dependencies for testing
- **Separation of concerns**: Business logic separate from infrastructure
- **Type safety**: Full IDE autocomplete and type checking
- **Composability**: Dependencies can depend on other dependencies

In production, DI solves real problems:
- Database session management
- Authentication and authorization
- Rate limiting and quotas
- Logging and metrics
- Configuration management
- External service clients

In this section, you'll master:
- Basic and advanced dependency patterns
- Dependency composition and hierarchies
- Class-based dependencies
- Dependency overrides for testing
- Performance optimization
- Production patterns

### The Dependency Tree

\`\`\`
Endpoint Handler
    ↓
get_current_user (depends on)
    ↓
verify_token (depends on)
    ↓
get_db (database session)
    ↓
get_redis (cache client)

FastAPI resolves this tree automatically!
\`\`\`

---

## Basic Dependencies

### Simple Dependencies

\`\`\`python
"""
Basic Dependency Injection
"""

from fastapi import FastAPI, Depends, Query
from typing import Optional

app = FastAPI()

# Simple dependency function
def common_parameters(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Common pagination parameters
    Reusable across multiple endpoints
    """
    return {"skip": skip, "limit": limit}

@app.get("/users")
async def list_users (commons: dict = Depends (common_parameters)):
    """
    commons = {"skip": 0, "limit": 10}
    Dependency injected automatically
    """
    return {"params": commons, "users": []}

@app.get("/products")
async def list_products (commons: dict = Depends (common_parameters)):
    """
    Same dependency, different endpoint
    No code duplication!
    """
    return {"params": commons, "products": []}

# Multiple dependencies
def get_api_key (api_key: str = Query(...)):
    """Validate API key"""
    if api_key != "secret":
        raise HTTPException (status_code=401, detail="Invalid API key")
    return api_key

def get_user_agent (user_agent: str = Header(...)):
    """Get user agent header"""
    return user_agent

@app.get("/secure-endpoint")
async def secure_endpoint(
    api_key: str = Depends (get_api_key),
    user_agent: str = Depends (get_user_agent),
    commons: dict = Depends (common_parameters)
):
    """
    Multiple dependencies injected
    All resolved before handler runs
    """
    return {
        "api_key": api_key,
        "user_agent": user_agent,
        "pagination": commons
    }
\`\`\`

### Database Session Dependency

\`\`\`python
"""
Database Session Management with Dependencies
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, Depends

# Database setup
DATABASE_URL = "postgresql://user:pass@localhost/db"
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker (bind=engine)

app = FastAPI()

def get_db() -> Session:
    """
    Database session dependency
    
    Yields session, ensures cleanup
    Called for every request
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Use in endpoints
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends (get_db)
):
    """
    db session automatically injected and closed
    No manual session management!
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    return user

@app.post("/users")
async def create_user(
    user: UserCreate,
    db: Session = Depends (get_db)
):
    """
    Same dependency, automatic cleanup
    """
    db_user = User(**user.dict())
    db.add (db_user)
    db.commit()
    db.refresh (db_user)
    return db_user

# Works with transactions
@app.post("/orders")
async def create_order(
    order: OrderCreate,
    db: Session = Depends (get_db)
):
    """
    Transaction automatically managed
    Rollback on exception, commit on success
    """
    try:
        # Create order
        db_order = Order(**order.dict())
        db.add (db_order)
        
        # Update inventory
        for item in order.items:
            product = db.query(Product).filter(Product.id == item.product_id).first()
            product.stock -= item.quantity
        
        db.commit()
        db.refresh (db_order)
        return db_order
    except Exception as e:
        db.rollback()
        raise HTTPException (status_code=400, detail=str (e))
\`\`\`

---

## Nested Dependencies

### Dependency Chains

\`\`\`python
"""
Dependencies Depending on Other Dependencies
"""

from fastapi import FastAPI, Depends, HTTPException, Header
from jose import JWTError, jwt
from datetime import datetime, timedelta

app = FastAPI()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

# Level 1: Database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Level 2: Get token from header
def get_token (authorization: str = Header(...)):
    """
    Extract token from Authorization header
    Depends on: Nothing
    """
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException (status_code=401, detail="Invalid auth scheme")
        return token
    except ValueError:
        raise HTTPException (status_code=401, detail="Invalid auth header")

# Level 3: Verify token
def verify_token(
    token: str = Depends (get_token)
):
    """
    Verify JWT token
    Depends on: get_token
    """
    try:
        payload = jwt.decode (token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException (status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException (status_code=401, detail="Invalid token")

# Level 4: Get current user
def get_current_user(
    user_id: int = Depends (verify_token),
    db: Session = Depends (get_db)
):
    """
    Fetch user from database
    Depends on: verify_token, get_db
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException (status_code=403, detail="User inactive")
    return user

# Level 5: Require admin
def get_current_admin(
    current_user: User = Depends (get_current_user)
):
    """
    Ensure user is admin
    Depends on: get_current_user (which depends on verify_token, get_db)
    """
    if not current_user.is_admin:
        raise HTTPException (status_code=403, detail="Admin required")
    return current_user

# Use in endpoint
@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends (get_current_user)
):
    """
    Dependency chain automatically resolved:
    1. get_token extracts token
    2. verify_token validates token
    3. get_db provides session
    4. get_current_user fetches user
    
    All automatic!
    """
    return current_user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends (get_current_admin),
    db: Session = Depends (get_db)
):
    """
    Admin-only endpoint
    Dependency chain:
    1. get_token
    2. verify_token
    3. get_db (called twice, but cached!)
    4. get_current_user
    5. get_current_admin
    6. get_db (for deletion)
    """
    user = db.query(User).filter(User.id == user_id).first()
    db.delete (user)
    db.commit()
    return {"deleted": user_id}
\`\`\`

### Dependency Caching

\`\`\`python
"""
Dependencies are Cached Within a Request
"""

from fastapi import FastAPI, Depends

app = FastAPI()

# This dependency is expensive
async def expensive_dependency():
    """
    Called only ONCE per request
    Even if used multiple times
    """
    print("Computing expensive result...")
    # Expensive computation
    result = await compute_something_expensive()
    return result

# Use multiple times
def service_a (data = Depends (expensive_dependency)):
    return f"Service A: {data}"

def service_b (data = Depends (expensive_dependency)):
    return f"Service B: {data}"

@app.get("/endpoint")
async def endpoint(
    a = Depends (service_a),
    b = Depends (service_b),
    direct = Depends (expensive_dependency)
):
    """
    expensive_dependency called only ONCE!
    Result cached and reused for:
    - service_a
    - service_b
    - direct
    
    Massive performance benefit
    """
    return {"a": a, "b": b, "direct": direct}

# Disable caching with use_cache=False
@app.get("/no-cache")
async def no_cache(
    result1 = Depends (expensive_dependency, use_cache=False),
    result2 = Depends (expensive_dependency, use_cache=False)
):
    """
    expensive_dependency called TWICE
    Use for dependencies that shouldn't be cached
    """
    return {"result1": result1, "result2": result2}
\`\`\`

---

## Class-Based Dependencies

### Callable Classes

\`\`\`python
"""
Class-Based Dependencies for Stateful Logic
"""

from fastapi import FastAPI, Depends, Query

app = FastAPI()

class Pagination:
    """
    Callable class for pagination
    """
    def __init__(
        self,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100)
    ):
        self.page = page
        self.page_size = page_size
    
    @property
    def offset (self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit (self) -> int:
        return self.page_size
    
    def apply_to_query (self, query):
        """Apply pagination to SQLAlchemy query"""
        return query.offset (self.offset).limit (self.limit)

@app.get("/users")
async def list_users(
    pagination: Pagination = Depends(),  # No need to call Pagination()
    db: Session = Depends (get_db)
):
    """
    Pagination instance injected
    Access properties and methods
    """
    query = db.query(User)
    query = pagination.apply_to_query (query)
    users = query.all()
    
    return {
        "data": users,
        "page": pagination.page,
        "page_size": pagination.page_size
    }

# More complex example
class FilterParams:
    """
    Complex filtering logic
    """
    def __init__(
        self,
        search: Optional[str] = Query(None, min_length=2),
        category: Optional[str] = None,
        min_price: Optional[float] = Query(None, ge=0),
        max_price: Optional[float] = Query(None, ge=0),
        in_stock: bool = True
    ):
        self.search = search
        self.category = category
        self.min_price = min_price
        self.max_price = max_price
        self.in_stock = in_stock
    
    def apply_filters (self, query):
        """Build filtered query"""
        if self.search:
            query = query.filter(Product.name.ilike (f"%{self.search}%"))
        if self.category:
            query = query.filter(Product.category == self.category)
        if self.min_price:
            query = query.filter(Product.price >= self.min_price)
        if self.max_price:
            query = query.filter(Product.price <= self.max_price)
        if self.in_stock:
            query = query.filter(Product.stock > 0)
        return query

@app.get("/products")
async def search_products(
    filters: FilterParams = Depends(),
    pagination: Pagination = Depends(),
    db: Session = Depends (get_db)
):
    """
    Multiple class-based dependencies
    Clean, organized, testable
    """
    query = db.query(Product)
    query = filters.apply_filters (query)
    query = pagination.apply_to_query (query)
    
    products = query.all()
    total = filters.apply_filters (db.query(Product)).count()
    
    return {
        "data": products,
        "pagination": {
            "page": pagination.page,
            "page_size": pagination.page_size,
            "total": total
        }
    }
\`\`\`

### Service Layer Pattern

\`\`\`python
"""
Service Layer with Dependency Injection
"""

from fastapi import FastAPI, Depends
from typing import List

app = FastAPI()

class UserService:
    """
    User service with injected dependencies
    """
    def __init__(self, db: Session = Depends (get_db)):
        self.db = db
    
    def get_user (self, user_id: int) -> User:
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException (status_code=404, detail="User not found")
        return user
    
    def list_users (self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.db.query(User).offset (skip).limit (limit).all()
    
    def create_user (self, user: UserCreate) -> User:
        db_user = User(**user.dict())
        self.db.add (db_user)
        self.db.commit()
        self.db.refresh (db_user)
        return db_user
    
    def update_user (self, user_id: int, user: UserUpdate) -> User:
        db_user = self.get_user (user_id)
        for key, value in user.dict (exclude_unset=True).items():
            setattr (db_user, key, value)
        self.db.commit()
        self.db.refresh (db_user)
        return db_user

class ProductService:
    """
    Product service with multiple dependencies
    """
    def __init__(
        self,
        db: Session = Depends (get_db),
        cache: Redis = Depends (get_redis)
    ):
        self.db = db
        self.cache = cache
    
    async def get_product (self, product_id: int) -> Product:
        # Check cache first
        cached = await self.cache.get (f"product:{product_id}")
        if cached:
            return Product(**json.loads (cached))
        
        # Fetch from database
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException (status_code=404)
        
        # Cache for 5 minutes
        await self.cache.setex(
            f"product:{product_id}",
            300,
            json.dumps (product.dict())
        )
        
        return product

# Use services in endpoints
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    user_service: UserService = Depends()
):
    """
    Service injected with its dependencies
    Clean separation of concerns
    """
    return user_service.get_user (user_id)

@app.post("/users")
async def create_user(
    user: UserCreate,
    user_service: UserService = Depends()
):
    return user_service.create_user (user)

@app.get("/products/{product_id}")
async def get_product(
    product_id: int,
    product_service: ProductService = Depends()
):
    """
    Service with multiple injected dependencies
    """
    return await product_service.get_product (product_id)
\`\`\`

---

## Router-Level Dependencies

### Shared Dependencies

\`\`\`python
"""
Apply Dependencies to All Routes in Router
"""

from fastapi import APIRouter, Depends, HTTPException

# Rate limiting dependency
async def rate_limit (api_key: str = Header(...)):
    """Check rate limit for API key"""
    if await is_rate_limited (api_key):
        raise HTTPException (status_code=429, detail="Rate limit exceeded")
    return api_key

# API key validation
async def verify_api_key (api_key: str = Header(...)):
    """Verify API key is valid"""
    if not await is_valid_api_key (api_key):
        raise HTTPException (status_code=401, detail="Invalid API key")
    return api_key

# Router with dependencies
api_router = APIRouter(
    prefix="/api",
    dependencies=[
        Depends (verify_api_key),  # All routes require API key
        Depends (rate_limit)        # All routes are rate limited
    ]
)

@api_router.get("/users")
async def list_users():
    """
    Automatically protected by:
    - verify_api_key
    - rate_limit
    
    No need to add Depends() to every endpoint!
    """
    return []

@api_router.get("/products")
async def list_products():
    """
    Same protection, zero duplication
    """
    return []

# Admin router with different dependencies
admin_router = APIRouter(
    prefix="/admin",
    dependencies=[
        Depends (verify_admin_token),  # Admin token required
        Depends (log_admin_action)      # Log all admin actions
    ]
)

@admin_router.delete("/users/{user_id}")
async def delete_user (user_id: int):
    """
    Admin-only, automatically logged
    """
    return {"deleted": user_id}

# Combine routers
app = FastAPI()
app.include_router (api_router)
app.include_router (admin_router)
\`\`\`

---

## Global Dependencies

### Application-Level Dependencies

\`\`\`python
"""
Dependencies Applied to Entire Application
"""

from fastapi import FastAPI, Depends, Request
import time

# Logging dependency
async def log_request (request: Request):
    """Log every request"""
    start = time.time()
    print(f"Request: {request.method} {request.url.path}")
    
    # This dependency doesn't return anything
    # Just performs side effects
    
    # Note: Use middleware for production logging
    # This is for demonstration

# Create app with global dependency
app = FastAPI(
    dependencies=[Depends (log_request)]
)

@app.get("/users")
async def list_users():
    """Logged automatically"""
    return []

@app.get("/products")
async def list_products():
    """Also logged automatically"""
    return []

# All endpoints in the app have log_request dependency
\`\`\`

---

## Testing with Dependencies

### Dependency Override

\`\`\`python
"""
Override Dependencies for Testing
"""

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

app = FastAPI()

# Production dependency
def get_db():
    db = ProductionDB()
    try:
        yield db
    finally:
        db.close()

def get_current_user (db = Depends (get_db)):
    # Complex logic to get user from database
    return db.query(User).first()

@app.get("/users/me")
async def read_users_me (current_user = Depends (get_current_user)):
    return current_user

# Testing
def test_read_users_me():
    # Mock dependencies
    def mock_get_db():
        return MockDB()
    
    def mock_get_current_user():
        return User (id=1, username="testuser")
    
    # Override dependencies
    app.dependency_overrides[get_db] = mock_get_db
    app.dependency_overrides[get_current_user] = mock_get_current_user
    
    client = TestClient (app)
    response = client.get("/users/me")
    
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    
    # Clean up
    app.dependency_overrides.clear()

# pytest fixture for common override
import pytest

@pytest.fixture
def client():
    """Test client with overridden dependencies"""
    def override_get_db():
        return MockDB()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient (app) as c:
        yield c
    
    app.dependency_overrides.clear()

def test_with_fixture (client):
    """Use fixture with overridden dependencies"""
    response = client.get("/users/me")
    assert response.status_code == 200
\`\`\`

---

## Advanced Patterns

### Optional Dependencies

\`\`\`python
"""
Optional Dependencies with Defaults
"""

from typing import Optional
from fastapi import FastAPI, Depends

app = FastAPI()

# Optional authentication
async def get_optional_user(
    authorization: Optional[str] = Header(None)
) -> Optional[User]:
    """
    Return user if authenticated, None otherwise
    """
    if not authorization:
        return None
    
    try:
        token = authorization.split()[1]
        user_id = verify_token (token)
        return await fetch_user (user_id)
    except:
        return None

@app.get("/products")
async def list_products(
    user: Optional[User] = Depends (get_optional_user)
):
    """
    Works for both authenticated and anonymous users
    Different behavior based on auth status
    """
    if user:
        # Show personalized products
        return await get_personalized_products (user.id)
    else:
        # Show default products
        return await get_default_products()
\`\`\`

### Factory Dependencies

\`\`\`python
"""
Dependency Factories for Configuration
"""

from typing import Callable

def get_query_limit (default: int, max: int) -> Callable:
    """
    Factory that creates limit dependency
    """
    def limit_dependency(
        limit: int = Query (default, ge=1, le=max)
    ) -> int:
        return limit
    
    return limit_dependency

# Create different limit dependencies
limit_10 = get_query_limit (default=10, max=100)
limit_50 = get_query_limit (default=50, max=500)

@app.get("/users")
async def list_users (limit: int = Depends (limit_10)):
    """Max 100 items"""
    return []

@app.get("/products")
async def list_products (limit: int = Depends (limit_50)):
    """Max 500 items"""
    return []
\`\`\`

---

## Performance Optimization

### Efficient Dependencies

\`\`\`python
"""
Optimizing Dependency Performance
"""

# ❌ Bad: Creating expensive object every request
def get_service_bad():
    # Expensive initialization
    service = ExpensiveService()
    return service

# ✅ Good: Reuse singleton
_service = None

def get_service_good():
    global _service
    if _service is None:
        _service = ExpensiveService()
    return _service

# ✅ Best: Use lru_cache (Python 3.9+)
from functools import lru_cache

@lru_cache()
def get_settings():
    """
    Settings loaded once, cached
    Perfect for configuration
    """
    return Settings()

@app.get("/")
async def root (settings = Depends (get_settings)):
    return {"app_name": settings.app_name}
\`\`\`

---

## Summary

### Key Takeaways

✅ **Depends() is magic**: Automatically resolves dependency trees  
✅ **Reusability**: Write once, use everywhere (db sessions, auth, pagination)  
✅ **Testability**: Override dependencies for testing with mocks  
✅ **Composability**: Dependencies can depend on dependencies (unlimited depth)  
✅ **Caching**: Dependencies cached per request (performance optimization)  
✅ **Type-safe**: Full IDE support, mypy validation  
✅ **Class-based**: Use callable classes for stateful dependencies

### Best Practices

**1. Separation of concerns**:
- Dependencies for infrastructure (DB, cache, auth)
- Business logic in services
- Endpoints as thin orchestration layer

**2. Reuse, don't repeat**:
- Common patterns (pagination, filtering) as dependencies
- Apply at router level for shared logic
- Use class-based dependencies for complex logic

**3. Testing**:
- Always use dependency_overrides for tests
- Create test fixtures for common mocks
- Test dependencies in isolation

**4. Performance**:
- Use @lru_cache for expensive initialization
- Leverage per-request caching
- Avoid recreating objects unnecessarily

### Next Steps

In the next section, we'll explore **Database Integration (SQLAlchemy + FastAPI)**: combining FastAPI's dependency injection with SQLAlchemy\'s ORM for production-grade database access. You'll learn repository patterns, transaction management, and async database operations.

**Production mindset**: Mastering dependency injection transforms how you write APIs. It's the difference between tangled spaghetti code and clean, testable architecture.
`,
};
