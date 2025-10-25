export const pathOperationsRouting = {
  title: 'Path Operations & Routing',
  id: 'path-operations-routing',
  content: `
# Path Operations & Routing

## Introduction

Routing is the backbone of any API—it maps URLs to handler functions. FastAPI's routing system is powerful, flexible, and designed for building well-structured, maintainable APIs. Unlike Flask\'s simple decorator approach or Django's complex URL configuration, FastAPI strikes the perfect balance between simplicity and features.

**Why routing matters in production:**
- **Organization**: Structure large APIs with routers and sub-applications
- **Versioning**: Handle API v1, v2, v3 cleanly
- **Maintainability**: Group related endpoints logically
- **Documentation**: Automatic OpenAPI tags and grouping
- **Performance**: Efficient path matching and parameter extraction

In this section, you'll master:
- HTTP methods and path operations
- Path parameters with validation
- Query parameters and request bodies
- APIRouter for organizing large applications
- Route prefixes, tags, and dependencies
- Advanced routing patterns for production

### The Routing Hierarchy

\`\`\`
FastAPI App
├── Root routes (@app.get("/"))
├── APIRouter: /api/v1
│   ├── /users (users router)
│   ├── /products (products router)
│   └── /orders (orders router)
└── Sub-applications (mount)
    └── /admin (admin app)
\`\`\`

---

## HTTP Methods & Path Operations

### Basic HTTP Methods

\`\`\`python
"""
RESTful HTTP Methods in FastAPI
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float

# In-memory storage (use database in production)
items_db = {}

@app.get("/items")
async def list_items() -> List[Item]:
    """
    GET: Retrieve collection
    
    - Idempotent (same result every time)
    - Cacheable
    - Safe (no side effects)
    """
    return list (items_db.values())

@app.get("/items/{item_id}")
async def get_item (item_id: int) -> Item:
    """
    GET: Retrieve single resource
    
    - Return 404 if not found
    - Return 200 with resource
    """
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    return items_db[item_id]

@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item (item: Item) -> Item:
    """
    POST: Create new resource
    
    - Return 201 Created
    - Return 409 Conflict if already exists
    - Include Location header (best practice)
    """
    if item.id in items_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Item {item.id} already exists"
        )
    items_db[item.id] = item
    return item

@app.put("/items/{item_id}")
async def update_item (item_id: int, item: Item) -> Item:
    """
    PUT: Replace entire resource
    
    - Idempotent
    - Create if not exists (optional)
    - Replace all fields
    """
    if item_id != item.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path ID doesn't match body ID"
        )
    items_db[item_id] = item
    return item

@app.patch("/items/{item_id}")
async def partial_update_item(
    item_id: int,
    name: str | None = None,
    price: float | None = None
) -> Item:
    """
    PATCH: Partial update
    
    - Update only provided fields
    - Keep other fields unchanged
    - More flexible than PUT
    """
    if item_id not in items_db:
        raise HTTPException (status_code=404, detail="Item not found")
    
    item = items_db[item_id]
    if name is not None:
        item.name = name
    if price is not None:
        item.price = price
    
    return item

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item (item_id: int):
    """
    DELETE: Remove resource
    
    - Idempotent (deleting twice = same result)
    - Return 204 No Content (no body)
    - Return 404 if not found
    """
    if item_id not in items_db:
        raise HTTPException (status_code=404, detail="Item not found")
    
    del items_db[item_id]
    return None  # 204 No Content

@app.head("/items/{item_id}")
async def check_item_exists (item_id: int):
    """
    HEAD: Check resource exists (same as GET but no body)
    
    - Returns headers only
    - Used for checking existence
    - Checking metadata (Last-Modified, ETag)
    """
    if item_id not in items_db:
        raise HTTPException (status_code=404)
    return None  # Headers only, no body

@app.options("/items")
async def options_items():
    """
    OPTIONS: Describe available methods
    
    - CORS preflight requests
    - API discovery
    - Usually handled automatically by CORS middleware
    """
    return {
        "methods": ["GET", "POST"],
        "description": "Items collection endpoint"
    }
\`\`\`

### HTTP Method Best Practices

\`\`\`python
"""
RESTful API Design Patterns
"""

from fastapi import FastAPI, status, Response
from typing import Optional

app = FastAPI()

# ✅ Good: Proper status codes
@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user (user: UserCreate):
    """POST returns 201 Created"""
    return {"id": 1, **user.dict()}

# ✅ Good: Idempotent PUT
@app.put("/users/{user_id}")
async def replace_user (user_id: int, user: UserCreate):
    """PUT replaces entire resource"""
    # Idempotent: same request = same result
    return {"id": user_id, **user.dict()}

# ✅ Good: PATCH for partial updates
@app.patch("/users/{user_id}")
async def update_user (user_id: int, user: UserUpdate):
    """PATCH updates only provided fields"""
    # More flexible than PUT
    return updated_user

# ✅ Good: DELETE returns 204
@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user (user_id: int):
    """DELETE returns 204 No Content"""
    delete_from_db (user_id)
    return None

# ❌ Bad: POST for updates (should use PUT/PATCH)
@app.post("/users/{user_id}/update")
async def bad_update (user_id: int, user: User):
    """Don't use POST for updates"""
    pass

# ❌ Bad: GET with side effects (should use POST)
@app.get("/users/{user_id}/delete")
async def bad_delete (user_id: int):
    """Don't use GET for actions with side effects"""
    pass
\`\`\`

---

## Path Parameters

### Basic Path Parameters

\`\`\`python
"""
Path Parameters with Type Validation
"""

from fastapi import FastAPI, Path
from typing import Optional
from enum import Enum

app = FastAPI()

# Simple path parameter
@app.get("/users/{user_id}")
async def get_user (user_id: int):
    """
    Path parameter with type conversion
    
    /users/123 → user_id = 123 (int)
    /users/abc → 422 Unprocessable Entity
    """
    return {"user_id": user_id}

# Multiple path parameters
@app.get("/users/{user_id}/posts/{post_id}")
async def get_user_post (user_id: int, post_id: int):
    """
    Multiple path parameters
    
    /users/123/posts/456
    """
    return {"user_id": user_id, "post_id": post_id}

# Path parameter with validation
@app.get("/users/{user_id}")
async def get_user_validated(
    user_id: int = Path(..., description="User ID", ge=1, le=1_000_000)
):
    """
    Path parameter with constraints
    
    - ge=1: Greater than or equal to 1
    - le=1_000_000: Less than or equal to 1,000,000
    """
    return {"user_id": user_id}

# String path parameters
@app.get("/users/{username}")
async def get_user_by_username(
    username: str = Path(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_]+$")
):
    """
    String path parameter with validation
    
    - 3-50 characters
    - Alphanumeric + underscore only
    """
    return {"username": username}

# Enum path parameters
class Category (str, Enum):
    ELECTRONICS = "electronics"
    BOOKS = "books"
    CLOTHING = "clothing"

@app.get("/products/{category}")
async def get_products_by_category (category: Category):
    """
    Enum path parameter
    
    Valid: /products/electronics, /products/books
    Invalid: /products/invalid → 422 error
    """
    return {"category": category.value}

# Path parameter order matters
@app.get("/users/me")
async def get_current_user():
    """
    Static route (no parameter)
    MUST come before /users/{user_id}
    """
    return {"user": "current user"}

@app.get("/users/{user_id}")
async def get_user (user_id: int):
    """
    Dynamic route (with parameter)
    Comes after static routes
    """
    return {"user_id": user_id}
\`\`\`

### Advanced Path Parameters

\`\`\`python
"""
Advanced Path Parameter Patterns
"""

from fastapi import FastAPI, Path, HTTPException
from datetime import date
from uuid import UUID

app = FastAPI()

# UUID path parameters
@app.get("/sessions/{session_id}")
async def get_session (session_id: UUID):
    """
    UUID path parameter
    
    Valid: /sessions/550e8400-e29b-41d4-a716-446655440000
    Invalid: /sessions/123 → 422 error
    """
    return {"session_id": session_id}

# Date path parameters
@app.get("/reports/{report_date}")
async def get_report (report_date: date):
    """
    Date path parameter (ISO 8601)
    
    Valid: /reports/2024-01-15
    Invalid: /reports/15-01-2024 → 422 error
    """
    return {"report_date": report_date}

# File paths as parameters
@app.get("/files/{file_path:path}")
async def get_file (file_path: str):
    """
    :path allows slashes in parameter
    
    /files/docs/api/readme.md
    file_path = "docs/api/readme.md"
    """
    return {"file_path": file_path}

# Regex validation
@app.get("/products/{sku}")
async def get_product_by_sku(
    sku: str = Path(..., regex=r"^[A-Z]{3}-\\d{6}$")
):
    """
    SKU format: ABC-123456
    
    Valid: /products/ABC-123456
    Invalid: /products/abc-123 → 422 error
    """
    return {"sku": sku}

# Multiple types with Union
from typing import Union

@app.get("/items/{item_identifier}")
async def get_item (item_identifier: Union[int, UUID]):
    """
    Accept either int or UUID
    
    /items/123 → int
    /items/550e8400-e29b-41d4-a716-446655440000 → UUID
    """
    if isinstance (item_identifier, int):
        return {"type": "id", "value": item_identifier}
    return {"type": "uuid", "value": str (item_identifier)}
\`\`\`

---

## Query Parameters

### Query Parameter Patterns

\`\`\`python
"""
Query Parameters with Validation
"""

from fastapi import FastAPI, Query
from typing import Optional, List
from datetime import date, datetime

app = FastAPI()

# Basic query parameters
@app.get("/search")
async def search(
    q: str,                          # Required
    page: int = 1,                   # Optional with default
    page_size: int = 20,             # Optional with default
    sort: Optional[str] = None       # Optional, no default
):
    """
    Query parameters
    
    /search?q=python&page=2&page_size=50
    /search?q=python (uses defaults)
    """
    return {
        "query": q,
        "page": page,
        "page_size": page_size,
        "sort": sort
    }

# Query parameter validation
@app.get("/products")
async def list_products(
    # String validation
    search: Optional[str] = Query(
        None,
        min_length=3,
        max_length=100,
        description="Search query"
    ),
    
    # Integer validation
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    
    # Enum query parameter
    sort: Optional[str] = Query(
        None,
        regex="^(price|name|created_at)$",
        description="Sort field"
    ),
    
    # Boolean flag
    in_stock: bool = Query(True, description="Show only in-stock items"),
    
    # Multiple values (list)
    tags: Optional[List[str]] = Query(
        None,
        description="Filter by tags",
        max_items=5
    ),
    
    # Date range
    created_after: Optional[date] = Query(None, description="Created after"),
    created_before: Optional[date] = Query(None, description="Created before"),
    
    # Deprecated query parameter
    old_param: Optional[str] = Query(
        None,
        deprecated=True,
        description="Deprecated: use new_param instead"
    )
):
    """
    Complex query parameter validation
    
    /products?search=laptop&page=2&tags=electronics&tags=sale&in_stock=true
    """
    return {
        "search": search,
        "pagination": {"page": page, "page_size": page_size},
        "filters": {
            "tags": tags,
            "in_stock": in_stock,
            "date_range": {
                "after": created_after,
                "before": created_before
            }
        }
    }

# Required query parameter with Query()
@app.get("/analytics")
async def get_analytics(
    user_id: int = Query(..., description="User ID"),  # Required with ...
    metric: str = Query(..., regex="^(views|clicks|conversions)$")
):
    """
    Required query parameters
    
    /analytics?user_id=123&metric=views
    /analytics → 422 error (missing required params)
    """
    return {"user_id": user_id, "metric": metric}

# List of primitive types
@app.get("/items")
async def filter_items(
    ids: Optional[List[int]] = Query(None, description="Item IDs")
):
    """
    List query parameter
    
    /items?ids=1&ids=2&ids=3
    ids = [1, 2, 3]
    """
    return {"ids": ids}
\`\`\`

### Query Parameter Best Practices

\`\`\`python
"""
Production Query Parameter Patterns
"""

from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

app = FastAPI()

# Pagination model
class Pagination(BaseModel):
    page: int = Field(1, ge=1, le=1000)
    page_size: int = Field(20, ge=1, le=100)
    
    @property
    def offset (self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit (self) -> int:
        return self.page_size

# Sorting model
class SortParam(BaseModel):
    field: str
    order: Literal["asc", "desc"] = "asc"

@app.get("/users")
async def list_users(
    # Pagination
    page: int = Query(1, ge=1, le=1000),
    page_size: int = Query(20, ge=1, le=100),
    
    # Filtering
    search: Optional[str] = Query(None, min_length=1, max_length=100),
    is_active: Optional[bool] = Query(None),
    role: Optional[List[str]] = Query(None, max_items=5),
    
    # Sorting
    sort_by: str = Query(
        "created_at",
        regex="^(username|email|created_at)$"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc"),
    
    # Date filters
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None
):
    """
    Production-ready query parameters
    
    - Pagination with reasonable limits
    - Multiple filters
    - Flexible sorting
    - Date range filtering
    """
    pagination = Pagination (page=page, page_size=page_size)
    
    # Validate date range
    if created_after and created_before and created_after > created_before:
        raise HTTPException(
            status_code=400,
            detail="created_after must be before created_before"
        )
    
    # Build query (pseudo-code)
    users = await query_users(
        offset=pagination.offset,
        limit=pagination.limit,
        search=search,
        is_active=is_active,
        roles=role,
        sort_by=sort_by,
        sort_order=sort_order,
        created_after=created_after,
        created_before=created_before
    )
    
    return {
        "data": users,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": await count_users()
        }
    }
\`\`\`

---

## APIRouter for Organization

### Basic APIRouter

\`\`\`python
"""
Organizing APIs with APIRouter
"""

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel

# Create router
users_router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "User not found"}}
)

@users_router.get("/")
async def list_users():
    """List all users"""
    return []

@users_router.get("/{user_id}")
async def get_user (user_id: int):
    """Get user by ID"""
    return {"user_id": user_id}

@users_router.post("/", status_code=201)
async def create_user (user: UserCreate):
    """Create new user"""
    return user

@users_router.put("/{user_id}")
async def update_user (user_id: int, user: UserUpdate):
    """Update user"""
    return user

@users_router.delete("/{user_id}", status_code=204)
async def delete_user (user_id: int):
    """Delete user"""
    return None

# Create app and include router
app = FastAPI()
app.include_router (users_router)

# Routes become:
# GET /users/
# GET /users/{user_id}
# POST /users/
# PUT /users/{user_id}
# DELETE /users/{user_id}
\`\`\`

### Multiple Routers

\`\`\`python
"""
Large Application Structure with Multiple Routers
"""

# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/")
async def list_users (db: Session = Depends (get_db)):
    return []

@router.get("/{user_id}")
async def get_user (user_id: int, db: Session = Depends (get_db)):
    return {"user_id": user_id}

# app/api/v1/endpoints/products.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_products():
    return []

@router.get("/{product_id}")
async def get_product (product_id: int):
    return {"product_id": product_id}

# app/api/v1/endpoints/orders.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_orders():
    return []

@router.post("/")
async def create_order (order: OrderCreate):
    return order

# app/api/v1/router.py
"""Combine all v1 routers"""
from fastapi import APIRouter
from app.api.v1.endpoints import users, products, orders

api_router = APIRouter()

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    products.router,
    prefix="/products",
    tags=["products"]
)

api_router.include_router(
    orders.router,
    prefix="/orders",
    tags=["orders"]
)

# app/main.py
"""Main application"""
from fastapi import FastAPI
from app.api.v1.router import api_router

app = FastAPI(title="My API")

app.include_router (api_router, prefix="/api/v1")

# Final routes:
# GET /api/v1/users/
# GET /api/v1/users/{user_id}
# GET /api/v1/products/
# GET /api/v1/products/{product_id}
# GET /api/v1/orders/
# POST /api/v1/orders/
\`\`\`

---

## API Versioning

### URL-Based Versioning

\`\`\`python
"""
API Versioning with Routers
"""

from fastapi import FastAPI, APIRouter

app = FastAPI()

# API v1
v1_router = APIRouter (prefix="/api/v1", tags=["v1"])

@v1_router.get("/users")
async def list_users_v1():
    """V1: Returns simple user list"""
    return [{"id": 1, "name": "Alice"}]

# API v2 (with pagination)
v2_router = APIRouter (prefix="/api/v2", tags=["v2"])

@v2_router.get("/users")
async def list_users_v2(page: int = 1, page_size: int = 20):
    """V2: Returns paginated user list"""
    return {
        "data": [{"id": 1, "first_name": "Alice", "last_name": "Johnson"}],
        "pagination": {"page": page, "page_size": page_size, "total": 100}
    }

# API v3 (with filtering)
v3_router = APIRouter (prefix="/api/v3", tags=["v3"])

@v3_router.get("/users")
async def list_users_v3(
    page: int = 1,
    page_size: int = 20,
    search: str = None,
    is_active: bool = None
):
    """V3: Returns filtered, paginated user list"""
    return {
        "data": [],
        "pagination": {},
        "filters": {"search": search, "is_active": is_active}
    }

app.include_router (v1_router)
app.include_router (v2_router)
app.include_router (v3_router)

# Routes:
# GET /api/v1/users  (simple)
# GET /api/v2/users  (paginated)
# GET /api/v3/users  (filtered + paginated)
\`\`\`

### Header-Based Versioning

\`\`\`python
"""
Header-Based API Versioning (like Stripe)
"""

from fastapi import FastAPI, Header, HTTPException
from typing import Optional

app = FastAPI()

@app.get("/users")
async def list_users (api_version: Optional[str] = Header(None)):
    """
    API version from header
    
    Request:
    GET /users
    API-Version: 2023-11-01
    """
    if api_version == "2023-01-01":
        # Return v1 format
        return [{"id": 1, "name": "Alice"}]
    
    elif api_version == "2023-06-01":
        # Return v2 format
        return [{"id": 1, "first_name": "Alice", "last_name": "Johnson"}]
    
    elif api_version == "2023-11-01" or api_version is None:
        # Return v3 format (default)
        return {"data": [{"id": 1, "first_name": "Alice", "last_name": "Johnson"}]}
    
    else:
        raise HTTPException (status_code=400, detail="Unsupported API version")
\`\`\`

---

## Advanced Routing Patterns

### Route Dependencies

\`\`\`python
"""
Router-Level Dependencies
"""

from fastapi import APIRouter, Depends, HTTPException

async def verify_api_key (api_key: str = Header(...)):
    """Verify API key for all routes"""
    if api_key != "secret":
        raise HTTPException (status_code=401, detail="Invalid API key")
    return api_key

async def rate_limit():
    """Check rate limit"""
    # Implementation here
    pass

# Apply dependencies to all routes in router
admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends (verify_api_key), Depends (rate_limit)]
)

@admin_router.get("/users")
async def admin_list_users():
    """
    Protected by API key and rate limit
    Both dependencies run before handler
    """
    return []

@admin_router.delete("/users/{user_id}")
async def admin_delete_user (user_id: int):
    """
    Also protected by same dependencies
    """
    return None
\`\`\`

### Sub-Applications

\`\`\`python
"""
Mounting Sub-Applications
"""

from fastapi import FastAPI

# Main application
app = FastAPI(title="Main API")

# Sub-application (completely separate)
admin_app = FastAPI(title="Admin API")

@admin_app.get("/users")
async def admin_users():
    return []

@admin_app.get("/settings")
async def admin_settings():
    return {}

# Mount sub-application
app.mount("/admin", admin_app)

# Routes:
# Main app: /
# Admin app: /admin/users, /admin/settings
# Each has separate OpenAPI docs:
# Main: /docs
# Admin: /admin/docs
\`\`\`

### Dynamic Route Generation

\`\`\`python
"""
Generate Routes Dynamically
"""

from fastapi import FastAPI, APIRouter

app = FastAPI()

# Generate routes for multiple resources
resources = ["users", "products", "orders", "customers"]

for resource in resources:
    router = APIRouter (prefix=f"/{resource}", tags=[resource])
    
    @router.get("/")
    async def list_items (resource=resource):
        return {f"{resource}": []}
    
    @router.get("/{item_id}")
    async def get_item (item_id: int, resource=resource):
        return {f"{resource}_id": item_id}
    
    app.include_router (router)

# Generates:
# GET /users/, /users/{item_id}
# GET /products/, /products/{item_id}
# GET /orders/, /orders/{item_id}
# GET /customers/, /customers/{item_id}
\`\`\`

---

## Summary

### Key Takeaways

✅ **HTTP methods**: GET (retrieve), POST (create), PUT (replace), PATCH (update), DELETE (remove)  
✅ **Path parameters**: Type-safe with validation, order matters (static before dynamic)  
✅ **Query parameters**: Flexible filtering, pagination, sorting with Query()  
✅ **APIRouter**: Organize large APIs, group related endpoints, apply shared dependencies  
✅ **Versioning**: URL-based (/api/v1) or header-based (API-Version header)  
✅ **Production patterns**: Router dependencies, sub-applications, dynamic generation

### Best Practices

**1. RESTful conventions**:
- Use proper HTTP methods
- Return appropriate status codes (201, 204, 404)
- Idempotent PUT/DELETE
- Safe GET/HEAD

**2. Route organization**:
- Group related endpoints in routers
- Use prefixes for versioning
- Apply shared dependencies at router level
- Keep routers in separate files

**3. Parameter validation**:
- Always validate path/query parameters
- Use Field constraints (ge, le, regex)
- Provide clear descriptions
- Set reasonable limits (pagination)

**4. Versioning strategy**:
- Choose URL or header-based early
- Maintain backward compatibility
- Deprecate gracefully (6-12 months)
- Document breaking changes

### Next Steps

In the next section, we'll explore **Dependency Injection System**: FastAPI's most powerful feature for writing clean, testable, reusable code. You'll learn how to inject database sessions, authentication, permissions, and shared logic into your endpoints.

**Production mindset**: Good routing is the foundation of maintainable APIs. Invest time in organization early—restructuring routes later is painful.
`,
};
