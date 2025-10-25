export const requestResponseModelsPydantic = {
  title: 'Request & Response Models (Pydantic)',
  id: 'request-response-models-pydantic',
  content: `
# Request & Response Models (Pydantic)

## Introduction

Pydantic is the **data validation powerhouse** behind FastAPI. It transforms Python type hints into runtime validation, serialization, and documentation. In production APIs, data validation is critical—invalid data causes crashes, security vulnerabilities, and data corruption. Pydantic ensures only valid data enters your system.

**Why Pydantic matters**:
- **Type-safe validation**: Catches errors before they reach your code
- **Performance**: 5-17x faster than alternatives (marshmallow, Django REST serializers)
- **Developer experience**: Natural Python syntax, IDE support, clear error messages
- **Production ready**: Used by FastAPI, Microsoft, Amazon, thousands of projects

In this section, you'll master:
- Pydantic models for requests and responses
- Advanced validation techniques
- Field constraints and custom validators
- Nested models and complex data structures
- Performance optimization
- Real-world production patterns

### The Data Flow

\`\`\`
Request → Pydantic Model → Validation → Your Code → Pydantic Model → Response

1. Client sends JSON
2. Pydantic parses and validates
3. If valid: model instance passed to handler
4. If invalid: 422 error with details
5. Handler returns data
6. Pydantic serializes to JSON
7. Response sent to client
\`\`\`

---

## Pydantic Basics

### Simple Models

\`\`\`python
"""
Basic Pydantic Model
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class User(BaseModel):
    """
    User model with automatic validation
    """
    id: int                          # Required integer
    username: str                    # Required string
    email: str                       # Required string
    is_active: bool = True          # Optional with default
    created_at: datetime            # Required datetime
    bio: Optional[str] = None       # Optional string

# Create instance
user = User(
    id=1,
    username="alice",
    email="alice@example.com",
    created_at=datetime.utcnow()
)

# Access fields
print(user.username)      # "alice"
print(user.is_active)     # True (default)
print(user.bio)           # None

# Validation happens automatically
try:
    invalid_user = User (id="not_an_int", username="bob")
except ValidationError as e:
    print(e.errors())
    # [{'loc': ('id',), 'msg': 'value is not a valid integer', 'type': 'type_error.integer'}]
\`\`\`

### Model Configuration

\`\`\`python
"""
Pydantic Model Config
"""

from pydantic import BaseModel

class User(BaseModel):
    id: int
    username: str
    email: str
    password_hash: str
    
    class Config:
        # Allow ORM models (SQLAlchemy)
        orm_mode = True
        
        # Validate on assignment
        validate_assignment = True
        
        # Use enum values instead of enum objects
        use_enum_values = True
        
        # Allow arbitrary types (non-standard)
        arbitrary_types_allowed = False
        
        # Example for docs
        schema_extra = {
            "example": {
                "id": 1,
                "username": "alice",
                "email": "alice@example.com",
                "password_hash": "hashed_password"
            }
        }

# orm_mode allows: User.from_orm (db_user_object)
# validate_assignment: user.email = "invalid" → raises ValidationError
\`\`\`

---

## Field Validation

### Built-in Field Constraints

\`\`\`python
"""
Field Constraints with Field()
"""

from pydantic import BaseModel, Field, EmailStr, HttpUrl
from typing import List
from datetime import datetime

class Product(BaseModel):
    """
    Product with comprehensive validation
    """
    # String constraints
    name: str = Field(
        ...,                                    # Required (... = Ellipsis)
        min_length=3,
        max_length=100,
        description="Product name"
    )
    
    # Integer constraints
    price: int = Field(
        ...,
        ge=0,                                   # >= 0 (greater than or equal)
        le=1_000_000,                          # <= 1,000,000
        description="Price in cents"
    )
    
    # Float constraints
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Product rating 0-5"
    )
    
    # Special string formats
    email: EmailStr                            # Validates email format
    website: HttpUrl                           # Validates URL format
    
    # List constraints
    tags: List[str] = Field(
        default=[],
        min_items=0,
        max_items=10,
        description="Product tags (max 10)"
    )
    
    # Regex validation
    sku: str = Field(
        ...,
        regex=r"^[A-Z]{3}-\d{6}$",           # Format: ABC-123456
        description="SKU format: ABC-123456"
    )
    
    # Datetime
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )

# Usage in FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/products", response_model=Product)
async def create_product (product: Product):
    """
    Automatic validation:
    - name: 3-100 chars
    - price: 0 to 1,000,000
    - rating: 0.0 to 5.0
    - email: valid email format
    - website: valid URL
    - tags: max 10 items
    - sku: matches regex pattern
    """
    return product
\`\`\`

### Custom Validators

\`\`\`python
"""
Custom Validation Logic
"""

from pydantic import BaseModel, validator, root_validator
from typing import List
import re

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    password_confirm: str
    tags: List[str] = []
    
    @validator('username')
    def username_alphanumeric (cls, v):
        """
        Validate username is alphanumeric
        """
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        if len (v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v.lower()  # Normalize to lowercase
    
    @validator('password')
    def password_strength (cls, v):
        """
        Validate password strength
        """
        if len (v) < 8:
            raise ValueError('Password must be at least 8 characters')
        
        if not any (c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        
        if not any (c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        
        if not any (c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        
        if not any (c in '!@#$%^&*()' for c in v):
            raise ValueError('Password must contain special character')
        
        return v
    
    @validator('tags', each_item=True)
    def validate_tag (cls, v):
        """
        Validate each tag (runs for each item in list)
        """
        if len (v) < 2:
            raise ValueError('Tag must be at least 2 characters')
        if not v.replace('-', '').isalnum():
            raise ValueError('Tag must be alphanumeric (hyphens allowed)')
        return v.lower()
    
    @root_validator
    def passwords_match (cls, values):
        """
        Validate entire model (all fields)
        """
        password = values.get('password')
        password_confirm = values.get('password_confirm')
        
        if password != password_confirm:
            raise ValueError('Passwords do not match')
        
        return values

# Usage
try:
    user = UserCreate(
        username="Alice123",
        email="alice@example.com",
        password="Weak",
        password_confirm="Weak",
        tags=["python", "fastapi"]
    )
except ValidationError as e:
    print(e.json())
    # Shows all validation errors with clear messages
\`\`\`

---

## Request Models

### Request Body Validation

\`\`\`python
"""
Request Body Models
"""

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

app = FastAPI()

class CreatePostRequest(BaseModel):
    """
    Request model for creating a blog post
    """
    title: str = Field(..., min_length=10, max_length=200)
    content: str = Field(..., min_length=50)
    tags: List[str] = Field (default=[], max_items=10)
    published: bool = False
    publish_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Getting Started with FastAPI",
                "content": "FastAPI is a modern, fast web framework...",
                "tags": ["python", "fastapi", "web"],
                "published": False,
                "publish_at": "2024-12-01T10:00:00Z"
            }
        }

@app.post("/posts")
async def create_post (post: CreatePostRequest):
    """
    Automatic validation:
    - title: 10-200 characters
    - content: minimum 50 characters
    - tags: maximum 10 tags
    - published: boolean (defaults to False)
    - publish_at: valid datetime or null
    
    Invalid requests return 422 Unprocessable Entity
    """
    # post is guaranteed to be valid here
    return {"id": 1, **post.dict()}

# Multiple body parameters
@app.post("/posts/advanced")
async def create_post_advanced(
    post: CreatePostRequest,
    metadata: dict = Body (default={}),
    importance: int = Body (default=1, ge=1, le=5)
):
    """
    Multiple body parameters:
    - post: CreatePostRequest model
    - metadata: arbitrary dict
    - importance: integer 1-5
    
    Request body:
    {
      "post": {...},
      "metadata": {...},
      "importance": 3
    }
    """
    return {
        "post": post.dict(),
        "metadata": metadata,
        "importance": importance
    }
\`\`\`

### Query Parameters

\`\`\`python
"""
Query Parameter Validation
"""

from fastapi import FastAPI, Query
from typing import Optional, List
from datetime import date

app = FastAPI()

@app.get("/posts")
async def list_posts(
    # Required query parameter
    user_id: int = Query(..., description="User ID", ge=1),
    
    # Optional with default
    page: int = Query(1, ge=1, le=1000, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    
    # Optional without default
    search: Optional[str] = Query(
        None,
        min_length=3,
        max_length=100,
        description="Search query"
    ),
    
    # List of values
    tags: Optional[List[str]] = Query(
        None,
        description="Filter by tags",
        max_items=5
    ),
    
    # Date range
    created_after: Optional[date] = Query(None, description="Created after date"),
    created_before: Optional[date] = Query(None, description="Created before date"),
    
    # Boolean flag
    published_only: bool = Query(True, description="Show only published posts")
):
    """
    Query parameters with validation
    
    GET /posts?user_id=1&page=2&page_size=50&search=python&tags=fastapi&tags=web
    
    Validation:
    - user_id: required, must be >= 1
    - page: 1-1000
    - page_size: 1-100
    - search: 3-100 characters if provided
    - tags: maximum 5 tags
    - dates: valid ISO date format
    """
    return {
        "user_id": user_id,
        "page": page,
        "page_size": page_size,
        "search": search,
        "tags": tags,
        "created_after": created_after,
        "created_before": created_before,
        "published_only": published_only
    }
\`\`\`

### Path Parameters

\`\`\`python
"""
Path Parameter Validation
"""

from fastapi import FastAPI, Path
from enum import Enum

app = FastAPI()

class Category (str, Enum):
    """Enum for valid categories"""
    TECH = "tech"
    SCIENCE = "science"
    BUSINESS = "business"

@app.get("/posts/{category}/{post_id}")
async def get_post(
    category: Category,  # Enum validation
    post_id: int = Path(..., description="Post ID", ge=1),
    version: int = Path(1, description="API version", ge=1, le=3)
):
    """
    Path parameters with validation
    
    GET /posts/tech/123?version=2
    
    Validation:
    - category: must be "tech", "science", or "business"
    - post_id: must be >= 1
    - version: must be 1, 2, or 3
    """
    return {
        "category": category,
        "post_id": post_id,
        "version": version
    }
\`\`\`

---

## Response Models

### Response Model Filtering

\`\`\`python
"""
Response Models - Filtering Fields
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserDB(BaseModel):
    """
    Complete user model (internal)
    """
    id: int
    username: str
    email: EmailStr
    password_hash: str              # Sensitive
    api_key: str                    # Sensitive
    is_active: bool
    is_superuser: bool              # Sensitive
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        orm_mode = True

class UserResponse(BaseModel):
    """
    Public user model (API response)
    Excludes sensitive fields
    """
    id: int
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        orm_mode = True

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user (user_id: int):
    """
    Returns user WITHOUT sensitive fields
    
    UserDB has: password_hash, api_key, is_superuser
    UserResponse excludes them automatically
    """
    # Fetch from database (has all fields)
    db_user = await get_user_from_db (user_id)
    
    # Return db_user, FastAPI filters to UserResponse fields only
    return db_user

# Alternative: Use response_model_exclude
@app.get("/users/{user_id}/full", response_model=UserDB, response_model_exclude={"password_hash", "api_key", "is_superuser"})
async def get_user_full (user_id: int):
    """
    Same result using response_model_exclude
    """
    db_user = await get_user_from_db (user_id)
    return db_user

# Alternative: Use response_model_include
@app.get("/users/{user_id}/minimal", response_model=UserDB, response_model_include={"id", "username"})
async def get_user_minimal (user_id: int):
    """
    Return only specific fields
    """
    db_user = await get_user_from_db (user_id)
    return db_user  # Only id and username in response
\`\`\`

### Multiple Response Models

\`\`\`python
"""
Different Response Models for Different Scenarios
"""

from typing import Union
from fastapi import status

class UserPublic(BaseModel):
    """Public user info (non-authenticated)"""
    id: int
    username: str
    created_at: datetime

class UserPrivate(BaseModel):
    """Private user info (authenticated)"""
    id: int
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class UserAdmin(BaseModel):
    """Admin user info (admins only)"""
    id: int
    username: str
    email: EmailStr
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime]
    login_count: int

@app.get("/users/{user_id}")
async def get_user_dynamic(
    user_id: int,
    current_user: Optional[User] = Depends (get_current_user)
):
    """
    Return different response based on auth level
    
    Anonymous: UserPublic
    Authenticated: UserPrivate
    Admin: UserAdmin
    """
    db_user = await get_user_from_db (user_id)
    
    if not current_user:
        # Anonymous: minimal info
        return UserPublic.from_orm (db_user)
    
    if current_user.is_superuser:
        # Admin: full info
        return UserAdmin.from_orm (db_user)
    
    # Authenticated: private info
    return UserPrivate.from_orm (db_user)

# Using Union for response_model
@app.get("/users/{user_id}/typed", response_model=Union[UserPublic, UserPrivate, UserAdmin])
async def get_user_typed (user_id: int):
    """
    Response can be any of the three types
    OpenAPI docs show all possibilities
    """
    pass
\`\`\`

---

## Nested Models

### Complex Data Structures

\`\`\`python
"""
Nested Pydantic Models
"""

from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    """Nested address model"""
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str = Field(..., regex=r"^\d{5}(-\d{4})?$")
    country: str = "USA"

class SocialMedia(BaseModel):
    """Nested social media model"""
    platform: str
    url: HttpUrl
    followers: int = Field (ge=0)

class Author(BaseModel):
    """Nested author model"""
    name: str
    email: EmailStr
    bio: Optional[str] = None
    social_media: List[SocialMedia] = []

class Comment(BaseModel):
    """Nested comment model"""
    id: int
    author: Author  # Nested Author
    content: str
    created_at: datetime

class BlogPost(BaseModel):
    """
    Complex nested model
    """
    id: int
    title: str = Field(..., min_length=10, max_length=200)
    content: str
    author: Author              # Nested object
    comments: List[Comment]     # List of nested objects
    tags: List[str] = []
    metadata: dict = {}
    published_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "title": "Advanced FastAPI Patterns",
                "content": "In this post...",
                "author": {
                    "name": "Alice Johnson",
                    "email": "alice@example.com",
                    "bio": "Python developer",
                    "social_media": [
                        {
                            "platform": "Twitter",
                            "url": "https://twitter.com/alice",
                            "followers": 1000
                        }
                    ]
                },
                "comments": [
                    {
                        "id": 1,
                        "author": {
                            "name": "Bob Smith",
                            "email": "bob@example.com"
                        },
                        "content": "Great post!",
                        "created_at": "2024-01-01T10:00:00Z"
                    }
                ],
                "tags": ["python", "fastapi"],
                "metadata": {"views": 100},
                "published_at": "2024-01-01T09:00:00Z"
            }
        }

@app.post("/posts", response_model=BlogPost)
async def create_post (post: BlogPost):
    """
    Automatically validates entire nested structure:
    - BlogPost fields
    - Author (nested) with email validation
    - SocialMedia (nested in Author) with URL validation
    - Comments (list of nested objects)
    - Each Comment has nested Author
    
    Deep validation at all levels!
    """
    return post
\`\`\`

---

## Advanced Patterns

### Generic Models

\`\`\`python
"""
Generic Response Models
"""

from typing import TypeVar, Generic, List, Optional
from pydantic import BaseModel
from pydantic.generics import GenericModel

T = TypeVar('T')

class Pagination(BaseModel):
    """Pagination metadata"""
    page: int
    page_size: int
    total: int
    total_pages: int

class PaginatedResponse(GenericModel, Generic[T]):
    """
    Generic paginated response
    Works with any model type
    """
    data: List[T]
    pagination: Pagination

class User(BaseModel):
    id: int
    username: str
    email: str

class Product(BaseModel):
    id: int
    name: str
    price: float

@app.get("/users", response_model=PaginatedResponse[User])
async def list_users (page: int = 1, page_size: int = 20):
    """
    Returns paginated list of users
    Type: PaginatedResponse[User]
    """
    users = await fetch_users (page, page_size)
    total = await count_users()
    
    return PaginatedResponse(
        data=users,
        pagination=Pagination(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size
        )
    )

@app.get("/products", response_model=PaginatedResponse[Product])
async def list_products (page: int = 1, page_size: int = 20):
    """
    Same generic response for products
    Type: PaginatedResponse[Product]
    """
    products = await fetch_products (page, page_size)
    total = await count_products()
    
    return PaginatedResponse(
        data=products,
        pagination=Pagination(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size
        )
    )
\`\`\`

### Model Inheritance

\`\`\`python
"""
Model Inheritance for DRY
"""

class BaseUser(BaseModel):
    """Base user fields"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(BaseUser):
    """User creation (includes password)"""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """User update (all fields optional)"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)

class UserInDB(BaseUser):
    """User in database (includes sensitive fields)"""
    id: int
    password_hash: str
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

class UserResponse(BaseUser):
    """User API response (excludes sensitive fields)"""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True

# No duplication of username/email validation!
\`\`\`

---

## Performance Optimization

### Pydantic v2 Performance

\`\`\`python
"""
Pydantic v2 Performance Features
"""

from pydantic import BaseModel, Field
from typing import List
import time

# Pydantic v2 uses Rust-based validation (pydantic-core)
# 5-17x faster than Pydantic v1

class Product(BaseModel):
    name: str
    price: float
    quantity: int

# Benchmark: Validate 10,000 products
products_data = [
    {"name": f"Product {i}", "price": 9.99, "quantity": 100}
    for i in range(10000)
]

start = time.time()
products = [Product(**data) for data in products_data]
elapsed = time.time() - start

print(f"Validated 10,000 products in {elapsed:.3f}s")
# Pydantic v2: ~0.05s (Rust-based)
# Pydantic v1: ~0.3s (Python-based)
# 6x improvement!

# Model reuse for performance
# Don't recreate models in hot loops
# Reuse compiled validators
\`\`\`

### Lazy Validation

\`\`\`python
"""
Optimize Validation Performance
"""

from pydantic import BaseModel, validator

class OptimizedModel(BaseModel):
    # Use simple types when possible (faster)
    id: int                    # Fast
    name: str                  # Fast
    # email: EmailStr          # Slower (regex validation)
    email: str                 # Fast
    
    # Custom validation only when needed
    @validator('email')
    def validate_email_simple (cls, v):
        """Simple email validation (faster than EmailStr)"""
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

# For bulk operations, consider:
# 1. Validate in batches
# 2. Skip validation for trusted data
# 3. Use parse_obj instead of __init__ when possible
\`\`\`

---

## Production Patterns

### Error Handling

\`\`\`python
"""
Pydantic Validation Errors in Production
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler (request, exc: RequestValidationError):
    """
    Custom validation error response
    
    Default: 422 with detailed errors
    Custom: User-friendly messages
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join (str (loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation failed",
            "errors": errors
        }
    )

# Example validation error response:
# {
#   "detail": "Validation failed",
#   "errors": [
#     {
#       "field": "body -> email",
#       "message": "value is not a valid email address",
#       "type": "value_error.email"
#     },
#     {
#       "field": "body -> age",
#       "message": "ensure this value is greater than or equal to 0",
#       "type": "value_error.number.not_ge"
#     }
#   ]
# }
\`\`\`

### Testing with Pydantic

\`\`\`python
"""
Testing Pydantic Models
"""

import pytest
from pydantic import ValidationError

def test_user_validation():
    """Test user model validation"""
    
    # Valid user
    user = User(
        username="alice",
        email="alice@example.com",
        age=25
    )
    assert user.username == "alice"
    
    # Invalid email
    with pytest.raises(ValidationError) as exc_info:
        User (username="bob", email="invalid", age=30)
    
    errors = exc_info.value.errors()
    assert any (e["loc"] == ("email",) for e in errors)
    
    # Invalid age
    with pytest.raises(ValidationError) as exc_info:
        User (username="charlie", email="charlie@example.com", age=-5)
    
    errors = exc_info.value.errors()
    assert any (e["loc"] == ("age",) for e in errors)
\`\`\`

---

## Summary

### Key Takeaways

✅ **Pydantic = Type-safe validation**: Transform type hints into runtime validation  
✅ **Field constraints**: min_length, max_length, ge, le, regex, and more  
✅ **Custom validators**: @validator for complex logic, @root_validator for multi-field  
✅ **Request models**: Validate body, query, path parameters automatically  
✅ **Response models**: Filter sensitive fields, ensure consistent API responses  
✅ **Nested models**: Validate complex, deeply nested data structures  
✅ **Performance**: Pydantic v2 is 5-17x faster (Rust-based validation)  
✅ **Production ready**: Custom error handling, testing, DRY with inheritance

### Best Practices

**1. Use separate models**:
- \`UserCreate\` (with password)
- \`UserUpdate\` (optional fields)
- \`UserResponse\` (without sensitive fields)
- \`UserInDB\` (with all fields)

**2. Validate at the boundary**:
- Let Pydantic validate ALL incoming data
- Never trust client input
- Fail fast with clear error messages

**3. Response models are contracts**:
- Always use \`response_model\`
- Never return raw database objects
- Filter sensitive fields automatically

**4. Optimize for performance**:
- Use simple types when possible
- Custom validators only when needed
- Reuse models, don't recreate

### Next Steps

In the next section, we'll explore **Path Operations & Routing**: organizing your API endpoints, route parameters, HTTP methods, status codes, and building a well-structured API architecture.

**Production mindset**: Pydantic is your first line of defense against invalid data. Master validation, and you'll prevent entire classes of bugs before they reach your business logic.
`,
};
