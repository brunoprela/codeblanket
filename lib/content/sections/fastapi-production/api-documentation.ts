export const apiDocumentation = {
  title: 'API Documentation (OpenAPI/Swagger)',
  id: 'api-documentation',
  content: `
# API Documentation (OpenAPI/Swagger)

## Introduction

Good API documentation is critical for adoption. FastAPI automatically generates interactive OpenAPI (Swagger) documentation, but production APIs need customization: detailed descriptions, examples, error documentation, and client SDK generation.

**Why API documentation matters:**
- **Developer experience**: Clear docs accelerate integration
- **Self-service**: Reduce support burden
- **Discoverability**: Show what your API can do
- **Client SDKs**: Auto-generate client libraries
- **Contract testing**: Documentation as source of truth
- **Onboarding**: New developers productive faster

**FastAPI documentation advantages:**
- **Automatic**: Generated from code (stays in sync)
- **Interactive**: Try endpoints directly in browser
- **Standards-based**: OpenAPI 3.0 specification
- **Customizable**: Add descriptions, examples, schemas

In this section, you'll master:
- OpenAPI specification fundamentals
- Customizing endpoint documentation
- Adding request/response examples
- Organizing with tags
- Security scheme documentation
- Custom schemas and models
- Generating client SDKs
- Production documentation patterns

---

## OpenAPI Basics

### Automatic Documentation

\`\`\`python
"""
FastAPI automatically generates OpenAPI docs
"""

from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="Production API with comprehensive documentation",
    version="1.0.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)

# Access documentation:
# http://localhost:8000/docs - Swagger UI
# http://localhost:8000/redoc - ReDoc
# http://localhost:8000/openapi.json - OpenAPI schema
\`\`\`

### Endpoint Documentation

\`\`\`python
"""
Document endpoints with descriptions and examples
"""

from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    """User model with field documentation"""
    id: int = Field(..., description="Unique user identifier", example=123)
    email: str = Field(..., description="User email address", example="user@example.com")
    username: str = Field(..., min_length=3, max_length=50, example="johndoe")
    is_active: bool = Field (default=True, description="Whether user account is active")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 123,
                "email": "user@example.com",
                "username": "johndoe",
                "is_active": True
            }
        }

@app.post(
    "/users/",
    response_model=User,
    status_code=201,
    summary="Create a new user",
    description="Create a new user account with email and username. Email must be unique.",
    response_description="The created user",
    tags=["users"],
    responses={
        201: {
            "description": "User created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 123,
                        "email": "user@example.com",
                        "username": "johndoe",
                        "is_active": True
                    }
                }
            }
        },
        400: {
            "description": "Bad request - validation error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "validation_error",
                        "message": "Email already exists"
                    }
                }
            }
        },
        422: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "validation_error",
                        "details": [
                            {
                                "field": "email",
                                "message": "invalid email format"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def create_user (user: User):
    """
    Create a new user account.
    
    - **email**: Must be unique and valid email format
    - **username**: 3-50 characters, alphanumeric
    - **is_active**: Optional, defaults to True
    
    Returns the created user with assigned ID.
    """
    return user
\`\`\`

---

## Advanced Documentation

### Tags and Organization

\`\`\`python
"""
Organize endpoints with tags and metadata
"""

from fastapi import APIRouter

# Define tags with descriptions
tags_metadata = [
    {
        "name": "users",
        "description": "User management endpoints. Create, read, update, delete users.",
    },
    {
        "name": "authentication",
        "description": "Authentication and authorization endpoints. Login, logout, token refresh.",
        "externalDocs": {
            "description": "Authentication guide",
            "url": "https://docs.example.com/auth"
        }
    },
    {
        "name": "posts",
        "description": "Blog post CRUD operations",
    }
]

app = FastAPI(
    title="Blog API",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# Group endpoints by router
users_router = APIRouter (prefix="/users", tags=["users"])
auth_router = APIRouter (prefix="/auth", tags=["authentication"])
posts_router = APIRouter (prefix="/posts", tags=["posts"])

@users_router.get("/")
async def list_users():
    """List all users (paginated)"""
    pass

@auth_router.post("/login")
async def login():
    """Authenticate user and return JWT token"""
    pass

app.include_router (users_router)
app.include_router (auth_router)
app.include_router (posts_router)
\`\`\`

### Security Schemes

\`\`\`python
"""
Document authentication/authorization
"""

from fastapi import FastAPI, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

app = FastAPI(
    title="Secure API",
    version="1.0.0"
)

# Automatic security documentation
@app.get(
    "/protected",
    dependencies=[Security (security)],
    summary="Protected endpoint",
    description="Requires valid JWT token in Authorization header"
)
async def protected_endpoint(
    credentials: HTTPAuthorizationCredentials = Security (security)
):
    """
    Access protected resource.
    
    Requires:
    - Authorization: Bearer <JWT_TOKEN>
    
    The token must be obtained from /auth/login endpoint.
    """
    return {"message": "Access granted"}

# OAuth2 documentation
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access"
    }
)

@app.get("/admin")
async def admin_only (token: str = Depends (oauth2_scheme)):
    """
    Admin-only endpoint.
    
    Required scope: admin
    """
    pass
\`\`\`

### Custom Examples

\`\`\`python
"""
Multiple examples for better documentation
"""

from fastapi import Body
from typing import Annotated

@app.post("/orders/")
async def create_order(
    order: Annotated[Order, Body(
        examples={
            "normal": {
                "summary": "Normal order",
                "description": "Standard order with single item",
                "value": {
                    "items": [{"product_id": 1, "quantity": 2}],
                    "shipping_address": "123 Main St",
                    "payment_method": "credit_card"
                }
            },
            "bulk": {
                "summary": "Bulk order",
                "description": "Large order with multiple items",
                "value": {
                    "items": [
                        {"product_id": 1, "quantity": 100},
                        {"product_id": 2, "quantity": 50}
                    ],
                    "shipping_address": "456 Business Ave",
                    "payment_method": "invoice"
                }
            },
            "international": {
                "summary": "International order",
                "description": "Order shipping internationally",
                "value": {
                    "items": [{"product_id": 3, "quantity": 5}],
                    "shipping_address": "Tokyo, Japan",
                    "payment_method": "paypal"
                }
            }
        }
    )]
):
    """Create a new order with various scenarios"""
    pass
\`\`\`

---

## Custom OpenAPI Schema

### Modifying OpenAPI Schema

\`\`\`python
"""
Customize the generated OpenAPI schema
"""

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Custom API",
        version="2.0.0",
        description="API with custom OpenAPI schema",
        routes=app.routes,
    )
    
    # Add custom server
    openapi_schema["servers"] = [
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Development"}
    ]
    
    # Add API key security
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
\`\`\`

---

## Generating Client SDKs

### OpenAPI Generator

\`\`\`bash
# Generate TypeScript client
openapi-generator-cli generate \\
  -i http://localhost:8000/openapi.json \\
  -g typescript-axios \\
  -o ./clients/typescript

# Generate Python client
openapi-generator-cli generate \\
  -i http://localhost:8000/openapi.json \\
  -g python \\
  -o ./clients/python

# Generate Go client
openapi-generator-cli generate \\
  -i http://localhost:8000/openapi.json \\
  -g go \\
  -o ./clients/go
\`\`\`

---

## Production Patterns

### Deprecation Notices

\`\`\`python
"""
Document deprecated endpoints
"""

@app.get(
    "/api/v1/users",
    deprecated=True,
    summary="List users (DEPRECATED)",
    description="This endpoint is deprecated. Use /api/v2/users instead."
)
async def list_users_v1():
    """
    **DEPRECATED**: Use /api/v2/users instead.
    
    This endpoint will be removed in version 3.0.
    Migration guide: https://docs.example.com/migration
    """
    pass
\`\`\`

### Documentation Hosting

\`\`\`python
"""
Serve documentation in production
"""

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Disable docs in production (optional)
app = FastAPI(
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Or serve from separate domain
# docs.example.com → static site with OpenAPI spec
\`\`\`

---

## Summary

✅ **Auto-generated docs**: FastAPI creates interactive documentation  
✅ **Customization**: Add descriptions, examples, tags  
✅ **Security docs**: Document authentication schemes  
✅ **Client SDKs**: Generate clients from OpenAPI spec  
✅ **Production patterns**: Deprecation, versioning, hosting  

### Best Practices

**1. Comprehensive descriptions**: Every endpoint, parameter, model
**2. Multiple examples**: Show common use cases
**3. Error documentation**: Document all error responses
**4. Security**: Document auth requirements clearly
**5. Versioning**: Handle API versions in docs
**6. Keep in sync**: Documentation from code (not separate)

### Next Steps

In the next section, we'll explore **Testing FastAPI Applications**: comprehensive testing strategies with pytest, TestClient, and mocking.
`,
};
