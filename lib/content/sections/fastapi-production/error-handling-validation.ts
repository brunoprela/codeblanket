export const errorHandlingValidation = {
  title: 'Error Handling & Validation',
  id: 'error-handling-validation',
  content: `
# Error Handling & Validation

## Introduction

APIs fail. Databases timeout, users send invalid data, external services crash. Production APIs must handle errors gracefully: informative messages, proper HTTP status codes, structured responses, and comprehensive logging.

**Why error handling matters:**
- **User experience**: Clear error messages help users fix issues
- **Debugging**: Detailed logs enable rapid issue resolution
- **Security**: Don't leak internal details in error responses
- **Reliability**: Graceful degradation instead of crashes
- **Monitoring**: Track error rates and patterns

In this section, you'll master:
- Custom exception handlers
- Validation error formatting
- HTTP status codes
- Structured error responses
- Production error tracking

---

## Exception Handling

### Custom Exception Handlers

\`\`\`python
"""
Custom exception handlers for production
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

app = FastAPI()

# Custom exceptions
class BusinessLogicError(Exception):
    """Raised for business rule violations"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

class ResourceNotFoundError(Exception):
    """Raised when resource doesn't exist"""
    pass

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler (request: Request, exc: Exception):
    """
    Catch all unhandled exceptions
    Prevents 500 errors with no details
    """
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id  # For tracking
        }
    )

# Business logic error handler
@app.exception_handler(BusinessLogicError)
async def business_logic_handler (request: Request, exc: BusinessLogicError):
    """Handle business rule violations"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "business_logic_error",
            "message": exc.message
        }
    )

# Not found handler
@app.exception_handler(ResourceNotFoundError)
async def not_found_handler (request: Request, exc: ResourceNotFoundError):
    """Handle 404s consistently"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": str (exc) or "Resource not found"
        }
    )

# Database error handler
@app.exception_handler(SQLAlchemyError)
async def database_exception_handler (request: Request, exc: SQLAlchemyError):
    """Handle database errors"""
    logger.error (f"Database error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "database_error",
            "message": "Database operation failed"
        }
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler (request: Request, exc: RequestValidationError):
    """
    Format Pydantic validation errors
    """
    errors = []
    
    for error in exc.errors():
        errors.append({
            "field": ".".join (str (loc) for loc in error["loc"][1:]),  # Skip 'body'
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid input data",
            "details": errors
        }
    )
\`\`\`

---

## HTTP Status Codes

### Status Code Guide

\`\`\`python
"""
Proper HTTP status codes for all scenarios
"""

# Success (2xx)
@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user (user: UserCreate):
    """201: Resource created"""
    pass

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user (user_id: int):
    """204: Success with no response body"""
    pass

# Client errors (4xx)
@app.get("/users/{user_id}")
async def get_user (user_id: int):
    """
    400: Bad Request - Invalid input
    401: Unauthorized - Not authenticated
    403: Forbidden - Authenticated but not allowed
    404: Not Found - Resource doesn't exist
    409: Conflict - Resource state conflict
    422: Unprocessable Entity - Validation failed
    429: Too Many Requests - Rate limited
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException (status_code=404, detail="User not found")
    return user

# Server errors (5xx)
# 500: Internal Server Error - Unhandled exception
# 503: Service Unavailable - Temporary outage
\`\`\`

---

## Structured Error Responses

### Error Response Format

\`\`\`python
"""
Consistent error response structure
"""

from pydantic import BaseModel
from typing import Optional, List, Dict

class ErrorDetail(BaseModel):
    """Single error detail"""
    field: Optional[str]  # Field that caused error
    message: str  # Human-readable message
    code: str  # Machine-readable code

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str  # Error type: "validation_error", "auth_error"
    message: str  # Human-readable summary
    details: Optional[List[ErrorDetail]] = None  # Detailed errors
    request_id: Optional[str] = None  # For tracking
    timestamp: str  # ISO timestamp

# Usage
@app.post("/orders")
async def create_order (order: OrderCreate, request: Request):
    """Return structured errors"""
    if order.quantity <= 0:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="validation_error",
                message="Invalid order quantity",
                details=[
                    ErrorDetail(
                        field="quantity",
                        message="Quantity must be positive",
                        code="min_value_error"
                    ).dict()
                ],
                request_id=request.state.request_id,
                timestamp=datetime.utcnow().isoformat()
            ).dict()
        )
\`\`\`

---

## Summary

✅ **Exception handlers**: Custom handlers for all error types  
✅ **HTTP status codes**: Proper codes for all scenarios  
✅ **Structured responses**: Consistent error format  
✅ **Validation**: Formatted Pydantic errors  
✅ **Logging**: Comprehensive error tracking  

### Next Steps

In the next section, we'll explore **Middleware & CORS**: implementing middleware for cross-cutting concerns and configuring CORS for frontend integration.
`,
};
