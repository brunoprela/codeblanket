export const errorHandlingValidationQuiz = {
  title: 'Error Handling & Validation - Discussion Questions',
  id: 'error-handling-validation-quiz',
  questions: [
    {
      id: 1,
      question:
        'Design a comprehensive error handling strategy for a production FastAPI application that includes: (1) custom exception handlers for different error types, (2) structured error responses with request IDs for tracking, (3) appropriate HTTP status codes, (4) logging without exposing sensitive data, and (5) integration with error tracking services like Sentry. Implement the complete solution showing how validation errors, business logic errors, database errors, and unexpected exceptions are handled differently.',
      answer: `**Comprehensive Error Handling Strategy**:

\`\`\`python
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from uuid import uuid4

# Initialize Sentry
sentry_sdk.init(
    dsn=SENTRY_DSN,
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment="production"
)

app = FastAPI()

# Custom exceptions
class BusinessLogicError(Exception):
    def __init__(self, message: str, code: str, status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code

class InsufficientFundsError(BusinessLogicError):
    def __init__(self, required: float, available: float):
        super().__init__(
            message=f"Insufficient funds. Required: {required}, Available: {available}",
            code="insufficient_funds",
            status_code=400
        )

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"][1:])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    logger.warning(
        "Validation error",
        extra={
            "request_id": request.state.request_id,
            "path": request.url.path,
            "errors": errors
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Invalid input data",
            "details": errors,
            "request_id": request.state.request_id
        }
    )

# Business logic error handler
@app.exception_handler(BusinessLogicError)
async def business_error_handler(request: Request, exc: BusinessLogicError):
    logger.info(
        f"Business logic error: {exc.code}",
        extra={
            "request_id": request.state.request_id,
            "error_code": exc.code,
            "message": exc.message
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "message": exc.message,
            "request_id": request.state.request_id
        }
    )

# Database error handler
@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    # Log full error server-side
    logger.error(
        "Database error",
        exc_info=True,
        extra={
            "request_id": request.state.request_id,
            "path": request.url.path
        }
    )
    
    # Send to Sentry
    sentry_sdk.capture_exception(exc)
    
    # Return generic error to client (don't leak DB details)
    return JSONResponse(
        status_code=500,
        content={
            "error": "database_error",
            "message": "A database error occurred",
            "request_id": request.state.request_id
        }
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log with full context
    logger.error(
        "Unhandled exception",
        exc_info=True,
        extra={
            "request_id": request.state.request_id,
            "path": request.url.path,
            "method": request.method,
            "user_id": getattr(request.state, "user_id", None)
        }
    )
    
    # Send to Sentry with context
    with sentry_sdk.push_scope() as scope:
        scope.set_context("request", {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        })
        sentry_sdk.capture_exception(exc)
    
    # Return generic error
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id
        }
    )
\`\`\`

**Key principles**: Validation errors → 422, Business errors → 4xx with code, Database errors → log+Sentry, never expose internals.`,
    },
    {
      id: 2,
      question:
        'Explain the difference between HTTP status codes 400, 401, 403, 404, 422, and 429. For each status code, provide specific use cases in a FastAPI application. Design an endpoint that handles user registration and can return each of these status codes in appropriate scenarios.',
      answer: `**HTTP Status Code Guide**:

**400 Bad Request**: Generic client error, malformed request
- Use when: Request syntax invalid, missing required header, invalid JSON
- Example: Missing Content-Type header, malformed JSON body

**401 Unauthorized**: Authentication required but not provided
- Use when: No credentials provided, invalid token, expired token
- Example: No Authorization header, invalid JWT token

**403 Forbidden**: Authenticated but not authorized
- Use when: Valid credentials but insufficient permissions
- Example: Regular user trying to access admin endpoint

**404 Not Found**: Resource doesn't exist
- Use when: URL path doesn't exist, resource ID not found
- Example: GET /users/99999 when user doesn't exist

**422 Unprocessable Entity**: Request valid but data fails validation
- Use when: Pydantic validation fails, business rule validation fails
- Example: Email format invalid, password too weak

**429 Too Many Requests**: Rate limit exceeded
- Use when: Client exceeds allowed request rate
- Example: > 100 requests per minute

**Registration Endpoint Example**:

\`\`\`python
@app.post("/register", status_code=201)
async def register_user(
    user: UserRegister,
    request: Request,
    db: Session = Depends(get_db)
):
    # 429: Rate limit exceeded
    if await is_rate_limited(request.client.host):
        raise HTTPException(
            status_code=429,
            detail="Too many registration attempts. Try again in 1 hour."
        )
    
    # 422: Validation failed (automatic via Pydantic)
    # user.email must be valid email format
    # user.password must meet requirements
    
    # 400: Bad request - email already exists
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create user
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    
    return {"user_id": new_user.id, "message": "Registration successful"}

# 401: Unauthorized - no token provided
@app.get("/profile")
async def get_profile(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(401, "Authentication required")
    
    user = decode_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    
    return user

# 403: Forbidden - not admin
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_user)
):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    # 404: User not found
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted"}
\`\`\`

**Summary**: 400=bad input, 401=not authenticated, 403=not authorized, 404=not found, 422=validation failed, 429=rate limited.`,
    },
    {
      id: 3,
      question:
        'Design an error response format that balances providing enough information for debugging while not exposing sensitive internal details. The format should support: field-level validation errors, error codes for programmatic handling, request IDs for tracking, and optional debug information in development. Show how this format would be used for different error scenarios and how it integrates with frontend error handling.',
      answer: `**Structured Error Response Format**:

\`\`\`python
from pydantic import BaseModel
from typing import Optional, List, Any
from enum import Enum

class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"

class FieldError(BaseModel):
    field: str
    message: str
    code: str
    value: Optional[Any] = None  # Only in dev

class ErrorResponse(BaseModel):
    error: ErrorCode
    message: str  # Human-readable
    details: Optional[List[FieldError]] = None
    request_id: str
    timestamp: str
    debug: Optional[dict] = None  # Only in development

# Error response builder
def build_error_response(
    error_code: ErrorCode,
    message: str,
    request: Request,
    details: List[FieldError] = None,
    debug_info: dict = None
) -> ErrorResponse:
    response = ErrorResponse(
        error=error_code,
        message=message,
        details=details,
        request_id=request.state.request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Add debug info only in development
    if settings.DEBUG and debug_info:
        response.debug = debug_info
    
    return response

# Usage examples
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    field_errors = [
        FieldError(
            field=".".join(str(loc) for loc in err["loc"][1:]),
            message=err["msg"],
            code=err["type"],
            value=err.get("input") if settings.DEBUG else None
        )
        for err in exc.errors()
    ]
    
    error_response = build_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Invalid input data",
        request=request,
        details=field_errors,
        debug_info={"raw_errors": exc.errors()} if settings.DEBUG else None
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict(exclude_none=True)
    )

# Frontend integration
"""
// TypeScript frontend error handler
interface ErrorResponse {
    error: string;
    message: string;
    details?: FieldError[];
    request_id: string;
    timestamp: string;
}

interface FieldError {
    field: string;
    message: string;
    code: string;
}

async function handleApiError(error: Response) {
    const errorData: ErrorResponse = await error.json();
    
    // Show user-friendly message
    toast.error(errorData.message);
    
    // Handle field-level errors
    if (errorData.details) {
        errorData.details.forEach(fieldError => {
            // Highlight field in form
            setFieldError(fieldError.field, fieldError.message);
        });
    }
    
    // Log for debugging
    console.error(\`Error \${errorData.request_id}:\`, errorData);
    
    // Programmatic handling based on error code
    switch (errorData.error) {
        case 'authentication_error':
            redirectToLogin();
            break;
        case 'rate_limited':
            showRateLimitMessage();
            break;
    }
}
"""
\`\`\`

**Production vs Development**:
- Production: Generic messages, no stack traces, no sensitive data
- Development: Full details, stack traces, input values
- Always include: request_id for support ticket correlation`,
    },
  ],
};
