/**
 * REST API Error Handling Section
 */

export const restapierrorhandlingSection = {
  id: 'rest-api-error-handling',
  title: 'REST API Error Handling',
  content: `Comprehensive error handling is critical for API usability and debugging. Well-designed error responses help developers quickly identify and fix issues.

## HTTP Status Code Strategy

### **Success Codes (2xx)**
- **200 OK**: Standard success response
- **201 Created**: Resource successfully created
- **202 Accepted**: Request accepted for processing (async)
- **204 No Content**: Success with no response body
- **206 Partial Content**: Range request (partial resource)

### **Client Error Codes (4xx)**
- **400 Bad Request**: Invalid syntax, validation error
- **401 Unauthorized**: Not authenticated
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource doesn't exist
- **405 Method Not Allowed**: Wrong HTTP method
- **409 Conflict**: Resource state conflict
- **422 Unprocessable Entity**: Validation error (alternative to 400)
- **429 Too Many Requests**: Rate limit exceeded

### **Server Error Codes (5xx)**
- **500 Internal Server Error**: Unexpected server error
- **502 Bad Gateway**: Invalid upstream response
- **503 Service Unavailable**: Temporary unavailability
- **504 Gateway Timeout**: Upstream timeout

## Error Response Structure

### **Standard Format**

\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": [
      {
        "field": "email",
        "code": "INVALID_FORMAT",
        "message": "Email address must be valid"
      },
      {
        "field": "age",
        "code": "OUT_OF_RANGE",
        "message": "Age must be between 18 and 120",
        "constraints": {"min": 18, "max": 120}
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z",
    "requestId": "req_abc123",
    "documentation": "https://docs.api.com/errors/validation"
  }
}
\`\`\`

### **Minimal Format (Alternative)**

\`\`\`json
{
  "error": "VALIDATION_ERROR",
  "message": "Invalid email address"
}
\`\`\`

## Validation Errors

**Detailed field-level feedback**:

\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "fields": {
      "email": ["Email is required", "Email must be valid"],
      "password": ["Password must be at least 8 characters"],
      "age": ["Age must be a number"]
    }
  }
}
\`\`\`

## Rate Limit Errors

**Provide retry information**:

\`\`\`
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640000000
Retry-After: 60

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Try again in 60 seconds",
    "retryAfter": 60,
    "limit": 1000,
    "window": "1 hour"
  }
}
\`\`\`

## Internal Server Errors

**Never expose internal details**:

\`\`\`json
❌ Bad:
{
  "error": "SQLException: Column 'usr_email' not found in table 'users'"
}

✅ Good:
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "requestId": "req_abc123"
  }
}
// Log full details server-side for debugging
\`\`\`

## Error Code Conventions

**Use consistent, descriptive error codes**:

\`\`\`
# Resource errors
RESOURCE_NOT_FOUND
RESOURCE_ALREADY_EXISTS
RESOURCE_CONFLICT

# Authentication errors
INVALID_CREDENTIALS
TOKEN_EXPIRED
TOKEN_INVALID
UNAUTHORIZED

# Authorization errors
FORBIDDEN
INSUFFICIENT_PERMISSIONS

# Validation errors
VALIDATION_ERROR
MISSING_REQUIRED_FIELD
INVALID_FORMAT
OUT_OF_RANGE

# Rate limiting
RATE_LIMIT_EXCEEDED
QUOTA_EXCEEDED

# Server errors
INTERNAL_ERROR
SERVICE_UNAVAILABLE
UPSTREAM_ERROR
\`\`\`

## Retry Strategies

**Idempotency keys for safe retries**:

\`\`\`
POST /api/payments
Idempotency-Key: unique-operation-id-123

# Server stores result by key
# Retry with same key returns cached result
\`\`\`

**Exponential backoff guidance**:

\`\`\`json
{
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Service temporarily unavailable",
    "retryable": true,
    "retryAfter": 5
  }
}
\`\`\`

## Real-World Examples

### **Stripe API**
\`\`\`json
{
  "error": {
    "type": "card_error",
    "code": "card_declined",
    "message": "Your card was declined",
    "decline_code": "insufficient_funds"
  }
}
\`\`\`

### **GitHub API**
\`\`\`json
{
  "message": "Validation Failed",
  "errors": [
    {
      "resource": "Issue",
      "field": "title",
      "code": "missing_field"
    }
  ]
}
\`\`\`

### **AWS API**
\`\`\`json
{
  "Error": {
    "Code": "InvalidParameterValue",
    "Message": "Invalid value for parameter 'instanceType'"
  }
}
\`\`\`

## Best Practices

1. **Use proper HTTP status codes**: Don't return 200 for errors
2. **Machine-readable error codes**: For programmatic handling
3. **Human-readable messages**: For developer debugging
4. **Field-level details**: For validation errors
5. **Request ID**: For support and debugging
6. **Documentation links**: Help developers fix issues
7. **Consistent structure**: Across all endpoints
8. **Never expose sensitive data**: Internal paths, database details, etc.
9. **Localization support**: i18n for error messages
10. **Versioning**: Error format may evolve with API`,
};
