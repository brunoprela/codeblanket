/**
 * API Documentation Section
 */

export const apidocumentationSection = {
  id: 'api-documentation',
  title: 'API Documentation',
  content: `Comprehensive API documentation is crucial for developer experience and API adoption. Well-documented APIs are easier to use, maintain, and debug.

## Why Documentation Matters

- **Developer onboarding**: Faster integration
- **Reduced support**: Self-service answers
- **API discoverability**: Users find all features
- **Fewer errors**: Clear usage examples
- **Maintainability**: Team understands existing APIs

## Documentation Standards

### **OpenAPI (Swagger)**

Industry-standard API documentation format:

\`\`\`yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for managing users
  
servers:
  - url: https://api.example.com/v1
    description: Production
  - url: https://staging-api.example.com/v1
    description: Staging

paths:
  /users:
    get:
      summary: List users
      description: Returns a paginated list of users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  users:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '429':
          $ref: '#/components/responses/RateLimitExceeded'
    
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserInput'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'

  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          example: "usr_123"
        name:
          type: string
          example: "John Doe"
        email:
          type: string
          format: email
          example: "john@example.com"
        created_at:
          type: string
          format: date-time
    
    CreateUserInput:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
          format: email
        password:
          type: string
          format: password
          minLength: 8
    
    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        hasMore:
          type: boolean
  
  responses:
    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                example: "Authentication required"
    
    RateLimitExceeded:
      description: Rate limit exceeded
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
        X-RateLimit-Remaining:
          schema:
            type: integer
        Retry-After:
          schema:
            type: integer
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                example: "Rate limit exceeded"
  
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
\`\`\`

## Code Examples

Provide examples in multiple languages:

\`\`\`markdown
## Create User

Creates a new user account.

### Request

\`\`\`http
POST /v1/users
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "name": "John Doe",
  "email": "john@example.com"
}
\`\`\`

### Response

\`\`\`json
{
  "id": "usr_123",
  "name": "John Doe",
  "email": "john@example.com",
  "created_at": "2024-01-01T00:00:00Z"
}
\`\`\`

### Code Examples

**JavaScript**
\`\`\`javascript
const response = await fetch('https://api.example.com/v1/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    name: 'John Doe',
    email: 'john@example.com'
  })
});

const user = await response.json();
console.log (user);
\`\`\`

**Python**
\`\`\`python
import requests

response = requests.post(
    'https://api.example.com/v1/users',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={'name': 'John Doe', 'email': 'john@example.com'}
)

user = response.json()
print(user)
\`\`\`

**cURL**
\`\`\`bash
curl -X POST https://api.example.com/v1/users \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"name":"John Doe","email":"john@example.com"}'
\`\`\`
\`\`\`

## Interactive Documentation

### **Swagger UI**

Auto-generated interactive documentation:

\`\`\`javascript
const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./openapi.json');

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup (swaggerDocument));
\`\`\`

Developers can:
- Browse all endpoints
- Try requests directly
- See request/response schemas
- Test authentication

### **Redoc**

Clean, responsive API documentation:

\`\`\`html
<!DOCTYPE html>
<html>
  <head>
    <title>API Documentation</title>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
  </head>
  <body>
    <redoc spec-url='/openapi.json'></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
  </body>
</html>
\`\`\`

## Best Practices

1. **Keep docs updated**: Auto-generate from code
2. **Include examples**: Show real requests/responses
3. **Error scenarios**: Document all error codes
4. **Rate limits**: Document limits clearly
5. **Authentication**: Show how to authenticate
6. **Versioning**: Document version changes
7. **Status codes**: Explain what each means
8. **Pagination**: Show how to navigate results
9. **Webhooks**: Document event payloads
10. **SDKs**: Provide client libraries`,
};
