/**
 * RESTful API Design Principles Section
 */

export const restfulapidesignSection = {
  id: 'restful-api-design',
  title: 'RESTful API Design Principles',
  content: `RESTful APIs are the backbone of modern web services. Understanding REST principles and design patterns is essential for building scalable, maintainable APIs that developers love to use.

## What is REST?

**REST (Representational State Transfer)** is an architectural style for designing networked applications. It was introduced by Roy Fielding in his 2000 PhD dissertation.

### Key Principle
REST treats server objects as **resources** that can be created, read, updated, or deleted—similar to CRUD operations in databases.

---

## REST Constraints

REST defines six architectural constraints:

### **1. Client-Server Architecture**
- **Separation of concerns**: UI separated from data storage
- **Independence**: Client and server evolve independently
- **Benefit**: Better scalability and portability

**Example**: A React frontend (client) consumes a Node.js backend API (server). The frontend can be rewritten in Vue.js without changing the API.

### **2. Stateless**
- **No session state** stored on the server
- Each request contains **all information** needed to process it
- Session state lives in the client

**Example**:
\`\`\`
❌ Bad (Stateful):
GET /api/next-page  // Server remembers which page you're on

✅ Good (Stateless):
GET /api/users?page=2&limit=20  // All info in request
\`\`\`

**Benefits**:
- Easier to scale (no session affinity needed)
- Better reliability (no state to lose)
- Simplified server architecture

### **3. Cacheable**
- Responses must define themselves as cacheable or not
- Reduces client-server interactions
- Improves performance and scalability

**Example**:
\`\`\`http
HTTP/1.1 200 OK
Cache-Control: max-age=3600
ETag: "33a64df551425fcc"
\`\`\`

### **4. Uniform Interface**
- Consistent API across all resources
- Four sub-constraints:
  - **Resource identification**: URLs identify resources
  - **Resource manipulation through representations**: JSON/XML representations
  - **Self-descriptive messages**: Each message includes enough info to process it
  - **HATEOAS**: Hypermedia as the Engine of Application State

### **5. Layered System**
- Client can't tell if connected directly to end server
- Allows for load balancers, caches, security layers

### **6. Code on Demand (Optional)**
- Server can extend client functionality
- Example: Sending JavaScript to execute in browser

---

## Resource Naming Conventions

Good resource names are critical for intuitive APIs.

### **Use Nouns, Not Verbs**

\`\`\`
❌ Bad:
POST /api/createUser
GET /api/getUser/123
DELETE /api/deleteUser/123

✅ Good:
POST /api/users
GET /api/users/123
DELETE /api/users/123
\`\`\`

### **Use Plural Nouns for Collections**

\`\`\`
✅ Consistent:
GET /api/users          # Get all users
GET /api/users/123      # Get specific user
POST /api/users         # Create user
\`\`\`

### **Use Hierarchical Relationships**

\`\`\`
GET /api/users/123/posts          # User's posts
GET /api/users/123/posts/456      # Specific post by user
GET /api/posts/456/comments       # Post's comments
\`\`\`

### **Use Hyphens for Multi-Word Resources**

\`\`\`
✅ Good:
/api/shopping-carts
/api/user-preferences

❌ Avoid:
/api/shoppingCarts      # camelCase
/api/shopping_carts     # snake_case
\`\`\`

### **Lowercase URLs**

\`\`\`
✅ Good: /api/users/123/profile-pictures
❌ Bad: /api/Users/123/ProfilePictures
\`\`\`

---

## HTTP Methods Semantics

REST uses HTTP methods to define operations:

### **GET - Retrieve Resource**
- **Idempotent**: Multiple identical requests have same effect
- **Safe**: Doesn't modify server state
- **Cacheable**: Yes

\`\`\`
GET /api/users/123
GET /api/users?role=admin&status=active
\`\`\`

### **POST - Create Resource**
- **Not idempotent**: Multiple requests create multiple resources
- Creates subordinate resources
- Can also be used for complex operations that don't fit CRUD

\`\`\`
POST /api/users
Body: { "name": "Alice", "email": "alice@example.com" }

Response:
201 Created
Location: /api/users/124
{ "id": 124, "name": "Alice", "email": "alice@example.com" }
\`\`\`

### **PUT - Update/Replace Resource**
- **Idempotent**: Multiple identical requests have same effect
- Replaces entire resource
- Client specifies resource ID

\`\`\`
PUT /api/users/123
Body: { "name": "Alice Updated", "email": "alice@new.com", "role": "admin" }
\`\`\`

**Important**: PUT replaces the **entire** resource. Missing fields may be set to null/default.

### **PATCH - Partial Update**
- **Can be idempotent** (depends on implementation)
- Updates only specified fields
- More flexible than PUT

\`\`\`
PATCH /api/users/123
Body: { "email": "newemail@example.com" }
# Only email is updated, other fields unchanged
\`\`\`

### **DELETE - Remove Resource**
- **Idempotent**: Deleting same resource multiple times has same effect
- Resource removed

\`\`\`
DELETE /api/users/123

Response:
204 No Content  or  200 OK with confirmation message
\`\`\`

---

## Idempotency

**Idempotent**: An operation that can be applied multiple times without changing the result beyond the initial application.

### **Why It Matters**
- **Network retries**: Safe to retry failed requests
- **Reliability**: Handles duplicate requests gracefully
- **Client simplicity**: No complex retry logic needed

### **Idempotency by Method**

| Method | Idempotent | Safe | Cacheable |
|--------|-----------|------|-----------|
| GET    | ✅ Yes    | ✅ Yes | ✅ Yes |
| POST   | ❌ No     | ❌ No  | ❌ Rarely |
| PUT    | ✅ Yes    | ❌ No  | ❌ No |
| PATCH  | ⚠️ Depends | ❌ No  | ❌ No |
| DELETE | ✅ Yes    | ❌ No  | ❌ No |

**Example of Idempotency**:
\`\`\`
PUT /api/users/123 with { "name": "Alice" }
# Call it once: name becomes "Alice"
# Call it 100 times: name is still "Alice" (not "AliceAliceAlice...")

POST /api/users with { "name": "Alice" }
# Call it once: Creates user with ID 124
# Call it 100 times: Creates 100 users with different IDs
\`\`\`

---

## HATEOAS (Hypermedia as the Engine of Application State)

The most debated REST constraint. Responses include **links** to related resources.

### **Example Without HATEOAS**
\`\`\`json
{
  "id": 123,
  "name": "Alice",
  "orderId": 456
}
\`\`\`

Client must know to construct: \`GET /api/orders/456\`

### **Example With HATEOAS**
\`\`\`json
{
  "id": 123,
  "name": "Alice",
  "order": {
    "id": 456,
    "href": "/api/orders/456"
  },
  "links": {
    "self": "/api/users/123",
    "orders": "/api/users/123/orders",
    "update": {
      "href": "/api/users/123",
      "method": "PUT"
    },
    "delete": {
      "href": "/api/users/123",
      "method": "DELETE"
    }
  }
}
\`\`\`

### **Benefits**
- **Self-documenting**: API tells clients what actions are available
- **Evolvability**: URLs can change without breaking clients
- **Discoverability**: Clients navigate API through links

### **Drawbacks**
- **Verbose**: Larger response payloads
- **Complexity**: More work to implement
- **Limited adoption**: Few APIs implement full HATEOAS

**Reality**: Most "REST APIs" don't implement HATEOAS and are technically "REST-like" or "HTTP APIs."

---

## Richardson Maturity Model

Leonard Richardson's model for REST API maturity:

### **Level 0: The Swamp of POX (Plain Old XML)**
- Single endpoint
- Single HTTP method (usually POST)
- RPC-style

\`\`\`
POST /api
Body: { "method": "getUser", "params": {"id": 123} }
\`\`\`

### **Level 1: Resources**
- Multiple endpoints (resources)
- Still mostly POST

\`\`\`
POST /api/users
POST /api/orders
\`\`\`

### **Level 2: HTTP Verbs**
- Proper use of HTTP methods
- Proper status codes
- Most APIs stop here

\`\`\`
GET /api/users/123
POST /api/users
PUT /api/users/123
DELETE /api/users/123
\`\`\`

### **Level 3: Hypermedia Controls (HATEOAS)**
- Responses include links
- True REST
- Rare in practice

---

## RESTful API Best Practices

### **1. Version Your API**
\`\`\`
/api/v1/users
/api/v2/users
\`\`\`

### **2. Use Query Parameters for Filtering, Sorting, Pagination**
\`\`\`
GET /api/users?role=admin&status=active
GET /api/users?sort=created_at:desc
GET /api/users?page=2&limit=20
\`\`\`

### **3. Return Appropriate Status Codes**
- \`200 OK\`: Success
- \`201 Created\`: Resource created
- \`204 No Content\`: Success with no body
- \`400 Bad Request\`: Client error
- \`401 Unauthorized\`: Not authenticated
- \`403 Forbidden\`: Authenticated but not authorized
- \`404 Not Found\`: Resource doesn't exist
- \`500 Internal Server Error\`: Server error

### **4. Use JSON for Responses**
Most modern APIs use JSON (not XML):
\`\`\`json
{
  "id": 123,
  "name": "Alice",
  "email": "alice@example.com"
}
\`\`\`

### **5. Provide Meaningful Error Messages**
\`\`\`json
{
  "error": {
    "code": "INVALID_EMAIL",
    "message": "Email address is invalid",
    "field": "email",
    "value": "not-an-email"
  }
}
\`\`\`

### **6. Support Filtering and Pagination for Collections**
Never return all records for large collections.

### **7. Use OAuth 2.0 for Authorization**
Industry standard for API authorization.

### **8. Rate Limit Your API**
Protect against abuse:
\`\`\`http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640000000
\`\`\`

---

## Real-World Examples

### **GitHub API** (Level 2 REST, HATEOAS for some endpoints)
\`\`\`
GET https://api.github.com/users/octocat
GET https://api.github.com/repos/owner/repo/issues
POST https://api.github.com/repos/owner/repo/issues
\`\`\`

### **Stripe API** (Excellent REST design)
\`\`\`
POST /v1/customers
GET /v1/customers/:id
POST /v1/customers/:id
DELETE /v1/customers/:id
\`\`\`

### **Twitter API v2** (Modern REST practices)
\`\`\`
GET /2/tweets/:id
GET /2/users/:id/tweets
POST /2/tweets
\`\`\`

---

## Common Mistakes

### **❌ Using Verbs in Resource Names**
\`\`\`
POST /api/createUser  # Wrong
POST /api/users       # Correct
\`\`\`

### **❌ Not Using Proper HTTP Methods**
\`\`\`
GET /api/users/delete/123  # Wrong
DELETE /api/users/123      # Correct
\`\`\`

### **❌ Not Returning Appropriate Status Codes**
\`\`\`
# Wrong: Returning 200 for errors
{ "success": false, "error": "Not found" }  # Status 200

# Correct: Use proper status
Status 404 Not Found
{ "error": "User not found" }
\`\`\`

### **❌ Not Versioning APIs**
Breaking changes without versioning break client applications.

### **❌ Over-fetching/Under-fetching**
- Over-fetching: Returning too much data
- Under-fetching: Requiring multiple requests
- Solution: Field selection or consider GraphQL

---

## Interview Tips

1. **Explain REST constraints clearly**: Especially stateless and uniform interface
2. **Know the difference between PUT and PATCH**: PUT replaces, PATCH updates
3. **Understand idempotency**: Critical for reliability
4. **Discuss trade-offs**: REST vs GraphQL vs gRPC
5. **Real-world experience**: Mention APIs you've designed
6. **Status codes**: Know the common ones (200, 201, 400, 401, 403, 404, 500)
7. **HATEOAS awareness**: Know what it is, but acknowledge limited adoption

### **Sample Interview Question**
"Design a RESTful API for a blog platform with users, posts, and comments."

**Good Answer**:
\`\`\`
# Users
GET    /api/v1/users           # List users
POST   /api/v1/users           # Create user
GET    /api/v1/users/:id       # Get user
PUT    /api/v1/users/:id       # Update user
DELETE /api/v1/users/:id       # Delete user

# Posts
GET    /api/v1/posts           # List posts
POST   /api/v1/posts           # Create post
GET    /api/v1/posts/:id       # Get post
PUT    /api/v1/posts/:id       # Update post
DELETE /api/v1/posts/:id       # Delete post

# User's posts
GET    /api/v1/users/:id/posts # Get user's posts

# Comments
GET    /api/v1/posts/:id/comments      # List comments on post
POST   /api/v1/posts/:id/comments      # Create comment
GET    /api/v1/comments/:id            # Get comment
PUT    /api/v1/comments/:id            # Update comment
DELETE /api/v1/comments/:id            # Delete comment

# With query parameters
GET /api/v1/posts?author=123&status=published&sort=created_at:desc&page=1&limit=20
\`\`\`

Discuss: Versioning strategy, authentication (OAuth 2.0), rate limiting, pagination, filtering, error handling.

---

**Key Takeaway**: RESTful API design is about creating intuitive, consistent, scalable APIs using HTTP standards effectively. Most successful APIs are "Level 2" REST with practical compromises.`,
};
