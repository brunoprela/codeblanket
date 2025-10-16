import { Module } from '../types';

export const systemDesignApiDesignModule: Module = {
  id: 'system-design-api-design',
  title: 'API Design & Management',
  description:
    'Master RESTful API design, GraphQL, gRPC, and API lifecycle management',
  icon: '🔌',
  sections: [
    {
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
      multipleChoice: [
        {
          id: 'rest-q1',
          question:
            "Which HTTP method should be used to partially update a user's email address without affecting other fields?",
          options: [
            'POST /api/users/123 with {"email": "new@example.com"}',
            'PUT /api/users/123 with {"email": "new@example.com"}',
            'PATCH /api/users/123 with {"email": "new@example.com"}',
            'GET /api/users/123/update-email?email=new@example.com',
          ],
          correctAnswer: 2,
          explanation:
            'PATCH is designed for partial updates, modifying only the specified fields. PUT would replace the entire resource, potentially clearing other fields. POST is for creating resources, and GET should never modify data.',
          difficulty: 'medium',
        },
        {
          id: 'rest-q2',
          question:
            'A client retries a failed DELETE request 3 times due to network issues. The resource was actually deleted on the first attempt. What makes this safe?',
          options: [
            'DELETE operations always return 404 after the first success',
            'DELETE is idempotent - multiple deletions have the same effect',
            'The server stores a request ID to prevent duplicates',
            'DELETE operations are automatically cached by HTTP',
          ],
          correctAnswer: 1,
          explanation:
            'DELETE is idempotent by design. Whether you delete a resource once or multiple times, the end state is the same - the resource is gone. This makes retries safe. The second attempt typically returns 404 (not found) or 204 (no content), but the key is that the operation is idempotent.',
          difficulty: 'medium',
        },
        {
          id: 'rest-q3',
          question:
            'Which URL follows RESTful naming conventions for retrieving the 5 most recent orders of a specific customer?',
          options: [
            'GET /api/getCustomerOrders?customerId=123&limit=5',
            'GET /api/customer/123/recent-orders',
            'GET /api/customers/123/orders?sort=created_at:desc&limit=5',
            'POST /api/customers/orders with {"customerId": 123, "limit": 5}',
          ],
          correctAnswer: 2,
          explanation:
            'Option 3 follows REST conventions: uses plural nouns (customers, orders), hierarchical structure showing relationship, GET method for retrieval, and query parameters for filtering/sorting. Option 1 uses verbs in URL, option 2 uses singular and vague naming, option 4 incorrectly uses POST for reading data.',
          difficulty: 'medium',
        },
        {
          id: 'rest-q4',
          question:
            'What is the main principle of the "stateless" constraint in REST?',
          options: [
            'The API should not store any data in a database',
            'Each request must contain all information needed to process it',
            'The server should not maintain any state at all',
            'Clients cannot maintain session cookies or tokens',
          ],
          correctAnswer: 1,
          explanation:
            'Stateless means each request contains all information needed (auth tokens, parameters, etc.) - the server does not store session state between requests. The server can still have databases (option 1 wrong), maintain application state like data (option 3 wrong), and clients can hold tokens (option 4 wrong). The key is no server-side session state.',
          difficulty: 'medium',
        },
        {
          id: 'rest-q5',
          question:
            'According to the Richardson Maturity Model, what distinguishes a Level 2 REST API from Level 3?',
          options: [
            'Level 2 uses JSON while Level 3 uses XML',
            'Level 2 uses HTTP verbs correctly, Level 3 adds hypermedia controls (HATEOAS)',
            'Level 2 supports pagination while Level 3 supports filtering',
            'Level 2 is synchronous while Level 3 supports async operations',
          ],
          correctAnswer: 1,
          explanation:
            'The Richardson Maturity Model levels are: Level 0 (single endpoint), Level 1 (multiple resources), Level 2 (HTTP verbs and status codes correctly), Level 3 (adds HATEOAS - responses include links to related resources). Most production APIs are Level 2. The other options are unrelated to the maturity model.',
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'rest-d1',
          question:
            'You\'re designing an API for a large e-commerce platform. A senior developer suggests making all operations use POST to "simplify the API." How would you respond, and what are the trade-offs of using proper HTTP methods vs. using only POST?',
          sampleAnswer: `I would respectfully disagree with using only POST, here's why:

**Problems with POST-only APIs:**

1. **Loss of Idempotency**: GET, PUT, and DELETE are idempotent, making retries safe. With POST-only, you need custom logic to prevent duplicate operations (like charging a customer twice).

2. **No HTTP Caching**: Browsers and CDNs automatically cache GET requests. POST requests are never cached, leading to unnecessary server load and slower responses for read operations.

3. **Breaking REST Conventions**: Developers expect GET for reads, POST for creates, PUT/PATCH for updates, DELETE for deletes. A POST-only API is confusing and harder to learn.

4. **Loss of Browser Features**: Browser history, bookmarking, and link sharing work with GET. A POST-only API breaks these features.

5. **Monitoring and Debugging**: HTTP methods in logs immediately show intent (GET = read, DELETE = delete). With POST-only, you need to inspect request bodies.

**When POST-only Might Make Sense:**
- RPC-style operations that don't map to CRUD (e.g., "run report", "process payment")
- Internal microservices where caching isn't needed
- Legacy system constraints

**Recommendation**: Use proper HTTP methods (Level 2 REST) as the default. It's an industry standard that provides caching, idempotency, and better developer experience with minimal additional complexity.`,
          keyPoints: [
            'HTTP methods provide idempotency (safe retries)',
            'GET enables caching at multiple layers',
            'Semantic meaning aids debugging and monitoring',
            'Industry conventions improve developer experience',
            'Trade-offs: slightly more complex routing vs. significant benefits',
          ],
        },
        {
          id: 'rest-d2',
          question:
            'When designing a REST API, when should you use PUT vs. PATCH for updates? Provide a real-world scenario where choosing the wrong method could cause production issues.',
          sampleAnswer: `**PUT vs. PATCH Decision:**

**Use PUT when:**
- Replacing the entire resource
- Client sends complete representation
- Idempotency is critical

**Use PATCH when:**
- Updating specific fields only
- Preserving other fields is important
- Reducing payload size matters

**Real-World Scenario - User Profile API:**

Imagine a user profile with fields: name, email, phone, address, bio, profilePicture, preferences, notifications, etc.

**Using PUT Incorrectly:**
\`\`\`javascript
// Mobile app wants to update only email
PUT /api/users/123
{ "email": "newemail@example.com" }

// Server interprets this as full replacement
// Result: name, phone, address, etc. all set to null!
\`\`\`

**Production Issue**: User's profile is wiped except for email. They lose their profile picture, bio, notification preferences, etc. This actually happened at a company I consulted for - a mobile app bug sent incomplete PUT requests, causing customer data loss.

**Solution - Use PATCH:**
\`\`\`javascript
PATCH /api/users/123
{ "email": "newemail@example.com" }

// Server updates only email, preserves other fields
\`\`\`

**Best Practice:**
1. Use PATCH for most update operations (safer, more flexible)
2. Document whether PUT requires full object or accepts partial
3. Server validation: reject PUT requests missing required fields
4. Consider using PATCH exclusively to avoid confusion
5. Version your API to change behavior without breaking clients

**Trade-off**: PATCH is more complex to implement (need to handle partial updates, merge logic) but much safer for clients.`,
          keyPoints: [
            'PUT replaces entire resource; PATCH updates specific fields',
            'Incorrect PUT usage can cause unintended data loss',
            'PATCH is safer for partial updates but more complex to implement',
            'Server should validate PUT requests contain complete resource',
            'Document expected behavior clearly in API documentation',
          ],
        },
        {
          id: 'rest-d3',
          question:
            'A client team complains your REST API requires too many round trips - they need to fetch a user, their posts, and comments on each post (3+ requests). How would you address this while maintaining RESTful principles? Discuss the trade-offs.',
          sampleAnswer: `This is a classic REST challenge: maintaining stateless, resource-oriented design while avoiding the N+1 query problem and excessive round trips.

**Solutions:**

**1. Query Parameter Expansion (Recommended for REST)**
\`\`\`
GET /api/users/123?expand=posts,posts.comments

Response:
{
  "id": 123,
  "name": "Alice",
  "posts": [
    {
      "id": 456,
      "title": "My Post",
      "comments": [...]
    }
  ]
}
\`\`\`

**Trade-offs:**
✅ Stays RESTful
✅ Backward compatible (expand is optional)
❌ Response can get large
❌ Complex to implement server-side

**2. Compound Documents (JSON:API Style)**
Include related resources in a single response with relationships.

**3. GraphQL (Paradigm Shift)**
Let clients specify exactly what they need:
\`\`\`graphql
query {
  user(id: 123) {
    name
    posts {
      title
      comments {
        text
      }
    }
  }
}
\`\`\`

**Trade-offs:**
✅ Perfect for this problem
✅ No over/under-fetching
❌ Not RESTful anymore
❌ Learning curve for team
❌ More complex backend

**4. Backend for Frontend (BFF) Pattern**
Create a custom endpoint for this specific UI need:
\`\`\`
GET /api/bff/user-profile-page/123
\`\`\`

**Trade-offs:**
✅ Single round trip
✅ Optimized for frontend
❌ Less generic/reusable
❌ More endpoints to maintain

**Recommendation:**
Start with **query parameter expansion** for REST APIs. If you have multiple clients with very different needs, consider GraphQL. Use BFF endpoints sparingly for complex aggregations.

**Reality Check**: This is a fundamental limitation of REST. Many companies (GitHub, Shopify, Facebook) offer both REST and GraphQL APIs for different use cases.`,
          keyPoints: [
            "REST's resource-oriented nature can require multiple requests",
            'Query parameter expansion (?expand=) maintains REST principles',
            'GraphQL solves this problem but is a paradigm shift',
            'BFF pattern creates custom endpoints for specific UI needs',
            'No perfect solution - choose based on team, clients, and use case',
          ],
        },
      ],
    },
    {
      id: 'api-request-response-design',
      title: 'API Request/Response Design',
      content: `Designing clear, consistent request and response structures is crucial for API usability. Well-designed APIs are intuitive, efficient, and handle edge cases gracefully.

## Request Structure Best Practices

### **URL Structure**
\`\`\`
https://api.example.com/v1/resources/123?filter=value
\`\`\`

### **Pagination Strategies**

**Offset-Based**:
\`\`\`
GET /api/users?page=2&limit=20
\`\`\`

**Cursor-Based (Recommended for Scale)**:
\`\`\`
GET /api/users?cursor=eyJpZCI6MTIzfQ&limit=20
\`\`\`

**Key Differences**:
- Offset: Simple, can jump to any page, but slow at high offsets and inconsistent with real-time data
- Cursor: Fast, consistent results, but can't jump to arbitrary page

### **Filtering and Sorting**

\`\`\`
GET /api/users?role=admin&status=active
GET /api/users?age[gte]=18&age[lte]=65
GET /api/users?sort=created_at:desc
\`\`\`

### **Field Selection**

\`\`\`
GET /api/users/123?fields=id,name,email
\`\`\`

Reduces bandwidth and improves performance for mobile clients.

### **Error Response Standards**

\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Email must be valid"
      }
    ]
  }
}
\`\`\``,
      multipleChoice: [
        {
          id: 'req-res-q1',
          question:
            'Which pagination approach best handles real-time data where items are frequently added/removed?',
          options: [
            'Offset-based: GET /posts?page=2&limit=20',
            'Cursor-based: GET /posts?cursor=abc123&limit=20',
            'Page numbers with caching',
            'Load all data client-side',
          ],
          correctAnswer: 1,
          explanation:
            'Cursor-based pagination uses stable references (like ID + timestamp), preventing duplicates or missing items when data changes. Offset-based can show duplicates if items are inserted between requests.',
          difficulty: 'medium',
        },
        {
          id: 'req-res-q2',
          question:
            "What's the correct HTTP status code and response for an invalid email during registration?",
          options: [
            '200 OK with {"success": false}',
            '400 Bad Request with structured error',
            '500 Internal Server Error',
            '422 Unprocessable Entity',
          ],
          correctAnswer: 1,
          explanation:
            '400 Bad Request is appropriate for client input errors. While 422 is also valid for validation errors, 400 is more commonly used and understood.',
          difficulty: 'easy',
        },
        {
          id: 'req-res-q3',
          question:
            'Which query syntax best supports filtering users by age range (18-65)?',
          options: [
            'GET /users?age=18-65',
            'GET /users?minAge=18&maxAge=65',
            'GET /users?age[gte]=18&age[lte]=65',
            'GET /users?filter=age BETWEEN 18 AND 65',
          ],
          correctAnswer: 2,
          explanation:
            'Operator-based filtering (age[gte]=18) is extensible, consistent, and widely used (MongoDB-style). It scales to other operators (gt, lt, ne, in) without parameter explosion.',
          difficulty: 'medium',
        },
        {
          id: 'req-res-q4',
          question:
            'A mobile app needs only id, name, avatar but not full profile. Best approach?',
          options: [
            'Create /users/123/mobile endpoint',
            'Use GET /users/123?fields=id,name,avatar',
            'Return everything, filter client-side',
            'Require migration to GraphQL',
          ],
          correctAnswer: 1,
          explanation:
            'Field selection (sparse fieldsets) is the REST way to optimize bandwidth without creating multiple endpoints or paradigm shifts.',
          difficulty: 'easy',
        },
        {
          id: 'req-res-q5',
          question: 'Why is OFFSET 10000 slow in pagination queries?',
          options: [
            'Database needs more memory',
            'Database must scan and skip first 10,000 rows',
            'Query optimizer is not configured',
            'Index is missing on the ID column',
          ],
          correctAnswer: 1,
          explanation:
            'OFFSET requires scanning and discarding rows. Cursor-based pagination uses WHERE id > X which uses indexes efficiently regardless of position.',
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'req-res-d1',
          question:
            'Design a response structure that serves both bandwidth-constrained mobile clients and data-rich web clients without maintaining two APIs.',
          sampleAnswer: `Use field selection with predefined field sets:

\`\`\`
GET /api/users/123?fields=basic    # Mobile: id, name, avatar
GET /api/users/123?fields=full     # Web: all fields
GET /api/users/123?fields=id,name,email  # Custom selection
\`\`\`

Implementation: Whitelist allowed fields, SELECT only requested columns from database, serialize only included fields.

Alternative: GraphQL allows perfect client-driven selection but requires paradigm shift.

Trade-off: Field selection is more complex backend but maintains REST principles and serves diverse clients efficiently.`,
          keyPoints: [
            'Field selection allows flexible data requirements',
            'Predefined field sets (basic, full) for common patterns',
            'GraphQL solves this elegantly but different paradigm',
            'Enable gzip compression for additional savings',
            'Trade complexity for flexibility and performance',
          ],
        },
        {
          id: 'req-res-d2',
          question:
            "Your API embeds a user's posts array inline, causing timeouts for users with thousands of posts. How to fix?",
          sampleAnswer: `Never include unbounded nested collections. Solutions:

1. **Remove nested collection, use links** (Recommended):
\`\`\`json
{
  "id": 123,
  "name": "Alice",
  "postsCount": 10000,
  "links": {"posts": "/api/users/123/posts"}
}
\`\`\`

2. **Limited preview with link**:
\`\`\`json
{
  "recentPosts": [/* 5 most recent */],
  "postsCount": 10000,
  "links": {"allPosts": "/api/users/123/posts"}
}
\`\`\`

3. **Separate endpoints**:
\`\`\`
GET /api/users/123        # User only
GET /api/posts?userId=123 # Posts paginated
\`\`\`

Real-world: GitHub, Twitter, Stripe all use separate endpoints for collections.

Rule: Always provide counts and links to paginated collections, never unbounded arrays.`,
          keyPoints: [
            'Unbounded nested collections cause performance issues',
            'Use separate endpoints for related collections',
            'Include counts and pagination links',
            'Limited previews provide convenience without risk',
            'Scalability always trumps convenience',
          ],
        },
        {
          id: 'req-res-d3',
          question:
            'Search results show inconsistent total counts (1,247 then 1,251 seconds later). Explain and propose solutions.',
          sampleAnswer: `Root cause: Real-time data changes between COUNT and SELECT queries, or eventual consistency in distributed databases.

Why exact counts are problematic:
1. COUNT(*) with filters is slow on large tables
2. Stale immediately in real-time systems
3. Requires locks for consistency (bad performance)

Solutions:

1. **Approximate counts** (Recommended):
\`\`\`json
{"approximateTotal": "~1,200"}
\`\`\`
Examples: Google ("About 1,240,000 results")

2. **Omit total** (Best for scale):
\`\`\`json
{"hasMore": true, "nextCursor": "..."}
\`\`\`
Examples: Twitter, Instagram feeds

3. **Cache with TTL**: Fast but still potentially inconsistent

Only use exact counts for:
- Small datasets (<10K records)
- Admin dashboards
- Static data

This isn't a bug—it's a fundamental trade-off between consistency, performance, and scale in distributed systems.`,
          keyPoints: [
            'Exact counts expensive and stale in real-time systems',
            'Approximate counts faster and honest about uncertainty',
            'Omitting counts scales best (infinite scroll pattern)',
            'Cursor pagination works without total counts',
            'Trade accuracy for performance at scale',
          ],
        },
      ],
    },
    {
      id: 'api-authentication',
      title: 'API Authentication Methods',
      content: `Authentication verifies who is making an API request. Choosing the right method depends on your use case, security requirements, and client types.

## Overview of Authentication Methods

### **1. API Keys**
Simple string tokens identifying the application.

\`\`\`http
GET /api/users
X-API-Key: sk_live_abc123xyz
\`\`\`

**Pros**: Simple, good for identifying applications
**Cons**: Not per-user, hard to rotate, easily leaked if not HTTPS

**Use Cases**: Public APIs with usage tracking, server-to-server communication

### **2. Bearer Tokens (JWT)**
JSON Web Tokens containing encoded claims.

\`\`\`http
GET /api/users
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
\`\`\`

**Structure**:
\`\`\`
header.payload.signature
\`\`\`

**Pros**: Stateless, contains user info, self-contained
**Cons**: Can't revoke easily (until expiry), size overhead

**Use Cases**: Modern web/mobile apps, microservices

### **3. OAuth 2.0**
Delegation protocol for third-party access.

**Flows**:
- Authorization Code: Web apps
- Client Credentials: Server-to-server
- Implicit: Legacy SPAs (deprecated)
- PKCE: Mobile/SPA (recommended)

**Use Cases**: Social login, third-party integrations

### **4. Basic Authentication**
Username:password in Base64.

\`\`\`http
Authorization: Basic dXNlcjpwYXNzd29yZA==
\`\`\`

**Pros**: Simple, built into HTTP
**Cons**: Credentials in every request, no built-in expiry

**Use Cases**: Simple internal APIs, dev/test environments

### **5. Mutual TLS (mTLS)**
Both client and server present certificates.

**Pros**: Very secure, cryptographic authentication
**Cons**: Complex setup, certificate management

**Use Cases**: High-security microservices, banking APIs

## API Keys Deep Dive

**Generation**:
\`\`\`javascript
const apiKey = 'sk_' + crypto.randomBytes(32).toString('hex');
\`\`\`

**Storage**: Hash before storing (like passwords)

**Best Practices**:
- Prefix keys (\`sk_\` for secret, \`pk_\` for public)
- Allow multiple keys per account
- Track last used date
- Enable rotation
- Rate limit by key

## JWT Best Practices

**Claims**:
\`\`\`json
{
  "sub": "user_123",
  "iat": 1640000000,
  "exp": 1640003600,
  "role": "admin"
}
\`\`\`

**Short-lived Access Tokens**:
- Access token: 15 minutes
- Refresh token: 7 days
- Rotate refresh tokens

**Validation**:
1. Check signature
2. Verify expiration
3. Validate issuer/audience
4. Check revocation list (if needed)

## OAuth 2.0 in Practice

**Authorization Code Flow**:
\`\`\`
1. Client → Authorization Server: Request auth code
2. User logs in, grants permission
3. Authorization Server → Client: Auth code
4. Client → Authorization Server: Exchange code for token
5. Authorization Server → Client: Access token + refresh token
\`\`\`

**Scopes**:
\`\`\`
read:users write:users admin:all
\`\`\`

## Security Best Practices

1. **Always use HTTPS**: Prevents token interception
2. **Short token lifetimes**: Limit damage if compromised
3. **Implement rate limiting**: Prevent brute force
4. **Log authentication attempts**: Detect attacks
5. **Support token revocation**: For compromised tokens
6. **Use secure storage**: HttpOnly cookies or secure storage
7. **Implement CORS properly**: Prevent unauthorized origins`,
      multipleChoice: [
        {
          id: 'auth-q1',
          question:
            'Which authentication method is most appropriate for a mobile app accessing your API with per-user permissions?',
          options: [
            'API Keys (one per app install)',
            'Basic Authentication with username/password',
            'JWT with OAuth 2.0 PKCE flow',
            'Mutual TLS certificates',
          ],
          correctAnswer: 2,
          explanation:
            "JWT with OAuth 2.0 PKCE flow is designed for mobile apps: provides per-user auth, secure without client secret, supports token refresh. API keys don't differentiate users, Basic Auth sends credentials repeatedly, mTLS is complex for mobile.",
          difficulty: 'medium',
        },
        {
          id: 'auth-q2',
          question:
            'Why should JWT access tokens have short lifetimes (e.g., 15 minutes)?',
          options: [
            'To reduce server storage requirements',
            'To force users to log in frequently for security',
            'To limit damage window if token is compromised',
            'To improve API performance',
          ],
          correctAnswer: 2,
          explanation:
            "Short-lived access tokens limit the time window an attacker can use a stolen token. Use refresh tokens for long-term access. JWTs are stateless (not stored on server), and short lifetimes don't require frequent user logins (refresh tokens handle renewal).",
          difficulty: 'medium',
        },
        {
          id: 'auth-q3',
          question:
            'You need to allow a third-party app to access user data on behalf of users without receiving user passwords. Which approach?',
          options: [
            'Share API keys with third party',
            'Have users share their passwords with third party',
            'Implement OAuth 2.0 authorization',
            'Use Basic Authentication with temporary passwords',
          ],
          correctAnswer: 2,
          explanation:
            'OAuth 2.0 is specifically designed for delegation - allowing third parties to access resources on behalf of users without sharing credentials. Users grant permission, and third party receives scoped access tokens.',
          difficulty: 'easy',
        },
        {
          id: 'auth-q4',
          question:
            'What is the main security advantage of mTLS over Bearer tokens?',
          options: [
            'mTLS tokens are shorter and faster',
            'mTLS provides cryptographic proof of identity from both sides',
            'mTLS tokens never expire',
            "mTLS doesn't require HTTPS",
          ],
          correctAnswer: 1,
          explanation:
            'mTLS (Mutual TLS) requires both client and server to present valid certificates, providing cryptographic proof of identity. Bearer tokens can be stolen and replayed. mTLS still requires TLS, and certificates do expire.',
          difficulty: 'hard',
        },
        {
          id: 'auth-q5',
          question:
            'Your API key was accidentally committed to a public GitHub repo. Best response?',
          options: [
            'Delete the commit immediately, problem solved',
            'Revoke the key, rotate to new key, audit usage logs',
            'Change GitHub repo to private',
            'Add .gitignore for future, keep current key',
          ],
          correctAnswer: 1,
          explanation:
            "Once exposed publicly, assume the key is compromised forever (GitHub history, scrapers, caches). Must revoke immediately, issue new key, and audit logs for unauthorized usage. Deleting commits doesn't remove from history.",
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'auth-d1',
          question:
            'Design an authentication system for an API serving web app, mobile app, and third-party integrations. What methods would you use for each and why?',
          sampleAnswer: `Different clients have different security requirements and constraints:

**Web App (SPA)**:
- OAuth 2.0 Authorization Code + PKCE
- Short-lived JWT access tokens (15 min)
- Refresh tokens in HttpOnly cookies
- Why: Secure for browser environment, PKCE prevents token interception, HttpOnly prevents XSS

**Mobile App**:
- OAuth 2.0 PKCE flow
- JWT access + refresh tokens in secure storage
- Biometric auth locally
- Why: No client secret security, native secure storage, good UX with biometrics

**Third-Party Integrations**:
- OAuth 2.0 Authorization Code (for user delegation)
- API Keys for server-to-server
- Scoped permissions (read:users, write:posts)
- Why: Users control what third parties can access, revocable, scoped

**Internal Microservices**:
- mTLS or service-to-service JWT
- Service mesh for automatic cert management
- Why: High security, no user involvement, automatic rotation

**Implementation**:
- Single Authorization Server (like Auth0, Keycloak)
- Multiple authentication flows supported
- Centralized token validation
- Audit logs for all auth events

Trade-offs: Complexity vs security. Multiple methods increase implementation and maintenance but provide appropriate security for each use case.`,
          keyPoints: [
            'Different clients need different auth methods',
            'OAuth 2.0 with PKCE for user-facing clients',
            'API keys for server-to-server integration',
            'mTLS for high-security internal services',
            'Centralized auth server for consistency',
          ],
        },
        {
          id: 'auth-d2',
          question:
            'JWTs are stateless, which makes them scalable but difficult to revoke. How would you handle immediate token revocation (e.g., user logs out or account compromised)?',
          sampleAnswer: `JWT revocation challenge: They're valid until expiration by design. Solutions:

**1. Short-Lived Tokens + Refresh Token Rotation** (Recommended):
- Access token: 15 minutes
- Refresh token: Stored in database, can be revoked
- On compromise: Revoke refresh token, access token expires soon

**2. Token Blacklist (Allow List)**:
\`\`\`javascript
// Check on every request
const isRevoked = await redis.get(\`revoked:\${jti}\`);
\`\`\`
Pros: Immediate revocation
Cons: Requires state (contradicts stateless), network call each request

**3. Token Versioning**:
\`\`\`json
{ "sub": "user_123", "version": 5 }
\`\`\`
Store current version in cache. Revoke all by incrementing version.
Pros: One cache lookup per request
Cons: Still requires state

**4. Short Expiry + Frequent Refresh**:
- Ultra-short access tokens (5 min)
- Check permissions on refresh
- Effective revocation within 5 minutes

**5. Event-Driven Revocation**:
- Broadcast revocation events to all servers
- Local in-memory cache of revoked tokens
- Reduces latency vs database check

**Best Practice Approach**:
- Short-lived access tokens (15 min) 
- Revocable refresh tokens in database
- Allow list for critical revocations (admin lockout)
- Accept eventual consistency for most cases

**Trade-off**: Balance between immediate revocation and scalability. Most apps can tolerate 15-minute window. High-security apps need blacklist despite scalability cost.`,
          keyPoints: [
            'JWT statelessness makes immediate revocation challenging',
            'Short-lived access tokens + revocable refresh tokens is standard',
            'Blacklist/allow list provides immediate revocation but adds state',
            'Token versioning reduces database load',
            'Accept eventual consistency for better scalability',
          ],
        },
        {
          id: 'auth-d3',
          question:
            'A third-party integration stores API keys in their database unencrypted, then gets hacked. How would you design your API key system to limit damage from such incidents?',
          sampleAnswer: `Defense in depth for API key security:

**1. Hash Keys Before Storage** (Like Passwords):
\`\`\`javascript
// Generation
const apiKey = 'sk_' + randomBytes(32).toString('hex');
const hash = bcrypt.hash(apiKey, 10);
// Store: id, hash, prefix 'sk_...xyz' (last 4), metadata

// Validation
const match = bcrypt.compare(providedKey, storedHash);
\`\`\`
Benefit: Even if YOUR database is compromised, keys can't be recovered

**2. Scope Limitations**:
\`\`\`json
{
  "key_id": "key_123",
  "scopes": ["read:public_data"],
  "rate_limit": "1000/hour"
}
\`\`\`
Benefit: Compromised key has limited permissions

**3. IP Whitelisting**:
\`\`\`json
{"key_id": "key_123", "allowed_ips": ["52.12.34.56"]}
\`\`\`
Benefit: Stolen key useless from other IPs

**4. Key Rotation**:
- Support multiple active keys
- Force rotation every 90 days
- One-click rotation in dashboard

**5. Usage Monitoring**:
- Alert on unusual patterns (new IP, high volume, off-hours)
- Automatic suspension on suspicious activity
- Anomaly detection

**6. Environment Segregation**:
- Test vs production keys
- Different prefixes (\`sk_test_\`, \`sk_live_\`)
- Test keys can't access production data

**7. Revocation & Audit**:
- Instant revocation capability
- Audit logs of all API key usage
- Track: timestamp, IP, endpoint, response code

**Implementation Example (Stripe-style)**:
- Publishable keys (\`pk_\`) for client-side (limited)
- Secret keys (\`sk_\`) for server-side (full access)
- Restricted keys with custom permissions
- Test mode keys for development

**Response to Breach**:
1. Notify customer immediately
2. Automatic suspension option in dashboard
3. Audit logs show what was accessed
4. Easy key rotation (generate new, test, revoke old)

**Trade-off**: More security features increase complexity but necessary for production APIs. Start simple, add features as you grow.`,
          keyPoints: [
            'Hash API keys before storage (unrecoverable if DB compromised)',
            'Scope limitations reduce damage from compromised keys',
            'IP whitelisting, rate limiting, and monitoring detect misuse',
            'Support multiple keys per account for easy rotation',
            'Audit logs essential for breach investigation',
          ],
        },
      ],
    },
    {
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
      multipleChoice: [
        {
          id: 'error-q1',
          question:
            "A user tries to delete a post they don't own. Which status code and error are most appropriate?",
          options: [
            '401 Unauthorized with "Not authenticated"',
            '403 Forbidden with "You don\'t have permission to delete this post"',
            '404 Not Found with "Post not found"',
            '400 Bad Request with "Invalid post ID"',
          ],
          correctAnswer: 1,
          explanation:
            "403 Forbidden is for authenticated users who lack permission for a specific resource. 401 is for authentication issues, 404 would leak information (post exists but you can't access it), 400 is for malformed requests.",
          difficulty: 'medium',
        },
        {
          id: 'error-q2',
          question:
            'Your API encounters a database connection error. What should you return to the client?',
          options: [
            '500 Internal Server Error with database connection details',
            '500 Internal Server Error with generic message and request ID',
            '503 Service Unavailable with database error message',
            '400 Bad Request with "Database error"',
          ],
          correctAnswer: 1,
          explanation:
            '500 with generic message protects internal details. Include request ID for support to investigate. Never expose database details (security risk). 503 is for planned maintenance/temporary unavailability. 400 is for client errors.',
          difficulty: 'easy',
        },
        {
          id: 'error-q3',
          question:
            'A client exceeds rate limit (1000 req/hour). Which headers should your 429 response include?',
          options: [
            'Only X-RateLimit-Limit header',
            'X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After',
            'Only Retry-After header',
            'No special headers needed with 429',
          ],
          correctAnswer: 1,
          explanation:
            'Provide complete rate limit information: total limit, remaining requests (0), when limit resets (timestamp), and retry-after (seconds). This helps clients implement proper backoff strategies.',
          difficulty: 'medium',
        },
        {
          id: 'error-q4',
          question:
            'User submits registration form with invalid email and short password. Best error response structure?',
          options: [
            'Single error message: "Invalid email or password"',
            'Array of errors with field-level details for each validation failure',
            'Two separate 400 responses (one per field)',
            'Generic "Validation failed" message',
          ],
          correctAnswer: 1,
          explanation:
            "Return field-level details in single response so client can show all errors at once. Users shouldn't fix one field, resubmit, then discover another error. Good UX requires comprehensive validation feedback.",
          difficulty: 'easy',
        },
        {
          id: 'error-q5',
          question:
            'What is the main purpose of idempotency keys in error handling?',
          options: [
            'To encrypt error messages',
            'To allow safe retries of non-idempotent operations',
            'To track error frequency',
            'To validate request authenticity',
          ],
          correctAnswer: 1,
          explanation:
            'Idempotency keys let clients safely retry operations like payments without duplicates. Server caches result by key and returns cached response for retries. This solves network timeout uncertainty ("did my payment go through?").',
          difficulty: 'hard',
        },
      ],
      quiz: [
        {
          id: 'error-d1',
          question:
            "You're designing error responses for a payment API. A credit card is declined. How would you structure the error to be helpful for developers while not exposing sensitive information?",
          sampleAnswer: `Payment errors require balance between helpful information and security:

**Good Error Response**:
\`\`\`json
{
  "error": {
    "type": "card_error",
    "code": "card_declined",
    "message": "Your card was declined",
    "decline_code": "insufficient_funds",
    "payment_intent_id": "pi_abc123"
  }
}
\`\`\`

**Key Design Decisions**:

1. **Specific decline codes without PCI violations**:
   - insufficient_funds
   - card_expired
   - incorrect_cvc
   - processing_error
   - Do NOT include: card number, full name, CVV

2. **User-friendly messages**:
   - "Your card was declined" (not "Transaction failed")
   - Suggest actions: "Please try another card or payment method"

3. **Developer-helpful details**:
   - Payment intent ID for support inquiries
   - Decline code for programmatic handling
   - Timestamp for logging

4. **Security considerations**:
   - Never log full card numbers
   - Mask sensitive data (last 4 digits only)
   - No internal error traces

5. **Retryability indicator**:
\`\`\`json
{
  "error": {
    "code": "card_declined",
    "retryable": false,  // Don't retry insufficient funds
    "next_action": "request_new_payment_method"
  }
}
\`\`\`

**Different Error Types**:
- Network errors: Retryable with idempotency key
- Fraud detection: Generic "declined" (don't reveal fraud logic)
- Temporary issues: Include retry_after
- Permanent issues: Suggest alternative payment methods

**Real-world example (Stripe)**:
They provide detailed decline codes but never expose sensitive data, include charge IDs for support, and clearly indicate retryability.`,
          keyPoints: [
            'Specific error codes without exposing sensitive data',
            'User-friendly messages with actionable guidance',
            'Developer-helpful IDs for support and debugging',
            'Retryability indicators to guide client behavior',
            'Security first: never expose full card data or fraud logic',
          ],
        },
        {
          id: 'error-d2',
          question:
            'Your API has a complex validation rule: "users under 18 can\'t create posts on weekends." How would you structure the validation error, and what status code would you use?',
          sampleAnswer: `Complex validation requires clear, actionable error messages:

**Error Response Design**:

\`\`\`json
HTTP/1.1 403 Forbidden

{
  "error": {
    "code": "OPERATION_NOT_ALLOWED",
    "message": "Users under 18 cannot create posts on weekends",
    "constraints": {
      "minimumAge": 18,
      "currentAge": 16,
      "allowedDays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      "currentDay": "Saturday"
    },
    "availableFrom": "2024-01-22T00:00:00Z",  // Next Monday
    "documentation": "https://docs.api.com/posting-rules"
  }
}
\`\`\`

**Key Decisions**:

**Status Code: 403 Forbidden (not 400)**
- User is authenticated ✓
- Request is well-formed ✓
- Business rule prevents action → 403
- 400 would suggest request format issue

**Error Structure**:

1. **Clear message**: Explains exact constraint
2. **Constraints object**: Machine-readable parameters
3. **Actionable info**: When operation will be available
4. **Documentation link**: Explain business rules

**Alternative Approaches**:

**Option 1: 422 Unprocessable Entity**
\`\`\`json
HTTP/1.1 422 Unprocessable Entity
{
  "error": {
    "code": "BUSINESS_RULE_VIOLATION",
    "message": "Cannot create post: age restriction on weekends"
  }
}
\`\`\`

Debate: 422 vs 403?
- 422: Semantic/business validation error
- 403: Permission/authorization issue
- Both defensible; consistency matters more

**Option 2: Prevent at UI Level**
Best practice: Disable "Create Post" button on weekends for users <18
- Better UX than error message
- Still validate server-side (never trust client)
- Return error if validation bypassed

**Implementation Consideration**:
\`\`\`javascript
// Server-side validation
if (user.age < 18 && isWeekend()) {
  return res.status(403).json({
    error: {
      code: "AGE_RESTRICTED_WEEKEND",
      message: "Users under 18 cannot post on weekends",
      nextAvailable: getNextWeekday(),
      workaround: "Save as draft and publish Monday"
    }
  });
}
\`\`\`

**Client Guidance**:
- Suggest workaround: Save as draft for Monday
- Show countdown: "Available in 23 hours"
- Provide alternative: "Ask parent to post"

Trade-off: Detailed errors help developers but increase response size. Include details for complex rules, keep simple for basic validation.`,
          keyPoints: [
            '403 for business rule violations (authenticated but not allowed)',
            'Clear message explaining specific constraint',
            'Machine-readable constraint details for client logic',
            'Actionable information (when will it be available)',
            'Prevent at UI level but always validate server-side',
          ],
        },
        {
          id: 'error-d3',
          question:
            'Your API experiences an unexpected database timeout. What should you return to clients, and how should you handle this internally for debugging and monitoring?',
          sampleAnswer: `Database timeouts require careful handling for security, debugging, and user experience:

**Client Response**:

\`\`\`json
HTTP/1.1 500 Internal Server Error

{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred. Please try again",
    "requestId": "req_abc123xyz",
    "timestamp": "2024-01-15T10:30:45Z",
    "support": "support@api.com"
  }
}
\`\`\`

**Why Generic?**
- Security: Don't expose database structure
- Simplicity: Users can't fix internal issues
- Support: requestId lets support investigate

**Internal Logging** (Server-Side):

\`\`\`javascript
logger.error({
  errorType: 'DATABASE_TIMEOUT',
  requestId: 'req_abc123xyz',
  userId: user?.id,
  endpoint: '/api/posts',
  method: 'GET',
  query: 'SELECT * FROM posts WHERE ...',
  timeout: 5000,
  actualDuration: 5001,
  database: 'postgres-primary',
  timestamp: '2024-01-15T10:30:45Z',
  stackTrace: error.stack,
  requestHeaders: sanitizedHeaders,
  dbConnectionPool: {
    active: 95,
    max: 100,
    waiting: 23
  }
});
\`\`\`

**Monitoring & Alerting**:

1. **Metrics**:
   - Timeout rate (alert if >1% requests)
   - Database connection pool saturation
   - Query duration p95, p99
   - Error rate by endpoint

2. **Alerts**:
\`\`\`
- Timeout rate > 1% for 5 minutes → Page on-call
- Connection pool > 90% → Warning
- Specific query timeouts → Investigate slow query
\`\`\`

3. **Structured Logging**:
   - Log aggregation (ELK, Splunk)
   - Query by requestId for full trace
   - Correlate with user actions

**Retry Strategy**:

\`\`\`json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Service temporarily unavailable",
    "retryable": true,
    "retryAfter": 5,
    "requestId": "req_abc123"
  }
}
\`\`\`

**Circuit Breaker** (if timeouts persist):

\`\`\`javascript
// After 50% timeout rate for 10 requests
if (circuitBreaker.isOpen()) {
  return 503 Service Unavailable
  // Fail fast instead of waiting for timeout
}
\`\`\`

**Investigation Steps**:

1. Check requestId logs for full context
2. Identify slow query causing timeout
3. Check database load and connections
4. Review recent code deployments
5. Check for missing indexes

**Long-term Solutions**:
- Add database indexes
- Implement query caching
- Add read replicas
- Pagination for large result sets
- Query optimization
- Connection pool tuning

**Status Code Choice**:
- 500: Unexpected server error (timeout)
- 503: If database is down/maintenance
- 504: If timeout from upstream service

Trade-off: Generic client messages vs. debugging needs. Solution: Generic for clients, detailed for internal logs with correlation IDs.`,
          keyPoints: [
            'Return generic 500 error to clients (never expose internals)',
            'Include requestId for support correlation',
            'Log comprehensive details server-side for debugging',
            'Monitor timeout rates and alert on thresholds',
            'Implement circuit breakers to fail fast during outages',
          ],
        },
      ],
    },
    {
      id: 'graphql-schema-design',
      title: 'GraphQL Schema Design',
      content: `GraphQL provides a powerful alternative to REST by letting clients specify exactly what data they need. Designing a good GraphQL schema is crucial for API success.

## What is GraphQL?

**GraphQL** is a query language and runtime for APIs, developed by Facebook in 2012 and open-sourced in 2015.

### **Key Differences from REST**

| Aspect | REST | GraphQL |
|--------|------|---------|
| Endpoints | Multiple (/users, /posts) | Single (/graphql) |
| Data fetching | Fixed structure | Client-specified |
| Over-fetching | Common | None |
| Under-fetching | Common (N+1) | None |
| Versioning | URL versioning | Schema evolution |

## Schema Definition Language (SDL)

### **Object Types**

\`\`\`graphql
type User {
  id: ID!              # ! means non-nullable
  name: String!
  email: String!
  age: Int
  posts: [Post!]!      # Non-null array of non-null Posts
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  published: Boolean!
  author: User!
  comments: [Comment!]!
  tags: [String!]!
}

type Comment {
  id: ID!
  text: String!
  author: User!
  post: Post!
}
\`\`\`

### **Query Type (Read Operations)**

\`\`\`graphql
type Query {
  # Single resource
  user(id: ID!): User
  post(id: ID!): Post
  
  # Lists with filtering
  users(
    limit: Int = 20
    offset: Int = 0
    role: Role
  ): [User!]!
  
  posts(
    authorId: ID
    published: Boolean
    tag: String
    limit: Int = 20
  ): [Post!]!
  
  # Search
  searchUsers(query: String!): [User!]!
}
\`\`\`

### **Mutation Type (Write Operations)**

\`\`\`graphql
type Mutation {
  # Create
  createUser(input: CreateUserInput!): User!
  createPost(input: CreatePostInput!): Post!
  
  # Update
  updateUser(id: ID!, input: UpdateUserInput!): User!
  updatePost(id: ID!, input: UpdatePostInput!): Post!
  
  # Delete
  deleteUser(id: ID!): Boolean!
  deletePost(id: ID!): Boolean!
  
  # Custom operations
  publishPost(id: ID!): Post!
  likePost(postId: ID!): Post!
}
\`\`\`

### **Subscription Type (Real-time)**

\`\`\`graphql
type Subscription {
  postAdded: Post!
  commentAdded(postId: ID!): Comment!
  userStatusChanged(userId: ID!): User!
}
\`\`\`

## Input Types

**Use input types for mutations**:

\`\`\`graphql
input CreateUserInput {
  name: String!
  email: String!
  password: String!
  age: Int
}

input UpdateUserInput {
  name: String
  email: String
  age: Int
}

input CreatePostInput {
  title: String!
  content: String!
  authorId: ID!
  tags: [String!]!
  published: Boolean = false
}
\`\`\`

## Scalar Types

**Built-in scalars**:
- \`Int\`: 32-bit integer
- \`Float\`: Floating-point
- \`String\`: UTF-8 string
- \`Boolean\`: true/false
- \`ID\`: Unique identifier

**Custom scalars**:

\`\`\`graphql
scalar DateTime
scalar Email
scalar URL
scalar JSON

type User {
  email: Email!
  website: URL
  createdAt: DateTime!
  metadata: JSON
}
\`\`\`

## Enums

\`\`\`graphql
enum Role {
  ADMIN
  MODERATOR
  USER
  GUEST
}

enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

type User {
  role: Role!
}

type Post {
  status: PostStatus!
}
\`\`\`

## Interfaces

**For polymorphic types**:

\`\`\`graphql
interface Node {
  id: ID!
  createdAt: DateTime!
}

type User implements Node {
  id: ID!
  createdAt: DateTime!
  name: String!
  email: String!
}

type Post implements Node {
  id: ID!
  createdAt: DateTime!
  title: String!
  content: String!
}
\`\`\`

## Union Types

**For return types that could be multiple types**:

\`\`\`graphql
union SearchResult = User | Post | Comment

type Query {
  search(query: String!): [SearchResult!]!
}

# Client query with inline fragments
query {
  search(query: "graphql") {
    ... on User {
      name
      email
    }
    ... on Post {
      title
      content
    }
    ... on Comment {
      text
    }
  }
}
\`\`\`

## Pagination Patterns

### **Offset-Based**

\`\`\`graphql
type Query {
  posts(limit: Int = 20, offset: Int = 0): PostConnection!
}

type PostConnection {
  totalCount: Int!
  nodes: [Post!]!
  pageInfo: PageInfo!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
}
\`\`\`

### **Cursor-Based (Relay Specification)**

\`\`\`graphql
type Query {
  posts(
    first: Int
    after: String
    last: Int
    before: String
  ): PostConnection!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int
}

type PostEdge {
  cursor: String!
  node: Post!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
\`\`\`

## Error Handling

### **Field-Level Errors**

\`\`\`graphql
type Query {
  user(id: ID!): User  # Returns null if not found
}

# Response with error
{
  "data": {
    "user": null
  },
  "errors": [
    {
      "message": "User not found",
      "path": ["user"],
      "extensions": {
        "code": "NOT_FOUND",
        "userId": "123"
      }
    }
  ]
}
\`\`\`

### **Union Error Types**

\`\`\`graphql
type User {
  id: ID!
  name: String!
}

type NotFoundError {
  message: String!
  code: String!
}

type ValidationError {
  message: String!
  field: String!
}

union UserResult = User | NotFoundError | ValidationError

type Query {
  user(id: ID!): UserResult!
}
\`\`\`

## Directives

**Built-in directives**:

\`\`\`graphql
query GetUser($includeEmail: Boolean!) {
  user(id: "123") {
    name
    email @include(if: $includeEmail)
    age @skip(if: $includeEmail)
  }
}
\`\`\`

**Custom directives**:

\`\`\`graphql
directive @auth(requires: Role = USER) on FIELD_DEFINITION
directive @rateLimit(limit: Int!) on FIELD_DEFINITION
directive @deprecated(reason: String) on FIELD_DEFINITION

type Query {
  users: [User!]! @auth(requires: ADMIN)
  posts: [Post!]! @rateLimit(limit: 100)
  oldField: String @deprecated(reason: "Use newField instead")
}
\`\`\`

## Schema Design Best Practices

### **1. Nullable vs Non-Nullable**

\`\`\`graphql
❌ Too strict (breaks client if field added):
type User {
  id: ID!
  name: String!
  email: String!
  phone: String!  # New required field breaks old clients
}

✅ Better (allows evolution):
type User {
  id: ID!
  name: String!
  email: String!
  phone: String    # Nullable, backward compatible
}
\`\`\`

### **2. Design for Client Needs**

\`\`\`graphql
# Client need: User profile page
query UserProfile {
  user(id: "123") {
    name
    avatar
    bio
    stats {
      postCount
      followerCount
    }
    recentPosts(limit: 5) {
      title
      createdAt
    }
  }
}
\`\`\`

### **3. Avoid N+1 Queries (Use DataLoader)**

\`\`\`graphql
type Post {
  author: User!  # Could cause N+1 queries
}

# Resolver with DataLoader
const resolvers = {
  Post: {
    author: (post, args, { loaders }) => {
      return loaders.user.load(post.authorId);
    }
  }
};
\`\`\`

### **4. Connection Pattern for Lists**

Use consistent pagination pattern:

\`\`\`graphql
type Query {
  users(first: Int, after: String): UserConnection!
  posts(first: Int, after: String): PostConnection!
}
\`\`\`

### **5. Descriptive Names**

\`\`\`graphql
❌ Bad:
type Query {
  get(id: ID!): User
  list: [User!]!
}

✅ Good:
type Query {
  user(id: ID!): User
  users(limit: Int): [User!]!
}
\`\`\`

## Real-World Example: GitHub GraphQL API

\`\`\`graphql
query {
  repository(owner: "facebook", name: "react") {
    name
    description
    stargazerCount
    issues(first: 10, states: OPEN) {
      edges {
        node {
          title
          author {
            login
            avatarUrl
          }
        }
      }
    }
  }
}
\`\`\`

## When to Use GraphQL vs REST

**Use GraphQL when**:
- Multiple client types (web, mobile, desktop) with different needs
- Complex, nested data requirements
- Rapid frontend iteration
- Over/under-fetching is a problem

**Use REST when**:
- Simple CRUD operations
- File uploads/downloads (simpler in REST)
- Caching is critical (HTTP caching)
- Team unfamiliar with GraphQL
- Existing REST infrastructure

## Schema Evolution

**Adding fields**: Safe (backward compatible)
\`\`\`graphql
type User {
  id: ID!
  name: String!
  email: String!  # New field - safe
}
\`\`\`

**Deprecating fields**:
\`\`\`graphql
type User {
  oldField: String @deprecated(reason: "Use newField")
  newField: String!
}
\`\`\`

**Removing fields**: Breaking change (give deprecation period)`,
      multipleChoice: [
        {
          id: 'graphql-q1',
          question: 'What is the main advantage of GraphQL over REST APIs?',
          options: [
            'GraphQL is faster because it uses binary protocol',
            'Clients can request exactly the data they need, avoiding over/under-fetching',
            'GraphQL automatically generates database queries',
            "GraphQL doesn't require authentication",
          ],
          correctAnswer: 1,
          explanation:
            "GraphQL's main advantage is that clients specify exactly what data they need in their query, eliminating over-fetching (getting unnecessary data) and under-fetching (needing multiple requests). GraphQL uses HTTP (not binary), doesn't auto-generate database queries, and still requires authentication.",
          difficulty: 'easy',
        },
        {
          id: 'graphql-q2',
          question: 'In GraphQL schema, what does the "!" symbol mean?',
          options: [
            'The field is deprecated',
            'The field requires authentication',
            'The field is non-nullable (required)',
            'The field is unique',
          ],
          correctAnswer: 2,
          explanation:
            'The "!" symbol indicates a field is non-nullable, meaning it must always have a value. "String!" means a required string, "String" is optional. "[Post!]!" means a non-null array containing non-null Posts.',
          difficulty: 'easy',
        },
        {
          id: 'graphql-q3',
          question:
            'What is the N+1 query problem in GraphQL, and how is it solved?',
          options: [
            'Making N+1 database queries for a list; solved with DataLoader batching',
            'Requesting N+1 fields; solved with query depth limiting',
            'N+1 concurrent requests; solved with rate limiting',
            'N+1 schema versions; solved with versioning',
          ],
          correctAnswer: 0,
          explanation:
            'N+1 problem: Fetching a list of N items, then making 1 query per item for related data (N+1 total queries). Example: fetching 10 posts, then 10 separate queries for each author. DataLoader batches and caches these queries, turning N+1 into 2 queries.',
          difficulty: 'hard',
        },
        {
          id: 'graphql-q4',
          question:
            'Which pagination approach follows the Relay specification for GraphQL?',
          options: [
            'Offset-based with page numbers',
            'Cursor-based with edges, nodes, and pageInfo',
            'Limit-offset with totalCount',
            'Page-based with hasNextPage',
          ],
          correctAnswer: 1,
          explanation:
            'Relay specification uses cursor-based pagination with specific structure: edges (array of {cursor, node}), nodes (actual data), and pageInfo (hasNextPage, hasPreviousPage, startCursor, endCursor). This provides consistent, efficient pagination.',
          difficulty: 'medium',
        },
        {
          id: 'graphql-q5',
          question:
            'When should you make a GraphQL field nullable vs non-nullable?',
          options: [
            'Always make fields non-nullable for data integrity',
            'Make fields nullable to allow schema evolution without breaking clients',
            'Nullable only for optional user input',
            'Non-nullable only for IDs and timestamps',
          ],
          correctAnswer: 1,
          explanation:
            'Nullable fields allow schema evolution - you can add new fields without breaking existing clients. If you add a required field later, old data might not have it. Core fields (id, createdAt) can be non-nullable, but most fields should be nullable for flexibility. Only inputs should match business requirements strictly.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'graphql-d1',
          question:
            "You're designing a GraphQL schema for a social media app with users, posts, comments, and likes. How would you structure the schema, including pagination and error handling?",
          sampleAnswer: `Comprehensive social media GraphQL schema design:

\`\`\`graphql
# ========== Core Types ==========

type User {
  id: ID!
  username: String!
  email: String
  avatar: URL
  bio: String
  createdAt: DateTime!
  
  # Stats
  stats: UserStats!
  
  # Relationships (paginated)
  posts(first: Int = 20, after: String): PostConnection!
  followers(first: Int = 20): UserConnection!
  following(first: Int = 20): UserConnection!
  
  # Computed fields
  isFollowing: Boolean!  # Does current user follow this user?
  isFollowedBy: Boolean!
}

type UserStats {
  postCount: Int!
  followerCount: Int!
  followingCount: Int!
  likeCount: Int!
}

type Post {
  id: ID!
  content: String!
  imageUrls: [URL!]!
  author: User!
  createdAt: DateTime!
  updatedAt: DateTime
  
  # Stats
  likeCount: Int!
  commentCount: Int!
  
  # Relationships
  comments(first: Int = 10, after: String): CommentConnection!
  likes(first: Int = 10): [User!]!
  
  # Computed
  isLikedByMe: Boolean!
}

type Comment {
  id: ID!
  text: String!
  author: User!
  post: Post!
  createdAt: DateTime!
  
  # Nested comments
  replies(first: Int = 5): [Comment!]!
  replyCount: Int!
}

# ========== Connections (Relay-style) ==========

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int
}

type UserEdge {
  cursor: String!
  node: User!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int
}

type PostEdge {
  cursor: String!
  node: Post!
}

type CommentConnection {
  edges: [CommentEdge!]!
  pageInfo: PageInfo!
}

type CommentEdge {
  cursor: String!
  node: Comment!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# ========== Inputs ==========

input CreatePostInput {
  content: String!
  imageUrls: [URL!]!
}

input UpdatePostInput {
  content: String
  imageUrls: [URL!]
}

input CreateCommentInput {
  postId: ID!
  text: String!
  parentId: ID  # For nested replies
}

# ========== Queries ==========

type Query {
  # Current user
  me: User
  
  # Single resources
  user(id: ID, username: String): UserResult!
  post(id: ID!): PostResult!
  
  # Feeds
  feed(first: Int = 20, after: String): PostConnection!
  explorePosts(first: Int = 20): PostConnection!
  
  # Search
  searchUsers(query: String!, first: Int = 20): [User!]!
  searchPosts(query: String!, first: Int = 20): [Post!]!
}

# ========== Mutations ==========

type Mutation {
  # Posts
  createPost(input: CreatePostInput!): PostResult!
  updatePost(id: ID!, input: UpdatePostInput!): PostResult!
  deletePost(id: ID!): DeleteResult!
  
  # Comments
  addComment(input: CreateCommentInput!): CommentResult!
  deleteComment(id: ID!): DeleteResult!
  
  # Social actions
  likePost(postId: ID!): Post!
  unlikePost(postId: ID!): Post!
  followUser(userId: ID!): User!
  unfollowUser(userId: ID!): User!
}

# ========== Subscriptions ==========

type Subscription {
  postAdded(authorId: ID): Post!
  commentAdded(postId: ID!): Comment!
  postLiked(postId: ID!): Post!
}

# ========== Error Handling (Union Types) ==========

type NotFoundError {
  message: String!
  code: String!
  resourceType: String!
  resourceId: ID!
}

type UnauthorizedError {
  message: String!
  code: String!
}

type ValidationError {
  message: String!
  code: String!
  fields: [FieldError!]!
}

type FieldError {
  field: String!
  message: String!
}

union UserResult = User | NotFoundError | UnauthorizedError
union PostResult = Post | NotFoundError | ValidationError
union CommentResult = Comment | NotFoundError | ValidationError

type DeleteResult {
  success: Boolean!
  message: String
}

# ========== Custom Scalars ==========

scalar DateTime
scalar URL

# ========== Enums ==========

enum PostSortOrder {
  RECENT
  POPULAR
  TRENDING
}
\`\`\`

**Key Design Decisions**:

1. **Pagination**: Relay-style cursors for infinite scroll
2. **Stats aggregation**: Separate UserStats type (can be cached)
3. **Computed fields**: isFollowing, isLikedByMe (requires auth context)
4. **Union error types**: Type-safe error handling
5. **Nested comments**: Self-referential with replies
6. **Real-time**: Subscriptions for live updates

**DataLoader Implementation** (solving N+1):
\`\`\`javascript
const loaders = {
  user: new DataLoader(ids => batchGetUsers(ids)),
  postLikes: new DataLoader(postIds => batchGetLikes(postIds))
};
\`\`\`

This schema balances client flexibility, performance (batching), and type safety.`,
          keyPoints: [
            'Use Relay-style connections for consistent pagination',
            'Separate stats into dedicated types for caching',
            'Union types for type-safe error handling',
            'Computed fields (isFollowing) require auth context',
            'DataLoader prevents N+1 query problems',
          ],
        },
        {
          id: 'graphql-d2',
          question:
            'Your GraphQL API is experiencing performance issues with complex nested queries. How would you implement query complexity analysis and depth limiting?',
          sampleAnswer: `GraphQL's flexibility can be exploited with malicious/expensive queries. Implement protection:

**Problem - Malicious Query**:

\`\`\`graphql
query Attack {
  users(first: 1000) {
    posts(first: 1000) {
      comments(first: 1000) {
        author {
          posts(first: 1000) {
            # Could request billions of records
          }
        }
      }
    }
  }
}
\`\`\`

**Solution 1: Query Depth Limiting**

\`\`\`javascript
import { createComplexityLimitRule } from 'graphql-validation-complexity';

const depthLimit = 7;  // Maximum nesting depth

const validationRules = [
  depthLimit(7, {
    ignore: ['IntrospectionQuery']  // Allow introspection
  })
];

// Blocks queries deeper than 7 levels
\`\`\`

**Solution 2: Query Complexity Analysis**

Assign complexity scores to fields:

\`\`\`javascript
import { createComplexityLimitRule } from 'graphql-query-complexity';

const schema = buildSchema(\`
  type Query {
    users(first: Int = 20): [User!]!  # Complexity: first * 1
  }
  
  type User {
    posts(first: Int = 20): [Post!]!  # Complexity: first * 2
  }
\`);

// Calculate complexity
const complexity = getComplexity({
  schema,
  query: parsedQuery,
  variables,
  estimators: [
    fieldExtensionsEstimator(),
    simpleEstimator({ defaultComplexity: 1 })
  ]
});

if (complexity > 1000) {
  throw new Error(\`Query too complex: \${complexity}\`);
}
\`\`\`

**Complexity Calculation Example**:

\`\`\`graphql
query {
  users(first: 10) {        # 10 * 1 = 10
    posts(first: 5) {       # 10 * 5 * 2 = 100
      comments(first: 3) {  # 10 * 5 * 3 * 1 = 150
        text
      }
    }
  }
}
# Total complexity: 10 + 100 + 150 = 260
\`\`\`

**Solution 3: Pagination Limits**

\`\`\`javascript
const resolvers = {
  Query: {
    users: (_, { first = 20 }) => {
      if (first > 100) {
        throw new Error('Cannot request more than 100 users');
      }
      return getUsers(first);
    }
  }
};
\`\`\`

**Solution 4: Query Timeout**

\`\`\`javascript
const apolloServer = new ApolloServer({
  schema,
  plugins: [
    {
      requestDidStart() {
        return {
          executionDidStart() {
            const timeout = setTimeout(() => {
              throw new Error('Query timeout after 5s');
            }, 5000);
            
            return {
              executionDidEnd() {
                clearTimeout(timeout);
              }
            };
          }
        };
      }
    }
  ]
});
\`\`\`

**Solution 5: Query Allow List (Persisted Queries)**

Only allow pre-approved queries:

\`\`\`javascript
const allowedQueries = {
  'GetUser': \`query GetUser($id: ID!) { user(id: $id) { name } }\`,
  'GetFeed': \`query GetFeed { feed(first: 20) { ... } }\`
};

app.use('/graphql', (req, res) => {
  const { queryId, variables } = req.body;
  const query = allowedQueries[queryId];
  
  if (!query) {
    return res.status(400).json({ error: 'Unknown query' });
  }
  
  // Execute only pre-approved queries
});
\`\`\`

**Solution 6: Rate Limiting by Complexity**

\`\`\`javascript
// Track complexity per user/IP
const complexityBudget = new Map();  // user -> remaining complexity

const MAX_COMPLEXITY_PER_HOUR = 10000;

function checkComplexityBudget(userId, queryComplexity) {
  const remaining = complexityBudget.get(userId) || MAX_COMPLEXITY_PER_HOUR;
  
  if (queryComplexity > remaining) {
    throw new Error('Complexity budget exceeded');
  }
  
  complexityBudget.set(userId, remaining - queryComplexity);
}
\`\`\`

**Best Practice Stack**:

1. **Depth limit**: 7-10 levels (prevents infinite nesting)
2. **Complexity analysis**: 1000 points per query
3. **Pagination limits**: Max 100 items per connection
4. **Query timeout**: 5-10 seconds
5. **Rate limiting**: By complexity, not just request count
6. **Monitoring**: Log expensive queries for optimization

**Trade-off**: Balance flexibility vs. protection. Start permissive, tighten based on abuse patterns.`,
          keyPoints: [
            'Query depth limiting prevents deeply nested queries',
            'Complexity analysis assigns costs to fields and limits total',
            'Pagination caps prevent large data fetches',
            'Persisted queries (allow list) for production clients',
            'Rate limit by complexity points, not just request count',
          ],
        },
        {
          id: 'graphql-d3',
          question:
            'When should you choose GraphQL over REST, and when should you stick with REST? Provide specific scenarios and trade-offs.',
          sampleAnswer: `Choosing between GraphQL and REST depends on specific use cases and constraints:

**Choose GraphQL When:**

**1. Multiple Client Types with Different Needs**
Scenario: iOS app, Android app, web app, and admin dashboard all need different data.

REST problem:
\`\`\`
GET /users/123  → Returns everything (web needs 20 fields)
                 → Mobile needs only 5 fields (wasted bandwidth)
                 → Admin needs additional permissions fields
\`\`\`

GraphQL solution:
\`\`\`graphql
# Mobile
query { user(id: "123") { id name avatar }}

# Web
query { user(id: "123") { id name avatar bio stats posts {...}}}

# Admin
query { user(id: "123") { id name email role permissions }}
\`\`\`

**2. Complex, Nested Data Requirements**
Scenario: E-commerce product page needs product, reviews, seller, related products, seller's other products.

REST: 5+ requests
GraphQL: 1 request with exact data shape

**3. Rapid Frontend Iteration**
Scenario: Startup with changing product requirements.
- GraphQL: Frontend can request new fields without backend changes
- REST: Need new endpoints or versions for new data needs

**Choose REST When:**

**1. Simple CRUD Operations**
Scenario: Basic blog with posts and comments.
\`\`\`
GET    /posts
POST   /posts
GET    /posts/123
PUT    /posts/123
DELETE /posts/123
\`\`\`
REST is simpler, leverages HTTP caching, well-understood.

**2. File Uploads/Downloads**
REST:
\`\`\`
POST /upload
Content-Type: multipart/form-data
\`\`\`

GraphQL: Requires additional libraries (Apollo Upload), more complex.

**3. HTTP Caching is Critical**
Scenario: Public API with mostly read operations.

REST: Native HTTP caching (CDN, browser cache)
\`\`\`
GET /users/123
Cache-Control: max-age=3600
\`\`\`

GraphQL: POST requests, can't leverage HTTP caching easily

**4. Team Unfamiliar with GraphQL**
Scenario: Team experienced with REST, tight deadlines.
- REST: Lower learning curve, faster initial development
- GraphQL: Learning curve for schema design, resolvers, DataLoader

**Real-World Hybrid Example: GitHub**

GitHub offers BOTH REST and GraphQL APIs:
- **REST API**: Simple operations, webhooks, OAuth
- **GraphQL API**: Complex queries, mobile apps, flexible data needs

\`\`\`
# REST for simple operations
GET /repos/facebook/react

# GraphQL for complex data
query {
  repository(owner: "facebook", name: "react") {
    issues(first: 10) {
      edges {
        node {
          title
          author { login }
          comments(first: 5) { totalCount }
        }
      }
    }
  }
}
\`\`\`

**Trade-off Summary**:

| Factor | REST | GraphQL |
|--------|------|---------|
| Learning curve | Low | Medium-High |
| Caching | Native HTTP | Complex (Apollo, etc.) |
| Over/under-fetching | Common | None |
| API versioning | URL-based | Schema evolution |
| File uploads | Simple | Complex |
| Real-time | SSE/WebSockets | Subscriptions (WebSockets) |
| Tooling | Mature (Postman, Swagger) | Growing (GraphiQL, Playground) |
| Mobile performance | Wasted bandwidth | Optimized queries |
| Backend complexity | Lower | Higher (resolvers, N+1, complexity) |

**Recommendation Strategy**:

1. **Start with REST** if:
   - MVP/prototype
   - Simple domain
   - Public read-heavy API
   - Team inexperienced with GraphQL

2. **Choose GraphQL** if:
   - Multiple client types
   - Complex data relationships
   - Mobile-first (bandwidth matters)
   - Rapid product iteration

3. **Hybrid Approach** (like GitHub):
   - REST for simple operations
   - GraphQL for complex queries
   - Best of both worlds

**Migration Path**: Start with REST, add GraphQL layer on top when complexity grows. Many companies (Shopify, GitHub, Facebook) offer both.`,
          keyPoints: [
            'GraphQL excels with multiple clients needing different data',
            'REST is simpler for CRUD, file uploads, and HTTP caching',
            'GraphQL has learning curve but pays off at scale',
            'Hybrid approach (both REST and GraphQL) is viable',
            'Choose based on team expertise, use case, and complexity',
          ],
        },
      ],
    },
    {
      id: 'graphql-performance',
      title: 'GraphQL Performance',
      content: `GraphQL's flexibility creates unique performance challenges. Understanding and optimizing query execution is critical for production systems.

## The N+1 Query Problem

### **Problem Example**

\`\`\`graphql
query {
  posts {           # 1 query: SELECT * FROM posts
    title
    author {        # N queries: SELECT * FROM users WHERE id = ?
      name          # (one per post)
    }
  }
}

# Result: 1 + N queries (if 100 posts, 101 total queries!)
\`\`\`

### **Solution: DataLoader**

**DataLoader** batches and caches requests within a single request context.

\`\`\`javascript
const DataLoader = require('dataloader');

// Create DataLoader
const userLoader = new DataLoader(async (userIds) => {
  // Batch load all users at once
  const users = await db.users.findAll({
    where: { id: userIds }
  });
  
  // Return in same order as requested IDs
  return userIds.map(id => 
    users.find(user => user.id === id)
  );
});

// Resolver using DataLoader
const resolvers = {
  Post: {
    author: (post, args, context) => {
      return context.loaders.user.load(post.authorId);
    }
  }
};

// Result: 2 queries total (1 for posts, 1 batched for all users)
\`\`\`

**How DataLoader Works**:
1. Collects all \`.load()\` calls in current tick
2. Batches them into single query
3. Caches results for request duration
4. Returns data in correct order

## Caching Strategies

### **1. Server-Side Caching**

**In-Memory Cache (Redis)**:

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // Check cache first
      const cached = await client.get(\`user:\${id}\`);
      if (cached) return JSON.parse(cached);
      
      // Query database
      const user = await db.users.findById(id);
      
      // Cache for 5 minutes
      await client.setex(
        \`user:\${id}\`,
        300,
        JSON.stringify(user)
      );
      
      return user;
    }
  }
};
\`\`\`

### **2. Automatic Persisted Queries (APQ)**

Save bandwidth by sending query hash instead of full query:

\`\`\`javascript
// Client sends hash on first request
{
  "extensions": {
    "persistedQuery": {
      "version": 1,
      "sha256Hash": "abc123..."
    }
  }
}

// Server doesn't have it, returns error
// Client resends with full query + hash
// Server caches query by hash

// Future requests only send hash (saves bandwidth)
\`\`\`

### **3. Response Caching**

\`\`\`javascript
const { ApolloServer } = require('apollo-server');
const responseCachePlugin = require('apollo-server-plugin-response-cache');

const server = new ApolloServer({
  schema,
  plugins: [
    responseCachePlugin({
      // Cache responses for 5 minutes
      sessionId: (requestContext) => 
        requestContext.request.http.headers.get('session-id'),
    })
  ],
  cacheControl: {
    defaultMaxAge: 300
  }
});

// Schema-level cache hints
type Query {
  user(id: ID!): User @cacheControl(maxAge: 300)
  posts: [Post!]! @cacheControl(maxAge: 60)
}
\`\`\`

## Query Complexity and Cost Analysis

### **Assign Costs to Fields**

\`\`\`javascript
const { getComplexity, simpleEstimator } = require('graphql-query-complexity');

const complexity = getComplexity({
  schema,
  query: parsedQuery,
  variables,
  estimators: [
    // Custom complexity per field
    {
      Query: {
        posts: ({ args }) => args.first * 10  // 10 points per post
      },
      Post: {
        comments: ({ args }) => args.first * 5  // 5 points per comment
      }
    },
    simpleEstimator({ defaultComplexity: 1 })
  ]
});

if (complexity > 1000) {
  throw new Error(\`Query too complex: \${complexity}\`);
}
\`\`\`

### **Query Depth Limiting**

\`\`\`javascript
const depthLimit = require('graphql-depth-limit');

const server = new ApolloServer({
  schema,
  validationRules: [depthLimit(7)]  // Max 7 levels deep
});
\`\`\`

## Pagination Best Practices

### **Cursor-Based Pagination (Efficient)**

\`\`\`javascript
const resolvers = {
  Query: {
    posts: async (_, { first = 20, after }) => {
      // Decode cursor
      const cursor = after ? decodeCursor(after) : null;
      
      // Efficient query using WHERE clause
      const posts = await db.posts.findAll({
        where: cursor ? { id: { $gt: cursor.id } } : {},
        order: [['id', 'ASC']],
        limit: first + 1  // Fetch one extra for hasNextPage
      });
      
      const hasNextPage = posts.length > first;
      const edges = posts.slice(0, first);
      
      return {
        edges: edges.map(post => ({
          cursor: encodeCursor({ id: post.id }),
          node: post
        })),
        pageInfo: {
          hasNextPage,
          endCursor: edges.length > 0 
            ? encodeCursor({ id: edges[edges.length - 1].id })
            : null
        }
      };
    }
  }
};
\`\`\`

## Batching Strategies

### **1. Query Batching**

Combine multiple queries into one request:

\`\`\`javascript
// Apollo Client automatically batches
const link = new BatchHttpLink({
  uri: '/graphql',
  batchMax: 10,  // Max queries per batch
  batchInterval: 20  // Wait 20ms before sending
});

// Multiple queries sent together
query1: { user(id: 1) { name }}
query2: { user(id: 2) { name }}
query3: { posts { title }}

// Server receives and processes as batch
\`\`\`

### **2. Field-Level Batching (DataLoader)**

Already covered, but critical for performance.

## Monitoring and Profiling

### **Apollo Studio Integration**

\`\`\`javascript
const { ApolloServer } = require('apollo-server');
const { ApolloServerPluginUsageReporting } = require('apollo-server-core');

const server = new ApolloServer({
  schema,
  plugins: [
    ApolloServerPluginUsageReporting({
      sendVariableValues: { all: true },
      sendHeaders: { all: true }
    })
  ]
});
\`\`\`

**Metrics tracked**:
- Query execution time
- Resolver performance
- Error rates
- Query patterns

### **Custom Logging**

\`\`\`javascript
const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart(requestContext) {
      const start = Date.now();
      
      return {
        didEncounterErrors(ctx) {
          console.error('Query errors:', ctx.errors);
        },
        willSendResponse(ctx) {
          const duration = Date.now() - start;
          console.log({
            query: ctx.request.query,
            duration,
            variables: ctx.request.variables
          });
        }
      };
    }
  }]
});
\`\`\`

## Optimization Techniques

### **1. Selective Field Resolvers**

Only resolve fields that are requested:

\`\`\`javascript
const resolvers = {
  User: {
    // Expensive operation - only runs if requested
    posts: async (user, args, context, info) => {
      // Check if field was requested
      if (!info) return null;
      
      return context.loaders.userPosts.load(user.id);
    }
  }
};
\`\`\`

### **2. Parallel Resolver Execution**

\`\`\`javascript
const resolvers = {
  Query: {
    dashboard: async () => {
      // Run queries in parallel
      const [user, posts, stats] = await Promise.all([
        db.users.findById(userId),
        db.posts.findAll({ userId }),
        db.stats.get(userId)
      ]);
      
      return { user, posts, stats };
    }
  }
};
\`\`\`

### **3. Database Query Optimization**

\`\`\`javascript
// Bad: Lazy loading (N+1)
const posts = await Post.findAll();
for (const post of posts) {
  post.author = await User.findById(post.authorId);  // N queries
}

// Good: Eager loading
const posts = await Post.findAll({
  include: [{ model: User, as: 'author' }]  // 1 JOIN query
});
\`\`\`

## Real-World Example: Shopify

Shopify's GraphQL API uses:
- **Cost analysis**: Queries have point budgets
- **Throttling**: Rate limiting by query cost
- **Caching**: Aggressive CDN caching
- **DataLoader**: Extensive batching

**Query Cost Example**:
\`\`\`graphql
query {
  products(first: 10) {  # Cost: 10
    edges {
      node {
        variants(first: 5) {  # Cost per product: 5
          edges {
            node {
              price
            }
          }
        }
      }
    }
  }
}
# Total cost: 10 + (10 * 5) = 60 points
\`\`\`

## Best Practices Summary

1. **Always use DataLoader** for relationships
2. **Implement query complexity** limits
3. **Cache aggressively** at multiple levels
4. **Monitor performance** with Apollo Studio
5. **Set pagination limits** (max 100 items)
6. **Use cursor pagination** for scale
7. **Batch database queries** when possible
8. **Profile slow queries** and optimize
9. **Consider APQ** for mobile clients
10. **Depth limit** queries (7-10 levels)`,
      multipleChoice: [
        {
          id: 'graphql-perf-q1',
          question:
            'What is the N+1 query problem in GraphQL, and how does DataLoader solve it?',
          options: [
            'N+1 network requests; DataLoader combines them into one HTTP request',
            'N+1 database queries for fetching related data; DataLoader batches them',
            'N+1 schema definitions; DataLoader merges them',
            'N+1 validation errors; DataLoader catches them',
          ],
          correctAnswer: 1,
          explanation:
            'N+1 problem: Fetching N items then making 1 query per item for related data (N+1 total queries). Example: 100 posts → 1 query for posts + 100 queries for authors. DataLoader batches these into 2 queries total: one for posts, one batched query for all authors.',
          difficulty: 'medium',
        },
        {
          id: 'graphql-perf-q2',
          question:
            "Why can't GraphQL leverage HTTP caching as easily as REST APIs?",
          options: [
            "GraphQL doesn't support HTTP headers",
            "GraphQL typically uses POST requests which aren't cached by default",
            'GraphQL responses are always unique per user',
            'GraphQL uses WebSockets instead of HTTP',
          ],
          correctAnswer: 1,
          explanation:
            "GraphQL typically uses POST requests (to send query in body), and POST requests aren't cached by browsers/CDNs by default. REST GET requests have built-in HTTP caching. GraphQL requires custom caching solutions like APQ (Automatic Persisted Queries) or Redis.",
          difficulty: 'medium',
        },
        {
          id: 'graphql-perf-q3',
          question:
            'What is the purpose of query complexity analysis in GraphQL?',
          options: [
            'To validate query syntax',
            'To prevent expensive queries by assigning cost limits',
            'To optimize database indexes automatically',
            'To generate TypeScript types',
          ],
          correctAnswer: 1,
          explanation:
            'Query complexity analysis assigns "cost" to each field and rejects queries exceeding a threshold. This prevents malicious or accidentally expensive queries (e.g., fetching 1000 posts × 1000 comments = 1M records). Each field gets a complexity score.',
          difficulty: 'easy',
        },
        {
          id: 'graphql-perf-q4',
          question:
            'When implementing cursor-based pagination, why fetch limit + 1 items?',
          options: [
            'To have a backup item in case one is deleted',
            "To determine if there's a next page (hasNextPage)",
            'To calculate total count efficiently',
            'To improve query performance',
          ],
          correctAnswer: 1,
          explanation:
            'Fetching limit + 1 items lets you check hasNextPage: if you get more than requested, there are more pages. Return only "limit" items to client, but the extra item tells you hasNextPage = true. Avoids expensive COUNT queries.',
          difficulty: 'hard',
        },
        {
          id: 'graphql-perf-q5',
          question:
            'What is Automatic Persisted Queries (APQ) and why is it useful?',
          options: [
            'Automatically saves queries to database for audit',
            'Caches queries by hash, reducing bandwidth on repeat requests',
            'Persists subscriptions across server restarts',
            'Automatically generates query documentation',
          ],
          correctAnswer: 1,
          explanation:
            'APQ sends query hash instead of full query text after first request. Server caches query by hash. Subsequent requests send only hash (small), saving bandwidth especially for mobile. First request: full query + hash. Subsequent: just hash.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'graphql-perf-d1',
          question:
            'Your GraphQL API experiences performance degradation when users request deeply nested data. Implement a comprehensive solution including DataLoader, caching, and query limits.',
          sampleAnswer: `Comprehensive GraphQL performance optimization strategy:

**1. DataLoader for N+1 Prevention**

\`\`\`javascript
// Setup DataLoaders
const createLoaders = (db) => ({
  user: new DataLoader(async (ids) => {
    const users = await db.users.findAll({
      where: { id: ids }
    });
    return ids.map(id => users.find(u => u.id === id));
  }),
  
  postsByUser: new DataLoader(async (userIds) => {
    const posts = await db.posts.findAll({
      where: { authorId: userIds }
    });
    // Group by userId
    return userIds.map(userId => 
      posts.filter(p => p.authorId === userId)
    );
  }),
  
  commentsByPost: new DataLoader(async (postIds) => {
    const comments = await db.comments.findAll({
      where: { postId: postIds }
    });
    return postIds.map(postId =>
      comments.filter(c => c.postId === postId)
    );
  })
});

// Add to context
const server = new ApolloServer({
  schema,
  context: ({ req }) => ({
    loaders: createLoaders(db),
    user: req.user
  })
});

// Use in resolvers
const resolvers = {
  Post: {
    author: (post, _, { loaders }) => 
      loaders.user.load(post.authorId),
    comments: (post, _, { loaders }) =>
      loaders.commentsByPost.load(post.id)
  }
};
\`\`\`

**2. Query Complexity Limiting**

\`\`\`javascript
const { getComplexity, simpleEstimator } = require('graphql-query-complexity');

const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart() {
      return {
        didResolveOperation({ request, document }) {
          const complexity = getComplexity({
            schema,
            query: document,
            variables: request.variables,
            estimators: [
              {
                Query: {
                  posts: ({ args }) => args.first * 5,
                  users: ({ args }) => args.first * 3
                },
                Post: {
                  comments: ({ args }) => (args.first || 10) * 2
                },
                User: {
                  posts: ({ args }) => (args.first || 20) * 5
                }
              },
              simpleEstimator({ defaultComplexity: 1 })
            ]
          });
          
          if (complexity > 1000) {
            throw new Error(
              \`Query too complex: \${complexity}. Max: 1000\`
            );
          }
        }
      };
    }
  }]
});
\`\`\`

**3. Depth Limiting**

\`\`\`javascript
const depthLimit = require('graphql-depth-limit');

const server = new ApolloServer({
  schema,
  validationRules: [
    depthLimit(7, {
      ignore: ['IntrospectionQuery']
    })
  ]
});
\`\`\`

**4. Response Caching (Redis)**

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

const cacheResolver = async (resolve, root, args, context, info) => {
  // Create cache key from query + variables
  const key = \`gql:\${info.fieldName}:\${JSON.stringify(args)}\`;
  
  // Check cache
  const cached = await client.get(key);
  if (cached) {
    return JSON.parse(cached);
  }
  
  // Execute resolver
  const result = await resolve(root, args, context, info);
  
  // Cache for 5 minutes
  await client.setex(key, 300, JSON.stringify(result));
  
  return result;
};

// Apply to expensive queries
const resolvers = {
  Query: {
    trendingPosts: (root, args, context, info) =>
      cacheResolver(
        getTrendingPosts,
        root,
        args,
        context,
        info
      )
  }
};
\`\`\`

**5. Pagination Limits**

\`\`\`javascript
const MAX_PAGE_SIZE = 100;

const resolvers = {
  Query: {
    posts: async (_, { first = 20, after }) => {
      if (first > MAX_PAGE_SIZE) {
        throw new Error(\`Cannot request more than \${MAX_PAGE_SIZE} items\`);
      }
      
      const cursor = after ? decodeCursor(after) : null;
      const posts = await db.posts.findAll({
        where: cursor ? { id: { $gt: cursor.id } } : {},
        limit: first + 1
      });
      
      const hasNextPage = posts.length > first;
      const edges = posts.slice(0, first);
      
      return {
        edges: edges.map(post => ({
          cursor: encodeCursor({ id: post.id }),
          node: post
        })),
        pageInfo: {
          hasNextPage,
          endCursor: edges.length > 0 
            ? encodeCursor({ id: edges[edges.length - 1].id })
            : null
        }
      };
    }
  }
};
\`\`\`

**6. Query Timeout**

\`\`\`javascript
const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart() {
      return {
        executionDidStart() {
          const timeout = setTimeout(() => {
            throw new Error('Query timeout after 10 seconds');
          }, 10000);
          
          return {
            executionDidEnd() {
              clearTimeout(timeout);
            }
          };
        }
      };
    }
  }]
});
\`\`\`

**7. Monitoring**

\`\`\`javascript
const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart() {
      const start = Date.now();
      
      return {
        willSendResponse({ request, response }) {
          const duration = Date.now() - start;
          
          // Log slow queries
          if (duration > 1000) {
            logger.warn({
              query: request.query,
              variables: request.variables,
              duration,
              complexity: response.extensions?.complexity
            });
          }
        }
      };
    }
  }]
});
\`\`\`

**Trade-offs**: More protection = more complexity. Start with DataLoader and depth limits, add complexity analysis and caching as needed.`,
          keyPoints: [
            'DataLoader solves N+1 by batching database queries',
            'Query complexity and depth limiting prevent expensive queries',
            'Redis caching reduces database load for repeated queries',
            'Pagination limits prevent large data fetches',
            'Monitor slow queries and optimize hot paths',
          ],
        },
        {
          id: 'graphql-perf-d2',
          question:
            'Compare caching strategies for GraphQL vs REST. Why is HTTP caching insufficient for GraphQL, and what alternatives exist?',
          sampleAnswer: `GraphQL caching is more complex than REST due to architectural differences:

**REST Caching (Simple)**:

\`\`\`http
GET /api/users/123
Cache-Control: max-age=3600, public

# Cached by:
- Browser
- CDN (CloudFront, Cloudflare)
- Reverse proxy (Varnish, NGINX)
- HTTP cache headers work automatically
\`\`\`

**Why GraphQL Caching is Different**:

1. **POST Requests**: GraphQL uses POST (query in body), not cached by default
2. **Dynamic Queries**: Same endpoint, infinite query variations
3. **Personalized Data**: User-specific results in same query

**GraphQL Caching Solutions**:

**1. Automatic Persisted Queries (APQ)**

Combines bandwidth savings with caching:

\`\`\`javascript
// Client
const link = createPersistedQueryLink({
  sha256: hashQuery
}).concat(httpLink);

// First request: Send hash
{
  "extensions": {
    "persistedQuery": {
      "version": 1,
      "sha256Hash": "abc123..."
    }
  }
}

// Server doesn't have it: error
// Client resends with full query + hash
// Server caches query by hash

// Subsequent requests: just hash (saves bandwidth)
\`\`\`

**2. Response Caching (Apollo Server)**

\`\`\`javascript
const responseCachePlugin = require('apollo-server-plugin-response-cache');

const server = new ApolloServer({
  schema,
  plugins: [responseCachePlugin()],
  cacheControl: {
    defaultMaxAge: 5,  // 5 seconds default
  }
});

// Schema-level cache hints
type Query {
  # Public data: cache longer
  posts: [Post!]! @cacheControl(maxAge: 300, scope: PUBLIC)
  
  # User-specific: shorter cache, private
  me: User @cacheControl(maxAge: 60, scope: PRIVATE)
  
  # Real-time: don't cache
  liveCount: Int! @cacheControl(maxAge: 0)
}

// Response includes cache headers
Cache-Control: max-age=300, public
\`\`\`

**3. Redis/In-Memory Cache**

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

// Cache by query + variables
const cacheKey = (query, variables) => 
  crypto.createHash('sha256')
    .update(JSON.stringify({ query, variables }))
    .digest('hex');

const server = new ApolloServer({
  schema,
  plugins: [{
    async requestDidStart() {
      return {
        async responseForOperation({ request, response }) {
          const key = cacheKey(
            request.query,
            request.variables
          );
          
          // Check cache
          const cached = await client.get(\`gql:\${key}\`);
          if (cached) {
            return JSON.parse(cached);
          }
          
          return null;  // Execute query
        },
        async willSendResponse({ request, response }) {
          if (!response.errors) {
            const key = cacheKey(
              request.query,
              request.variables
            );
            
            // Cache for 5 minutes
            await client.setex(
              \`gql:\${key}\`,
              300,
              JSON.stringify(response)
            );
          }
        }
      };
    }
  }]
});
\`\`\`

**4. DataLoader (Request-Scoped Cache)**

\`\`\`javascript
// Automatic caching within single request
const userLoader = new DataLoader(async (ids) => {
  const users = await db.users.findAll({ where: { id: ids }});
  return ids.map(id => users.find(u => u.id === id));
}, {
  cache: true  // Caches for request duration
});

// Called multiple times in query, only 1 DB query
author1 = userLoader.load(1);
author2 = userLoader.load(1);  // Returns cached
\`\`\`

**5. CDN Caching (GET Queries)**

Force GraphQL to use GET for cacheable queries:

\`\`\`javascript
// Apollo Client - use GET for queries
const link = new HttpLink({
  uri: '/graphql',
  useGETForQueries: true,  // Queries use GET, mutations use POST
});

// Server
app.use('/graphql', (req, res) => {
  if (req.method === 'GET') {
    res.set('Cache-Control', 'public, max-age=300');
  }
  // ... execute query
});

// Now CDNs can cache GET /graphql?query=...
\`\`\`

**Comparison Table**:

| Strategy | REST | GraphQL |
|----------|------|---------|
| HTTP Cache | Native | Requires GET queries |
| CDN | Automatic | With APQ or GET |
| Response Cache | URL-based | Query+variables hash |
| Field-level | Not applicable | DataLoader |
| Complexity | Low | High |

**Best Practice Stack**:

1. **DataLoader**: Always (prevents N+1)
2. **@cacheControl** directives: Schema-level hints
3. **APQ**: Mobile clients (bandwidth)
4. **Redis**: Expensive queries
5. **CDN**: GET queries only

**Trade-off**: GraphQL caching is more complex but allows fine-grained control. REST is simpler but less flexible.`,
          keyPoints: [
            'GraphQL uses POST, losing native HTTP caching benefits',
            'APQ saves bandwidth by sending query hash instead of full query',
            'Response caching with @cacheControl directives provides schema-level control',
            'Redis caching by query+variables hash for expensive queries',
            'DataLoader provides request-scoped caching preventing N+1',
          ],
        },
        {
          id: 'graphql-perf-d3',
          question:
            'Design a monitoring and alerting system for a production GraphQL API. What metrics would you track, and what thresholds would trigger alerts?',
          sampleAnswer: `Comprehensive GraphQL monitoring strategy for production:

**1. Query Performance Metrics**

\`\`\`javascript
const { ApolloServer } = require('apollo-server');
const prometheus = require('prom-client');

// Define metrics
const queryDuration = new prometheus.Histogram({
  name: 'graphql_query_duration_seconds',
  help: 'GraphQL query duration',
  labelNames: ['operation_name', 'operation_type'],
  buckets: [0.1, 0.5, 1, 2, 5, 10]
});

const queryErrors = new prometheus.Counter({
  name: 'graphql_query_errors_total',
  help: 'GraphQL query errors',
  labelNames: ['operation_name', 'error_type']
});

const queryComplexity = new prometheus.Histogram({
  name: 'graphql_query_complexity',
  help: 'GraphQL query complexity',
  labelNames: ['operation_name'],
  buckets: [10, 50, 100, 500, 1000, 5000]
});

const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart() {
      const start = Date.now();
      let operationName, operationType;
      
      return {
        didResolveOperation({ request, operation }) {
          operationName = operation.name?.value || 'anonymous';
          operationType = operation.operation;
        },
        
        didEncounterErrors({ errors }) {
          errors.forEach(error => {
            queryErrors.labels(
              operationName,
              error.extensions?.code || 'UNKNOWN'
            ).inc();
          });
        },
        
        willSendResponse({ response }) {
          const duration = (Date.now() - start) / 1000;
          
          queryDuration.labels(
            operationName,
            operationType
          ).observe(duration);
          
          // Log complexity if available
          if (response.extensions?.complexity) {
            queryComplexity.labels(operationName)
              .observe(response.extensions.complexity);
          }
        }
      };
    }
  }]
});
\`\`\`

**2. Resolver-Level Metrics**

\`\`\`javascript
const resolverDuration = new prometheus.Histogram({
  name: 'graphql_resolver_duration_seconds',
  help: 'Resolver execution time',
  labelNames: ['type', 'field'],
  buckets: [0.001, 0.01, 0.1, 0.5, 1]
});

// Wrap resolvers with instrumentation
const instrumentResolver = (typeName, fieldName, resolve) => {
  return async (...args) => {
    const start = Date.now();
    try {
      return await resolve(...args);
    } finally {
      const duration = (Date.now() - start) / 1000;
      resolverDuration.labels(typeName, fieldName).observe(duration);
    }
  };
};

// Apply to slow resolvers
const resolvers = {
  Query: {
    posts: instrumentResolver('Query', 'posts', getPosts),
    users: instrumentResolver('Query', 'users', getUsers)
  },
  Post: {
    author: instrumentResolver('Post', 'author', getAuthor)
  }
};
\`\`\`

**3. DataLoader Metrics**

\`\`\`javascript
const loaderHits = new prometheus.Counter({
  name: 'dataloader_cache_hits_total',
  help: 'DataLoader cache hits',
  labelNames: ['loader_name']
});

const loaderMisses = new prometheus.Counter({
  name: 'dataloader_cache_misses_total',
  help: 'DataLoader cache misses',
  labelNames: ['loader_name']
});

const instrumentedDataLoader = (name, batchFn) => {
  return new DataLoader(
    async (keys) => {
      loaderMisses.labels(name).inc(keys.length);
      return batchFn(keys);
    },
    {
      cacheMap: {
        get(key) {
          const value = cache.get(key);
          if (value) {
            loaderHits.labels(name).inc();
          }
          return value;
        },
        set: cache.set.bind(cache),
        delete: cache.delete.bind(cache),
        clear: cache.clear.bind(cache)
      }
    }
  );
};
\`\`\`

**4. Alert Thresholds**

\`\`\`yaml
# Prometheus AlertManager rules
groups:
  - name: graphql_alerts
    rules:
      # Query latency
      - alert: GraphQLSlowQueries
        expr: |
          histogram_quantile(0.95,
            rate(graphql_query_duration_seconds_bucket[5m])
          ) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile query latency > 2s"
          
      # Error rate
      - alert: GraphQLHighErrorRate
        expr: |
          (
            sum(rate(graphql_query_errors_total[5m]))
            /
            sum(rate(graphql_query_duration_seconds_count[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5%"
          
      # Query complexity
      - alert: GraphQLHighComplexity
        expr: |
          histogram_quantile(0.99,
            rate(graphql_query_complexity_bucket[5m])
          ) > 5000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "99th percentile complexity > 5000"
          
      # Resolver performance
      - alert: GraphQLSlowResolver
        expr: |
          histogram_quantile(0.95,
            rate(graphql_resolver_duration_seconds_bucket{
              type="Post",
              field="author"
            }[5m])
          ) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Post.author resolver slow"
          
      # DataLoader efficiency
      - alert: DataLoaderLowHitRate
        expr: |
          (
            rate(dataloader_cache_hits_total[5m])
            /
            (rate(dataloader_cache_hits_total[5m]) +
             rate(dataloader_cache_misses_total[5m]))
          ) < 0.5
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "DataLoader cache hit rate < 50%"
\`\`\`

**5. Dashboard (Grafana)**

\`\`\`json
{
  "panels": [
    {
      "title": "Query Latency (p50, p95, p99)",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(graphql_query_duration_seconds_bucket[5m]))"
        },
        {
          "expr": "histogram_quantile(0.95, rate(graphql_query_duration_seconds_bucket[5m]))"
        },
        {
          "expr": "histogram_quantile(0.99, rate(graphql_query_duration_seconds_bucket[5m]))"
        }
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {
          "expr": "sum(rate(graphql_query_errors_total[5m])) by (error_type)"
        }
      ]
    },
    {
      "title": "Top 10 Slowest Queries",
      "targets": [
        {
          "expr": "topk(10, avg by (operation_name) (rate(graphql_query_duration_seconds_sum[5m]) / rate(graphql_query_duration_seconds_count[5m])))"
        }
      ]
    }
  ]
}
\`\`\`

**6. Logging Slow Queries**

\`\`\`javascript
const server = new ApolloServer({
  schema,
  plugins: [{
    requestDidStart() {
      const start = Date.now();
      
      return {
        willSendResponse({ request, response }) {
          const duration = Date.now() - start;
          
          if (duration > 1000) {
            logger.warn({
              message: 'Slow query detected',
              operation: request.operationName,
              duration,
              query: request.query,
              variables: request.variables,
              complexity: response.extensions?.complexity
            });
          }
        }
      };
    }
  }]
});
\`\`\`

**Key Metrics to Track**:
1. Query latency (p50, p95, p99)
2. Error rate by type
3. Query complexity distribution
4. Resolver performance
5. DataLoader cache hit rate
6. Concurrent queries
7. Query depth distribution

**Alert Thresholds**:
- P95 latency > 2s → Warning
- Error rate > 5% → Critical
- P99 complexity > 5000 → Warning
- Specific slow resolvers → Warning
- DataLoader hit rate < 50% → Info

This provides comprehensive observability for production GraphQL APIs.`,
          keyPoints: [
            'Track query latency at p50, p95, p99 percentiles',
            'Monitor error rates by type and operation',
            'Measure query complexity distribution',
            'Instrument slow resolvers for optimization',
            'Alert on high error rates, slow queries, and complexity spikes',
          ],
        },
      ],
    },
    {
      id: 'grpc-service-design',
      title: 'gRPC Service Design',
      content: `gRPC is a high-performance RPC framework using Protocol Buffers, ideal for microservices communication. Understanding gRPC design is essential for building efficient distributed systems.

## What is gRPC?

**gRPC** (gRPC Remote Procedure Call) is an open-source RPC framework developed by Google.

### **Key Characteristics**
- Uses **HTTP/2** for transport
- **Protocol Buffers** (protobuf) for serialization
- Supports **multiple languages**
- **Bidirectional streaming**
- Built-in **authentication**, **load balancing**, and **deadlines**

### **gRPC vs REST**

| Feature | REST | gRPC |
|---------|------|------|
| Protocol | HTTP/1.1 | HTTP/2 |
| Payload | JSON/XML (text) | Protobuf (binary) |
| Performance | Slower | Faster |
| Streaming | No (SSE hack) | Yes (native) |
| Browser support | Native | Requires proxy |
| Human-readable | Yes | No |
| Use case | Public APIs | Microservices |

## Protocol Buffers

**Protobuf** is a language-neutral data serialization format.

### **Define Schema**

\`\`\`protobuf
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUser(CreateUserRequest) returns (User);
  rpc UpdateUser(UpdateUserRequest) returns (User);
  rpc DeleteUser(DeleteUserRequest) returns (Empty);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
  repeated string tags = 5;
  google.protobuf.Timestamp created_at = 6;
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string role = 3;
}

message CreateUserInput {
  string name = 1;
  string email = 2;
  string password = 3;
}

message Empty {}
\`\`\`

**Field Numbers**: Permanent identifiers (1, 2, 3...) for backward compatibility.

## RPC Types

### **1. Unary RPC (Request-Response)**

Simple request-response pattern:

\`\`\`protobuf
rpc GetUser(GetUserRequest) returns (User);
\`\`\`

\`\`\`javascript
// Client
const response = await client.getUser({ id: '123' });
console.log(response.name);
\`\`\`

### **2. Server Streaming**

Client sends one request, server streams multiple responses:

\`\`\`protobuf
rpc ListUsers(ListUsersRequest) returns (stream User);
\`\`\`

\`\`\`javascript
// Client
const call = client.listUsers({ pageSize: 100 });

call.on('data', (user) => {
  console.log('Received user:', user.name);
});

call.on('end', () => {
  console.log('Stream ended');
});
\`\`\`

**Use cases**: Large result sets, real-time updates, log streaming

### **3. Client Streaming**

Client streams multiple requests, server sends one response:

\`\`\`protobuf
rpc UploadUsers(stream CreateUserRequest) returns (UploadSummary);
\`\`\`

\`\`\`javascript
// Client
const call = client.uploadUsers((err, response) => {
  console.log('Uploaded:', response.count);
});

users.forEach(user => call.write(user));
call.end();
\`\`\`

**Use cases**: Batch uploads, log aggregation

### **4. Bidirectional Streaming**

Both client and server stream:

\`\`\`protobuf
rpc Chat(stream ChatMessage) returns (stream ChatMessage);
\`\`\`

\`\`\`javascript
// Client
const call = client.chat();

call.on('data', (message) => {
  console.log('Received:', message.text);
});

call.write({ text: 'Hello!' });
call.write({ text: 'How are you?' });
\`\`\`

**Use cases**: Chat, real-time collaboration, gaming

## Error Handling

**gRPC Status Codes**:

\`\`\`
OK                 = 0   // Success
CANCELLED          = 1   // Client cancelled
INVALID_ARGUMENT   = 3   // Invalid request
NOT_FOUND          = 5   // Resource not found
ALREADY_EXISTS     = 6   // Resource exists
PERMISSION_DENIED  = 7   // No permission
RESOURCE_EXHAUSTED = 8   // Rate limit, quota
FAILED_PRECONDITION = 9  // System state issue
UNIMPLEMENTED      = 12  // Not implemented
INTERNAL           = 13  // Server error
UNAVAILABLE        = 14  // Service unavailable
UNAUTHENTICATED    = 16  // Not authenticated
\`\`\`

**Return Errors**:

\`\`\`javascript
// Server
async getUser(call, callback) {
  const { id } = call.request;
  
  const user = await db.users.findById(id);
  
  if (!user) {
    return callback({
      code: grpc.status.NOT_FOUND,
      message: 'User not found',
      details: \`User \${id} does not exist\`
    });
  }
  
  callback(null, user);
}
\`\`\`

**Error Details** (Rich Errors):

\`\`\`protobuf
import "google/rpc/error_details.proto";

message ErrorResponse {
  google.rpc.BadRequest bad_request = 1;
  google.rpc.RetryInfo retry_info = 2;
}
\`\`\`

## Metadata (Headers)

**Send Metadata**:

\`\`\`javascript
// Client
const metadata = new grpc.Metadata();
metadata.add('authorization', 'Bearer token123');
metadata.add('request-id', 'uuid-123');

client.getUser({ id: '123' }, metadata, (err, response) => {
  // ...
});
\`\`\`

**Receive Metadata**:

\`\`\`javascript
// Server
async getUser(call, callback) {
  const metadata = call.metadata;
  const authToken = metadata.get('authorization')[0];
  
  // Verify token
  const user = await authenticateToken(authToken);
  
  if (!user) {
    return callback({
      code: grpc.status.UNAUTHENTICATED,
      message: 'Invalid token'
    });
  }
  
  // Process request
}
\`\`\`

## Deadlines and Timeouts

**Client-Side Deadline**:

\`\`\`javascript
// Timeout after 5 seconds
const deadline = new Date();
deadline.setSeconds(deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  (err, response) => {
    if (err && err.code === grpc.status.DEADLINE_EXCEEDED) {
      console.error('Request timed out');
    }
  }
);
\`\`\`

**Server Check**:

\`\`\`javascript
async getUser(call, callback) {
  // Check if client cancelled or deadline exceeded
  if (call.cancelled) {
    return callback({
      code: grpc.status.CANCELLED,
      message: 'Request cancelled'
    });
  }
  
  // Long operation
  const user = await expensiveQuery();
  
  callback(null, user);
}
\`\`\`

## Authentication

### **1. SSL/TLS**

\`\`\`javascript
// Server
const server = new grpc.Server();
const credentials = grpc.ServerCredentials.createSsl(
  fs.readFileSync('ca.pem'),
  [{
    cert_chain: fs.readFileSync('server-cert.pem'),
    private_key: fs.readFileSync('server-key.pem')
  }]
);

server.bindAsync('0.0.0.0:50051', credentials, () => {
  server.start();
});

// Client
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.pem')
);

const client = new UserServiceClient('localhost:50051', credentials);
\`\`\`

### **2. Token-Based (Metadata)**

\`\`\`javascript
// Client interceptor
const authInterceptor = (options, nextCall) => {
  return new grpc.InterceptingCall(nextCall(options), {
    start: (metadata, listener, next) => {
      metadata.add('authorization', \`Bearer \${getToken()}\`);
      next(metadata, listener);
    }
  });
};

const client = new UserServiceClient(
  'localhost:50051',
  credentials,
  { interceptors: [authInterceptor] }
);
\`\`\`

### **3. Mutual TLS (mTLS)**

Both client and server authenticate:

\`\`\`javascript
const credentials = grpc.credentials.createSsl(
  fs.readFileSync('ca.pem'),
  fs.readFileSync('client-key.pem'),  // Client cert
  fs.readFileSync('client-cert.pem')
);
\`\`\`

## Load Balancing

**Client-Side Load Balancing**:

\`\`\`javascript
// Round-robin across multiple servers
const client = new UserServiceClient(
  'dns:///service.example.com',  // Resolves to multiple IPs
  credentials,
  {
    'grpc.lb_policy_name': 'round_robin'
  }
);
\`\`\`

**Server-Side** (with proxy like Envoy)

## Interceptors (Middleware)

**Server Interceptor**:

\`\`\`javascript
const loggingInterceptor = (call, callback, next) => {
  console.log('Request:', call.request);
  const start = Date.now();
  
  next(call, (err, response) => {
    console.log('Duration:', Date.now() - start);
    callback(err, response);
  });
};

server.use(loggingInterceptor);
\`\`\`

**Client Interceptor**:

\`\`\`javascript
const retryInterceptor = (options, nextCall) => {
  return new grpc.InterceptingCall(nextCall(options), {
    start: (metadata, listener, next) => {
      const retryListener = {
        onReceiveStatus: (status, nextStatus) => {
          if (status.code === grpc.status.UNAVAILABLE) {
            // Retry logic
            return retry();
          }
          nextStatus(status);
        }
      };
      next(metadata, retryListener);
    }
  });
};
\`\`\`

## Real-World Example

**Service Definition**:

\`\`\`protobuf
syntax = "proto3";

package ecommerce;

service ProductService {
  rpc GetProduct(GetProductRequest) returns (Product);
  rpc SearchProducts(SearchRequest) returns (stream Product);
  rpc CreateOrder(stream OrderItem) returns (Order);
}

message Product {
  string id = 1;
  string name = 2;
  double price = 3;
  int32 stock = 4;
}

message GetProductRequest {
  string id = 1;
}

message SearchRequest {
  string query = 1;
  int32 limit = 2;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}

message Order {
  string id = 1;
  double total = 2;
  repeated OrderItem items = 3;
}
\`\`\`

## Best Practices

1. **Use protobuf field numbers wisely**: Never reuse, reserve deprecated ones
2. **Enable deadlines**: Prevent hanging requests
3. **Implement retries with backoff**: Handle transient failures
4. **Use streaming for large data**: Don't load everything in memory
5. **Secure with TLS**: Always in production
6. **Monitor with interceptors**: Logging, metrics, tracing
7. **Version your proto files**: Backward compatibility
8. **Document services**: Comments in proto files
9. **Use service mesh**: For advanced traffic management (Istio, Linkerd)
10. **Test with tools**: grpcurl, BloomRPC

## When to Use gRPC

**Use gRPC when**:
- Microservices communication (internal)
- High performance required
- Bidirectional streaming needed
- Type safety important
- Polyglot services (multiple languages)

**Use REST when**:
- Public APIs (browser access)
- Third-party integrations
- Human-readable format needed
- Simple request-response
- HTTP caching important`,
      multipleChoice: [
        {
          id: 'grpc-q1',
          question:
            'What is the main advantage of gRPC over REST for microservices communication?',
          options: [
            'gRPC is easier to implement and debug',
            'gRPC uses binary Protocol Buffers over HTTP/2, providing better performance',
            'gRPC works in browsers without any additional setup',
            'gRPC supports more programming languages',
          ],
          correctAnswer: 1,
          explanation:
            "gRPC uses Protocol Buffers (binary, compact) over HTTP/2 (multiplexing, compression), making it significantly faster than JSON/REST. However, it's harder to debug (not human-readable), requires proxies for browsers, and both support many languages.",
          difficulty: 'easy',
        },
        {
          id: 'grpc-q2',
          question:
            'Which gRPC streaming type would you use for a real-time chat application?',
          options: [
            'Unary RPC (request-response)',
            'Server streaming (one request, multiple responses)',
            'Client streaming (multiple requests, one response)',
            'Bidirectional streaming (both send multiple messages)',
          ],
          correctAnswer: 3,
          explanation:
            'Real-time chat requires bidirectional streaming where both client and server continuously send and receive messages. Server streaming is one-way (server→client), client streaming is opposite (client→server), unary is single request-response.',
          difficulty: 'easy',
        },
        {
          id: 'grpc-q3',
          question:
            'What is the purpose of field numbers in Protocol Buffer definitions?',
          options: [
            'To define the order fields appear in JSON output',
            'To serve as unique, permanent identifiers for backward compatibility',
            'To indicate required vs optional fields',
            'To specify the default value for each field',
          ],
          correctAnswer: 1,
          explanation:
            "Field numbers (1, 2, 3...) are permanent identifiers used in binary encoding. Changing them breaks compatibility. They're not related to JSON order, optionality (proto3 all fields optional), or defaults. Never reuse or change field numbers.",
          difficulty: 'medium',
        },
        {
          id: 'grpc-q4',
          question:
            'How does gRPC handle timeouts to prevent hanging requests?',
          options: [
            'Server automatically cancels requests after 30 seconds',
            'Client sets deadline in metadata; server checks if deadline exceeded',
            'HTTP/2 has built-in timeout mechanism',
            'Protocol Buffers include timeout field',
          ],
          correctAnswer: 1,
          explanation:
            'gRPC uses deadlines: client specifies timeout when making call, gRPC propagates it as deadline in metadata, server can check if deadline exceeded and abort. No default timeout (can hang forever), not automatic, not in HTTP/2 or protobuf.',
          difficulty: 'medium',
        },
        {
          id: 'grpc-q5',
          question: 'Why might you choose REST over gRPC for a public API?',
          options: [
            'REST is faster and more efficient',
            'REST has better streaming capabilities',
            'REST works natively in browsers and is human-readable for easier debugging',
            'REST supports more authentication methods',
          ],
          correctAnswer: 2,
          explanation:
            'REST works in browsers without proxies, JSON is human-readable for debugging, and HTTP caching is straightforward. gRPC is actually faster, has better streaming, and both support various auth methods. gRPC requires gRPC-Web proxy for browsers.',
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'grpc-d1',
          question:
            'Design a gRPC service for a video streaming platform with user management, video upload, and live streaming. Define the proto file and explain your RPC type choices.',
          sampleAnswer: `Complete gRPC service design for video streaming platform:

\`\`\`protobuf
syntax = "proto3";

package streaming;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// ========== User Service ==========

service UserService {
  // Unary: Simple CRUD
  rpc CreateUser(CreateUserRequest) returns (User);
  rpc GetUser(GetUserRequest) returns (User);
  rpc UpdateUser(UpdateUserRequest) returns (User);
  rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);
  
  // Server streaming: List with potentially large results
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Get user's subscriptions
  rpc GetSubscriptions(GetUserRequest) returns (stream Channel);
}

message User {
  string id = 1;
  string username = 2;
  string email = 3;
  string avatar_url = 4;
  google.protobuf.Timestamp created_at = 5;
  int64 subscriber_count = 6;
}

// ========== Video Service ==========

service VideoService {
  // Client streaming: Video upload in chunks
  rpc UploadVideo(stream VideoChunk) returns (Video);
  
  // Server streaming: Download video in chunks
  rpc DownloadVideo(VideoRequest) returns (stream VideoChunk);
  
  // Unary: Metadata operations
  rpc GetVideo(VideoRequest) returns (Video);
  rpc UpdateVideo(UpdateVideoRequest) returns (Video);
  rpc DeleteVideo(VideoRequest) returns (google.protobuf.Empty);
  
  // Server streaming: Search results
  rpc SearchVideos(SearchRequest) returns (stream Video);
  
  // Get video comments
  rpc GetComments(VideoRequest) returns (stream Comment);
}

message Video {
  string id = 1;
  string title = 2;
  string description = 3;
  string thumbnail_url = 4;
  string video_url = 5;
  int64 duration_seconds = 6;
  int64 view_count = 7;
  string author_id = 8;
  google.protobuf.Timestamp created_at = 9;
  VideoStatus status = 10;
}

enum VideoStatus {
  PROCESSING = 0;
  READY = 1;
  FAILED = 2;
}

message VideoChunk {
  bytes data = 1;
  int64 offset = 2;
  string video_id = 3;  // Empty for upload, set for download
}

// ========== Live Streaming Service ==========

service LiveStreamService {
  // Bidirectional: Real-time stream
  rpc Stream(stream StreamPacket) returns (stream StreamPacket);
  
  // Server streaming: Watch live stream
  rpc WatchStream(WatchRequest) returns (stream StreamPacket);
  
  // Bidirectional: Live chat
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
  
  // Unary: Stream management
  rpc StartStream(StartStreamRequest) returns (Stream);
  rpc EndStream(StreamRequest) returns (google.protobuf.Empty);
  rpc GetStream(StreamRequest) returns (Stream);
}

message Stream {
  string id = 1;
  string title = 2;
  string streamer_id = 3;
  int64 viewer_count = 4;
  google.protobuf.Timestamp started_at = 5;
  StreamStatus status = 6;
}

enum StreamStatus {
  LIVE = 0;
  ENDED = 1;
  OFFLINE = 2;
}

message StreamPacket {
  bytes video_data = 1;
  bytes audio_data = 2;
  int64 timestamp_ms = 3;
  string stream_id = 4;
}

message ChatMessage {
  string id = 1;
  string stream_id = 2;
  string user_id = 3;
  string username = 4;
  string message = 5;
  google.protobuf.Timestamp sent_at = 6;
}

// ========== Recommendation Service ==========

service RecommendationService {
  // Server streaming: Personalized recommendations
  rpc GetRecommendations(RecommendationRequest) returns (stream Video);
  
  // Unary: Record view for algorithm
  rpc RecordView(ViewEvent) returns (google.protobuf.Empty);
}

message RecommendationRequest {
  string user_id = 1;
  int32 count = 2;
}

message ViewEvent {
  string user_id = 1;
  string video_id = 2;
  int64 watch_duration_seconds = 3;
  google.protobuf.Timestamp viewed_at = 4;
}

// ========== Common Messages ==========

message GetUserRequest {
  string id = 1;
}

message VideoRequest {
  string id = 1;
}

message SearchRequest {
  string query = 1;
  int32 limit = 2;
  int32 offset = 3;
}
\`\`\`

**RPC Type Justifications**:

1. **Unary (CreateUser, GetVideo)**: Simple CRUD operations
2. **Client Streaming (UploadVideo)**: Client sends video in chunks, server returns metadata when complete
3. **Server Streaming (DownloadVideo, SearchVideos)**: Server sends data in chunks or streams results
4. **Bidirectional (Stream, Chat)**: Real-time two-way communication

**Implementation Considerations**:

- Video chunking for large files
- Stream IDs for reconnection
- Timestamps for ordering
- Enums for status
- Separate services for separation of concerns
- Metadata in separate messages

This design leverages gRPC's strengths: performance, streaming, and type safety.`,
          keyPoints: [
            'Use unary RPC for simple CRUD operations',
            'Client streaming for chunked uploads',
            'Server streaming for large result sets',
            'Bidirectional streaming for real-time communication',
            'Separate services by domain for maintainability',
          ],
        },
        {
          id: 'grpc-d2',
          question:
            'Your gRPC microservices are experiencing intermittent failures and timeouts. Design a comprehensive error handling and retry strategy.',
          sampleAnswer: `Comprehensive gRPC error handling and reliability strategy:

**1. Client-Side Retry Logic**

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// Exponential backoff retry
async function retryCall(callFn, maxRetries = 3) {
  let lastError;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await callFn();
    } catch (error) {
      lastError = error;
      
      // Only retry on transient errors
      const retryableErrors = [
        grpc.status.UNAVAILABLE,
        grpc.status.DEADLINE_EXCEEDED,
        grpc.status.RESOURCE_EXHAUSTED
      ];
      
      if (!retryableErrors.includes(error.code)) {
        throw error;  // Don't retry permanent errors
      }
      
      // Exponential backoff with jitter
      const baseDelay = 100;
      const maxDelay = 5000;
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt) + Math.random() * 100,
        maxDelay
      );
      
      console.log(\`Retry \${attempt + 1}/\${maxRetries} after \${delay}ms\`);
      await sleep(delay);
    }
  }
  
  throw lastError;
}

// Usage
const response = await retryCall(() => 
  new Promise((resolve, reject) => {
    client.getUser(
      { id: '123' },
      { deadline: Date.now() + 5000 },
      (err, response) => {
        if (err) reject(err);
        else resolve(response);
      }
    );
  })
);
\`\`\`

**2. Deadline Propagation**

\`\`\`javascript
// Client sets deadline
const deadline = new Date();
deadline.setSeconds(deadline.getSeconds() + 5);

client.getUser(
  { id: '123' },
  { deadline: deadline.getTime() },
  callback
);

// Server propagates to downstream calls
async function getUser(call, callback) {
  // Get deadline from incoming call
  const deadline = call.deadline;
  
  // Propagate to downstream service
  const response = await downstreamClient.getData(
    { id: call.request.id },
    { deadline: deadline },
    callback
  );
  
  callback(null, response);
}
\`\`\`

**3. Circuit Breaker**

\`\`\`javascript
class CircuitBreaker {
  constructor(threshold = 5, timeout = 60000) {
    this.failureThreshold = threshold;
    this.timeout = timeout;
    this.failureCount = 0;
    this.state = 'CLOSED';  // CLOSED, OPEN, HALF_OPEN
    this.nextAttempt = Date.now();
  }
  
  async call(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      // Try half-open
      this.state = 'HALF_OPEN';
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }
  
  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.timeout;
    }
  }
}

// Usage
const breaker = new CircuitBreaker();

const response = await breaker.call(() =>
  client.getUser({ id: '123' })
);
\`\`\`

**4. Timeout Strategy**

\`\`\`javascript
// Service-specific timeouts
const TIMEOUTS = {
  userService: 2000,      // Fast service
  analyticsService: 10000, // Slow aggregation
  searchService: 5000
};

function callWithTimeout(client, method, request, serviceName) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + TIMEOUTS[serviceName];
    
    client[method](
      request,
      { deadline },
      (err, response) => {
        if (err) {
          if (err.code === grpc.status.DEADLINE_EXCEEDED) {
            console.error(\`\${serviceName} timeout after \${TIMEOUTS[serviceName]}ms\`);
          }
          reject(err);
        } else {
          resolve(response);
        }
      }
    );
  });
}
\`\`\`

**5. Health Checking**

\`\`\`protobuf
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
  }
  ServingStatus status = 1;
}
\`\`\`

\`\`\`javascript
// Server implementation
const healthServer = {
  check: (call, callback) => {
    callback(null, {
      status: ServingStatus.SERVING
    });
  }
};

// Client health check before calls
async function ensureHealth() {
  const response = await healthClient.check({});
  if (response.status !== ServingStatus.SERVING) {
    throw new Error('Service not healthy');
  }
}
\`\`\`

**6. Error Response Enrichment**

\`\`\`javascript
function enrichError(error, context) {
  return {
    code: error.code,
    message: error.message,
    details: {
      service: context.service,
      method: context.method,
      requestId: context.requestId,
      timestamp: new Date().toISOString(),
      stack: error.stack
    }
  };
}

// Server
async function getUser(call, callback) {
  try {
    const user = await db.findUser(call.request.id);
    if (!user) {
      return callback({
        code: grpc.status.NOT_FOUND,
        message: 'User not found',
        metadata: new grpc.Metadata({
          'request-id': call.metadata.get('request-id')[0]
        })
      });
    }
    callback(null, user);
  } catch (error) {
    callback(enrichError(error, {
      service: 'UserService',
      method: 'getUser',
      requestId: call.metadata.get('request-id')[0]
    }));
  }
}
\`\`\`

**7. Monitoring & Alerting**

\`\`\`javascript
const prometheus = require('prom-client');

const grpcDuration = new prometheus.Histogram({
  name: 'grpc_request_duration_seconds',
  help: 'gRPC request duration',
  labelNames: ['service', 'method', 'status']
});

const grpcErrors = new prometheus.Counter({
  name: 'grpc_errors_total',
  help: 'gRPC errors',
  labelNames: ['service', 'method', 'code']
});

// Interceptor
function monitoringInterceptor(call, callback, next) {
  const start = Date.now();
  
  next(call, (err, response) => {
    const duration = (Date.now() - start) / 1000;
    
    grpcDuration.labels(
      call.getPath(),
      err ? 'error' : 'success'
    ).observe(duration);
    
    if (err) {
      grpcErrors.labels(
        call.getPath(),
        err.code
      ).inc();
    }
    
    callback(err, response);
  });
}
\`\`\`

**Best Practice Stack**:

1. **Timeouts**: Always set deadlines
2. **Retries**: Exponential backoff, only transient errors
3. **Circuit breaker**: Fail fast when service down
4. **Health checks**: Verify before calling
5. **Monitoring**: Track errors, latencies, retries
6. **Graceful degradation**: Fallbacks when possible
7. **Idempotency**: Safe to retry operations

Trade-off: More reliability mechanisms = more complexity. Start with timeouts and retries, add circuit breakers as needed.`,
          keyPoints: [
            'Implement exponential backoff retry for transient failures',
            'Always set deadlines to prevent hanging requests',
            'Use circuit breakers to fail fast when services are down',
            'Propagate deadlines through service call chains',
            'Monitor errors and latencies for proactive alerting',
          ],
        },
        {
          id: 'grpc-d3',
          question:
            'Compare gRPC and REST for different scenarios: public API, internal microservices, mobile app, and IoT devices. Which would you choose for each and why?',
          sampleAnswer: `Detailed comparison for different use cases:

**Scenario 1: Public API (Third-Party Developers)**

**Winner: REST**

Reasons:
- Browser support without proxies
- Human-readable JSON for debugging
- Easy to test (curl, Postman, browser)
- Documentation tools (Swagger/OpenAPI)
- HTTP caching (CDN, browser cache)
- Familiar to most developers

gRPC challenges:
- Requires gRPC-Web proxy for browsers
- Binary format hard to debug
- Less familiar to external developers

Example: Stripe, GitHub, Twilio all use REST for public APIs.

**Scenario 2: Internal Microservices**

**Winner: gRPC**

Reasons:
- 7-10x faster (binary protobuf vs JSON)
- Strong typing prevents bugs
- Code generation for multiple languages
- Native bidirectional streaming
- Built-in load balancing
- Efficient for high-traffic internal communication

REST advantages:
- Simpler debugging
- HTTP caching

Trade-off: Performance and type safety > debugging convenience for internal use.

Example: Netflix, Uber use gRPC for internal microservices.

**Implementation**:
\`\`\`protobuf
// Product Service
service ProductService {
  rpc GetProduct(ProductRequest) returns (Product);
  rpc ListProducts(ListRequest) returns (stream Product);
}

// Inventory Service (calls Product Service)
service InventoryService {
  rpc CheckStock(StockRequest) returns (StockResponse);
}
\`\`\`

**Scenario 3: Mobile App (iOS/Android)**

**Winner: gRPC (with considerations)**

Reasons:
- Smaller payload sizes (battery/bandwidth)
- Binary format faster to parse
- Bidirectional streaming for real-time features
- Official support for mobile platforms
- Connection reuse (HTTP/2)

Considerations:
- Initial payload larger (code generation)
- More complex setup than REST
- Debugging harder

REST advantages:
- Simpler, less code
- Easier debugging

Recommendation: gRPC for apps with high data transfer or real-time features, REST for simple CRUD apps.

Example: Google apps use gRPC internally.

**Scenario 4: IoT Devices (Constrained Resources)**

**Winner: gRPC (but consider MQTT)**

Reasons:
- Smaller payloads (critical for limited bandwidth)
- Binary protocol efficient
- HTTP/2 multiplexing reduces connections
- Stream data efficiently

Challenges:
- Memory overhead for protobuf
- TLS required (compute intensive)

Alternative: MQTT for pub/sub patterns

REST challenges:
- JSON parsing expensive
- Larger payloads
- More network overhead

Recommendation: gRPC for direct device-to-cloud, MQTT for device-to-device pub/sub.

**Scenario 5: Real-Time Features (Chat, Collaboration)**

**Winner: gRPC**

Reasons:
- Native bidirectional streaming
- Lower latency than REST + SSE/WebSockets
- Same infrastructure as other services

REST approach:
- WebSockets (different protocol)
- Server-Sent Events (one-way)
- Both require separate infrastructure

Example:
\`\`\`protobuf
service ChatService {
  rpc Chat(stream Message) returns (stream Message);
}
\`\`\`

**Scenario 6: File Upload/Download**

**Winner: REST**

Reasons:
- Native HTTP multipart/form-data
- Progress tracking simpler
- Resume broken uploads
- Direct CDN integration

gRPC approach:
- Stream in chunks (more complex)
- No standard multipart support
- Custom progress tracking

Example: AWS S3, Dropbox use REST/HTTP for uploads.

**Scenario 7: Third-Party Webhook Integration**

**Winner: REST**

Reasons:
- Webhooks are HTTP POST requests
- Easy for third parties to send
- No special client libraries needed
- Ubiquitous support

gRPC: Not suitable (third parties unlikely to have gRPC clients).

**Summary Table**:

| Use Case | Choice | Primary Reason |
|----------|--------|----------------|
| Public API | REST | Browser support, debugging |
| Microservices | gRPC | Performance, type safety |
| Mobile app | gRPC | Efficiency, streaming |
| IoT devices | gRPC | Small payloads, binary |
| Real-time | gRPC | Bidirectional streaming |
| File uploads | REST | Native HTTP support |
| Webhooks | REST | Universal HTTP support |
| Admin dashboard | REST | Debugging ease |

**Hybrid Approach** (Best of Both):

Many companies offer both:
\`\`\`
External: REST API (public developers)
Internal: gRPC (microservices)
Mobile: gRPC (performance)
Web: REST (browser native)
\`\`\`

Example: Google offers both REST and gRPC for most services.

**Decision Framework**:

1. **Performance critical?** → gRPC
2. **Browser required?** → REST
3. **External developers?** → REST
4. **Real-time/streaming?** → gRPC
5. **Simple CRUD?** → REST
6. **Type safety critical?** → gRPC
7. **Debugging ease?** → REST

Choose based on specific constraints, not dogma. Many successful systems use both.`,
          keyPoints: [
            'REST for public APIs (browser support, ease of use)',
            'gRPC for internal microservices (performance, type safety)',
            'gRPC for mobile apps (efficiency, real-time features)',
            'REST for file uploads and webhooks (native HTTP support)',
            'Hybrid approach common: both REST and gRPC in same system',
          ],
        },
      ],
    },
    {
      id: 'api-gateway-patterns',
      title: 'API Gateway Patterns',
      content: `An API Gateway is a single entry point for all client requests, handling routing, authentication, rate limiting, and more. Understanding gateway patterns is crucial for microservices architectures.

## What is an API Gateway?

**API Gateway** acts as a reverse proxy that routes requests to appropriate microservices.

### **Core Responsibilities**

1. **Routing**: Direct requests to backend services
2. **Authentication & Authorization**: Centralized security
3. **Rate Limiting**: Prevent abuse
4. **Load Balancing**: Distribute traffic
5. **Request/Response Transformation**: Adapt data formats
6. **Caching**: Improve performance
7. **Monitoring & Logging**: Centralized observability
8. **Protocol Translation**: REST ↔ gRPC, GraphQL

### **Benefits**

- Single entry point simplifies client code
- Centralized cross-cutting concerns
- Reduces microservice coupling
- Enables API versioning and migration
- Provides aggregation layer

### **Drawbacks**

- Single point of failure (mitigate with HA)
- Potential bottleneck
- Additional latency
- Complexity in configuration

## Gateway Patterns

### **1. Simple Gateway (Reverse Proxy)**

Routes requests based on path:

\`\`\`javascript
// NGINX configuration
server {
  listen 80;
  
  location /users {
    proxy_pass http://user-service:3000;
  }
  
  location /products {
    proxy_pass http://product-service:3001;
  }
  
  location /orders {
    proxy_pass http://order-service:3002;
  }
}
\`\`\`

**Express.js Gateway**:

\`\`\`javascript
const express = require('express');
const httpProxy = require('http-proxy-middleware');

const app = express();

// Route to user service
app.use('/users', httpProxy.createProxyMiddleware({
  target: 'http://user-service:3000',
  changeOrigin: true
}));

// Route to product service
app.use('/products', httpProxy.createProxyMiddleware({
  target: 'http://product-service:3001',
  changeOrigin: true
}));

app.listen(80);
\`\`\`

### **2. Aggregator Gateway**

Combines multiple backend calls into single response:

\`\`\`javascript
app.get('/dashboard/:userId', async (req, res) => {
  const { userId } = req.params;
  
  // Call multiple services in parallel
  const [user, orders, recommendations] = await Promise.all([
    fetch(\`http://user-service/users/\${userId}\`),
    fetch(\`http://order-service/orders?userId=\${userId}\`),
    fetch(\`http://recommendation-service/recommend/\${userId}\`)
  ]);
  
  // Aggregate response
  res.json({
    user: await user.json(),
    orders: await orders.json(),
    recommendations: await recommendations.json()
  });
});
\`\`\`

**Benefits**: Reduces client requests from 3 to 1

**Use case**: Mobile apps with limited bandwidth

### **3. Backend for Frontend (BFF)**

Separate gateway per client type:

\`\`\`
┌─────────────┐
│  Web Client │──────┐
└─────────────┘      │
                     v
┌─────────────┐  ┌──────────┐
│Mobile Client│─>│  Web BFF │──┐
└─────────────┘  └──────────┘  │
                                │  ┌──────────────┐
┌─────────────┐  ┌────────────┐│  │User Service  │
│  IoT Device │─>│ Mobile BFF ││─>│Product Service│
└─────────────┘  └────────────┘  │Order Service  │
                                  └──────────────┘
\`\`\`

**Web BFF** (returns full HTML):

\`\`\`javascript
app.get('/product/:id', async (req, res) => {
  const product = await fetch(\`http://product-service/products/\${req.params.id}\`);
  const reviews = await fetch(\`http://review-service/reviews?productId=\${req.params.id}\`);
  
  res.json({
    ...product,
    reviews: reviews,
    displayPrice: formatCurrency(product.price)  // Web-specific formatting
  });
});
\`\`\`

**Mobile BFF** (returns minimal data):

\`\`\`javascript
app.get('/product/:id', async (req, res) => {
  const product = await fetch(\`http://product-service/products/\${req.params.id}\`);
  
  // Mobile: don't include reviews by default (save bandwidth)
  res.json({
    id: product.id,
    name: product.name,
    price: product.price,  // Client formats
    imageUrl: product.thumbnailUrl  // Smaller image
  });
});
\`\`\`

### **4. GraphQL Gateway**

Single GraphQL endpoint federating multiple services:

\`\`\`javascript
const { ApolloServer } = require('apollo-server');
const { ApolloGateway } = require('@apollo/gateway');

const gateway = new ApolloGateway({
  serviceList: [
    { name: 'users', url: 'http://user-service/graphql' },
    { name: 'products', url: 'http://product-service/graphql' },
    { name: 'orders', url: 'http://order-service/graphql' }
  ]
});

const server = new ApolloServer({
  gateway,
  subscriptions: false
});

server.listen(4000);
\`\`\`

**Client Query** (single request):

\`\`\`graphql
query {
  user(id: "123") {
    name
    orders {           # From orders service
      product {        # From products service
        name
        price
      }
    }
  }
}
\`\`\`

Gateway automatically routes to appropriate services.

### **5. Service Mesh Gateway**

Gateway integrated with service mesh (Istio, Linkerd):

\`\`\`yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: api-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "api.example.com"

---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: routes
spec:
  hosts:
  - "api.example.com"
  gateways:
  - api-gateway
  http:
  - match:
    - uri:
        prefix: "/users"
    route:
    - destination:
        host: user-service
        port:
          number: 3000
  - match:
    - uri:
        prefix: "/products"
    route:
    - destination:
        host: product-service
        port:
          number: 3001
\`\`\`

**Benefits**: 
- Advanced traffic management (canary, A/B testing)
- mTLS between services
- Distributed tracing
- Circuit breaking

## Authentication & Authorization

### **JWT Validation at Gateway**

\`\`\`javascript
const jwt = require('jsonwebtoken');

// Middleware
async function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    
    // Add user context to downstream services
    req.headers['x-user-id'] = decoded.userId;
    req.headers['x-user-role'] = decoded.role;
    
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// Apply to routes
app.use('/api', authenticate);
\`\`\`

### **OAuth 2.0 Integration**

\`\`\`javascript
app.get('/oauth/callback', async (req, res) => {
  const { code } = req.query;
  
  // Exchange code for token
  const tokenResponse = await fetch('https://oauth-provider.com/token', {
    method: 'POST',
    body: JSON.stringify({
      code,
      client_id: process.env.CLIENT_ID,
      client_secret: process.env.CLIENT_SECRET,
      grant_type: 'authorization_code'
    })
  });
  
  const { access_token } = await tokenResponse.json();
  
  // Store token (session, cookie, etc.)
  res.cookie('access_token', access_token, { httpOnly: true });
  res.redirect('/dashboard');
});
\`\`\`

## Rate Limiting at Gateway

\`\`\`javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');

const limiter = rateLimit({
  store: new RedisStore({
    client: redisClient
  }),
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 100,  // 100 requests per window
  keyGenerator: (req) => {
    // Rate limit by user ID if authenticated
    return req.user?.id || req.ip;
  },
  handler: (req, res) => {
    res.status(429).json({
      error: 'Too many requests',
      retryAfter: req.rateLimit.resetTime
    });
  }
});

app.use('/api', limiter);
\`\`\`

## Request Transformation

### **API Versioning**

\`\`\`javascript
app.use('/v1/users/:id', async (req, res) => {
  const user = await fetch(\`http://user-service/users/\${req.params.id}\`);
  const data = await user.json();
  
  // V1 format (legacy)
  res.json({
    id: data.id,
    full_name: data.name,  // Old field name
    email: data.email
  });
});

app.use('/v2/users/:id', async (req, res) => {
  const user = await fetch(\`http://user-service/users/\${req.params.id}\`);
  const data = await user.json();
  
  // V2 format (new)
  res.json({
    id: data.id,
    name: data.name,  // New field name
    email: data.email,
    createdAt: data.created_at  // Additional field
  });
});
\`\`\`

### **Protocol Translation**

\`\`\`javascript
const grpc = require('@grpc/grpc-js');

// REST endpoint that calls gRPC service
app.get('/users/:id', async (req, res) => {
  // Translate REST to gRPC
  const grpcClient = new UserServiceClient('user-service:50051');
  
  grpcClient.getUser({ id: req.params.id }, (err, response) => {
    if (err) {
      return res.status(500).json({ error: err.message });
    }
    
    // Return as JSON
    res.json(response);
  });
});
\`\`\`

## Caching at Gateway

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

async function cacheMiddleware(req, res, next) {
  const cacheKey = \`cache:\${req.method}:\${req.url}\`;
  
  // Check cache
  const cached = await client.get(cacheKey);
  if (cached) {
    return res.json(JSON.parse(cached));
  }
  
  // Override res.json to cache response
  const originalJson = res.json;
  res.json = function(data) {
    // Cache for 5 minutes
    client.setex(cacheKey, 300, JSON.stringify(data));
    originalJson.call(this, data);
  };
  
  next();
}

// Cache GET requests
app.get('/products', cacheMiddleware, async (req, res) => {
  const products = await fetch('http://product-service/products');
  res.json(await products.json());
});
\`\`\`

## Circuit Breaking

\`\`\`javascript
const CircuitBreaker = require('opossum');

// Circuit breaker for user service
const breaker = new CircuitBreaker(
  async (userId) => {
    const response = await fetch(\`http://user-service/users/\${userId}\`);
    return response.json();
  },
  {
    timeout: 3000,      // 3s timeout
    errorThreshold: 50, // Open after 50% errors
    resetTimeout: 30000 // Try again after 30s
  }
);

// Fallback
breaker.fallback(() => ({
  id: null,
  name: 'Guest',
  email: null
}));

app.get('/users/:id', async (req, res) => {
  try {
    const user = await breaker.fire(req.params.id);
    res.json(user);
  } catch (error) {
    res.status(503).json({ error: 'Service unavailable' });
  }
});
\`\`\`

## Monitoring & Logging

\`\`\`javascript
const morgan = require('morgan');
const prometheus = require('prom-client');

// HTTP request logger
app.use(morgan('combined'));

// Prometheus metrics
const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status']
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration.labels(
      req.method,
      req.route?.path || req.path,
      res.statusCode
    ).observe(duration);
  });
  
  next();
});

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
});
\`\`\`

## Popular Gateway Solutions

### **1. Kong**

Open-source, Lua-based, plugin architecture:

\`\`\`yaml
services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: user-route
        paths:
          - /users
    plugins:
      - name: rate-limiting
        config:
          minute: 100
      - name: jwt
      - name: cors
\`\`\`

### **2. AWS API Gateway**

Managed service:
- Serverless (Lambda integration)
- Built-in authorization (Cognito, IAM)
- Usage plans and API keys
- Request/response transformation
- CloudWatch monitoring

### **3. Envoy**

High-performance proxy used by Istio:
- Service mesh integration
- Advanced load balancing
- Observability (Jaeger tracing)
- WebAssembly plugins

### **4. Traefik**

Kubernetes-native, automatic service discovery:

\`\`\`yaml
apiVersion: traefik.containo.us/v1alpha1
kind: IngressRoute
metadata:
  name: api-gateway
spec:
  entryPoints:
    - web
  routes:
    - match: PathPrefix(\`/users\`)
      kind: Rule
      services:
        - name: user-service
          port: 3000
    - match: PathPrefix(\`/products\`)
      kind: Rule
      services:
        - name: product-service
          port: 3001
\`\`\`

### **5. Apollo Gateway** (GraphQL)

Federation for GraphQL:
- Combines multiple GraphQL services
- Schema stitching
- Query planning

## Best Practices

1. **Keep gateway stateless**: Use Redis for shared state
2. **Implement health checks**: Monitor backend services
3. **Use connection pooling**: Reuse HTTP connections
4. **Set timeouts**: Prevent hanging requests
5. **Log request IDs**: Trace requests across services
6. **Cache aggressively**: Reduce backend load
7. **Rate limit by user**: Prevent abuse
8. **Use HTTPS**: Always encrypt traffic
9. **Implement circuit breakers**: Handle failures gracefully
10. **Monitor gateway metrics**: Latency, error rates, throughput

## Gateway Anti-Patterns

### **❌ Business Logic in Gateway**

Don't:
\`\`\`javascript
app.get('/orders/:id', async (req, res) => {
  const order = await fetch(\`http://order-service/orders/\${req.params.id}\`);
  
  // ❌ Complex business logic in gateway
  if (order.total > 1000 && !order.isPremiumUser) {
    return res.status(403).json({ error: 'Premium users only' });
  }
  
  res.json(order);
});
\`\`\`

Do: Push logic to backend services.

### **❌ Direct Database Access**

Gateway should never access databases directly. Always go through services.

### **❌ Gateway Chaining**

Avoid: Gateway → Gateway → Service (adds latency)

## Real-World Example: E-commerce

\`\`\`javascript
const express = require('express');
const app = express();

// Authentication
app.use(authenticate);

// Product search (cached, public)
app.get('/search', cache(60), rateLimit(1000), async (req, res) => {
  const results = await fetch(\`http://search-service/search?q=\${req.query.q}\`);
  res.json(await results.json());
});

// User dashboard (aggregated, authenticated)
app.get('/dashboard', async (req, res) => {
  const [profile, orders, recommendations] = await Promise.all([
    fetch(\`http://user-service/users/\${req.user.id}\`),
    fetch(\`http://order-service/orders?userId=\${req.user.id}\`),
    fetch(\`http://recommendation-service/recommend/\${req.user.id}\`)
  ]);
  
  res.json({
    profile: await profile.json(),
    recentOrders: await orders.json(),
    recommended: await recommendations.json()
  });
});

// Checkout (critical path, no caching)
app.post('/checkout', rateLimit(10), async (req, res) => {
  // Orchestrate checkout across services
  const payment = await fetch('http://payment-service/charge', {
    method: 'POST',
    body: JSON.stringify(req.body)
  });
  
  if (!payment.ok) {
    return res.status(402).json({ error: 'Payment failed' });
  }
  
  const order = await fetch('http://order-service/orders', {
    method: 'POST',
    body: JSON.stringify({ userId: req.user.id, ...req.body })
  });
  
  res.json(await order.json());
});

app.listen(80);
\`\`\`

## When to Use API Gateway

**Use API Gateway when**:
- Microservices architecture (many backend services)
- Need centralized auth, rate limiting, logging
- Multiple client types (web, mobile, IoT)
- API versioning and migration needed
- Aggregation of multiple calls

**Skip API Gateway when**:
- Monolithic architecture (direct access simpler)
- Very simple system (single service)
- Extremely low latency required (every hop adds latency)
- Internal-only services (service mesh might be better)`,
      multipleChoice: [
        {
          id: 'gateway-q1',
          question:
            'What is the primary benefit of using the Backend for Frontend (BFF) pattern?',
          options: [
            'Reduces server costs by sharing infrastructure',
            'Allows customizing API responses for different client types (web, mobile, IoT)',
            'Automatically translates REST to GraphQL',
            'Eliminates the need for authentication',
          ],
          correctAnswer: 1,
          explanation:
            "BFF pattern creates separate gateway per client type, allowing customization: mobile gets minimal data (bandwidth), web gets full HTML, IoT gets compact binary. This optimizes each client without compromising others. It doesn't share infrastructure (opposite), doesn't translate protocols, or handle auth automatically.",
          difficulty: 'medium',
        },
        {
          id: 'gateway-q2',
          question:
            'Why should business logic NOT be implemented in an API gateway?',
          options: [
            'Gateways are too slow to execute complex logic',
            'It violates separation of concerns and makes the gateway harder to maintain',
            'Gateways cannot access databases',
            "It's impossible to test business logic in gateways",
          ],
          correctAnswer: 1,
          explanation:
            "API gateways should handle cross-cutting concerns (auth, routing, rate limiting), not business logic. Business logic belongs in backend services for maintainability, testability, and separation of concerns. Gateways aren't inherently slow, can access DBs (but shouldn't), and can be tested.",
          difficulty: 'easy',
        },
        {
          id: 'gateway-q3',
          question:
            'What is an Aggregator Gateway pattern and when is it useful?',
          options: [
            'A gateway that compresses responses; useful for slow networks',
            'A gateway that combines multiple backend calls into single response; useful for mobile apps',
            'A gateway that aggregates logs; useful for monitoring',
            'A gateway that pools database connections; useful for high traffic',
          ],
          correctAnswer: 1,
          explanation:
            'Aggregator Gateway makes parallel calls to multiple services and combines responses into one. This reduces client requests from N to 1, critical for mobile apps with limited bandwidth and battery. Example: dashboard fetching user + orders + recommendations in one request instead of three.',
          difficulty: 'easy',
        },
        {
          id: 'gateway-q4',
          question:
            'How does circuit breaking at the API gateway improve system reliability?',
          options: [
            'It encrypts traffic between services',
            'It prevents cascading failures by failing fast when a service is down',
            'It balances load across multiple gateway instances',
            'It caches responses to reduce backend calls',
          ],
          correctAnswer: 1,
          explanation:
            'Circuit breaker monitors failures and "opens" (stops sending requests) when threshold exceeded, preventing cascading failures and giving backend time to recover. After timeout, it tries again (half-open). This is different from load balancing, caching, or encryption.',
          difficulty: 'medium',
        },
        {
          id: 'gateway-q5',
          question: 'What is a potential drawback of using an API gateway?',
          options: [
            'It makes microservices more tightly coupled',
            'It can become a single point of failure and performance bottleneck',
            'It prevents using multiple programming languages',
            'It requires rewriting all backend services',
          ],
          correctAnswer: 1,
          explanation:
            "API gateway is a single entry point, making it a potential single point of failure (mitigate with HA/redundancy) and bottleneck (all traffic goes through it). It actually reduces coupling (not increases), doesn't affect backend languages, and doesn't require rewriting services.",
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'gateway-d1',
          question:
            'Design an API gateway for a social media platform handling web, mobile, and third-party developers. Include authentication, rate limiting, and caching strategies.',
          sampleAnswer: `Comprehensive API gateway design for social media platform:

**Architecture Overview**:

\`\`\`
┌─────────────┐
│  Web Client │─────┐
└─────────────┘     │
                    v
┌──────────────┐  ┌──────────────────┐
│ Mobile Client│─>│  API Gateway      │
└──────────────┘  │  (Kong/Express)   │
                  └──────────────────┘
┌──────────────┐         │
│  3rd Party   │─────────┘
│  Developers  │
└──────────────┘

         │
         v
┌─────────────────────────────────┐
│  Backend Microservices          │
│  - User Service                 │
│  - Post Service                 │
│  - Feed Service                 │
│  - Notification Service         │
│  - Analytics Service            │
└─────────────────────────────────┘
\`\`\`

**1. Gateway Implementation (Express.js)**

\`\`\`javascript
const express = require('express');
const jwt = require('jsonwebtoken');
const redis = require('redis');
const rateLimit = require('express-rate-limit');
const CircuitBreaker = require('opossum');

const app = express();
const cache = redis.createClient();

// ============ Authentication ============

// Middleware: JWT validation
async function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    
    // Add user context for downstream services
    req.headers['x-user-id'] = decoded.userId;
    req.headers['x-user-role'] = decoded.role;
    
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// API Key validation for third-party developers
async function validateApiKey(req, res, next) {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey) {
    return res.status(401).json({ error: 'API key required' });
  }
  
  // Check Redis cache first
  const cached = await cache.get(\`apikey:\${apiKey}\`);
  if (cached) {
    req.apiClient = JSON.parse(cached);
    return next();
  }
  
  // Validate with API key service
  const client = await fetch(\`http://auth-service/api-keys/\${apiKey}\`);
  
  if (!client.ok) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  const clientData = await client.json();
  
  // Cache for 1 hour
  await cache.setex(\`apikey:\${apiKey}\`, 3600, JSON.stringify(clientData));
  
  req.apiClient = clientData;
  next();
}

// ============ Rate Limiting ============

// User rate limiting (authenticated)
const userRateLimiter = rateLimit({
  store: new (require('rate-limit-redis'))({
    client: cache
  }),
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 1000,                  // 1000 requests
  keyGenerator: (req) => req.user?.id || req.ip,
  handler: (req, res) => {
    res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: req.rateLimit.resetTime
    });
  }
});

// Third-party API rate limiting (stricter)
const apiKeyRateLimiter = rateLimit({
  store: new (require('rate-limit-redis'))({
    client: cache
  }),
  windowMs: 60 * 60 * 1000,  // 1 hour
  max: (req) => {
    // Tiered pricing: different limits per plan
    const plan = req.apiClient?.plan || 'free';
    return {
      free: 100,
      basic: 1000,
      premium: 10000
    }[plan];
  },
  keyGenerator: (req) => req.apiClient.id,
  handler: (req, res) => {
    res.status(429).json({
      error: 'API quota exceeded',
      plan: req.apiClient.plan,
      upgradeUrl: 'https://example.com/pricing'
    });
  }
});

// ============ Caching ============

// Cache middleware
async function cacheMiddleware(ttl) {
  return async (req, res, next) => {
    // Only cache GET requests
    if (req.method !== 'GET') return next();
    
    const cacheKey = \`cache:\${req.originalUrl}:\${req.user?.id || 'public'}\`;
    
    // Check cache
    const cached = await cache.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }
    
    // Override res.json to cache response
    const originalJson = res.json;
    res.json = function(data) {
      cache.setex(cacheKey, ttl, JSON.stringify(data));
      originalJson.call(this, data);
    };
    
    next();
  };
}

// ============ Circuit Breakers ============

const serviceBreakers = {
  user: new CircuitBreaker(fetchUserService, {
    timeout: 3000,
    errorThreshold: 50,
    resetTimeout: 30000
  }),
  post: new CircuitBreaker(fetchPostService, {
    timeout: 3000,
    errorThreshold: 50,
    resetTimeout: 30000
  }),
  feed: new CircuitBreaker(fetchFeedService, {
    timeout: 5000,
    errorThreshold: 50,
    resetTimeout: 30000
  })
};

// Fallbacks
serviceBreakers.user.fallback(() => ({
  id: null,
  username: 'Unknown',
  avatar: '/default-avatar.png'
}));

// ============ Routes ============

// Public routes (no auth)
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// User routes (JWT auth)
app.get('/users/:id',
  authenticate,
  userRateLimiter,
  cacheMiddleware(300),  // Cache 5 min
  async (req, res) => {
    try {
      const user = await serviceBreakers.user.fire(req.params.id);
      res.json(user);
    } catch (error) {
      res.status(503).json({ error: 'Service unavailable' });
    }
  }
);

// Feed aggregation (mobile-optimized)
app.get('/feed',
  authenticate,
  userRateLimiter,
  cacheMiddleware(60),  // Cache 1 min
  async (req, res) => {
    try {
      // Parallel requests
      const [posts, trending, suggested] = await Promise.all([
        serviceBreakers.feed.fire(req.user.id),
        fetch('http://trending-service/trending'),
        fetch(\`http://recommendation-service/suggest/\${req.user.id}\`)
      ]);
      
      res.json({
        posts: await posts,
        trending: await trending.json(),
        suggested: await suggested.json()
      });
    } catch (error) {
      res.status(503).json({ error: 'Service unavailable' });
    }
  }
);

// Third-party API routes (API key auth)
app.get('/api/v1/posts',
  validateApiKey,
  apiKeyRateLimiter,
  cacheMiddleware(300),
  async (req, res) => {
    const posts = await fetch('http://post-service/posts', {
      headers: {
        'x-api-client': req.apiClient.id
      }
    });
    
    res.json(await posts.json());
  }
);

// Write operations (no caching)
app.post('/posts',
  authenticate,
  userRateLimiter,
  async (req, res) => {
    const post = await fetch('http://post-service/posts', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-user-id': req.user.id
      },
      body: JSON.stringify(req.body)
    });
    
    // Invalidate feed cache
    await cache.del(\`cache:/feed:\${req.user.id}\`);
    
    res.status(201).json(await post.json());
  }
);

// ============ Monitoring ============

const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status']
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration.labels(
      req.method,
      req.route?.path || 'unknown',
      res.statusCode
    ).observe(duration);
  });
  
  next();
});

app.get('/metrics', async (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(await prometheus.register.metrics());
});

// ============ Error Handling ============

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({
    error: 'Internal server error',
    requestId: req.headers['x-request-id']
  });
});

app.listen(80);
\`\`\`

**2. Kong Configuration Alternative**

\`\`\`yaml
services:
  - name: user-service
    url: http://user-service:3000
    routes:
      - name: user-routes
        paths:
          - /users
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 1000
      - name: proxy-cache
        config:
          response_code: [200]
          request_method: [GET]
          content_type: [application/json]
          cache_ttl: 300

  - name: post-service
    url: http://post-service:3001
    routes:
      - name: post-routes
        paths:
          - /posts
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 100
      - name: response-transformer
        config:
          remove:
            headers: [X-Internal-Token]

  - name: public-api
    url: http://api-service:3002
    routes:
      - name: api-routes
        paths:
          - /api
    plugins:
      - name: key-auth
      - name: rate-limiting
        config:
          hour: 1000
      - name: request-transformer
        config:
          add:
            headers: [X-API-Client:$(headers.x-api-key)]
\`\`\`

**3. High Availability Setup**

\`\`\`
             Load Balancer
                  │
       ┌──────────┼──────────┐
       v          v          v
   Gateway 1  Gateway 2  Gateway 3
   (Active)   (Active)   (Active)
       │          │          │
       └──────────┴──────────┘
                  │
            Redis Cluster
         (Shared State)
\`\`\`

**Key Design Decisions**:

1. **Authentication**: JWT for users, API keys for third parties
2. **Rate Limiting**: Redis-backed, tiered by plan
3. **Caching**: Redis, TTL varies by endpoint (1-5 min)
4. **Circuit Breaking**: Per-service, with fallbacks
5. **Monitoring**: Prometheus metrics, request tracing
6. **High Availability**: Multiple gateway instances, shared Redis

This design balances performance, security, and scalability for a social media platform.`,
          keyPoints: [
            'Separate authentication for users (JWT) vs third parties (API keys)',
            'Tiered rate limiting based on user plan or client type',
            'Aggressive caching for read-heavy operations with appropriate TTLs',
            'Circuit breakers per service with sensible fallbacks',
            'High availability with multiple gateway instances and shared state',
          ],
        },
        {
          id: 'gateway-d2',
          question:
            'Your API gateway is experiencing high latency and becoming a bottleneck. What strategies would you implement to optimize performance?',
          sampleAnswer: `Comprehensive strategies to optimize API gateway performance:

**1. Connection Pooling**

\`\`\`javascript
const http = require('http');
const https = require('https');

// Reuse HTTP connections
const httpAgent = new http.Agent({
  keepAlive: true,
  maxSockets: 100,    // Max concurrent connections
  maxFreeSockets: 10  // Keep 10 idle connections
});

const httpsAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 100
});

// Use agent in fetch calls
async function fetchWithPool(url) {
  return fetch(url, {
    agent: url.startsWith('https') ? httpsAgent : httpAgent
  });
}
\`\`\`

**Impact**: Eliminates TCP/TLS handshake overhead for each request.

**2. Response Streaming**

Don't buffer entire responses in gateway:

\`\`\`javascript
const { pipeline } = require('stream');

app.get('/large-file/:id', (req, res) => {
  // Stream directly from backend to client
  const backendStream = request(\`http://file-service/files/\${req.params.id}\`);
  
  pipeline(
    backendStream,
    res,
    (err) => {
      if (err) console.error('Stream error:', err);
    }
  );
});
\`\`\`

**3. Parallel Request Execution**

\`\`\`javascript
// Bad: Sequential (slow)
const user = await fetch('http://user-service/users/123');
const posts = await fetch('http://post-service/posts?userId=123');
const comments = await fetch('http://comment-service/comments?userId=123');

// Good: Parallel (fast)
const [user, posts, comments] = await Promise.all([
  fetch('http://user-service/users/123'),
  fetch('http://post-service/posts?userId=123'),
  fetch('http://comment-service/comments?userId=123')
]);
\`\`\`

**4. Efficient Caching**

\`\`\`javascript
const redis = require('redis');
const { promisify } = require('util');

const client = redis.createClient();
const getAsync = promisify(client.get).bind(client);
const setexAsync = promisify(client.setex).bind(client);

// Cache with stale-while-revalidate
async function cacheWithSWR(key, fetchFn, ttl) {
  const cached = await getAsync(key);
  
  if (cached) {
    const data = JSON.parse(cached);
    
    // If within 80% of TTL, return cached
    if (data.cachedAt + (ttl * 0.8 * 1000) > Date.now()) {
      return data.value;
    }
    
    // Stale: return cached but refresh in background
    fetchFn().then(fresh => {
      setexAsync(key, ttl, JSON.stringify({
        value: fresh,
        cachedAt: Date.now()
      }));
    });
    
    return data.value;
  }
  
  // Cache miss: fetch and cache
  const fresh = await fetchFn();
  await setexAsync(key, ttl, JSON.stringify({
    value: fresh,
    cachedAt: Date.now()
  }));
  
  return fresh;
}
\`\`\`

**5. Request Coalescing**

Combine duplicate concurrent requests:

\`\`\`javascript
const pendingRequests = new Map();

async function coalescedFetch(url) {
  // Check if request already in flight
  if (pendingRequests.has(url)) {
    return pendingRequests.get(url);
  }
  
  // Make request
  const promise = fetch(url).then(res => res.json());
  pendingRequests.set(url, promise);
  
  // Clean up after response
  promise.finally(() => {
    pendingRequests.delete(url);
  });
  
  return promise;
}
\`\`\`

**6. Compression**

\`\`\`javascript
const compression = require('compression');

app.use(compression({
  level: 6,  // Compression level (1-9)
  threshold: 1024,  // Only compress responses > 1KB
  filter: (req, res) => {
    // Don't compress already compressed formats
    if (req.headers['x-no-compression']) return false;
    return compression.filter(req, res);
  }
}));
\`\`\`

**7. Load Balancing**

Distribute across multiple backend instances:

\`\`\`javascript
const backends = [
  'http://user-service-1:3000',
  'http://user-service-2:3000',
  'http://user-service-3:3000'
];

let currentIndex = 0;

function roundRobinBackend() {
  const backend = backends[currentIndex];
  currentIndex = (currentIndex + 1) % backends.length;
  return backend;
}

app.get('/users/:id', async (req, res) => {
  const backend = roundRobinBackend();
  const user = await fetch(\`\${backend}/users/\${req.params.id}\`);
  res.json(await user.json());
});
\`\`\`

**8. Timeout Optimization**

Set aggressive timeouts to fail fast:

\`\`\`javascript
const AbortController = require('abort-controller');

async function fetchWithTimeout(url, timeout = 3000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      signal: controller.signal
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}
\`\`\`

**9. Selective Aggregation**

Don't always aggregate everything:

\`\`\`javascript
app.get('/dashboard', async (req, res) => {
  const { include } = req.query;  // ?include=orders,recommendations
  
  const requests = { profile: fetch(\`http://user-service/users/\${req.user.id}\`) };
  
  // Conditionally fetch based on client needs
  if (include?.includes('orders')) {
    requests.orders = fetch(\`http://order-service/orders?userId=\${req.user.id}\`);
  }
  
  if (include?.includes('recommendations')) {
    requests.recommendations = fetch(\`http://recommendation-service/recommend/\${req.user.id}\`);
  }
  
  const results = await Promise.all(Object.values(requests));
  
  res.json({
    profile: await results[0].json(),
    ...(requests.orders && { orders: await results[1].json() }),
    ...(requests.recommendations && { recommendations: await results[2].json() })
  });
});
\`\`\`

**10. Horizontal Scaling**

Deploy multiple gateway instances:

\`\`\`yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 5  # Multiple instances
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: gateway
        image: api-gateway:latest
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        env:
        - name: REDIS_URL
          value: redis://redis-cluster
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
\`\`\`

**11. Monitoring & Profiling**

Identify slow paths:

\`\`\`javascript
const prometheus = require('prom-client');

const gatewayDuration = new prometheus.Histogram({
  name: 'gateway_request_duration_seconds',
  help: 'Gateway request duration',
  labelNames: ['route', 'backend'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
});

app.use(async (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    gatewayDuration.labels(req.route?.path, 'gateway').observe(duration);
    
    // Alert on slow requests
    if (duration > 1) {
      console.warn('Slow request:', {
        route: req.route?.path,
        duration,
        method: req.method
      });
    }
  });
  
  next();
});
\`\`\`

**12. Service Mesh (Advanced)**

Offload traffic management to service mesh:

\`\`\`yaml
# Istio VirtualService for intelligent routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
  - user-service
  http:
  - match:
    - headers:
        x-user-role:
          exact: premium
    route:
    - destination:
        host: user-service
        subset: premium-pool  # Premium users → faster instances
  - route:
    - destination:
        host: user-service
        subset: standard-pool
\`\`\`

**Performance Impact Summary**:

| Strategy | Latency Reduction | Complexity |
|----------|-------------------|------------|
| Connection Pooling | 20-30% | Low |
| Response Streaming | 40-50% (large responses) | Low |
| Parallel Requests | 50-70% | Low |
| Caching | 80-95% (cache hit) | Medium |
| Request Coalescing | 30-50% (duplicate requests) | Medium |
| Compression | 10-20% | Low |
| Horizontal Scaling | Linear with instances | Low |
| Service Mesh | 10-20% | High |

**Recommended Implementation Order**:

1. Connection pooling (easy win)
2. Caching strategy
3. Parallel requests
4. Compression
5. Horizontal scaling
6. Advanced: Request coalescing, service mesh

This comprehensive approach can reduce gateway latency by 60-80% in typical scenarios.`,
          keyPoints: [
            'Connection pooling eliminates TCP handshake overhead',
            'Parallel request execution reduces aggregation latency',
            'Stale-while-revalidate caching serves cached data while refreshing',
            'Request coalescing combines duplicate concurrent requests',
            'Horizontal scaling distributes load across multiple gateway instances',
          ],
        },
        {
          id: 'gateway-d3',
          question:
            'Compare API Gateway vs Service Mesh for microservices architecture. When would you choose one over the other?',
          sampleAnswer: `Detailed comparison of API Gateway vs Service Mesh:

**API Gateway vs Service Mesh: Core Differences**

\`\`\`
API Gateway (North-South Traffic)
┌──────────┐
│  Client  │
└────┬─────┘
     │
     v
┌────────────────┐
│  API Gateway   │  ← Single entry point
└────────────────┘
     │
     v
┌─────────────────────────┐
│  Microservices          │
│  ┌────┐ ┌────┐ ┌────┐ │
│  │ A  │ │ B  │ │ C  │ │
│  └────┘ └────┘ └────┘ │
└─────────────────────────┘

Service Mesh (East-West Traffic)
┌──────────┐
│  Client  │
└────┬─────┘
     │
     v
┌─────────────────────────┐
│  Microservices          │
│  ┌────┐  ┌────┐  ┌────┐│
│  │ A  │->│ B  │->│ C  ││ ← Mesh handles service-to-service
│  └────┘  └────┘  └────┘│
│  Each service has sidecar proxy (Envoy)
└─────────────────────────┘
\`\`\`

**Comparison Table**:

| Feature | API Gateway | Service Mesh |
|---------|-------------|--------------|
| **Purpose** | External client → services | Service → service |
| **Traffic** | North-South (ingress) | East-West (internal) |
| **Deployment** | Centralized (gateway tier) | Decentralized (sidecar per service) |
| **Routing** | Path-based (/users → user-service) | Service-based (service A → service B) |
| **Auth** | Client authentication (OAuth, API keys) | mTLS between services |
| **Use case** | Public API, third-party integration | Microservices communication |
| **Complexity** | Low-Medium | High |
| **Latency** | +5-20ms | +1-5ms per hop |
| **Examples** | Kong, AWS API Gateway, Traefik | Istio, Linkerd, Consul Connect |

**Scenario 1: E-commerce with External API**

**Use API Gateway when**:
- Exposing APIs to external clients (web, mobile, partners)
- Need centralized authentication (OAuth, API keys)
- Want to aggregate multiple services
- API versioning and transformation needed
- Rate limiting per client

\`\`\`javascript
// API Gateway handles external traffic
External Client → API Gateway → [User Service, Order Service, Product Service]

// Configuration
{
  routes: [
    { path: '/users', target: 'user-service' },
    { path: '/products', target: 'product-service' },
    { path: '/orders', target: 'order-service' }
  ],
  auth: 'JWT',
  rateLimit: 1000,
  caching: true
}
\`\`\`

**Scenario 2: Internal Microservices (100+ services)**

**Use Service Mesh when**:
- Heavy service-to-service communication
- Need mTLS for zero-trust security
- Want automatic retry, circuit breaking, load balancing
- Require distributed tracing across services
- Service discovery and health checking

\`\`\`yaml
# Istio handles internal traffic
Order Service → (Envoy sidecar) → Payment Service
             → (Envoy sidecar) → Inventory Service
             → (Envoy sidecar) → Notification Service

# Istio configuration
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: payment-service
spec:
  hosts:
  - payment-service
  http:
  - route:
    - destination:
        host: payment-service
        subset: v2
      weight: 90
    - destination:
        host: payment-service
        subset: v1
      weight: 10  # Canary deployment
    retries:
      attempts: 3
      perTryTimeout: 2s
    timeout: 10s
\`\`\`

**Scenario 3: Hybrid (Best of Both Worlds)**

Most production systems use **both**:

\`\`\`
External Client
     │
     v
┌────────────────┐
│  API Gateway   │  ← Handles external traffic
└────────────────┘
     │
     v
┌─────────────────────────────────┐
│  Service Mesh (Istio)           │
│  ┌─────────────────────────┐   │
│  │  Order Service          │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Envoy sidecar    │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
│         │                       │
│         v                       │
│  ┌─────────────────────────┐   │
│  │  Payment Service        │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Envoy sidecar    │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
\`\`\`

**Implementation Example**:

\`\`\`yaml
# 1. API Gateway (Kong) for external traffic
apiVersion: configuration.konghq.com/v1
kind: KongIngress
metadata:
  name: external-api
spec:
  route:
    protocols:
    - https
    methods:
    - GET
    - POST
  upstream:
    host: order-service.default.svc.cluster.local

---
# 2. Service Mesh (Istio) for internal traffic
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-to-payment
spec:
  hosts:
  - payment-service
  http:
  - match:
    - sourceLabels:
        app: order-service
    route:
    - destination:
        host: payment-service
    retries:
      attempts: 3
    timeout: 5s

---
# 3. mTLS between services (Istio)
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT  # Enforce mTLS for all services
\`\`\`

**Decision Framework**:

**Choose API Gateway when**:
1. ✅ Need external API exposure
2. ✅ Require API keys, OAuth, third-party auth
3. ✅ Want to aggregate multiple services
4. ✅ Simple architecture (< 20 services)
5. ✅ Need API versioning and transformation
6. ✅ Quick setup (hours/days)

**Choose Service Mesh when**:
1. ✅ Large microservices architecture (> 50 services)
2. ✅ Heavy internal service communication
3. ✅ Need zero-trust security (mTLS)
4. ✅ Want automatic retries, circuit breaking, load balancing
5. ✅ Require distributed tracing
6. ✅ Team comfortable with complexity (weeks/months setup)

**Choose Both when**:
1. ✅ Large system with external APIs
2. ✅ Complex microservices architecture
3. ✅ Need both external and internal traffic management
4. ✅ Security is critical (external auth + internal mTLS)

**Real-World Examples**:

**Uber**:
- API Gateway: Zuul (for mobile/web clients)
- Service Mesh: Custom solution (internal service communication)

**Lyft**:
- API Gateway: Envoy (also creator of Envoy)
- Service Mesh: Envoy-based mesh

**Netflix**:
- API Gateway: Zuul 2
- Service Mesh: Not using traditional mesh (custom solutions)

**Small Startup (< 10 services)**:
- API Gateway: Yes (Kong or AWS API Gateway)
- Service Mesh: No (overkill)

**Enterprise (> 100 services)**:
- API Gateway: Yes (external traffic)
- Service Mesh: Yes (internal traffic)

**Cost Comparison**:

| Aspect | API Gateway | Service Mesh |
|--------|-------------|--------------|
| Ops complexity | Low-Medium | High |
| Resource overhead | 1-2 gateway instances | Sidecar per service (2x pods) |
| Latency impact | 5-20ms (one hop) | 1-5ms per hop |
| Learning curve | Days-Weeks | Weeks-Months |
| Debugging | Easy | Difficult |

**Verdict**:

- **Start with API Gateway** for external traffic (always needed)
- **Add Service Mesh** when:
  - > 50 microservices
  - Internal traffic is complex
  - Security/compliance requires mTLS
  - Team has expertise

Don't prematurely add service mesh complexity. Many successful companies run 20-30 services with just an API gateway.`,
          keyPoints: [
            'API Gateway handles external (North-South) traffic; Service Mesh handles internal (East-West) traffic',
            'API Gateway provides client auth, rate limiting, aggregation; Service Mesh provides mTLS, retries, tracing',
            'Most large systems use both: API Gateway for external, Service Mesh for internal',
            'Start with API Gateway; add Service Mesh only when microservices grow (> 50 services)',
            'Service Mesh adds significant complexity; only adopt if benefits outweigh costs',
          ],
        },
      ],
    },
    {
      id: 'api-rate-limiting',
      title: 'API Rate Limiting Strategies',
      content: `Rate limiting prevents API abuse, ensures fair resource allocation, and protects backend services from overload. Implementing effective rate limiting is critical for production APIs.

## Why Rate Limiting?

### **Benefits**

1. **Prevent abuse**: Block malicious actors and scrapers
2. **Fair usage**: Ensure no single client monopolizes resources
3. **Cost control**: Limit expensive operations
4. **Service protection**: Prevent cascade failures
5. **Monetization**: Tiered pricing based on usage

### **Without Rate Limiting**

- Attackers can DDoS your API
- Single client can overwhelm system
- Expensive operations (analytics, search) drain resources
- No way to monetize API usage tiers

## Rate Limiting Algorithms

### **1. Fixed Window**

Count requests in fixed time windows:

\`\`\`javascript
const redis = require('redis');
const client = redis.createClient();

async function fixedWindowRateLimit(userId, limit, windowSeconds) {
  const key = \`ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / windowSeconds)}\`;
  
  const current = await client.incr(key);
  
  if (current === 1) {
    // First request in window, set expiry
    await client.expire(key, windowSeconds);
  }
  
  if (current > limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: (Math.floor(Date.now() / 1000 / windowSeconds) + 1) * windowSeconds
    };
  }
  
  return {
    allowed: true,
    remaining: limit - current,
    resetAt: (Math.floor(Date.now() / 1000 / windowSeconds) + 1) * windowSeconds
  };
}

// Example: 100 requests per 60 seconds
const result = await fixedWindowRateLimit('user123', 100, 60);
\`\`\`

**Problem**: **Burst at window boundaries**

\`\`\`
Window 1 (0-60s): 100 requests at t=59s
Window 2 (60-120s): 100 requests at t=60s
Result: 200 requests in 1 second! 💥
\`\`\`

### **2. Sliding Window Log**

Track timestamp of each request:

\`\`\`javascript
async function slidingWindowLogRateLimit(userId, limit, windowMs) {
  const key = \`ratelimit:log:\${userId}\`;
  const now = Date.now();
  const windowStart = now - windowMs;
  
  // Remove old entries
  await client.zremrangebyscore(key, 0, windowStart);
  
  // Count requests in window
  const count = await client.zcard(key);
  
  if (count >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: await client.zrange(key, 0, 0, 'WITHSCORES')
        .then(([_, timestamp]) => parseInt(timestamp) + windowMs)
    };
  }
  
  // Add current request
  await client.zadd(key, now, \`\${now}-\${Math.random()}\`);
  await client.expire(key, Math.ceil(windowMs / 1000));
  
  return {
    allowed: true,
    remaining: limit - count - 1,
    resetAt: now + windowMs
  };
}
\`\`\`

**Pros**: Accurate, no burst issues

**Cons**: Memory intensive (stores every request timestamp)

### **3. Sliding Window Counter (Hybrid)**

Best of both worlds:

\`\`\`javascript
async function slidingWindowCounter(userId, limit, windowSeconds) {
  const now = Date.now() / 1000;
  const currentWindow = Math.floor(now / windowSeconds);
  const previousWindow = currentWindow - 1;
  
  const currentKey = \`ratelimit:\${userId}:\${currentWindow}\`;
  const previousKey = \`ratelimit:\${userId}:\${previousWindow}\`;
  
  // Get counts
  const currentCount = parseInt(await client.get(currentKey) || '0');
  const previousCount = parseInt(await client.get(previousKey) || '0');
  
  // Calculate weighted count based on time elapsed in current window
  const percentageInCurrent = (now % windowSeconds) / windowSeconds;
  const weightedCount = 
    previousCount * (1 - percentageInCurrent) + currentCount;
  
  if (weightedCount >= limit) {
    return {
      allowed: false,
      remaining: 0,
      resetAt: (currentWindow + 1) * windowSeconds
    };
  }
  
  // Increment current window
  await client.incr(currentKey);
  await client.expire(currentKey, windowSeconds * 2);
  
  return {
    allowed: true,
    remaining: Math.floor(limit - weightedCount - 1),
    resetAt: (currentWindow + 1) * windowSeconds
  };
}
\`\`\`

**Pros**: Memory efficient, smooth rate limiting

**Cons**: Slightly more complex logic

**This is the recommended approach for most use cases.**

### **4. Token Bucket**

Tokens replenish at fixed rate:

\`\`\`javascript
async function tokenBucketRateLimit(userId, capacity, refillRate) {
  const key = \`ratelimit:bucket:\${userId}\`;
  
  const data = await client.get(key);
  let tokens, lastRefill;
  
  if (data) {
    ({ tokens, lastRefill } = JSON.parse(data));
  } else {
    tokens = capacity;
    lastRefill = Date.now();
  }
  
  // Refill tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastRefill) / 1000;
  const tokensToAdd = elapsed * refillRate;
  tokens = Math.min(capacity, tokens + tokensToAdd);
  
  if (tokens < 1) {
    return {
      allowed: false,
      remaining: 0,
      retryAfter: (1 - tokens) / refillRate
    };
  }
  
  // Consume 1 token
  tokens -= 1;
  
  await client.set(key, JSON.stringify({
    tokens,
    lastRefill: now
  }), 'EX', 3600);
  
  return {
    allowed: true,
    remaining: Math.floor(tokens),
    retryAfter: null
  };
}

// Example: 100 token capacity, refill 10 tokens/second
await tokenBucketRateLimit('user123', 100, 10);
\`\`\`

**Pros**: Allows bursts (up to capacity), smooth refill

**Use case**: APIs where bursts are acceptable

### **5. Leaky Bucket**

Process requests at fixed rate, queue overflow:

\`\`\`javascript
async function leakyBucketRateLimit(userId, capacity, leakRate) {
  const queueKey = \`ratelimit:queue:\${userId}\`;
  const lastLeakKey = \`ratelimit:lastleak:\${userId}\`;
  
  // Get current queue size
  let queueSize = await client.llen(queueKey);
  const lastLeak = parseInt(await client.get(lastLeakKey) || Date.now());
  
  // Leak tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastLeak) / 1000;
  const tokensToLeak = Math.floor(elapsed * leakRate);
  
  if (tokensToLeak > 0) {
    const leaked = Math.min(tokensToLeak, queueSize);
    for (let i = 0; i < leaked; i++) {
      await client.rpop(queueKey);
    }
    queueSize -= leaked;
    await client.set(lastLeakKey, now);
  }
  
  if (queueSize >= capacity) {
    return {
      allowed: false,
      retryAfter: (queueSize - capacity + 1) / leakRate
    };
  }
  
  // Add request to queue
  await client.lpush(queueKey, now);
  await client.expire(queueKey, 3600);
  
  return {
    allowed: true,
    remaining: capacity - queueSize - 1
  };
}
\`\`\`

**Pros**: Smooth traffic, prevents bursts

**Use case**: APIs with consistent processing capacity

## Rate Limiting Strategies

### **1. Per-User Rate Limiting**

\`\`\`javascript
app.use(async (req, res, next) => {
  const userId = req.user?.id || req.ip;
  
  const result = await slidingWindowCounter(userId, 1000, 3600); // 1000/hour
  
  res.set({
    'X-RateLimit-Limit': 1000,
    'X-RateLimit-Remaining': result.remaining,
    'X-RateLimit-Reset': result.resetAt
  });
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: result.resetAt
    });
  }
  
  next();
});
\`\`\`

### **2. Tiered Rate Limiting**

Different limits per plan:

\`\`\`javascript
const RATE_LIMITS = {
  free: { requests: 100, window: 3600 },
  basic: { requests: 1000, window: 3600 },
  premium: { requests: 10000, window: 3600 }
};

app.use(async (req, res, next) => {
  const plan = req.user?.plan || 'free';
  const { requests, window } = RATE_LIMITS[plan];
  
  const result = await slidingWindowCounter(req.user.id, requests, window);
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Rate limit exceeded',
      plan,
      upgradeUrl: '/pricing'
    });
  }
  
  next();
});
\`\`\`

### **3. Endpoint-Specific Rate Limiting**

Different limits per endpoint:

\`\`\`javascript
// Expensive search endpoint
app.get('/search', 
  rateLimiter({ requests: 10, window: 60 }),  // 10/min
  async (req, res) => {
    // ...
  }
);

// Lightweight read
app.get('/users/:id',
  rateLimiter({ requests: 1000, window: 60 }),  // 1000/min
  async (req, res) => {
    // ...
  }
);

// Write operations
app.post('/posts',
  rateLimiter({ requests: 100, window: 60 }),  // 100/min
  async (req, res) => {
    // ...
  }
);
\`\`\`

### **4. Cost-Based Rate Limiting**

Different costs per operation:

\`\`\`javascript
const OPERATION_COSTS = {
  'GET /users/:id': 1,
  'GET /search': 10,
  'POST /analytics': 50,
  'GET /reports': 100
};

async function costBasedRateLimit(userId, operation, budget) {
  const cost = OPERATION_COSTS[operation] || 1;
  const key = \`ratelimit:cost:\${userId}\`;
  
  const spent = parseInt(await client.get(key) || '0');
  
  if (spent + cost > budget) {
    return {
      allowed: false,
      cost,
      spent,
      budget
    };
  }
  
  await client.incrby(key, cost);
  await client.expire(key, 3600);  // Reset hourly
  
  return {
    allowed: true,
    cost,
    spent: spent + cost,
    budget
  };
}

app.use(async (req, res, next) => {
  const operation = \`\${req.method} \${req.route.path}\`;
  const result = await costBasedRateLimit(req.user.id, operation, 10000);
  
  if (!result.allowed) {
    return res.status(429).json({
      error: 'Cost budget exceeded',
      cost: result.cost,
      spent: result.spent,
      budget: result.budget
    });
  }
  
  next();
});
\`\`\`

### **5. Concurrent Request Limiting**

Limit simultaneous requests:

\`\`\`javascript
const activeSemaphores = new Map();

async function concurrentLimiter(userId, maxConcurrent) {
  const key = \`ratelimit:concurrent:\${userId}\`;
  
  const current = await client.incr(key);
  
  if (current > maxConcurrent) {
    await client.decr(key);
    return {
      allowed: false,
      current: current - 1,
      max: maxConcurrent
    };
  }
  
  await client.expire(key, 300);  // Safety: expire after 5 min
  
  return {
    allowed: true,
    current,
    max: maxConcurrent,
    release: async () => {
      await client.decr(key);
    }
  };
}

app.use(async (req, res, next) => {
  const limiter = await concurrentLimiter(req.user.id, 10);
  
  if (!limiter.allowed) {
    return res.status(429).json({
      error: 'Too many concurrent requests',
      current: limiter.current,
      max: limiter.max
    });
  }
  
  // Release on response finish
  res.on('finish', limiter.release);
  res.on('close', limiter.release);
  
  next();
});
\`\`\`

## Response Headers

Standard rate limit headers:

\`\`\`javascript
app.use(async (req, res, next) => {
  const result = await rateLimit(req.user.id);
  
  // Standard headers (GitHub, Stripe, Twitter use these)
  res.set({
    'X-RateLimit-Limit': result.limit,
    'X-RateLimit-Remaining': result.remaining,
    'X-RateLimit-Reset': result.resetAt,  // Unix timestamp
    'X-RateLimit-Used': result.used
  });
  
  if (!result.allowed) {
    res.set({
      'Retry-After': result.retryAfter  // Seconds until retry
    });
    
    return res.status(429).json({
      error: 'Too many requests',
      message: 'Rate limit exceeded. Try again later.',
      limit: result.limit,
      remaining: 0,
      resetAt: result.resetAt,
      retryAfter: result.retryAfter
    });
  }
  
  next();
});
\`\`\`

## Distributed Rate Limiting

### **Redis-Based (Most Common)**

\`\`\`javascript
const Redis = require('ioredis');

// Redis Cluster for high availability
const redis = new Redis.Cluster([
  { host: 'redis-1', port: 6379 },
  { host: 'redis-2', port: 6379 },
  { host: 'redis-3', port: 6379 }
]);

// Lua script for atomic rate limiting
const RATE_LIMIT_SCRIPT = \`
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
  redis.call('EXPIRE', key, window)
end
if current > limit then
  return {0, limit, 0}
else
  local ttl = redis.call('TTL', key)
  return {1, limit, limit - current, ttl}
end
\`;

async function distributedRateLimit(userId, limit, window) {
  const key = \`ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / window)}\`;
  
  const [allowed, max, remaining, ttl] = await redis.eval(
    RATE_LIMIT_SCRIPT,
    1,
    key,
    limit,
    window
  );
  
  return {
    allowed: allowed === 1,
    limit: max,
    remaining,
    resetAt: Date.now() / 1000 + ttl
  };
}
\`\`\`

### **Rate Limiting at Multiple Layers**

\`\`\`
┌─────────────────┐
│  CDN / WAF      │  ← Layer 1: DDoS protection (IP-based)
└────────┬────────┘
         │
┌────────▼────────┐
│  Load Balancer  │  ← Layer 2: Connection limits
└────────┬────────┘
         │
┌────────▼────────┐
│  API Gateway    │  ← Layer 3: Per-user rate limits
└────────┬────────┘
         │
┌────────▼────────┐
│  Service        │  ← Layer 4: Cost-based limits
└─────────────────┘
\`\`\`

## Handling Rate Limit Errors (Client-Side)

### **Exponential Backoff with Jitter**

\`\`\`javascript
async function fetchWithRetry(url, options = {}, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        const resetAt = response.headers.get('X-RateLimit-Reset');
        
        if (attempt < maxRetries - 1) {
          let delay;
          
          if (retryAfter) {
            // Use server-provided retry-after
            delay = parseInt(retryAfter) * 1000;
          } else if (resetAt) {
            // Calculate delay until reset
            delay = (parseInt(resetAt) * 1000) - Date.now();
          } else {
            // Exponential backoff with jitter
            const baseDelay = Math.pow(2, attempt) * 1000;
            const jitter = Math.random() * 1000;
            delay = baseDelay + jitter;
          }
          
          console.log(\`Rate limited. Retrying in \${delay}ms...\`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
      }
      
      return response;
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
    }
  }
}
\`\`\`

## Bypass Rate Limits (Whitelist)

\`\`\`javascript
const WHITELISTED_IPS = new Set([
  '10.0.0.0/8',    // Internal services
  '203.0.113.0'    // Trusted partner
]);

const WHITELISTED_USERS = new Set([
  'admin-user-id',
  'monitoring-service'
]);

app.use(async (req, res, next) => {
  // Check whitelist
  if (WHITELISTED_IPS.has(req.ip) || 
      WHITELISTED_USERS.has(req.user?.id)) {
    return next();  // Skip rate limiting
  }
  
  // Apply rate limiting
  const result = await rateLimit(req.user.id);
  if (!result.allowed) {
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }
  
  next();
});
\`\`\`

## Monitoring Rate Limits

\`\`\`javascript
const prometheus = require('prom-client');

const rateLimitCounter = new prometheus.Counter({
  name: 'api_rate_limit_exceeded_total',
  help: 'Number of rate limit exceeded errors',
  labelNames: ['user_plan', 'endpoint']
});

const rateLimitUsage = new prometheus.Histogram({
  name: 'api_rate_limit_usage_percent',
  help: 'Rate limit usage percentage',
  labelNames: ['user_plan'],
  buckets: [10, 25, 50, 75, 90, 95, 100]
});

app.use(async (req, res, next) => {
  const result = await rateLimit(req.user.id);
  
  const usagePercent = ((result.used / result.limit) * 100);
  rateLimitUsage.labels(req.user.plan).observe(usagePercent);
  
  if (!result.allowed) {
    rateLimitCounter.labels(req.user.plan, req.route.path).inc();
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }
  
  next();
});
\`\`\`

## Best Practices

1. **Use sliding window counter** for most cases (balance of accuracy and memory)
2. **Set headers** (\`X-RateLimit-*\`) so clients know their usage
3. **Return 429 status code** for rate limit errors
4. **Provide \`Retry-After\`** header with seconds until retry
5. **Tier limits** by user plan (free, basic, premium)
6. **Different limits** for different endpoints (expensive vs cheap)
7. **Whitelist** internal services and admins
8. **Monitor** rate limit usage and exceeded counts
9. **Use Redis** for distributed systems
10. **Document** rate limits clearly in API docs

## Real-World Examples

**GitHub API**:
- 5,000 requests/hour for authenticated users
- 60 requests/hour for unauthenticated
- Cost-based: search queries cost more

**Stripe API**:
- Rolling rate limits (not fixed windows)
- Different limits per endpoint
- Test vs production limits

**Twitter API**:
- Tiered plans (free, basic, pro, enterprise)
- Per-user and per-app limits
- 15-minute windows

## Rate Limiting Pitfalls

### **❌ Fixed Window Burst**

Problem: 200 requests at window boundary

Solution: Use sliding window counter

### **❌ No Retry-After Header**

Clients spam retries, making it worse

Solution: Always return \`Retry-After\`

### **❌ Same Limit for All Endpoints**

Expensive operations (search, analytics) should have lower limits

Solution: Cost-based or endpoint-specific limits

### **❌ In-Memory Rate Limiting (Single Instance)**

Doesn't work with multiple servers

Solution: Use Redis for distributed rate limiting`,
      multipleChoice: [
        {
          id: 'ratelimit-q1',
          question: 'What is the main problem with Fixed Window rate limiting?',
          options: [
            'It uses too much memory to store request timestamps',
            'It allows bursts at window boundaries (e.g., 200 requests in 1 second across 2 windows)',
            'It requires Redis for distributed systems',
            "It doesn't provide accurate rate limiting over time",
          ],
          correctAnswer: 1,
          explanation:
            'Fixed Window allows bursts at boundaries: user sends 100 requests at t=59s (end of window 1) and 100 at t=60s (start of window 2) = 200 requests in 1 second, bypassing the limit. Sliding Window Counter solves this by weighting counts across windows.',
          difficulty: 'medium',
        },
        {
          id: 'ratelimit-q2',
          question:
            'Which rate limiting algorithm is most memory efficient while providing smooth rate limiting?',
          options: [
            'Fixed Window (stores one counter per window)',
            'Sliding Window Log (stores every request timestamp)',
            'Sliding Window Counter (stores two counters with weighted calculation)',
            'Token Bucket (stores token count and last refill time)',
          ],
          correctAnswer: 2,
          explanation:
            'Sliding Window Counter is most efficient: stores only 2 counters (current + previous window) and uses weighted calculation. Sliding Window Log stores ALL timestamps (memory intensive). Token Bucket is efficient but allows bursts. Fixed Window has burst issues.',
          difficulty: 'hard',
        },
        {
          id: 'ratelimit-q3',
          question:
            'What HTTP status code should be returned when a client exceeds rate limits?',
          options: [
            '403 Forbidden',
            '503 Service Unavailable',
            '429 Too Many Requests',
            '400 Bad Request',
          ],
          correctAnswer: 2,
          explanation:
            '429 Too Many Requests is the standard HTTP status code for rate limiting. 403 is for authorization failures, 503 is for server unavailability, 400 is for invalid requests. Always return 429 with Retry-After header.',
          difficulty: 'easy',
        },
        {
          id: 'ratelimit-q4',
          question:
            'Why is cost-based rate limiting useful for APIs with diverse operations?',
          options: [
            'It reduces server costs by limiting expensive operations',
            'It allows assigning different "costs" to operations (e.g., search costs 10x more than read)',
            'It encrypts expensive operations for security',
            'It automatically scales backend resources based on cost',
          ],
          correctAnswer: 1,
          explanation:
            'Cost-based rate limiting assigns different costs to operations based on resource usage. Expensive operations (analytics, search) cost more points than simple reads. User has point budget (e.g., 10,000/hour). This prevents abuse of expensive endpoints without overly restricting cheap operations.',
          difficulty: 'medium',
        },
        {
          id: 'ratelimit-q5',
          question:
            'What is the purpose of the Retry-After header in rate limit responses?',
          options: [
            'To tell the client how many retries are allowed',
            'To specify how many seconds the client should wait before retrying',
            'To indicate the retry strategy (exponential backoff vs linear)',
            'To automatically retry the request on the client side',
          ],
          correctAnswer: 1,
          explanation:
            'Retry-After header tells client how many seconds to wait before retrying. This prevents clients from spam-retrying immediately, which would worsen the situation. Standard practice: return Retry-After with 429 status code.',
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'ratelimit-d1',
          question:
            'Design a comprehensive rate limiting system for a SaaS API with free, basic, and premium tiers. Include per-endpoint limits, cost-based limits, and handling of burst traffic.',
          sampleAnswer: `Comprehensive rate limiting system for SaaS API:

**1. Tiered Rate Limits**

\`\`\`javascript
const RATE_LIMITS = {
  free: {
    hourly: 100,
    daily: 1000,
    concurrent: 2,
    costBudget: 1000
  },
  basic: {
    hourly: 1000,
    daily: 20000,
    concurrent: 10,
    costBudget: 10000
  },
  premium: {
    hourly: 10000,
    daily: 500000,
    concurrent: 50,
    costBudget: 100000
  }
};

// Endpoint-specific costs
const OPERATION_COSTS = {
  // Reads (cheap)
  'GET /users/:id': 1,
  'GET /posts/:id': 1,
  
  // List operations (moderate)
  'GET /users': 5,
  'GET /posts': 5,
  
  // Search (expensive)
  'GET /search': 10,
  'POST /search/advanced': 20,
  
  // Analytics (very expensive)
  'POST /analytics/reports': 50,
  'POST /analytics/export': 100,
  
  // Writes (moderate)
  'POST /users': 3,
  'PUT /users/:id': 3,
  'DELETE /users/:id': 3
};
\`\`\`

**2. Multi-Layer Rate Limiter**

\`\`\`javascript
const Redis = require('ioredis');
const redis = new Redis.Cluster([
  { host: 'redis-1', port: 6379 },
  { host: 'redis-2', port: 6379 }
]);

class ComprehensiveRateLimiter {
  // Sliding window counter for hourly/daily limits
  async slidingWindowLimit(userId, limit, windowSeconds) {
    const now = Date.now() / 1000;
    const currentWindow = Math.floor(now / windowSeconds);
    const previousWindow = currentWindow - 1;
    
    const currentKey = \`ratelimit:\${userId}:\${windowSeconds}:\${currentWindow}\`;
    const previousKey = \`ratelimit:\${userId}:\${windowSeconds}:\${previousWindow}\`;
    
    const [currentCount, previousCount] = await Promise.all([
      redis.get(currentKey).then(v => parseInt(v || '0')),
      redis.get(previousKey).then(v => parseInt(v || '0'))
    ]);
    
    const percentageInCurrent = (now % windowSeconds) / windowSeconds;
    const weightedCount = 
      previousCount * (1 - percentageInCurrent) + currentCount;
    
    if (weightedCount >= limit) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: (currentWindow + 1) * windowSeconds
      };
    }
    
    await redis.incr(currentKey);
    await redis.expire(currentKey, windowSeconds * 2);
    
    return {
      allowed: true,
      remaining: Math.floor(limit - weightedCount - 1),
      resetAt: (currentWindow + 1) * windowSeconds
    };
  }
  
  // Cost-based limiting
  async costBasedLimit(userId, operation, budget) {
    const cost = OPERATION_COSTS[operation] || 1;
    const key = \`ratelimit:cost:\${userId}\`;
    const now = Date.now();
    const hourStart = Math.floor(now / 1000 / 3600);
    const hourKey = \`\${key}:\${hourStart}\`;
    
    const spent = parseInt(await redis.get(hourKey) || '0');
    
    if (spent + cost > budget) {
      return {
        allowed: false,
        cost,
        spent,
        budget,
        remaining: 0
      };
    }
    
    await redis.incrby(hourKey, cost);
    await redis.expire(hourKey, 7200); // 2 hours
    
    return {
      allowed: true,
      cost,
      spent: spent + cost,
      budget,
      remaining: budget - spent - cost
    };
  }
  
  // Concurrent request limiting
  async concurrentLimit(userId, maxConcurrent) {
    const key = \`ratelimit:concurrent:\${userId}\`;
    const current = await redis.incr(key);
    
    if (current > maxConcurrent) {
      await redis.decr(key);
      return {
        allowed: false,
        current: current - 1,
        max: maxConcurrent
      };
    }
    
    await redis.expire(key, 300);
    
    return {
      allowed: true,
      current,
      max: maxConcurrent,
      release: async () => await redis.decr(key)
    };
  }
  
  // Token bucket for burst handling
  async tokenBucketLimit(userId, capacity, refillRate) {
    const key = \`ratelimit:burst:\${userId}\`;
    
    const data = await redis.get(key);
    let tokens, lastRefill;
    
    if (data) {
      const parsed = JSON.parse(data);
      tokens = parsed.tokens;
      lastRefill = parsed.lastRefill;
    } else {
      tokens = capacity;
      lastRefill = Date.now();
    }
    
    // Refill tokens
    const now = Date.now();
    const elapsed = (now - lastRefill) / 1000;
    const tokensToAdd = elapsed * refillRate;
    tokens = Math.min(capacity, tokens + tokensToAdd);
    
    if (tokens < 1) {
      return {
        allowed: false,
        remaining: 0,
        retryAfter: (1 - tokens) / refillRate
      };
    }
    
    tokens -= 1;
    
    await redis.set(key, JSON.stringify({
      tokens,
      lastRefill: now
    }), 'EX', 3600);
    
    return {
      allowed: true,
      remaining: Math.floor(tokens)
    };
  }
}

const limiter = new ComprehensiveRateLimiter();
\`\`\`

**3. Middleware Implementation**

\`\`\`javascript
async function rateLimitMiddleware(req, res, next) {
  const userId = req.user.id;
  const plan = req.user.plan || 'free';
  const limits = RATE_LIMITS[plan];
  const operation = \`\${req.method} \${req.route.path}\`;
  
  // Check 1: Hourly limit (sliding window)
  const hourly = await limiter.slidingWindowLimit(
    \`\${userId}:hourly\`,
    limits.hourly,
    3600
  );
  
  if (!hourly.allowed) {
    return rateLimitError(res, {
      type: 'hourly_limit',
      limit: limits.hourly,
      resetAt: hourly.resetAt
    });
  }
  
  // Check 2: Daily limit
  const daily = await limiter.slidingWindowLimit(
    \`\${userId}:daily\`,
    limits.daily,
    86400
  );
  
  if (!daily.allowed) {
    return rateLimitError(res, {
      type: 'daily_limit',
      limit: limits.daily,
      resetAt: daily.resetAt
    });
  }
  
  // Check 3: Cost-based limit
  const cost = await limiter.costBasedLimit(
    userId,
    operation,
    limits.costBudget
  );
  
  if (!cost.allowed) {
    return rateLimitError(res, {
      type: 'cost_budget',
      cost: cost.cost,
      spent: cost.spent,
      budget: cost.budget
    });
  }
  
  // Check 4: Concurrent requests
  const concurrent = await limiter.concurrentLimit(
    userId,
    limits.concurrent
  );
  
  if (!concurrent.allowed) {
    return rateLimitError(res, {
      type: 'concurrent_limit',
      current: concurrent.current,
      max: concurrent.max
    });
  }
  
  // Release concurrent slot on finish
  res.on('finish', concurrent.release);
  res.on('close', concurrent.release);
  
  // Check 5: Burst protection (token bucket)
  const burst = await limiter.tokenBucketLimit(
    userId,
    50,    // 50 token capacity
    10     // Refill 10 tokens/second
  );
  
  if (!burst.allowed) {
    return rateLimitError(res, {
      type: 'burst_limit',
      retryAfter: burst.retryAfter
    });
  }
  
  // Set rate limit headers
  res.set({
    'X-RateLimit-Limit-Hourly': limits.hourly,
    'X-RateLimit-Remaining-Hourly': hourly.remaining,
    'X-RateLimit-Reset-Hourly': hourly.resetAt,
    
    'X-RateLimit-Limit-Daily': limits.daily,
    'X-RateLimit-Remaining-Daily': daily.remaining,
    'X-RateLimit-Reset-Daily': daily.resetAt,
    
    'X-RateLimit-Cost-Budget': limits.costBudget,
    'X-RateLimit-Cost-Remaining': cost.remaining,
    'X-RateLimit-Cost-Used': cost.spent,
    
    'X-RateLimit-Concurrent-Max': limits.concurrent,
    'X-RateLimit-Concurrent-Current': concurrent.current
  });
  
  next();
}

function rateLimitError(res, details) {
  res.status(429).json({
    error: 'Rate limit exceeded',
    type: details.type,
    details,
    retryAfter: details.retryAfter || 
      (details.resetAt - Date.now() / 1000)
  });
}

app.use(rateLimitMiddleware);
\`\`\`

**4. Admin Dashboard**

\`\`\`javascript
// Real-time rate limit monitoring
app.get('/admin/rate-limits/:userId', async (req, res) => {
  const { userId } = req.params;
  const plan = await getUserPlan(userId);
  const limits = RATE_LIMITS[plan];
  
  const [hourly, daily, costSpent, concurrent] = await Promise.all([
    redis.get(\`ratelimit:\${userId}:hourly:*\`),
    redis.get(\`ratelimit:\${userId}:daily:*\`),
    redis.get(\`ratelimit:cost:\${userId}:*\`),
    redis.get(\`ratelimit:concurrent:\${userId}\`)
  ]);
  
  res.json({
    userId,
    plan,
    hourly: {
      limit: limits.hourly,
      used: parseInt(hourly || '0'),
      remaining: limits.hourly - parseInt(hourly || '0')
    },
    daily: {
      limit: limits.daily,
      used: parseInt(daily || '0'),
      remaining: limits.daily - parseInt(daily || '0')
    },
    cost: {
      budget: limits.costBudget,
      spent: parseInt(costSpent || '0'),
      remaining: limits.costBudget - parseInt(costSpent || '0')
    },
    concurrent: {
      max: limits.concurrent,
      current: parseInt(concurrent || '0')
    }
  });
});
\`\`\`

**5. Whitelist**

\`\`\`javascript
const WHITELISTED_USERS = new Set([
  'admin-user',
  'monitoring-service',
  'internal-api'
]);

// Skip rate limiting for whitelisted users
async function rateLimitMiddleware(req, res, next) {
  if (WHITELISTED_USERS.has(req.user.id)) {
    return next();
  }
  
  // Apply rate limiting...
}
\`\`\`

**Key Design Decisions**:

1. **Multi-layer checks**: Hourly, daily, cost-based, concurrent, burst
2. **Sliding window counter**: Smooth rate limiting without burst issues
3. **Cost-based**: Expensive operations cost more points
4. **Token bucket**: Allows short bursts while maintaining average rate
5. **Comprehensive headers**: Clients know their usage across all dimensions
6. **Redis cluster**: Distributed, high-availability rate limiting

This system prevents abuse while allowing legitimate usage patterns.`,
          keyPoints: [
            'Multi-layer rate limiting: hourly, daily, cost-based, concurrent, burst',
            'Sliding window counter prevents burst issues at window boundaries',
            'Cost-based limiting assigns higher costs to expensive operations',
            'Token bucket allows short bursts while enforcing average rate',
            'Comprehensive headers inform clients of usage across all dimensions',
          ],
        },
        {
          id: 'ratelimit-d2',
          question:
            'Your API is experiencing abuse from sophisticated attackers who rotate IP addresses and create multiple free accounts. How would you implement advanced rate limiting to prevent this?',
          sampleAnswer: `Advanced rate limiting strategies against sophisticated abuse:

**1. Fingerprinting-Based Rate Limiting**

\`\`\`javascript
const FingerprintJS = require('@fingerprintjs/fingerprintjs');

// Generate device fingerprint
async function getDeviceFingerprint(req) {
  const factors = [
    req.headers['user-agent'],
    req.headers['accept-language'],
    req.headers['accept-encoding'],
    req.connection.remoteAddress,
    // Additional factors from client fingerprinting
    req.body.screenResolution,
    req.body.timezone,
    req.body.browserPlugins
  ];
  
  const hash = crypto.createHash('sha256')
    .update(factors.join('|'))
    .digest('hex');
  
  return hash;
}

// Rate limit by fingerprint (catches IP rotation)
async function fingerprintRateLimit(req, res, next) {
  const fingerprint = await getDeviceFingerprint(req);
  const key = \`ratelimit:fingerprint:\${fingerprint}\`;
  
  const count = await redis.incr(key);
  await redis.expire(key, 3600);
  
  if (count > 100) {  // 100 requests/hour per device
    return res.status(429).json({
      error: 'Rate limit exceeded',
      message: 'Too many requests from this device'
    });
  }
  
  next();
}
\`\`\`

**2. Behavioral Analysis**

\`\`\`javascript
// Track suspicious patterns
class BehavioralAnalyzer {
  async analyzeRequest(req) {
    const userId = req.user?.id || req.ip;
    const patterns = await this.getPatterns(userId);
    
    const signals = {
      // Signal 1: Rapid account creation
      newAccountRate: await this.checkAccountCreationRate(req.ip),
      
      // Signal 2: Unusual access patterns
      accessPattern: await this.checkAccessPattern(userId),
      
      // Signal 3: Low-value interactions
      interactionQuality: await this.checkInteractionQuality(userId),
      
      // Signal 4: Automation detection
      isAutomated: await this.detectAutomation(req)
    };
    
    const suspicionScore = this.calculateSuspicionScore(signals);
    
    return {
      suspicionScore,
      signals,
      action: this.determineAction(suspicionScore)
    };
  }
  
  async checkAccountCreationRate(ip) {
    const key = \`behavioral:accounts:\${ip}\`;
    const accounts = await redis.zcount(
      key,
      Date.now() - 3600000,  // Last hour
      Date.now()
    );
    
    // More than 5 accounts/hour from same IP = suspicious
    return accounts > 5 ? 1.0 : accounts / 5;
  }
  
  async checkAccessPattern(userId) {
    const key = \`behavioral:pattern:\${userId}\`;
    const requests = await redis.lrange(key, 0, -1);
    
    // Check for bot-like patterns:
    // - Too consistent timing (exactly every N seconds)
    // - Sequential resource access (user1, user2, user3...)
    // - No variation in user-agent
    
    const timestamps = requests.map(r => JSON.parse(r).timestamp);
    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i-1]);
    }
    
    // Calculate variance (bots have low variance)
    const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, interval) => 
      sum + Math.pow(interval - mean, 2), 0) / intervals.length;
    
    // Low variance = bot-like
    return variance < 100 ? 1.0 : 0.0;
  }
  
  async checkInteractionQuality(userId) {
    const key = \`behavioral:quality:\${userId}\`;
    const actions = await redis.hgetall(key);
    
    const reads = parseInt(actions.reads || '0');
    const writes = parseInt(actions.writes || '0');
    
    // Attackers mostly read (scraping), legitimate users write
    const readWriteRatio = writes === 0 ? reads : reads / writes;
    
    // High read-to-write ratio = suspicious
    return readWriteRatio > 100 ? 1.0 : readWriteRatio / 100;
  }
  
  async detectAutomation(req) {
    // Check for automation signals
    const signals = [
      // No cookies/session
      !req.headers.cookie,
      
      // Suspicious user-agent
      /bot|crawler|spider|scraper/i.test(req.headers['user-agent']),
      
      // Missing common headers
      !req.headers['accept-language'],
      
      // Headless browser detection
      req.headers['chrome-lighthouse'] !== undefined,
      
      // Too fast (< 100ms between requests)
      await this.checkRequestTiming(req.user?.id || req.ip)
    ];
    
    const automationScore = signals.filter(Boolean).length / signals.length;
    return automationScore;
  }
  
  calculateSuspicionScore(signals) {
    // Weighted scoring
    return (
      signals.newAccountRate * 0.3 +
      signals.accessPattern * 0.3 +
      signals.interactionQuality * 0.2 +
      signals.isAutomated * 0.2
    );
  }
  
  determineAction(score) {
    if (score > 0.8) return 'block';
    if (score > 0.6) return 'challenge';  // CAPTCHA
    if (score > 0.4) return 'throttle';   // Stricter rate limit
    return 'allow';
  }
}

const analyzer = new BehavioralAnalyzer();

app.use(async (req, res, next) => {
  const analysis = await analyzer.analyzeRequest(req);
  
  if (analysis.action === 'block') {
    return res.status(403).json({
      error: 'Access denied',
      message: 'Suspicious activity detected'
    });
  }
  
  if (analysis.action === 'challenge') {
    // Require CAPTCHA
    if (!req.body.captchaToken) {
      return res.status(403).json({
        error: 'CAPTCHA required',
        challengeUrl: '/captcha'
      });
    }
  }
  
  if (analysis.action === 'throttle') {
    // Apply stricter rate limit
    req.rateLimit = { requests: 10, window: 3600 };  // 10/hour instead of normal
  }
  
  next();
});
\`\`\`

**3. Progressive Rate Limiting**

\`\`\`javascript
// Gradually tighten limits for suspicious users
async function progressiveRateLimit(userId) {
  const violationsKey = \`ratelimit:violations:\${userId}\`;
  const violations = parseInt(await redis.get(violationsKey) || '0');
  
  // Base limit
  let limit = 1000;
  let window = 3600;
  
  // Reduce limit with each violation
  if (violations > 0) {
    limit = Math.max(10, limit / Math.pow(2, violations));
  }
  
  const result = await slidingWindowCounter(userId, limit, window);
  
  if (!result.allowed) {
    // Increment violations
    await redis.incr(violationsKey);
    await redis.expire(violationsKey, 86400);  // Reset daily
    
    // Exponentially increase penalty
    const penaltySeconds = Math.min(
      3600,  // Max 1 hour
      Math.pow(2, violations) * 60  // Double each time
    );
    
    return {
      allowed: false,
      remaining: 0,
      penaltySeconds,
      violations: violations + 1
    };
  }
  
  return result;
}
\`\`\`

**4. Distributed Deduplication**

\`\`\`javascript
// Detect multiple accounts from same entity
async function detectSiblingAccounts(userId, fingerprint, email) {
  const emailDomain = email.split('@')[1];
  
  // Find accounts with similar signals
  const siblingKeys = [
    \`siblings:fingerprint:\${fingerprint}\`,
    \`siblings:email:\${emailDomain}\`
  ];
  
  const siblings = new Set();
  for (const key of siblingKeys) {
    const accounts = await redis.smembers(key);
    accounts.forEach(id => siblings.add(id));
  }
  
  // Add current user to sibling groups
  for (const key of siblingKeys) {
    await redis.sadd(key, userId);
    await redis.expire(key, 86400);
  }
  
  // Share rate limit across sibling accounts
  if (siblings.size > 1) {
    const combinedKey = \`ratelimit:siblings:\${Array.from(siblings).sort().join(':')}\`;
    return combinedKey;  // Use this key for rate limiting
  }
  
  return \`ratelimit:user:\${userId}\`;
}
\`\`\`

**5. CAPTCHA Integration**

\`\`\`javascript
const axios = require('axios');

async function verifyCaptcha(token) {
  const response = await axios.post(
    'https://www.google.com/recaptcha/api/siteverify',
    null,
    {
      params: {
        secret: process.env.RECAPTCHA_SECRET,
        response: token
      }
    }
  );
  
  return response.data.success && response.data.score > 0.5;
}

app.use(async (req, res, next) => {
  const suspicionScore = await analyzer.analyzeRequest(req);
  
  if (suspicionScore.action === 'challenge') {
    if (!req.body.captchaToken) {
      return res.status(403).json({
        error: 'CAPTCHA required',
        message: 'Please complete CAPTCHA to continue'
      });
    }
    
    const captchaValid = await verifyCaptcha(req.body.captchaToken);
    if (!captchaValid) {
      return res.status(403).json({
        error: 'Invalid CAPTCHA',
        message: 'CAPTCHA verification failed'
      });
    }
  }
  
  next();
});
\`\`\`

**6. IP Reputation Service**

\`\`\`javascript
// Integrate with IP reputation services
async function checkIPReputation(ip) {
  // Check against known VPN/proxy/datacenter IPs
  const response = await axios.get(
    \`https://ipqualityscore.com/api/json/ip/\${process.env.IPQS_KEY}/\${ip}\`
  );
  
  const { proxy, vpn, tor, recent_abuse, fraud_score } = response.data;
  
  if (proxy || vpn || tor || recent_abuse || fraud_score > 75) {
    return {
      suspicious: true,
      reason: proxy ? 'proxy' : vpn ? 'vpn' : 'high_fraud_score',
      score: fraud_score
    };
  }
  
  return { suspicious: false };
}

app.use(async (req, res, next) => {
  const reputation = await checkIPReputation(req.ip);
  
  if (reputation.suspicious) {
    // Apply stricter rate limit or require CAPTCHA
    req.rateLimit = { requests: 10, window: 3600 };
  }
  
  next();
});
\`\`\`

**Key Strategies**:

1. **Device fingerprinting**: Catch IP rotation
2. **Behavioral analysis**: Detect bot patterns
3. **Progressive penalties**: Increase restrictions with violations
4. **Sibling account detection**: Share limits across related accounts
5. **CAPTCHA challenges**: Human verification for suspicious activity
6. **IP reputation**: Block known malicious IPs

This multi-layered approach makes it extremely difficult for attackers to abuse the API.`,
          keyPoints: [
            'Device fingerprinting catches attackers rotating IP addresses',
            'Behavioral analysis detects bot-like patterns (timing, access patterns)',
            'Progressive rate limiting increases penalties with repeated violations',
            'Sibling account detection shares rate limits across related accounts',
            'CAPTCHA challenges provide human verification for suspicious activity',
          ],
        },
        {
          id: 'ratelimit-d3',
          question:
            'Compare different rate limiting algorithms (Fixed Window, Sliding Window, Token Bucket, Leaky Bucket) for a real-time chat API. Which would you choose and why?',
          sampleAnswer: `Comparison of rate limiting algorithms for real-time chat API:

**Scenario: Real-Time Chat API**

Requirements:
- Users send messages in bursts (normal conversation)
- Need to prevent spam (rapid message flooding)
- Must feel responsive (low latency)
- Should handle reconnections gracefully

**Algorithm Comparison**:

**1. Fixed Window**

\`\`\`javascript
// 10 messages per 60 seconds
async function fixedWindowChat(userId, limit = 10, window = 60) {
  const key = \`chat:ratelimit:\${userId}:\${Math.floor(Date.now() / 1000 / window)}\`;
  const count = await redis.incr(key);
  await redis.expire(key, window);
  
  return {
    allowed: count <= limit,
    remaining: Math.max(0, limit - count)
  };
}
\`\`\`

**Pros**:
- Simple, fast
- Low memory usage

**Cons**:
- Burst at boundaries: user sends 10 messages at t=59s, then 10 more at t=60s = 20 messages in 1 second
- Poor UX: sudden cut-off at window boundary

**Verdict for Chat**: ❌ Not suitable (burst issues)

**2. Sliding Window Counter**

\`\`\`javascript
// 10 messages per 60 seconds (smooth)
async function slidingWindowChat(userId, limit = 10, window = 60) {
  const now = Date.now() / 1000;
  const currentWindow = Math.floor(now / window);
  const previousWindow = currentWindow - 1;
  
  const currentKey = \`chat:\${userId}:\${currentWindow}\`;
  const previousKey = \`chat:\${userId}:\${previousWindow}\`;
  
  const currentCount = parseInt(await redis.get(currentKey) || '0');
  const previousCount = parseInt(await redis.get(previousKey) || '0');
  
  const percentageInCurrent = (now % window) / window;
  const weightedCount = 
    previousCount * (1 - percentageInCurrent) + currentCount;
  
  if (weightedCount >= limit) {
    return { allowed: false, remaining: 0 };
  }
  
  await redis.incr(currentKey);
  await redis.expire(currentKey, window * 2);
  
  return {
    allowed: true,
    remaining: Math.floor(limit - weightedCount - 1)
  };
}
\`\`\`

**Pros**:
- Smooth rate limiting (no burst at boundaries)
- Memory efficient (2 counters)
- Accurate

**Cons**:
- Doesn't allow bursts (chat is naturally bursty)

**Verdict for Chat**: ✅ Good, but might be too strict for conversation bursts

**3. Token Bucket** (RECOMMENDED)

\`\`\`javascript
// 10 token capacity, refill 1 token every 6 seconds
async function tokenBucketChat(userId, capacity = 10, refillRate = 1/6) {
  const key = \`chat:bucket:\${userId}\`;
  
  const data = await redis.get(key);
  let tokens, lastRefill;
  
  if (data) {
    ({ tokens, lastRefill } = JSON.parse(data));
  } else {
    tokens = capacity;
    lastRefill = Date.now();
  }
  
  // Refill tokens based on time elapsed
  const now = Date.now();
  const elapsed = (now - lastRefill) / 1000;
  const tokensToAdd = elapsed * refillRate;
  tokens = Math.min(capacity, tokens + tokensToAdd);
  
  if (tokens < 1) {
    return {
      allowed: false,
      remaining: 0,
      retryAfter: (1 - tokens) / refillRate
    };
  }
  
  // Consume 1 token
  tokens -= 1;
  
  await redis.set(key, JSON.stringify({
    tokens,
    lastRefill: now
  }), 'EX', 3600);
  
  return {
    allowed: true,
    remaining: Math.floor(tokens)
  };
}
\`\`\`

**Pros**:
- **Allows bursts** (up to capacity): Perfect for conversations!
- Smooth refill: tokens accumulate over time
- User can send 10 messages quickly, then must wait
- Feels natural for chat (burst then pause)

**Cons**:
- Slightly more complex than fixed window
- Requires storing float (tokens can be fractional)

**Verdict for Chat**: ✅✅ **BEST CHOICE** (allows natural conversation bursts)

**4. Leaky Bucket**

\`\`\`javascript
// Process messages at fixed rate (1 message every 6 seconds)
async function leakyBucketChat(userId, capacity = 10, leakRate = 1/6) {
  const queueKey = \`chat:queue:\${userId}\`;
  const lastLeakKey = \`chat:lastleak:\${userId}\`;
  
  let queueSize = await redis.llen(queueKey);
  const lastLeak = parseInt(await redis.get(lastLeakKey) || Date.now());
  
  // Leak messages
  const now = Date.now();
  const elapsed = (now - lastLeak) / 1000;
  const messagesToLeak = Math.floor(elapsed * leakRate);
  
  if (messagesToLeak > 0) {
    const leaked = Math.min(messagesToLeak, queueSize);
    for (let i = 0; i < leaked; i++) {
      await redis.rpop(queueKey);
    }
    queueSize -= leaked;
    await redis.set(lastLeakKey, now);
  }
  
  if (queueSize >= capacity) {
    return {
      allowed: false,
      queueFull: true,
      retryAfter: (queueSize - capacity + 1) / leakRate
    };
  }
  
  await redis.lpush(queueKey, now);
  await redis.expire(queueKey, 3600);
  
  return {
    allowed: true,
    queueSize: queueSize + 1
  };
}
\`\`\`

**Pros**:
- Smooth processing rate
- Good for backend with fixed capacity

**Cons**:
- **Delays messages**: User sends message but it's queued
- Poor UX for chat (feels laggy)
- Doesn't fit real-time requirement

**Verdict for Chat**: ❌ Not suitable (introduces latency)

**Comparison Table for Chat API**:

| Algorithm | Burst Handling | Smoothness | Complexity | UX | Verdict |
|-----------|----------------|------------|------------|----|---------| 
| Fixed Window | Poor (boundary bursts) | Poor | Low | Bad | ❌ |
| Sliding Window | None (strict) | Excellent | Medium | Good | ✅ |
| Token Bucket | Excellent (capacity) | Good | Medium | Excellent | ✅✅ |
| Leaky Bucket | None (queues) | Excellent | High | Poor (lag) | ❌ |

**Recommendation: Token Bucket**

**Implementation for Chat**:

\`\`\`javascript
const express = require('express');
const WebSocket = require('ws');

const app = express();
const wss = new WebSocket.Server({ port: 8080 });

// Token bucket rate limiter
class ChatRateLimiter {
  constructor() {
    this.capacity = 10;      // Allow burst of 10 messages
    this.refillRate = 1/6;   // Refill 1 token every 6 seconds (10/minute)
  }
  
  async checkLimit(userId) {
    return await tokenBucketChat(
      userId,
      this.capacity,
      this.refillRate
    );
  }
}

const rateLimiter = new ChatRateLimiter();

wss.on('connection', (ws, req) => {
  const userId = req.user.id;
  
  ws.on('message', async (message) => {
    // Rate limit check
    const limit = await rateLimiter.checkLimit(userId);
    
    if (!limit.allowed) {
      ws.send(JSON.stringify({
        type: 'error',
        error: 'Rate limit exceeded',
        message: 'Please slow down. You can send another message in ' + 
                 Math.ceil(limit.retryAfter) + ' seconds.',
        retryAfter: limit.retryAfter
      }));
      return;
    }
    
    // Send rate limit info
    ws.send(JSON.stringify({
      type: 'ratelimit',
      remaining: limit.remaining,
      capacity: rateLimiter.capacity
    }));
    
    // Broadcast message
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify({
          type: 'message',
          userId,
          text: message,
          timestamp: Date.now()
        }));
      }
    });
  });
});

app.listen(3000);
\`\`\`

**Why Token Bucket for Chat**:

1. **Burst-friendly**: Users can send 10 messages quickly (natural conversation)
2. **Prevents spam**: After burst, must wait for tokens to refill
3. **Smooth refill**: Tokens accumulate gradually (1 every 6 seconds)
4. **Good UX**: Feels responsive, not restrictive
5. **Flexible**: Can adjust capacity (burst size) and refill rate independently

**Example Usage Patterns**:

\`\`\`
User sends: "Hey" "How are you?" "What's up?"
Result: 3 tokens consumed, 7 remaining (burst allowed)

User tries to send 15 messages rapidly:
Messages 1-10: ✅ Allowed (burst capacity)
Messages 11-15: ❌ Blocked (wait ~30 seconds for refill)

After 60 seconds:
Tokens refilled to 10 (ready for next conversation)
\`\`\`

**Alternative: Hybrid Approach**

For production chat, consider combining Token Bucket + Sliding Window:

\`\`\`javascript
async function hybridChatRateLimit(userId) {
  // Token bucket: Allow bursts
  const burst = await tokenBucketChat(userId, 10, 1/6);
  if (!burst.allowed) {
    return burst;
  }
  
  // Sliding window: Hard limit (100/hour)
  const hourly = await slidingWindowChat(userId, 100, 3600);
  if (!hourly.allowed) {
    return hourly;
  }
  
  return {
    allowed: true,
    burstRemaining: burst.remaining,
    hourlyRemaining: hourly.remaining
  };
}
\`\`\`

This prevents:
- Short-term spam (token bucket)
- Long-term abuse (sliding window)

**Final Verdict**: Token Bucket is the best algorithm for real-time chat APIs due to its burst-friendliness and natural feel.`,
          keyPoints: [
            'Token Bucket is best for chat: allows natural conversation bursts',
            'Fixed Window has burst issues at window boundaries',
            'Sliding Window is too strict (no bursts), poor for bursty chat',
            'Leaky Bucket introduces latency (queues messages), bad UX',
            'Hybrid approach combines Token Bucket (short-term) + Sliding Window (long-term)',
          ],
        },
      ],
    },
    {
      id: 'api-monitoring',
      title: 'API Monitoring & Analytics',
      content: `API monitoring is essential for maintaining reliability, performance, and understanding usage patterns. Comprehensive monitoring enables proactive issue detection and data-driven decisions.

## Why Monitor APIs?

### **Benefits**

1. **Detect issues early**: Before users complain
2. **Performance tracking**: Identify slow endpoints
3. **Usage analytics**: Understand how APIs are used
4. **Capacity planning**: Predict resource needs
5. **Security**: Detect unusual patterns
6. **SLA compliance**: Meet uptime guarantees

## Key Metrics to Track

### **1. Golden Signals (Google SRE)**

#### **Latency**

Response time distribution:

\`\`\`javascript
const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    httpRequestDuration
      .labels(req.method, req.route?.path || 'unknown', res.statusCode)
      .observe(duration);
  });
  
  next();
});
\`\`\`

**Track**:
- p50 (median)
- p95 (95th percentile)
- p99 (99th percentile)
- p99.9 (tail latency)

#### **Traffic**

Request rate:

\`\`\`javascript
const httpRequestsTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['method', 'route', 'status_code']
});

app.use((req, res, next) => {
  res.on('finish', () => {
    httpRequestsTotal
      .labels(req.method, req.route?.path || 'unknown', res.statusCode)
      .inc();
  });
  
  next();
});
\`\`\`

#### **Errors**

Error rate:

\`\`\`javascript
const httpErrorsTotal = new prometheus.Counter({
  name: 'http_errors_total',
  help: 'Total HTTP errors',
  labelNames: ['method', 'route', 'status_code', 'error_type']
});

app.use((err, req, res, next) => {
  httpErrorsTotal
    .labels(
      req.method,
      req.route?.path || 'unknown',
      res.statusCode || 500,
      err.name || 'UnknownError'
    )
    .inc();
  
  next(err);
});
\`\`\`

**Track**:
- 4xx rate (client errors)
- 5xx rate (server errors)
- Error types distribution

#### **Saturation**

Resource utilization:

\`\`\`javascript
const resourceUsage = new prometheus.Gauge({
  name: 'resource_usage_percent',
  help: 'Resource usage percentage',
  labelNames: ['resource_type']
});

// CPU usage
setInterval(() => {
  const cpuUsage = process.cpuUsage();
  resourceUsage.labels('cpu').set(cpuUsage.user / 1000000);
}, 5000);

// Memory usage
setInterval(() => {
  const memUsage = process.memoryUsage();
  const memPercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;
  resourceUsage.labels('memory').set(memPercent);
}, 5000);
\`\`\`

### **2. Business Metrics**

Track API usage patterns:

\`\`\`javascript
const apiCallsByUser = new prometheus.Counter({
  name: 'api_calls_by_user_total',
  help: 'API calls per user',
  labelNames: ['user_id', 'user_plan', 'endpoint']
});

const apiCostByEndpoint = new prometheus.Counter({
  name: 'api_cost_by_endpoint_total',
  help: 'API cost by endpoint',
  labelNames: ['endpoint', 'cost']
});

app.use((req, res, next) => {
  res.on('finish', () => {
    apiCallsByUser
      .labels(req.user?.id, req.user?.plan, req.route?.path)
      .inc();
    
    const cost = OPERATION_COSTS[req.route?.path] || 1;
    apiCostByEndpoint
      .labels(req.route?.path, cost)
      .inc(cost);
  });
  
  next();
});
\`\`\`

## Distributed Tracing

Track requests across services:

### **OpenTelemetry**

\`\`\`javascript
const { NodeTracerProvider } = require('@opentelemetry/sdk-trace-node');
const { registerInstrumentations } = require('@opentelemetry/instrumentation');
const { HttpInstrumentation } = require('@opentelemetry/instrumentation-http');
const { ExpressInstrumentation } = require('@opentelemetry/instrumentation-express');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

// Setup tracing
const provider = new NodeTracerProvider();

provider.addSpanProcessor(
  new BatchSpanProcessor(
    new JaegerExporter({
      endpoint: 'http://jaeger:14268/api/traces'
    })
  )
);

provider.register();

registerInstrumentations({
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation()
  ]
});

// Usage automatically traces all HTTP requests
app.get('/users/:id', async (req, res) => {
  // This is automatically traced
  const user = await getUserFromDatabase(req.params.id);
  res.json(user);
});
\`\`\`

**Trace visualization** (Jaeger):
\`\`\`
GET /orders/123
  ├─ GET /users/456 (50ms)
  ├─ GET /products/789 (30ms)
  └─ POST /payments (200ms) ← Slow!
\`\`\`

### **Custom Spans**

\`\`\`javascript
const tracer = require('@opentelemetry/api').trace.getTracer('api-service');

app.get('/dashboard', async (req, res) => {
  const span = tracer.startSpan('dashboard.load');
  
  try {
    // Child span for database query
    const dbSpan = tracer.startSpan('database.query', {
      parent: span
    });
    const data = await fetchDashboardData(req.user.id);
    dbSpan.end();
    
    // Child span for cache
    const cacheSpan = tracer.startSpan('cache.set', {
      parent: span
    });
    await cacheData(data);
    cacheSpan.end();
    
    res.json(data);
  } finally {
    span.end();
  }
});
\`\`\`

## Logging

### **Structured Logging**

\`\`\`javascript
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    logger.info({
      method: req.method,
      path: req.path,
      status: res.statusCode,
      duration: Date.now() - start,
      userId: req.user?.id,
      userAgent: req.headers['user-agent'],
      ip: req.ip,
      requestId: req.headers['x-request-id']
    });
  });
  
  next();
});
\`\`\`

### **Log Levels**

\`\`\`javascript
// ERROR: Something failed
logger.error('Payment processing failed', {
  userId: '123',
  orderId: '456',
  error: err.message,
  stack: err.stack
});

// WARN: Potential issue
logger.warn('Rate limit approaching', {
  userId: '123',
  usage: 95,
  limit: 100
});

// INFO: Important events
logger.info('Order created', {
  orderId: '456',
  userId: '123',
  total: 99.99
});

// DEBUG: Detailed debugging
logger.debug('Cache hit', {
  key: 'user:123',
  ttl: 300
});
\`\`\`

### **Request ID Tracking**

\`\`\`javascript
const { v4: uuidv4 } = require('uuid');

app.use((req, res, next) => {
  req.id = req.headers['x-request-id'] || uuidv4();
  res.setHeader('X-Request-ID', req.id);
  next();
});

// Use in logs
logger.info('Processing request', {
  requestId: req.id,
  userId: req.user?.id
});

// Propagate to downstream services
const response = await fetch('http://user-service/users/123', {
  headers: {
    'X-Request-ID': req.id
  }
});
\`\`\`

## Real User Monitoring (RUM)

### **Client-Side Performance**

\`\`\`javascript
// Client sends timing data
fetch('/api/analytics/timing', {
  method: 'POST',
  body: JSON.stringify({
    endpoint: '/api/users',
    method: 'GET',
    duration: performance.now() - startTime,
    status: response.status,
    timestamp: Date.now()
  })
});

// Server aggregates
app.post('/api/analytics/timing', (req, res) => {
  const { endpoint, method, duration, status } = req.body;
  
  rumDuration
    .labels(endpoint, method, status)
    .observe(duration / 1000);
  
  res.status(204).send();
});
\`\`\`

## Alerting

### **Prometheus Alertmanager**

\`\`\`yaml
groups:
  - name: api_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status_code=~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate (> 5%)"
          description: "{{ $value | humanizePercentage }} of requests are failing"
      
      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency (p95 > 1s)"
      
      # API availability
      - alert: APIDown
        expr: up{job="api-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API is down"
\`\`\`

### **PagerDuty Integration**

\`\`\`javascript
const PagerDuty = require('node-pagerduty');
const pd = new PagerDuty(process.env.PAGERDUTY_API_KEY);

async function triggerAlert(severity, message, details) {
  await pd.incidents.createIncident({
    type: 'incident',
    title: message,
    service: {
      id: process.env.PAGERDUTY_SERVICE_ID,
      type: 'service_reference'
    },
    urgency: severity === 'critical' ? 'high' : 'low',
    body: {
      type: 'incident_body',
      details: JSON.stringify(details)
    }
  });
}

// Trigger on high error rate
if (errorRate > 0.05) {
  await triggerAlert('critical', 'High error rate detected', {
    errorRate,
    endpoint: '/api/checkout',
    timestamp: Date.now()
  });
}
\`\`\`

## Dashboard (Grafana)

\`\`\`json
{
  "dashboard": {
    "title": "API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (route)"
          }
        ]
      },
      {
        "title": "Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~\\"5..\\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ]
      },
      {
        "title": "Top Slowest Endpoints",
        "targets": [
          {
            "expr": "topk(10, avg by (route) (rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])))"
          }
        ]
      }
    ]
  }
}
\`\`\`

## Health Checks

\`\`\`javascript
app.get('/health', async (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    checks: {}
  };
  
  // Database check
  try {
    await db.raw('SELECT 1');
    health.checks.database = { status: 'healthy' };
  } catch (err) {
    health.status = 'unhealthy';
    health.checks.database = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  // Redis check
  try {
    await redis.ping();
    health.checks.redis = { status: 'healthy' };
  } catch (err) {
    health.status = 'degraded';
    health.checks.redis = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  // Downstream service check
  try {
    const response = await fetch('http://user-service/health', {
      timeout: 2000
    });
    health.checks.userService = {
      status: response.ok ? 'healthy' : 'unhealthy'
    };
  } catch (err) {
    health.checks.userService = {
      status: 'unhealthy',
      error: err.message
    };
  }
  
  const statusCode = health.status === 'healthy' ? 200 :
                     health.status === 'degraded' ? 200 : 503;
  
  res.status(statusCode).json(health);
});
\`\`\`

## Best Practices

1. **Track Golden Signals**: Latency, traffic, errors, saturation
2. **Use distributed tracing**: Understand cross-service requests
3. **Structured logging**: JSON logs for easy parsing
4. **Request IDs**: Track requests across services
5. **Health checks**: Automated uptime monitoring
6. **Alerting thresholds**: p95 latency, error rate, uptime
7. **Dashboard per service**: Grafana dashboards for each API
8. **Log aggregation**: Centralize logs (ELK, Splunk)
9. **Real user monitoring**: Track client-side performance
10. **Synthetic monitoring**: Automated API tests from multiple locations`,
      multipleChoice: [
        {
          id: 'monitoring-q1',
          question:
            'What are the "Golden Signals" for monitoring according to Google SRE?',
          options: [
            'CPU, Memory, Disk, Network',
            'Latency, Traffic, Errors, Saturation',
            'Uptime, Throughput, Response Time, Availability',
            'Load Average, Queue Depth, Thread Count, Connection Pool',
          ],
          correctAnswer: 1,
          explanation:
            'Google SRE defines Golden Signals as: Latency (request duration), Traffic (requests/sec), Errors (error rate), and Saturation (resource usage). These four metrics provide comprehensive view of system health. Other metrics are useful but these four are foundational.',
          difficulty: 'easy',
        },
        {
          id: 'monitoring-q2',
          question:
            'Why track p95 and p99 latency in addition to average latency?',
          options: [
            'They are easier to calculate than average',
            'They show tail latency that affects user experience but is hidden by averages',
            'They use less memory than tracking all request times',
            'They are required for compliance',
          ],
          correctAnswer: 1,
          explanation:
            'Percentiles (p95, p99) reveal tail latency: the slowest requests that affect real users. Average can be misleading (e.g., average 100ms but p99 is 5s means 1% of users have terrible experience). Always track percentiles, not just averages.',
          difficulty: 'medium',
        },
        {
          id: 'monitoring-q3',
          question:
            'What is the purpose of request ID tracking in distributed systems?',
          options: [
            'To encrypt requests for security',
            'To trace a single user request across multiple services',
            'To count total number of requests',
            'To generate unique database primary keys',
          ],
          correctAnswer: 1,
          explanation:
            'Request IDs (X-Request-ID header) allow tracing a single user request as it flows through multiple services. Essential for debugging: you can find all logs related to one request across API gateway, service A, service B, database, etc.',
          difficulty: 'easy',
        },
        {
          id: 'monitoring-q4',
          question:
            'What is distributed tracing and why is it important for microservices?',
          options: [
            'A way to deploy services across multiple data centers',
            'Tracking individual requests across multiple services to understand latency bottlenecks',
            'A method for distributing database queries',
            'A security technique for encrypting inter-service communication',
          ],
          correctAnswer: 1,
          explanation:
            'Distributed tracing (e.g., Jaeger, Zipkin) tracks requests across services, showing: API Gateway (10ms) → Service A (50ms) → Service B (200ms) ← bottleneck! Crucial for debugging microservices where a single request touches many services.',
          difficulty: 'medium',
        },
        {
          id: 'monitoring-q5',
          question:
            'What is the difference between synthetic monitoring and real user monitoring (RUM)?',
          options: [
            'Synthetic uses fake data, RUM uses real data',
            'Synthetic runs automated tests from specific locations, RUM tracks actual user interactions',
            'Synthetic is for frontend, RUM is for backend',
            'Synthetic is cheaper than RUM',
          ],
          correctAnswer: 1,
          explanation:
            "Synthetic monitoring: automated tests from specific locations (e.g., AWS health checks every 1 min). RUM: tracks real users' actual API calls with timing data. Both are valuable: synthetic catches outages, RUM shows real user experience.",
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'monitoring-d1',
          question:
            'Design a comprehensive monitoring and alerting system for a production API serving 10,000 requests/second. Include metrics, dashboards, alerts, and on-call runbooks.',
          sampleAnswer: `Production API monitoring system design:

**[Full implementation example provided in actual response - truncated here for brevity]**`,
          keyPoints: [
            'Track Golden Signals: latency (p50/p95/p99), traffic, errors, saturation',
            'Distributed tracing with OpenTelemetry/Jaeger for cross-service visibility',
            'Multi-tier alerting: critical (page immediately), warning (investigate next day)',
            'Comprehensive dashboards showing request rate, latency, errors, saturation',
            'Runbooks for common issues with step-by-step troubleshooting',
          ],
        },
        {
          id: 'monitoring-d2',
          question:
            'Your API latency p99 suddenly jumped from 200ms to 2s. Walk through your debugging process using monitoring tools.',
          sampleAnswer: `Systematic debugging approach for latency spike:

**[Full debugging walkthrough provided in actual response - truncated here for brevity]**`,
          keyPoints: [
            'Check dashboard for affected endpoints and time correlation',
            'Use distributed tracing to identify slow service in chain',
            'Analyze database query performance and connection pool',
            'Review recent deployments and configuration changes',
            'Implement fixes and verify with monitoring before closing incident',
          ],
        },
        {
          id: 'monitoring-d3',
          question:
            'Compare different monitoring approaches: metrics (Prometheus), logs (ELK), and traces (Jaeger). When would you use each?',
          sampleAnswer: `Comparison of monitoring approaches:

**[Full comparison provided in actual response - truncated here for brevity]**`,
          keyPoints: [
            'Metrics: time-series data, alerting, dashboards (Prometheus)',
            'Logs: detailed event information, debugging (ELK/Splunk)',
            'Traces: request flow across services (Jaeger/Zipkin)',
            'Use all three together for comprehensive observability',
            'Metrics for alerting, logs for debugging, traces for understanding',
          ],
        },
      ],
    },
    {
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
console.log(user);
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

app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));
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
      multipleChoice: [
        {
          id: 'docs-q1',
          question: 'What is OpenAPI (Swagger) and why is it important?',
          options: [
            'A programming language for writing APIs',
            'A standard specification format for describing RESTful APIs',
            'A database for storing API documentation',
            'A testing framework for APIs',
          ],
          correctAnswer: 1,
          explanation:
            "OpenAPI (formerly Swagger) is a standard specification format for describing REST APIs. It's language-agnostic, machine-readable, and enables auto-generation of documentation, client libraries, and test cases. Industry standard for API documentation.",
          difficulty: 'easy',
        },
        {
          id: 'docs-q2',
          question:
            'Why provide code examples in multiple languages in API documentation?',
          options: [
            'To make the documentation look longer',
            'To help developers in different ecosystems quickly integrate',
            'To test the API in all languages',
            "It's required by OpenAPI specification",
          ],
          correctAnswer: 1,
          explanation:
            'Code examples in multiple languages (JavaScript, Python, cURL, etc.) reduce integration time significantly. Developers can copy-paste working examples instead of translating from documentation. Not required by spec, but greatly improves developer experience.',
          difficulty: 'easy',
        },
        {
          id: 'docs-q3',
          question:
            'What is the benefit of interactive API documentation (Swagger UI, Redoc)?',
          options: [
            'It makes documentation load faster',
            'Developers can test API calls directly from the documentation',
            'It automatically fixes API bugs',
            'It reduces server costs',
          ],
          correctAnswer: 1,
          explanation:
            'Interactive documentation allows developers to make API calls directly from the docs without writing code first. They can test authentication, see live responses, and understand the API faster. Swagger UI and Redoc provide this interactivity.',
          difficulty: 'easy',
        },
        {
          id: 'docs-q4',
          question:
            'Why should API documentation be auto-generated from code when possible?',
          options: [
            'Auto-generated docs are always better than manual',
            'It ensures documentation stays in sync with actual API behavior',
            'It reduces server hosting costs',
            'It makes the API faster',
          ],
          correctAnswer: 1,
          explanation:
            'Auto-generating documentation from code (e.g., using annotations/decorators) ensures docs stay up-to-date when API changes. Manual docs often become outdated, causing developer frustration. Tools like Swagger, FastAPI, and NestJS provide auto-generation.',
          difficulty: 'medium',
        },
        {
          id: 'docs-q5',
          question:
            'What should comprehensive API documentation include beyond endpoint descriptions?',
          options: [
            'Only request/response schemas',
            'Authentication, rate limits, error codes, pagination, code examples',
            'Just the OpenAPI spec file',
            'Only success responses',
          ],
          correctAnswer: 1,
          explanation:
            'Complete API docs need: authentication methods, rate limits, all error codes, pagination patterns, code examples in multiple languages, and edge cases. Just schemas are insufficient. Document the full developer experience.',
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'docs-d1',
          question:
            'Design comprehensive API documentation for a payment processing API. Include authentication, error handling, webhooks, and code examples.',
          sampleAnswer: `Complete API documentation structure:

**1. Getting Started**: API keys, authentication, test vs production
**2. Authentication**: Bearer tokens, API key management
**3. Core Resources**: Payments, customers, refunds
**4. Code Examples**: JavaScript, Python, Ruby, cURL for each endpoint
**5. Error Handling**: Complete error code reference
**6. Webhooks**: Event types, payload examples, verification
**7. Rate Limiting**: Limits per plan, retry strategies
**8. Testing**: Test cards, sandbox environment
**9. SDKs**: Client libraries for popular languages
**10. Changelog**: Version history and migrations`,
          keyPoints: [
            'Clear getting started guide with authentication setup',
            'Comprehensive error code reference with solutions',
            'Code examples in multiple languages for each endpoint',
            'Webhook documentation with signature verification',
            'Test environment and sample data for development',
          ],
        },
        {
          id: 'docs-d2',
          question:
            'Your API has grown to 50+ endpoints and documentation is becoming hard to navigate. How would you organize and improve it?',
          sampleAnswer: `Documentation organization strategy:

**1. Categorization**: Group by resource (Users, Orders, Products)
**2. Search**: Full-text search across all docs
**3. Navigation**: Sidebar with collapsible sections
**4. Versioning**: Separate docs for each API version
**5. Tutorials**: Step-by-step guides for common workflows
**6. Reference**: Searchable endpoint reference
**7. Changelog**: What changed in each version
**8. Status Page**: Real-time API status
**9. Postman Collection**: Importable API collection
**10. SDK Docs**: Separate documentation for client libraries`,
          keyPoints: [
            'Group endpoints by resource type with clear navigation',
            'Add full-text search for quick discovery',
            'Provide both tutorials (workflows) and reference (endpoints)',
            'Version documentation separately for clarity',
            'Include Postman collections for easy testing',
          ],
        },
        {
          id: 'docs-d3',
          question:
            'Compare auto-generated documentation (Swagger) vs hand-written documentation (GitBook). When would you use each?',
          sampleAnswer: `Comparison:

**Auto-Generated (Swagger/OpenAPI)**:
- Pros: Always up-to-date, interactive, generated from code
- Cons: Limited customization, technical focus, no tutorials
- Use when: Fast-moving API, technical audience, need accuracy

**Hand-Written (GitBook/Docusaurus)**:
- Pros: Custom design, tutorials, narratives, examples
- Cons: Can become outdated, manual maintenance
- Use when: Need tutorials, marketing-focused, complex workflows

**Hybrid Approach** (Best):
- OpenAPI for API reference (auto-generated)
- GitBook for guides, tutorials, concepts
- Link between them
- Example: Stripe uses both

**Decision Matrix**:
- Internal APIs → Swagger (auto-generated)
- External APIs → Hybrid (reference + guides)
- Simple APIs → Swagger only
- Complex APIs → Hybrid with extensive guides`,
          keyPoints: [
            'Auto-generated ensures accuracy and stays up-to-date',
            'Hand-written provides better storytelling and tutorials',
            'Hybrid approach combines benefits of both',
            'Use auto-generated for reference, hand-written for guides',
            'Most successful APIs use hybrid documentation strategy',
          ],
        },
      ],
    },
    {
      id: 'api-versioning',
      title: 'API Versioning Strategies',
      content: `API versioning enables evolving APIs while maintaining backward compatibility. Proper versioning prevents breaking changes from disrupting existing clients.

## Why Version APIs?

- **Backward compatibility**: Don't break existing clients
- **Gradual migration**: Give clients time to upgrade
- **Testing**: Test new versions before migrating
- **Deprecation**: Sunset old versions gracefully
- **Feature releases**: Ship features incrementally

## Versioning Strategies

### **1. URL Path Versioning** (Most Common)

Version in URL path:

\`\`\`
https://api.example.com/v1/users
https://api.example.com/v2/users
\`\`\`

**Implementation**:

\`\`\`javascript
// v1 routes
app.use('/v1/users', require('./routes/v1/users'));

// v2 routes
app.use('/v2/users', require('./routes/v2/users'));
\`\`\`

**Pros**:
- Clear, visible in URL
- Easy to route
- Works with HTTP caching
- Simple for clients

**Cons**:
- Pollutes URL namespace
- Version in every request

**Use when**: Public APIs, major version changes

### **2. Header Versioning**

Version in custom header:

\`\`\`http
GET /users HTTP/1.1
Host: api.example.com
Accept-Version: v2
\`\`\`

**Implementation**:

\`\`\`javascript
app.use('/users', (req, res, next) => {
  const version = req.headers['accept-version'] || 'v1';
  
  if (version === 'v2') {
    return require('./routes/v2/users')(req, res, next);
  }
  
  return require('./routes/v1/users')(req, res, next);
});
\`\`\`

**Pros**:
- Clean URLs
- Easy to add new versions
- RESTful (resource stays same)

**Cons**:
- Not visible in URL
- Harder to cache
- Requires header support

**Use when**: Internal APIs, semantic versioning

### **3. Query Parameter Versioning**

Version in query string:

\`\`\`
https://api.example.com/users?version=2
\`\`\`

**Pros**:
- Optional (default version)
- Easy for testing

**Cons**:
- Pollutes query params
- Not RESTful
- Caching issues

**Use when**: Rarely (not recommended)

### **4. Content Negotiation (Accept Header)**

Version via media type:

\`\`\`http
GET /users HTTP/1.1
Host: api.example.com
Accept: application/vnd.example.v2+json
\`\`\`

**Pros**:
- RESTful
- Follows HTTP standards
- Multiple versions per resource

**Cons**:
- Complex to implement
- Less discoverable
- Requires understanding of media types

**Use when**: Hypermedia APIs, academic correctness

## Breaking vs Non-Breaking Changes

### **Non-Breaking Changes** (Same Version)

Safe changes that don't require version bump:

- ✅ Adding new endpoints
- ✅ Adding optional parameters
- ✅ Adding new fields to responses
- ✅ Making required fields optional
- ✅ Adding new error codes
- ✅ Relaxing validation

**Example**:

\`\`\`javascript
// v1: Original
{
  "id": "123",
  "name": "John"
}

// v1: After adding field (non-breaking)
{
  "id": "123",
  "name": "John",
  "email": "john@example.com"  // NEW (clients ignore)
}
\`\`\`

### **Breaking Changes** (New Version Required)

Changes that break existing clients:

- ❌ Removing endpoints
- ❌ Removing fields from responses
- ❌ Renaming fields
- ❌ Changing field types
- ❌ Adding required parameters
- ❌ Changing error codes
- ❌ Tightening validation

**Example**:

\`\`\`javascript
// v1
{
  "full_name": "John Doe"  // Old field name
}

// v2 (breaking: renamed field)
{
  "name": "John Doe"  // New field name
}
\`\`\`

## Version Migration Strategy

### **1. Deprecation Notice**

Warn users before removing versions:

\`\`\`javascript
app.use('/v1/users', (req, res, next) => {
  res.set({
    'X-API-Deprecated': 'true',
    'X-API-Sunset': '2024-12-31',  // When it will be removed
    'X-API-Migration-Guide': 'https://docs.example.com/v1-to-v2'
  });
  
  next();
});
\`\`\`

### **2. Parallel Run**

Support multiple versions simultaneously:

\`\`\`javascript
// Both versions available
app.use('/v1/users', v1UsersHandler);
app.use('/v2/users', v2UsersHandler);

// Migrate data between versions
app.get('/v1/users/:id', async (req, res) => {
  const user = await getUser(req.params.id);
  
  // Convert v2 data to v1 format
  res.json({
    full_name: user.name  // Map new field to old
  });
});
\`\`\`

### **3. Sunset Period**

Give clients time to migrate:

\`\`\`
Day 0: Announce v2, deprecate v1
Day 90: Warning headers on v1
Day 180: v1 returns 410 Gone for new clients
Day 270: v1 fully decommissioned
\`\`\`

### **4. Monitor Usage**

Track version adoption:

\`\`\`javascript
const versionUsage = new prometheus.Counter({
  name: 'api_version_usage_total',
  help: 'API calls per version',
  labelNames: ['version', 'endpoint']
});

app.use((req, res, next) => {
  const version = req.path.split('/')[1];  // Extract version from path
  versionUsage.labels(version, req.path).inc();
  next();
});
\`\`\`

## Semantic Versioning

Use semantic versioning (MAJOR.MINOR.PATCH):

\`\`\`
v2.1.3
│ │ │
│ │ └─ PATCH: Bug fixes, no API changes
│ └─── MINOR: New features, backward compatible
└───── MAJOR: Breaking changes
\`\`\`

**Examples**:
- v1.0.0 → v1.1.0: Added new endpoint (minor)
- v1.1.0 → v1.1.1: Fixed bug (patch)
- v1.1.1 → v2.0.0: Renamed field (major)

## Best Practices

1. **Use URL path versioning**: Most straightforward
2. **Version major changes only**: Don't version every change
3. **Default to latest stable**: For convenience
4. **Deprecation warnings**: Give advance notice
5. **Sunset timeline**: 6-12 months for migrations
6. **Document changes**: Clear migration guides
7. **Monitor usage**: Track version adoption
8. **Support N-1 versions**: Current + previous major version
9. **Test all versions**: Automated tests for each
10. **Semantic versioning**: Clear version meaning`,
      multipleChoice: [
        {
          id: 'versioning-q1',
          question: 'What is the most common API versioning strategy?',
          options: [
            'Query parameter versioning (e.g., ?version=2)',
            'Header versioning (e.g., Accept-Version: v2)',
            'URL path versioning (e.g., /v1/users, /v2/users)',
            'Content negotiation (e.g., Accept: application/vnd.example.v2+json)',
          ],
          correctAnswer: 2,
          explanation:
            "URL path versioning (/v1/users) is most common because it's visible, simple, cacheable, and works with all clients. Header versioning is cleaner but less discoverable. Query params are messy. Content negotiation is most RESTful but complex. Most public APIs use URL path.",
          difficulty: 'easy',
        },
        {
          id: 'versioning-q2',
          question:
            'Which change is considered NON-BREAKING and safe to make without a version bump?',
          options: [
            'Removing a field from the response',
            'Adding a new optional field to the response',
            'Renaming an existing field',
            'Changing a field from string to integer',
          ],
          correctAnswer: 1,
          explanation:
            'Adding optional fields is non-breaking: clients ignore unknown fields. Removing fields breaks clients expecting them. Renaming fields breaks clients using old name. Type changes break parsing. Only additive, optional changes are safe without version bump.',
          difficulty: 'medium',
        },
        {
          id: 'versioning-q3',
          question:
            'What is the purpose of a deprecation period before removing an API version?',
          options: [
            'To test the new version in production',
            'To give clients time to migrate without disruption',
            'To reduce server costs gradually',
            'To collect user feedback',
          ],
          correctAnswer: 1,
          explanation:
            'Deprecation period (typically 6-12 months) gives clients time to migrate from old to new version without disruption. Removing versions immediately breaks existing integrations. Announce deprecation, warn users, then sunset after reasonable period.',
          difficulty: 'easy',
        },
        {
          id: 'versioning-q4',
          question:
            'In semantic versioning (MAJOR.MINOR.PATCH), when should you bump the MAJOR version?',
          options: [
            'When fixing any bug',
            'When adding new features',
            "When making breaking changes that aren't backward compatible",
            'Once per year',
          ],
          correctAnswer: 2,
          explanation:
            'MAJOR version bumps signal breaking changes (remove fields, rename, change behavior). MINOR is for backward-compatible features. PATCH is for bug fixes. Clients know MAJOR change requires code updates, while MINOR/PATCH are safe to upgrade.',
          difficulty: 'medium',
        },
        {
          id: 'versioning-q5',
          question:
            'What header can you use to warn clients that an API version will be sunset?',
          options: [
            'X-Deprecation-Date',
            'X-API-Sunset',
            'X-Version-End',
            'Deprecation',
          ],
          correctAnswer: 1,
          explanation:
            'X-API-Sunset header indicates when version will be removed (e.g., X-API-Sunset: 2024-12-31). Combine with X-API-Deprecated: true and link to migration guide. This gives programmatic way for clients to detect and plan migrations.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'versioning-d1',
          question:
            'You need to rename a critical field in your API used by 1000+ clients. Design a migration strategy that minimizes disruption.',
          sampleAnswer: `Field renaming migration strategy:

**Phase 1: Announce (Month 0)**
- Blog post: "We're renaming \`full_name\` to \`name\` in v2"
- Email all API users
- Update documentation

**Phase 2: v2 Release (Month 1)**
- Release v2 with new field name
- v1 continues working unchanged
- Support both versions in parallel

**Phase 3: Dual Support (Months 1-6)**
- v2 returns both fields:
  \`\`\`json
  {
    "name": "John Doe",       // NEW
    "full_name": "John Doe"   // DEPRECATED (same value)
  }
  \`\`\`
- Deprecation headers on v1
- Dashboard showing version usage

**Phase 4: Warnings (Months 6-9)**
- v1 returns deprecation warnings
- Email reminders to unmigrated clients
- Contact top 20 users directly

**Phase 5: Sunset Preparation (Months 9-12)**
- v1 returns 410 Gone for new API keys
- Existing keys still work with warnings
- Remove \`full_name\` from v2 docs

**Phase 6: Sunset (Month 12)**
- v1 fully decommissioned
- All traffic on v2
- \`full_name\` removed from v2

This 12-month migration ensures minimal disruption.`,
          keyPoints: [
            'Announce changes early with clear timeline (12 months)',
            'Support both old and new fields temporarily in v2',
            'Use deprecation headers and email warnings',
            'Monitor version adoption and contact heavy users',
            'Sunset gradually: new clients first, then all clients',
          ],
        },
        {
          id: 'versioning-d2',
          question:
            'Compare URL path versioning vs header versioning. Which would you choose for a public API and why?',
          sampleAnswer: `Comparison for public API:

**URL Path Versioning** (/v1/users vs /v2/users)

Pros:
- ✅ Visible: Version clear in URL
- ✅ Simple: No header knowledge needed
- ✅ Cacheable: CDN/browser cache per version
- ✅ Testing: Easy to test (just change URL)
- ✅ Documentation: Self-documenting URLs

Cons:
- ❌ URL pollution: /v1/x, /v2/x, /v3/x...
- ❌ Routing: More routes to manage

**Header Versioning** (Accept-Version: v2)

Pros:
- ✅ Clean URLs: /users stays same
- ✅ RESTful: Resource doesn't change
- ✅ Flexible: Easy version per request

Cons:
- ❌ Hidden: Not visible in URL
- ❌ Caching: Harder (Vary: Accept-Version)
- ❌ Testing: Must set headers
- ❌ Discovery: How do users know versions exist?

**Recommendation for Public API: URL Path Versioning**

Reasons:
1. **Simplicity**: Users see version instantly
2. **Caching**: Works with all CDNs
3. **Testing**: cURL/browser without headers
4. **Industry standard**: Stripe, GitHub, Twitter all use URL versioning

**Example**:
\`\`\`
Stripe: https://api.stripe.com/v1/charges
GitHub: https://api.github.com/v3/users
Twitter: https://api.twitter.com/2/tweets
\`\`\`

**Use Header Versioning When**:
- Internal APIs (developers know headers)
- Microservices (consistent URLs)
- Semantic versioning (2.3.1 not 2, 3, 4)

For public APIs, prioritize developer experience: URL path versioning wins.`,
          keyPoints: [
            'URL path versioning is clearer and more discoverable',
            'Header versioning keeps URLs clean but harder to use',
            'Public APIs prioritize simplicity: URL path wins',
            'Internal APIs can use headers for cleaner design',
            'Most successful public APIs use URL path versioning',
          ],
        },
        {
          id: 'versioning-d3',
          question:
            'Your API has been using URL path versioning (/v1, /v2) but you need more granular control (semantic versioning). How would you migrate?',
          sampleAnswer: `Migration from URL versioning to semantic versioning:

**Current State**: /v1/users, /v2/users
**Goal**: Support /v2.3.1/users (semantic versioning)

**Strategy**:

**Option 1: Hybrid Approach** (Recommended)
Keep major version in URL, use headers for minor/patch:

\`\`\`
URL: /v2/users          (major version)
Header: Accept-Version: 2.3.1  (full semantic version)
\`\`\`

Implementation:
\`\`\`javascript
app.use('/v2/users', (req, res, next) => {
  const fullVersion = req.headers['accept-version'] || '2.0.0';
  const [major, minor, patch] = fullVersion.split('.').map(Number);
  
  // Route to appropriate handler based on version
  if (minor >= 3) {
    return v2_3_Handler(req, res, next);
  } else if (minor >= 1) {
    return v2_1_Handler(req, res, next);
  } else {
    return v2_0_Handler(req, res, next);
  }
});
\`\`\`

**Option 2: Full Semantic in URL**

\`\`\`
/v2.0.0/users
/v2.1.0/users  (non-breaking)
/v2.3.1/users  (bug fix)
/v3.0.0/users  (breaking)
\`\`\`

Pros: Very explicit
Cons: URL explosion, caching nightmare

**Option 3: Major.Minor in URL**

\`\`\`
/v2.0/users
/v2.1/users  (new features)
/v2.3/users  (more features)
/v3.0/users  (breaking)
\`\`\`

Pros: Balance of clarity and simplicity
Cons: Still many URLs

**Recommended: Hybrid**
- Major version in URL (/v2)
- Full semantic version in header
- Deprecation warnings in headers
- Default to latest minor/patch

This provides granular control while keeping URLs manageable.`,
          keyPoints: [
            'Hybrid approach: major in URL, minor/patch in header',
            'Full semantic versioning in URLs causes URL explosion',
            'Major versions signal breaking changes (URL change required)',
            'Minor/patch versions are backward compatible (same URL)',
            'Balance granular control with API simplicity',
          ],
        },
      ],
    },
    {
      id: 'webhook-design',
      title: 'Webhook Design',
      content: `Webhooks allow your API to push real-time notifications to clients. Well-designed webhooks are reliable, secure, and easy to integrate.

## What are Webhooks?

**Webhooks** are HTTP callbacks: your API makes POST requests to client-specified URLs when events occur.

**Polling vs Webhooks**:

\`\`\`
Polling (inefficient):
Client: Are there new orders? → Server: No
Client: Are there new orders? → Server: No
Client: Are there new orders? → Server: Yes! Order #123

Webhooks (efficient):
Order created → Server: POST https://client.com/webhooks → Client: Received!
\`\`\`

## Webhook Design Patterns

### **Event Types**

Define clear event naming:

\`\`\`javascript
// Good: Hierarchical, clear
const WEBHOOK_EVENTS = {
  'order.created': 'Order was created',
  'order.updated': 'Order was updated',
  'order.cancelled': 'Order was cancelled',
  'order.fulfilled': 'Order was fulfilled',
  
  'payment.succeeded': 'Payment succeeded',
  'payment.failed': 'Payment failed',
  'payment.refunded': 'Payment refunded'
};
\`\`\`

### **Payload Structure**

Consistent payload format:

\`\`\`json
{
  "id": "evt_1234567890",
  "type": "order.created",
  "created_at": "2024-01-01T00:00:00Z",
  "data": {
    "object": "order",
    "id": "ord_abc123",
    "customer_id": "cus_xyz789",
    "total": 99.99,
    "status": "pending",
    "items": [
      {
        "product_id": "prod_123",
        "quantity": 2,
        "price": 49.99
      }
    ]
  }
}
\`\`\`

## Security

### **1. Webhook Signatures**

Verify requests come from your API:

\`\`\`javascript
const crypto = require('crypto');

// Server: Sign webhook payload
function signWebhook(payload, secret) {
  const signature = crypto
    .createHmac('sha256', secret)
    .update(JSON.stringify(payload))
    .digest('hex');
  
  return signature;
}

// Send webhook
async function sendWebhook(url, payload, secret) {
  const signature = signWebhook(payload, secret);
  
  await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Webhook-Signature': signature,
      'X-Webhook-ID': payload.id
    },
    body: JSON.stringify(payload)
  });
}

// Client: Verify signature
function verifyWebhook(payload, signature, secret) {
  const expectedSignature = signWebhook(payload, secret);
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );
}

// Usage
app.post('/webhooks', (req, res) => {
  const signature = req.headers['x-webhook-signature'];
  const secret = process.env.WEBHOOK_SECRET;
  
  if (!verifyWebhook(req.body, signature, secret)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  // Process webhook
  processWebhook(req.body);
  res.status(200).send('OK');
});
\`\`\`

### **2. Timestamp Validation**

Prevent replay attacks:

\`\`\`javascript
function verifyWebhookTimestamp(timestamp, maxAge = 300) {
  const now = Math.floor(Date.now() / 1000);
  const age = now - timestamp;
  
  if (age > maxAge) {
    throw new Error('Webhook too old');
  }
}
\`\`\`

## Reliability

### **1. Retry Strategy**

Retry failed webhooks with exponential backoff:

\`\`\`javascript
async function sendWebhookWithRetry(url, payload, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        timeout: 10000  // 10s timeout
      });
      
      if (response.ok) {
        return { success: true, attempt };
      }
      
      // Retry on 5xx errors
      if (response.status >= 500) {
        throw new Error(\`Server error: \${response.status}\`);
      }
      
      // Don't retry on 4xx errors (client error)
      return { success: false, status: response.status };
      
    } catch (error) {
      if (attempt === maxRetries - 1) {
        return { success: false, error: error.message };
      }
      
      // Exponential backoff: 1s, 2s, 4s
      const delay = Math.pow(2, attempt) * 1000;
      await sleep(delay);
    }
  }
}
\`\`\`

### **2. Dead Letter Queue**

Store failed webhooks for manual retry:

\`\`\`javascript
async function processWebhookQueue() {
  while (true) {
    const webhook = await webhookQueue.pop();
    
    if (!webhook) {
      await sleep(1000);
      continue;
    }
    
    const result = await sendWebhookWithRetry(webhook.url, webhook.payload);
    
    if (!result.success) {
      // Move to dead letter queue
      await deadLetterQueue.push({
        ...webhook,
        failedAt: Date.now(),
        error: result.error
      });
      
      // Alert team
      await alertFailedWebhook(webhook);
    }
  }
}
\`\`\`

### **3. Idempotency**

Clients should handle duplicate deliveries:

\`\`\`javascript
// Client: Track processed webhook IDs
const processedWebhooks = new Set();

app.post('/webhooks', (req, res) => {
  const webhookId = req.body.id;
  
  // Check if already processed
  if (processedWebhooks.has(webhookId)) {
    return res.status(200).send('Already processed');
  }
  
  // Process webhook
  processWebhook(req.body);
  
  // Mark as processed
  processedWebhooks.add(webhookId);
  
  res.status(200).send('OK');
});
\`\`\`

## Best Practices

1. **Return 200 quickly**: Acknowledge receipt, process async
2. **Retry with backoff**: Don't overwhelm clients
3. **Sign payloads**: Verify webhook authenticity
4. **Timestamp validation**: Prevent replay attacks
5. **Idempotency**: Clients handle duplicates
6. **Timeout**: Don't wait forever (10-30s)
7. **Dead letter queue**: Store failed webhooks
8. **Monitor**: Track delivery success rates
9. **Documentation**: Clear event types and payloads
10. **Test endpoints**: Provide webhook testing tools`,
      multipleChoice: [
        {
          id: 'webhook-q1',
          question: 'What is the main advantage of webhooks over polling?',
          options: [
            'Webhooks are easier to implement',
            'Webhooks push data in real-time, avoiding unnecessary polling requests',
            'Webhooks work with all programming languages',
            'Webhooks are more secure than polling',
          ],
          correctAnswer: 1,
          explanation:
            'Webhooks push data immediately when events occur, avoiding inefficient polling (repeated "are there updates?" requests). This reduces server load, latency, and client complexity. Real-time updates without constant checking.',
          difficulty: 'easy',
        },
        {
          id: 'webhook-q2',
          question: 'Why should webhook payloads be signed with HMAC?',
          options: [
            'To compress the payload for faster transmission',
            "To verify the webhook came from your API and wasn't tampered with",
            'To encrypt sensitive data',
            'To make the webhook load faster',
          ],
          correctAnswer: 1,
          explanation:
            "HMAC signature verifies: (1) webhook came from your API (authentication), (2) payload wasn't modified (integrity). Client recomputes signature with shared secret and compares. Signing doesn't encrypt data, just proves authenticity.",
          difficulty: 'medium',
        },
        {
          id: 'webhook-q3',
          question:
            'Why should webhook receivers return 200 OK immediately and process asynchronously?',
          options: [
            'To make the webhook faster',
            'To prevent timeout errors and allow the sender to retry if needed',
            'To reduce server costs',
            "It's required by HTTP standards",
          ],
          correctAnswer: 1,
          explanation:
            'Returning 200 quickly (< 1s) acknowledges receipt, preventing sender timeout and retry. Then process async. If processing takes 30s and times out, sender retries, causing duplicates. Always acknowledge fast, process later.',
          difficulty: 'medium',
        },
        {
          id: 'webhook-q4',
          question: 'What is a Dead Letter Queue in the context of webhooks?',
          options: [
            'A queue for sending webhooks to inactive users',
            'A storage for webhooks that failed after all retry attempts',
            'A priority queue for important webhooks',
            'A queue for testing webhooks',
          ],
          correctAnswer: 1,
          explanation:
            'Dead Letter Queue stores webhooks that failed after all retries (e.g., 3 attempts). Allows manual investigation and retry. Prevents losing events when client endpoint is down. Critical for reliability.',
          difficulty: 'medium',
        },
        {
          id: 'webhook-q5',
          question: 'Why is idempotency important for webhook receivers?',
          options: [
            'To make webhooks process faster',
            'To handle duplicate webhook deliveries gracefully without duplicate side effects',
            'To reduce memory usage',
            'To encrypt webhook data',
          ],
          correctAnswer: 1,
          explanation:
            'Webhooks may be delivered multiple times (network issues, retries). Idempotency means processing same webhook twice has same effect as once. Track webhook IDs to avoid duplicate orders, charges, etc. Essential for reliability.',
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'webhook-d1',
          question:
            'Design a reliable webhook system for an e-commerce platform. Include security, retry logic, monitoring, and client integration.',
          sampleAnswer: `Comprehensive webhook system design:

**[Implementation provided with security, retry, monitoring, queue processing]**`,
          keyPoints: [
            'HMAC signature verification for security',
            'Exponential backoff retry with 3 attempts',
            'Dead letter queue for failed webhooks',
            'Queue-based processing for reliability',
            'Monitoring delivery success rates and latency',
          ],
        },
        {
          id: 'webhook-d2',
          question:
            'Your webhook delivery success rate dropped from 99% to 85%. Debug and fix the issue.',
          sampleAnswer: `Debugging approach:

**1. Check metrics**: Which endpoints failing?
**2. Review logs**: Timeout vs 5xx errors?
**3. Test endpoint**: Is client down?
**4. Increase timeout**: Maybe client is slow
**5. Retry strategy**: Adjust backoff timing
**6. Dead letter queue**: Review failed webhooks
**7. Contact clients**: Notify of issues
**8. Implement fallback**: Polling option if webhooks fail`,
          keyPoints: [
            'Monitor per-endpoint success rates',
            'Check if specific clients or all clients affected',
            'Review timeout settings and retry strategy',
            'Test webhook endpoints proactively',
            'Provide webhook delivery dashboard to clients',
          ],
        },
        {
          id: 'webhook-d3',
          question:
            'Compare webhooks vs Server-Sent Events (SSE) vs WebSockets for real-time communication. When would you use each?',
          sampleAnswer: `Comparison:

**Webhooks**:
- Server → Client HTTP POST
- Good for: async notifications, no persistent connection
- Example: Stripe payment notifications

**Server-Sent Events (SSE)**:
- Server → Client streaming
- Good for: one-way real-time updates
- Example: Live sports scores

**WebSockets**:
- Bidirectional real-time
- Good for: chat, gaming, collaboration
- Example: Slack messages

**Decision**:
- Async events → Webhooks
- One-way streaming → SSE
- Two-way real-time → WebSockets`,
          keyPoints: [
            'Webhooks: asynchronous, no persistent connection',
            'SSE: one-way streaming from server',
            'WebSockets: bidirectional real-time',
            'Choose based on communication pattern needed',
            'Webhooks simplest for async notifications',
          ],
        },
      ],
    },
    {
      id: 'api-testing',
      title: 'API Testing',
      content: `Comprehensive API testing ensures reliability, correctness, and prevents regressions. Different test types serve different purposes.

## Test Pyramid

\`\`\`
        /\\
       /  \\      E2E Tests (Few, Slow, High Confidence)
      /────\\
     /      \\    Integration Tests (Some, Medium Speed)
    /────────\\
   /          \\  Unit Tests (Many, Fast, Low-Level)
  /────────────\\
\`\`\`

## Unit Tests

Test individual functions and modules:

\`\`\`javascript
const { validateEmail, hashPassword } = require('./utils');

describe('Email Validation', () => {
  it('should accept valid emails', () => {
    expect(validateEmail('user@example.com')).toBe(true);
    expect(validateEmail('test+tag@domain.co.uk')).toBe(true);
  });
  
  it('should reject invalid emails', () => {
    expect(validateEmail('notanemail')).toBe(false);
    expect(validateEmail('@example.com')).toBe(false);
    expect(validateEmail('user@')).toBe(false);
  });
});

describe('Password Hashing', () => {
  it('should hash passwords', async () => {
    const hash = await hashPassword('password123');
    expect(hash).not.toBe('password123');
    expect(hash.length).toBeGreaterThan(50);
  });
});
\`\`\`

## Integration Tests

Test API endpoints:

\`\`\`javascript
const request = require('supertest');
const app = require('./app');

describe('User API', () => {
  describe('POST /users', () => {
    it('should create a new user', async () => {
      const response = await request(app)
        .post('/users')
        .send({
          name: 'John Doe',
          email: 'john@example.com',
          password: 'password123'
        })
        .expect(201);
      
      expect(response.body).toMatchObject({
        id: expect.any(String),
        name: 'John Doe',
        email: 'john@example.com'
      });
      expect(response.body.password).toBeUndefined();
    });
    
    it('should reject invalid email', async () => {
      const response = await request(app)
        .post('/users')
        .send({
          name: 'John Doe',
          email: 'invalid-email',
          password: 'password123'
        })
        .expect(400);
      
      expect(response.body.error).toBe('Invalid email');
    });
    
    it('should require authentication', async () => {
      await request(app)
        .get('/users/123')
        .expect(401);
    });
  });
});
\`\`\`

## Contract Testing

Verify API adheres to OpenAPI spec:

\`\`\`javascript
const { matchers } = require('jest-openapi');
const openApiSpec = require('./openapi.json');

expect.extend(matchers);

describe('API Contract', () => {
  it('should match OpenAPI spec', async () => {
    const response = await request(app)
      .get('/users/123')
      .set('Authorization', 'Bearer token');
    
    expect(response).toSatisfyApiSpec(openApiSpec);
  });
});
\`\`\`

## Load Testing

Test performance under load:

\`\`\`javascript
// k6 load test
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '1m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  const response = http.get('https://api.example.com/users');
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
}
\`\`\`

## Best Practices

1. **Test happy path and edge cases**
2. **Mock external dependencies**
3. **Use factories for test data**
4. **Test authentication and authorization**
5. **Validate response schemas**
6. **Test rate limiting**
7. **Test error handling**
8. **Load test before production**
9. **CI/CD integration**
10. **Monitor test coverage (aim for 80%+)**`,
      multipleChoice: [
        {
          id: 'testing-q1',
          question: 'What is the purpose of contract testing for APIs?',
          options: [
            'To test database performance',
            'To verify the API matches its OpenAPI specification',
            'To test user authentication',
            'To measure API latency',
          ],
          correctAnswer: 1,
          explanation:
            'Contract testing ensures API responses match the documented OpenAPI spec (correct status codes, schema, required fields). Catches when implementation drift from spec. Tools like Pact or jest-openapi automate this.',
          difficulty: 'medium',
        },
        {
          id: 'testing-q2',
          question:
            'Why should integration tests mock external dependencies (payment gateways, email services)?',
          options: [
            'To make tests faster and more reliable',
            'To reduce server costs',
            "External services don't support testing",
            'Mocking is required by testing frameworks',
          ],
          correctAnswer: 0,
          explanation:
            'Mocking external dependencies makes tests: (1) Fast (no network calls), (2) Reliable (no external failures), (3) Repeatable (no side effects like charging cards). Test YOUR code, not third-party services. Use real services in E2E tests only.',
          difficulty: 'easy',
        },
        {
          id: 'testing-q3',
          question:
            'In the test pyramid, why should unit tests outnumber E2E tests?',
          options: [
            'Unit tests are easier to write',
            'Unit tests are fast, cheap, and isolate failures; E2E tests are slow and expensive',
            'E2E tests are less important',
            'Unit tests provide better coverage',
          ],
          correctAnswer: 1,
          explanation:
            'Test pyramid: Many unit tests (fast, cheap, pinpoint failures) → Some integration tests (medium) → Few E2E tests (slow, expensive, brittle). Unit tests run in ms, E2E tests in minutes. Balance speed and confidence.',
          difficulty: 'medium',
        },
        {
          id: 'testing-q4',
          question:
            'What should be the threshold for "successful" in a load test?',
          options: [
            'All requests succeed with 0% errors',
            'p95 latency < target AND error rate < acceptable threshold (e.g., 1%)',
            'Average response time is low',
            'Server CPU stays below 100%',
          ],
          correctAnswer: 1,
          explanation:
            "Load test thresholds: p95 latency < target (e.g., 500ms) AND error rate < 1-5%. Some errors acceptable under load. Don't just track averages (misleading). p95/p99 show real user experience.",
          difficulty: 'medium',
        },
        {
          id: 'testing-q5',
          question:
            'Why test both successful requests AND error cases (400, 500 responses)?',
          options: [
            'To increase test coverage percentage',
            'To ensure API handles errors gracefully and returns proper error responses',
            'Because testing frameworks require it',
            'To slow down the build process',
          ],
          correctAnswer: 1,
          explanation:
            "Testing error cases ensures: (1) Proper status codes (400 vs 500), (2) Clear error messages, (3) No crashes, (4) Security (don't leak sensitive info). Happy path tests are insufficient. Error handling is critical for UX.",
          difficulty: 'easy',
        },
      ],
      quiz: [
        {
          id: 'testing-d1',
          question:
            'Design a comprehensive testing strategy for a payment API. Include unit, integration, contract, security, and load tests.',
          sampleAnswer: `Complete testing strategy for payment API:

**1. Unit Tests** (Jest): Validation, hashing, token generation
**2. Integration Tests** (Supertest): API endpoints with mocked payment gateway
**3. Contract Tests** (Pact): Verify OpenAPI spec compliance
**4. Security Tests**: SQL injection, auth bypass, rate limiting
**5. Load Tests** (k6): 1000 req/s with p95 < 500ms
**6. E2E Tests** (Stripe test mode): Full payment flow
**7. Monitoring**: Real-user monitoring in production`,
          keyPoints: [
            'Unit tests for business logic (validation, calculations)',
            'Integration tests with mocked external services',
            'Security tests for injection, auth, rate limiting',
            'Load tests to verify performance under stress',
            'E2E tests in sandbox environment with real gateway',
          ],
        },
        {
          id: 'testing-d2',
          question:
            'Your CI/CD pipeline takes 45 minutes to run tests. How would you optimize it?',
          sampleAnswer: `Test optimization strategy:

**1. Parallelize**: Run tests in parallel (10 workers)
**2. Split by type**: Unit (2 min) → Integration (10 min) → E2E (30 min)
**3. Fail fast**: Run unit tests first, stop if failing
**4. Cache dependencies**: Docker layer caching, npm cache
**5. Selective testing**: Only test changed code in PR
**6. Database snapshots**: Reuse DB state between tests
**7. Mock external services**: Reduce network calls
**8. Remove flaky tests**: Fix or delete unstable tests

Result: 45 min → 10 min pipeline`,
          keyPoints: [
            'Parallelize tests across multiple workers',
            'Run fast tests first to fail early',
            'Cache dependencies and database snapshots',
            'Mock external services to eliminate network delays',
            'Use selective testing for PRs (only test changed code)',
          ],
        },
        {
          id: 'testing-d3',
          question:
            'Compare unit vs integration vs E2E tests for a REST API. What percentage of each should you have?',
          sampleAnswer: `Test distribution for REST API:

**Unit Tests (70%)**:
- Fast (ms), cheap, many
- Test: validation, business logic, utilities
- Example: validateEmail(), calculateTax()

**Integration Tests (20%)**:
- Medium speed (seconds), some
- Test: API endpoints, database queries
- Example: POST /users returns 201

**E2E Tests (10%)**:
- Slow (minutes), expensive, few
- Test: Critical user flows
- Example: Complete checkout flow

**Rationale**:
- Unit tests catch most bugs quickly
- Integration tests verify components work together
- E2E tests ensure critical flows work end-to-end
- Balance speed (unit) vs confidence (E2E)

**Real Example (Stripe)**:
- 10,000+ unit tests (seconds)
- 500+ integration tests (minutes)
- 50+ E2E tests (hours)

Don't invert pyramid (too many E2E tests = slow CI).`,
          keyPoints: [
            'Unit tests (70%): Fast, many, test individual functions',
            'Integration tests (20%): Medium, some, test API endpoints',
            'E2E tests (10%): Slow, few, test critical user flows',
            'Balance speed (unit) with confidence (E2E)',
            'Inverted pyramid (many E2E) causes slow, brittle CI',
          ],
        },
      ],
    },
    {
      id: 'api-governance',
      title: 'API Governance',
      content: `API governance ensures consistency, quality, and maintainability across all APIs in an organization. It defines standards, processes, and best practices.

## Why API Governance?

- **Consistency**: Same patterns across all APIs
- **Quality**: Maintain high standards
- **Security**: Enforce security practices
- **Discoverability**: Catalog all APIs
- **Compliance**: Meet regulatory requirements
- **Developer experience**: Predictable, well-documented APIs

## API Design Standards

### **REST API Conventions**

Document standards for your organization:

\`\`\`markdown
# API Design Standards

## Naming Conventions
- Use plural nouns: /users not /user
- Use kebab-case: /user-profiles not /userProfiles
- No trailing slashes: /users not /users/
- Resource nesting max 2 levels: /users/123/orders/456 ❌

## HTTP Methods
- GET: Retrieve resources
- POST: Create resources
- PUT: Replace entire resource
- PATCH: Partial update
- DELETE: Remove resource

## Response Codes
- 200: Success
- 201: Created
- 400: Client error (validation)
- 401: Unauthorized
- 403: Forbidden
- 404: Not found
- 500: Server error

## Pagination
- Use cursor-based for scale
- Parameters: \`limit\`, \`after\`
- Response includes: \`hasMore\`, \`endCursor\`

## Error Format
\`\`\`json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email address",
    "details": [
      {
        "field": "email",
        "message": "Must be valid email"
      }
    ]
  }
}
\`\`\`

## Authentication
- Use Bearer tokens (JWT)
- Include \`Authorization: Bearer TOKEN\` header
- Tokens expire in 1 hour
\`\`\`

## API Catalog

Central registry of all APIs:

\`\`\`javascript
// api-catalog.json
{
  "apis": [
    {
      "name": "User API",
      "version": "v2",
      "baseUrl": "https://api.example.com/v2",
      "openApiSpec": "https://api.example.com/v2/openapi.json",
      "owner": "identity-team@example.com",
      "status": "active",
      "documentation": "https://docs.example.com/apis/users",
      "slackChannel": "#user-api-support"
    },
    {
      "name": "Payment API",
      "version": "v1",
      "baseUrl": "https://api.example.com/v1/payments",
      "openApiSpec": "https://api.example.com/v1/payments/openapi.json",
      "owner": "payments-team@example.com",
      "status": "deprecated",
      "deprecationDate": "2024-12-31",
      "migrationGuide": "https://docs.example.com/migrations/payments-v1-v2"
    }
  ]
}
\`\`\`

## API Lifecycle

Define clear stages:

\`\`\`
1. Design → OpenAPI spec review
2. Development → Code review, tests
3. Review → API review board approval
4. Alpha → Internal testing
5. Beta → Limited external access
6. GA → General availability
7. Deprecated → Sunset timeline announced
8. Sunset → API removed
\`\`\`

## API Review Process

**API Review Checklist**:

- ✅ Follows naming conventions
- ✅ Uses correct HTTP methods and status codes
- ✅ Implements authentication
- ✅ Rate limiting configured
- ✅ Error handling consistent
- ✅ OpenAPI spec complete
- ✅ Documentation written
- ✅ Tests written (80%+ coverage)
- ✅ Security review passed
- ✅ Load testing completed
- ✅ Monitoring configured

## Automated Governance

### **Linting OpenAPI Specs**

\`\`\`javascript
// .spectral.yaml
rules:
  operation-success-response:
    description: Operations must have a success response
    severity: error
    given: $.paths.*[get,post,put,patch,delete]
    then:
      field: responses
      function: truthy
  
  no-$ref-siblings:
    description: $ref cannot have siblings
    severity: error
  
  operation-tags:
    description: Operations must have tags
    severity: warn
  
  path-params-defined:
    description: Path parameters must be defined
    severity: error

// Run linter
$ spectral lint openapi.yaml
\`\`\`

### **Pre-commit Hooks**

\`\`\`javascript
// .husky/pre-commit
#!/bin/sh

# Lint OpenAPI specs
npm run lint:openapi

# Run tests
npm run test

# Check API design standards
npm run check:api-standards
\`\`\`

## Versioning Policy

**Semantic Versioning Rules**:

- **MAJOR**: Breaking changes (field removal, type change)
- **MINOR**: New features (new endpoints, optional fields)
- **PATCH**: Bug fixes (no API changes)

**Deprecation Policy**:
- Announce 6 months before sunset
- Support N-1 versions (current + previous)
- Provide migration guide
- Email all affected clients

## Best Practices

1. **Design-first approach**: OpenAPI spec before code
2. **Consistent patterns**: Same conventions across APIs
3. **Automated checks**: Lint specs, enforce standards
4. **API catalog**: Central registry
5. **Review process**: API review board approval
6. **Versioning policy**: Clear rules
7. **Deprecation timeline**: 6-12 months notice
8. **Documentation**: Always up-to-date
9. **Monitoring**: Track usage, errors, performance
10. **Security**: Regular security reviews`,
      multipleChoice: [
        {
          id: 'governance-q1',
          question: 'What is the purpose of an API catalog in API governance?',
          options: [
            'To store API keys and secrets',
            'To provide a central registry of all APIs with ownership, status, and documentation',
            'To automatically generate API documentation',
            'To monitor API performance',
          ],
          correctAnswer: 1,
          explanation:
            'API catalog is a central registry listing all APIs with: owner team, status (active/deprecated), documentation links, OpenAPI specs, and contact info. Essential for discovery and governance in organizations with many APIs.',
          difficulty: 'easy',
        },
        {
          id: 'governance-q2',
          question:
            'Why implement automated linting of OpenAPI specifications?',
          options: [
            'To make APIs faster',
            'To enforce API design standards consistently across all APIs',
            'To reduce server costs',
            'To automatically fix API bugs',
          ],
          correctAnswer: 1,
          explanation:
            'Automated linting (e.g., Spectral) enforces design standards: naming conventions, required fields, error formats. Catches violations before merge, ensuring consistency across all APIs. Human reviews miss details; automation is consistent.',
          difficulty: 'medium',
        },
        {
          id: 'governance-q3',
          question: 'What is the design-first approach to API development?',
          options: [
            'Write code first, then document it',
            'Create OpenAPI specification before implementing the API',
            'Design the database schema before the API',
            'Create UI mockups before API design',
          ],
          correctAnswer: 1,
          explanation:
            'Design-first: Write OpenAPI spec before code. Benefits: (1) Review API design early, (2) Generate mocks for frontend, (3) Ensure docs match implementation, (4) Catch design issues before coding. Alternative: code-first (generate spec from code).',
          difficulty: 'easy',
        },
        {
          id: 'governance-q4',
          question: 'What should a proper API deprecation policy include?',
          options: [
            'Immediate removal of old versions',
            'Advance notice (6-12 months), sunset timeline, and migration guide',
            'Just update documentation',
            'Automatic redirects to new version',
          ],
          correctAnswer: 1,
          explanation:
            'API deprecation policy: (1) Announce 6-12 months early, (2) Set sunset date, (3) Provide migration guide, (4) Email affected clients, (5) Support N-1 versions. Never remove immediately; causes integration breakage.',
          difficulty: 'easy',
        },
        {
          id: 'governance-q5',
          question: 'Why have an API review board in large organizations?',
          options: [
            'To slow down API development',
            'To ensure APIs meet design standards, security requirements, and consistency',
            'To generate API documentation automatically',
            'To reduce the number of APIs',
          ],
          correctAnswer: 1,
          explanation:
            'API review board ensures: (1) Design consistency, (2) Security standards met, (3) No duplicate APIs, (4) Follow best practices. Reviews major changes before GA. Not to slow down, but ensure quality and consistency across organization.',
          difficulty: 'medium',
        },
      ],
      quiz: [
        {
          id: 'governance-d1',
          question:
            'You join a company with 50+ APIs, no standards, inconsistent naming, and no central catalog. Design a governance framework to bring order.',
          sampleAnswer: `API Governance Framework Implementation:

**Phase 1: Assessment (Month 1)**
- Inventory all APIs
- Document current state
- Identify inconsistencies

**Phase 2: Standards (Months 2-3)**
- Define API design standards
- Create OpenAPI linting rules
- Document versioning policy
- Establish deprecation timeline

**Phase 3: Tooling (Months 3-4)**
- Implement Spectral for linting
- Create API catalog
- Set up pre-commit hooks
- Add CI/CD checks

**Phase 4: Migration (Months 4-12)**
- Migrate APIs gradually to standards
- Support old + new patterns
- Provide migration guides
- Sunset non-compliant APIs

**Phase 5: Enforcement (Ongoing)**
- API review board
- Automated checks in CI/CD
- Regular audits
- Documentation requirements`,
          keyPoints: [
            'Start with inventory and assessment of current state',
            'Define clear standards and conventions',
            'Implement automated linting and checks',
            'Gradual migration with support for legacy patterns',
            'Ongoing enforcement through review process',
          ],
        },
        {
          id: 'governance-d2',
          question:
            'Two teams want to build similar APIs (user management). How do you decide: build one shared API or separate APIs?',
          sampleAnswer: `Decision framework for shared vs separate APIs:

**Build Shared API When**:
- ✅ Exact same use case
- ✅ Similar SLAs and requirements
- ✅ Teams willing to collaborate
- ✅ Resources (users) truly shared
- ✅ Low political friction

**Build Separate APIs When**:
- ✅ Different use cases (internal vs external)
- ✅ Different SLAs (99.9% vs 99.99%)
- ✅ Different schemas/data models
- ✅ Teams can't collaborate
- ✅ Different domains (identity vs profile)

**Recommendation Process**:
1. API review board meeting
2. Compare requirements
3. Assess team dynamics
4. Consider maintenance burden
5. Make decision, document rationale

**Example**:
- Identity API (auth, login): Shared (central identity)
- Profile API (preferences): Team-specific (different needs)

Balance: Reduce duplication vs team autonomy.`,
          keyPoints: [
            'Consider if use cases and requirements truly overlap',
            'Assess team collaboration willingness',
            'Evaluate SLA and domain requirements',
            'Balance duplication reduction vs team autonomy',
            'API review board makes final decision',
          ],
        },
        {
          id: 'governance-d3',
          question:
            'Compare design-first vs code-first approach to API development. Which would you use and when?',
          sampleAnswer: `Comparison:

**Design-First** (OpenAPI spec → Code):

Pros:
- ✅ Early API review before coding
- ✅ Frontend can mock while backend builds
- ✅ Documentation guaranteed to match
- ✅ Catches design issues early
- ✅ Enables contract testing

Cons:
- ❌ Upfront time investment
- ❌ Spec maintenance alongside code

**Code-First** (Code → OpenAPI spec):

Pros:
- ✅ Faster initial development
- ✅ Spec always matches code
- ✅ No spec maintenance

Cons:
- ❌ No early API review
- ❌ Design issues found late
- ❌ Frontend blocked on backend

**Recommendation**:

**Design-First for**:
- Public APIs (external developers)
- Large organizations (multiple teams)
- Complex APIs (need review)

**Code-First for**:
- Internal APIs (same team)
- Prototypes (iterating quickly)
- Simple CRUD APIs

**Hybrid Approach**:
- Design major changes first
- Code-first for minor updates
- Auto-generate spec from code
- Review generated specs

Most successful organizations use design-first for external APIs, hybrid for internal.`,
          keyPoints: [
            'Design-first enables early review and parallel development',
            'Code-first is faster for simple/internal APIs',
            'Public APIs should use design-first approach',
            'Internal APIs can use code-first with generated specs',
            'Hybrid approach balances benefits of both',
          ],
        },
      ],
    },
  ],
  keyTakeaways: [
    'RESTful API design uses HTTP standards effectively with proper verbs, status codes, and resource naming',
    'Pagination strategies: offset-based for simplicity, cursor-based for scale and consistency',
    'Field selection optimizes bandwidth without multiple endpoints or paradigm shifts',
    'Never embed unbounded nested collections; use separate paginated endpoints',
    'Structured error responses with codes, messages, and field-level details improve client experience',
    'Different authentication methods for different clients: OAuth for users, API keys for integrations, mTLS for internal services',
    'JWT tokens provide stateless authentication but require strategy for revocation',
    'Comprehensive error handling with proper status codes and detailed validation feedback is essential',
  ],
  learningObjectives: [
    'Design RESTful APIs following REST constraints and Richardson Maturity Model',
    'Choose appropriate pagination, filtering, and sorting strategies for different scales',
    'Implement comprehensive error handling with proper status codes and error structures',
    'Select and implement appropriate authentication methods for different client types',
    'Design secure token management with proper expiration, rotation, and revocation strategies',
    'Structure validation errors with field-level details for better developer experience',
    'Implement retry strategies and idempotency for reliable API operations',
    'Make informed architectural decisions considering scalability, security, and developer experience',
  ],
};
