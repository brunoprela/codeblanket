/**
 * Quiz questions for RESTful API Design Principles section
 */

export const restfulapidesignQuiz = [
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

**Recommendation**: Use proper HTTP methods (Level 2 REST) as the default. It\'s an industry standard that provides caching, idempotency, and better developer experience with minimal additional complexity.`,
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

**Production Issue**: User\'s profile is wiped except for email. They lose their profile picture, bio, notification preferences, etc. This actually happened at a company I consulted for - a mobile app bug sent incomplete PUT requests, causing customer data loss.

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
  user (id: 123) {
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
];
