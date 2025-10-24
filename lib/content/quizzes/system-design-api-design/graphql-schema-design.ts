/**
 * Quiz questions for GraphQL Schema Design section
 */

export const graphqlschemadesignQuiz = [
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
    ignore: ['IntrospectionQuery',]  // Allow introspection
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
];
