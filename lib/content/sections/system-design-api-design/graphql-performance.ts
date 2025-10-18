/**
 * GraphQL Performance Section
 */

export const graphqlperformanceSection = {
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
};
