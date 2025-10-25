/**
 * Quiz questions for GraphQL Performance section
 */

export const graphqlperformanceQuiz = [
  {
    id: 'graphql-perf-d1',
    question:
      'Your GraphQL API experiences performance degradation when users request deeply nested data. Implement a comprehensive solution including DataLoader, caching, and query limits.',
    sampleAnswer: `Comprehensive GraphQL performance optimization strategy:

**1. DataLoader for N+1 Prevention**

\`\`\`javascript
// Setup DataLoaders
const createLoaders = (db) => ({
  user: new DataLoader (async (ids) => {
    const users = await db.users.findAll({
      where: { id: ids }
    });
    return ids.map (id => users.find (u => u.id === id));
  }),
  
  postsByUser: new DataLoader (async (userIds) => {
    const posts = await db.posts.findAll({
      where: { authorId: userIds }
    });
    // Group by userId
    return userIds.map (userId => 
      posts.filter (p => p.authorId === userId)
    );
  }),
  
  commentsByPost: new DataLoader (async (postIds) => {
    const comments = await db.comments.findAll({
      where: { postId: postIds }
    });
    return postIds.map (postId =>
      comments.filter (c => c.postId === postId)
    );
  })
});

// Add to context
const server = new ApolloServer({
  schema,
  context: ({ req }) => ({
    loaders: createLoaders (db),
    user: req.user
  })
});

// Use in resolvers
const resolvers = {
  Post: {
    author: (post, _, { loaders }) => 
      loaders.user.load (post.authorId),
    comments: (post, _, { loaders }) =>
      loaders.commentsByPost.load (post.id)
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
      ignore: ['IntrospectionQuery',]
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
  const key = \`gql:\${info.fieldName}:\${JSON.stringify (args)}\`;
  
  // Check cache
  const cached = await client.get (key);
  if (cached) {
    return JSON.parse (cached);
  }
  
  // Execute resolver
  const result = await resolve (root, args, context, info);
  
  // Cache for 5 minutes
  await client.setex (key, 300, JSON.stringify (result));
  
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
      
      const cursor = after ? decodeCursor (after) : null;
      const posts = await db.posts.findAll({
        where: cursor ? { id: { $gt: cursor.id } } : {},
        limit: first + 1
      });
      
      const hasNextPage = posts.length > first;
      const edges = posts.slice(0, first);
      
      return {
        edges: edges.map (post => ({
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
              clearTimeout (timeout);
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
}).concat (httpLink);

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
  posts: [Post!]! @cacheControl (maxAge: 300, scope: PUBLIC)
  
  # User-specific: shorter cache, private
  me: User @cacheControl (maxAge: 60, scope: PRIVATE)
  
  # Real-time: don't cache
  liveCount: Int! @cacheControl (maxAge: 0)
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
            return JSON.parse (cached);
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
              JSON.stringify (response)
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
const userLoader = new DataLoader (async (ids) => {
  const users = await db.users.findAll({ where: { id: ids }});
  return ids.map (id => users.find (u => u.id === id));
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
  labelNames: ['operation_name', 'operation_type',],
  buckets: [0.1, 0.5, 1, 2, 5, 10]
});

const queryErrors = new prometheus.Counter({
  name: 'graphql_query_errors_total',
  help: 'GraphQL query errors',
  labelNames: ['operation_name', 'error_type',]
});

const queryComplexity = new prometheus.Histogram({
  name: 'graphql_query_complexity',
  help: 'GraphQL query complexity',
  labelNames: ['operation_name',],
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
          errors.forEach (error => {
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
          ).observe (duration);
          
          // Log complexity if available
          if (response.extensions?.complexity) {
            queryComplexity.labels (operationName)
              .observe (response.extensions.complexity);
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
  labelNames: ['type', 'field',],
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
      resolverDuration.labels (typeName, fieldName).observe (duration);
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
  labelNames: ['loader_name',]
});

const loaderMisses = new prometheus.Counter({
  name: 'dataloader_cache_misses_total',
  help: 'DataLoader cache misses',
  labelNames: ['loader_name',]
});

const instrumentedDataLoader = (name, batchFn) => {
  return new DataLoader(
    async (keys) => {
      loaderMisses.labels (name).inc (keys.length);
      return batchFn (keys);
    },
    {
      cacheMap: {
        get (key) {
          const value = cache.get (key);
          if (value) {
            loaderHits.labels (name).inc();
          }
          return value;
        },
        set: cache.set.bind (cache),
        delete: cache.delete.bind (cache),
        clear: cache.clear.bind (cache)
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
            rate (graphql_query_duration_seconds_bucket[5m])
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
            sum (rate (graphql_query_errors_total[5m]))
            /
            sum (rate (graphql_query_duration_seconds_count[5m]))
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
            rate (graphql_query_complexity_bucket[5m])
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
            rate (graphql_resolver_duration_seconds_bucket{
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
            rate (dataloader_cache_hits_total[5m])
            /
            (rate (dataloader_cache_hits_total[5m]) +
             rate (dataloader_cache_misses_total[5m]))
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
          "expr": "histogram_quantile(0.50, rate (graphql_query_duration_seconds_bucket[5m]))"
        },
        {
          "expr": "histogram_quantile(0.95, rate (graphql_query_duration_seconds_bucket[5m]))"
        },
        {
          "expr": "histogram_quantile(0.99, rate (graphql_query_duration_seconds_bucket[5m]))"
        }
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {
          "expr": "sum (rate (graphql_query_errors_total[5m])) by (error_type)"
        }
      ]
    },
    {
      "title": "Top 10 Slowest Queries",
      "targets": [
        {
          "expr": "topk(10, avg by (operation_name) (rate (graphql_query_duration_seconds_sum[5m]) / rate (graphql_query_duration_seconds_count[5m])))"
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
];
