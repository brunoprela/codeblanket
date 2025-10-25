/**
 * Quiz questions for GraphQL section
 */

export const graphqlQuiz = [
  {
    id: 'graphql-migration',
    question:
      "Your company has a large REST API with 150+ endpoints serving web and mobile clients. The mobile team complains about slow performance due to multiple round trips and bandwidth waste. You're considering migrating to GraphQL. Design a migration strategy that minimizes risk, explain how you'd handle the N+1 problem, implement caching, and handle authentication. Include specific technical approaches and timeline.",
    sampleAnswer: `**GraphQL Migration Strategy**
    
    **Phase 1: Assessment & Planning** (Week 1-2)
    
    1. **Analyze REST API Usage**:
       - Which endpoints are called most frequently?
       - Which screens make multiple requests?
       - Measure current: response sizes, latency, number of requests per screen
    
    2. **Identify High-Impact Screens**:
       - Mobile app home feed (calls 5-7 endpoints)
       - User profile page (calls 4 endpoints)
       - Product detail page (calls 6 endpoints)
       - **Expected improvement**: 70% reduction in requests, 50% reduction in data transfer
    
    3. **Choose Architecture**:
       - **Hybrid approach**: GraphQL alongside REST (not replacing)
       - GraphQL for mobile and new web features
       - Keep REST for backward compatibility and public API
       - Use GraphQL as BFF (Backend for Frontend)
    
    **Phase 2: Infrastructure Setup** (Week 3-4)
    
    1. **GraphQL Server Setup**:
    \`\`\`javascript
    // Apollo Server with existing REST API as data sources
    const { ApolloServer } = require('apollo-server-express');
    const { RESTDataSource } = require('apollo-datasource-rest');
    
    // Wrap existing REST API
    class UsersAPI extends RESTDataSource {
      constructor() {
        super();
        this.baseURL = 'https://api.example.com/v1/';
      }
    
      async getUser (id) {
        return this.get(\`users/\${id}\`);
      }
    
      async getUserPosts (userId) {
        return this.get(\`posts?userId=\${userId}\`);
      }
    }
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      dataSources: () => ({
        usersAPI: new UsersAPI(),
        postsAPI: new PostsAPI(),
        commentsAPI: new CommentsAPI()
      })
    });
    \`\`\`
    
    2. **Authentication**:
    \`\`\`javascript
    const server = new ApolloServer({
      context: ({ req }) => {
        // Extract JWT from header
        const token = req.headers.authorization?.replace('Bearer ', ');
        
        if (!token) {
          throw new AuthenticationError('Missing auth token');
        }
        
        try {
          const user = jwt.verify (token, SECRET_KEY);
          return { user };
        } catch (error) {
          throw new AuthenticationError('Invalid token');
        }
      }
    });
    
    // Use in resolvers
    const resolvers = {
      Query: {
        me: (parent, args, { user }) => {
          // user available from context
          return getUserById (user.id);
        }
      }
    };
    \`\`\`
    
    **Phase 3: Solve N+1 Problem** (Week 4)
    
    1. **Implement DataLoader**:
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // Batch load users
    const createUserLoader = () => new DataLoader (async (userIds) => {
      const users = await db.users.findByIds (userIds);
      // Return in same order as requested
      const userMap = new Map (users.map (user => [user.id, user]));
      return userIds.map (id => userMap.get (id));
    });
    
    // Batch load posts by user IDs
    const createPostsByUserLoader = () => new DataLoader (async (userIds) => {
      const posts = await db.posts.findByUserIds (userIds);
      // Group by userId
      const postsByUser = userIds.map (userId => 
        posts.filter (post => post.userId === userId)
      );
      return postsByUser;
    });
    
    // Add to context
    const server = new ApolloServer({
      context: ({ req }) => ({
        user: getUserFromToken (req),
        loaders: {
          user: createUserLoader(),
          postsByUser: createPostsByUserLoader()
        }
      })
    });
    
    // Use in resolvers
    const resolvers = {
      Post: {
        author: (post, args, { loaders }) => {
          // Batched! Multiple calls in same request are combined
          return loaders.user.load (post.userId);
        }
      },
      User: {
        posts: (user, args, { loaders }) => {
          return loaders.postsByUser.load (user.id);
        }
      }
    };
    \`\`\`
    
    **Monitoring N+1**:
    \`\`\`javascript
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        {
          requestDidStart() {
            let queryCount = 0;
            return {
              willSendResponse({ context }) {
                console.log(\`Query count: \${queryCount}\`);
                if (queryCount > 10) {
                  console.warn('Potential N+1 problem detected!');
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Phase 4: Implement Caching** (Week 5)
    
    1. **Response Caching**:
    \`\`\`javascript
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      plugins: [
        responseCachePlugin({
          sessionId: (context) => context.user?.id || null
        })
      ],
      cacheControl: {
        defaultMaxAge: 300 // 5 minutes
      }
    });
    
    // Cache hints in schema
    type User @cacheControl (maxAge: 600) {
      id: ID!
      name: String!
    }
    
    type Post @cacheControl (maxAge: 60) {
      id: ID!
      title: String!
    }
    \`\`\`
    
    2. **Persisted Queries** (for mobile):
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    
    const server = new ApolloServer({
      persistedQueries: {
        cache: redis, // Use Redis for query storage
        ttl: 900 // 15 minutes
      }
    });
    
    // Mobile client sends query hash instead of full query
    // First request: sends query + hash
    // Subsequent requests: only hash
    // Benefits: smaller payload, enables CDN caching (GET request)
    \`\`\`
    
    3. **Client-Side Caching** (Apollo Client):
    \`\`\`javascript
    import { ApolloClient, InMemoryCache } from '@apollo/client';
    
    const client = new ApolloClient({
      uri: 'https://api.example.com/graphql',
      cache: new InMemoryCache({
        typePolicies: {
          Query: {
            fields: {
              user: {
                merge (existing, incoming) {
                  return incoming;
                }
              }
            }
          }
        }
      })
    });
    \`\`\`
    
    **Phase 5: Pilot Implementation** (Week 6-8)
    
    1. **Choose Pilot Screen**: Mobile app home feed
       - Currently makes 7 REST requests
       - 500KB total data transfer
       - 2-3 second load time
    
    2. **GraphQL Query**:
    \`\`\`graphql
    query HomeFeed {
      me {
        id
        name
        avatar
        unreadNotifications
      }
      
      feed (first: 20) {
        edges {
          node {
            id
            content
            imageUrl
            author {
              username
              avatar
            }
            likeCount
            commentCount
            likedByMe
            comments (first: 3) {
              text
              author {
                username
              }
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
      
      suggestions {
        id
        username
        avatar
        mutualFriends
      }
    }
    \`\`\`
    
    3. **Measure Results**:
       - Requests: 7 → 1 (86% reduction)
       - Data transfer: 500KB → 180KB (64% reduction)
       - Load time: 2.5s → 0.9s (64% improvement)
       - Mobile bandwidth saved: ~2GB per user per month
    
    **Phase 6: Gradual Rollout** (Week 9-16)
    
    1. **Week 9-10**: User profile page
    2. **Week 11-12**: Product detail page
    3. **Week 13-14**: Search results
    4. **Week 15-16**: Messaging and notifications
    
    **Phase 7: Monitoring & Optimization** (Ongoing)
    
    1. **Metrics to Track**:
       - Query execution time (p50, p95, p99)
       - Resolver execution time
       - DataLoader hit rate
       - Cache hit rate
       - Error rate by query type
       - N+1 query detection
    
    2. **Observability**:
    \`\`\`javascript
    const { ApolloServerPluginLandingPageGraphQLPlayground } = require('apollo-server-core');
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        ApolloServerPluginInlineTrace(),
        {
          requestDidStart() {
            const start = Date.now();
            return {
              willSendResponse({ metrics, errors }) {
                const duration = Date.now() - start;
                
                metricsCollector.record({
                  duration,
                  query: metrics.queryPlanTrace,
                  errors: errors?.length || 0
                });
                
                if (duration > 1000) {
                  logger.warn(\`Slow query: \${duration}ms\`);
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Key Risks & Mitigations**:
    
    | **Risk** | **Mitigation** |
    |----------|----------------|
    | N+1 queries slow down API | Implement DataLoader for all relationships; monitor query counts |
    | Breaking changes to mobile app | Version GraphQL schema; use @deprecated directive; gradual rollout |
    | Caching complexity | Start with simple response caching; add persisted queries later |
    | Team learning curve | Training sessions; pair programming; comprehensive documentation |
    | Performance regression | A/B testing; rollback plan; monitor all metrics |
    
    **Expected Results**:
    - **Mobile app**: 60-70% reduction in data transfer
    - **Load times**: 40-50% improvement
    - **User satisfaction**: Higher (faster app, less lag)
    - **Infrastructure**: 30% reduction in API calls
    
    **Timeline**: 16 weeks total (4 months)
    
    **Final Architecture**:
    \`\`\`
    Mobile/Web Clients
             ↓
        GraphQL API (Apollo Server)
             ↓
       +---> DataLoader (batching)
       |
       +---> Response Cache (Redis)
       |
       +---> Existing REST APIs (data sources)
             ↓
        Database / Microservices
    \`\`\``,
    keyPoints: [
      'Migrate gradually: start with new features, then migrate high-impact endpoints',
      'Use DataLoader to solve N+1 query problem with batching',
      'Implement response caching with Redis for repeated queries',
      'Add query complexity analysis to prevent expensive operations',
      'Monitor query patterns and optimize resolvers based on real usage',
    ],
  },
  {
    id: 'graphql-security',
    question:
      'Design a comprehensive security strategy for a GraphQL API that prevents common attacks like query depth attacks, query cost attacks, and data exposure. Include specific code examples for: query complexity limits, depth limits, rate limiting, field-level authorization, and monitoring for malicious queries.',
    sampleAnswer: `**Comprehensive GraphQL Security Strategy**
    
    **1. Query Depth Limiting**
    
    **Attack Scenario**:
    \`\`\`graphql
    # Malicious query with infinite recursion
    query {
      user (id: "1") {
        friends {
          friends {
            friends {
              friends {
                # ... 1000 levels deep
              }
            }
          }
        }
      }
    }
    \`\`\`
    
    **Defense**:
    \`\`\`javascript
    const depthLimit = require('graphql-depth-limit');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [depthLimit(7)] // Max depth of 7
    });
    
    // Custom implementation
    function depthLimit (maxDepth) {
      return function (context) {
        return {
          Field (node, key, parent, path, ancestors) {
            const depth = ancestors.filter(
              ancestor => ancestor.kind === 'Field'
            ).length;
            
            if (depth > maxDepth) {
              throw new Error(
                \`Query depth limit of \${maxDepth} exceeded (depth: \${depth})\`
              );
            }
          }
        };
      };
    }
    \`\`\`
    
    **2. Query Cost Analysis**
    
    **Attack Scenario**:
    \`\`\`graphql
    # Expensive query (1 billion operations!)
    query {
      users (first: 1000) {       # 1,000 users
        posts (first: 1000) {      # × 1,000 posts each
          comments (first: 1000) { # × 1,000 comments each
            # Total: 1,000 × 1,000 × 1,000 = 1 billion
          }
        }
      }
    }
    \`\`\`
    
    **Defense**:
    \`\`\`javascript
    const { createComplexityLimitRule } = require('graphql-validation-complexity');
    
    // Define costs in schema
    const typeDefs = gql\`
      type Query {
        users (first: Int = 20): [User!]! @cost (complexity: 10, multipliers: ["first",])
        user (id: ID!): User @cost (complexity: 1)
      }
      
      type User {
        id: ID!
        name: String!
        posts (first: Int = 20): [Post!]! @cost (complexity: 5, multipliers: ["first",])
      }
      
      type Post {
        id: ID!
        comments (first: Int = 20): [Comment!]! @cost (complexity: 3, multipliers: ["first",])
      }
    \`;
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [
        createComplexityLimitRule(1000, {
          onCost: (cost) => {
            console.log(\`Query cost: \${cost}\`);
          }
        })
      ]
    });
    
    // Custom cost calculation
    function calculateQueryCost (query, schema) {
      let cost = 0;
      
      visit (query, {
        Field: (node) => {
          const fieldDef = schema.getField (node.name.value);
          const fieldCost = fieldDef?.cost || 1;
          
          // Apply multipliers (e.g., \`first\` argument)
          const multiplier = node.arguments.find(
            arg => arg.name.value === 'first'
          )?.value.value || 1;
          
          cost += fieldCost * multiplier;
        }
      });
      
      return cost;
    }
    
    // Validate cost before execution
    const validationRules = [
      (context) => ({
        Document (node) {
          const cost = calculateQueryCost (node, context.getSchema());
          
          if (cost > 1000) {
            throw new Error(\`Query cost \${cost} exceeds limit of 1000\`);
          }
          
          console.log(\`Query cost: \${cost}\`);
        }
      })
    ];
    \`\`\`
    
    **3. Rate Limiting**
    
    **Implementation**:
    \`\`\`javascript
    const redis = require('redis');
    const redisClient = redis.createClient();
    
    // Per-user rate limiting
    async function rateLimitPlugin() {
      return {
        requestDidStart: async ({ context }) => {
          const userId = context.user?.id || context.ip;
          const key = \`rate_limit:\${userId}\`;
          
          const requests = await redisClient.incr (key);
          
          if (requests === 1) {
            await redisClient.expire (key, 60); // 1 minute window
          }
          
          if (requests > 100) {
            throw new Error('Rate limit exceeded: 100 requests per minute');
          }
          
          return {
            willSendResponse: async ({ response }) => {
              response.http.headers.set('X-RateLimit-Limit', '100');
              response.http.headers.set('X-RateLimit-Remaining', String(100 - requests));
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [rateLimitPlugin()]
    });
    
    // Per-query-cost rate limiting
    async function costBasedRateLimitPlugin() {
      return {
        requestDidStart: async ({ context, request }) => {
          const userId = context.user?.id;
          const cost = calculateQueryCost (request.query);
          
          const key = \`cost_limit:\${userId}\`;
          const currentCost = await redisClient.incrBy (key, cost);
          
          if (currentCost === cost) {
            await redisClient.expire (key, 60); // 1 minute window
          }
          
          if (currentCost > 10000) {
            throw new Error(\`Query cost budget exceeded: \${currentCost}/10000\`);
          }
        }
      };
    }
    \`\`\`
    
    **4. Field-Level Authorization**
    
    **Schema with permissions**:
    \`\`\`javascript
    const { SchemaDirectiveVisitor } = require('apollo-server');
    const { defaultFieldResolver } = require('graphql');
    
    // Define @auth directive
    const typeDefs = gql\`
      directive @auth (requires: Role = USER) on FIELD_DEFINITION
      
      enum Role {
        USER
        ADMIN
        MODERATOR
      }
      
      type Query {
        me: User @auth (requires: USER)
        users: [User!]! @auth (requires: ADMIN)
        adminStats: Stats @auth (requires: ADMIN)
      }
      
      type User {
        id: ID!
        name: String!
        email: String! @auth (requires: USER)
        privateNotes: String @auth (requires: ADMIN)
      }
    \`;
    
    // Implement directive
    class AuthDirective extends SchemaDirectiveVisitor {
      visitFieldDefinition (field) {
        const { resolve = defaultFieldResolver } = field;
        const { requires } = this.args;
        
        field.resolve = async function (...args) {
          const context = args[2];
          const user = context.user;
          
          if (!user) {
            throw new AuthenticationError('Not authenticated');
          }
          
          if (!user.roles.includes (requires)) {
            throw new ForbiddenError(\`Requires role: \${requires}\`);
          }
          
          return resolve.apply (this, args);
        };
      }
    }
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      schemaDirectives: {
        auth: AuthDirective
      }
    });
    
    // Alternative: Check in resolver
    const resolvers = {
      User: {
        email: (parent, args, { user }) => {
          // Only return email if requesting own profile or admin
          if (user.id === parent.id || user.roles.includes('ADMIN')) {
            return parent.email;
          }
          throw new ForbiddenError('Cannot access email');
        },
        
        privateNotes: (parent, args, { user }) => {
          if (!user.roles.includes('ADMIN')) {
            throw new ForbiddenError('Admin only');
          }
          return parent.privateNotes;
        }
      }
    };
    \`\`\`
    
    **5. Input Validation**
    
    \`\`\`javascript
    const Joi = require('joi');
    
    const resolvers = {
      Mutation: {
        createUser: async (parent, args, context) => {
          // Validate input
          const schema = Joi.object({
            name: Joi.string().min(2).max(50).required(),
            email: Joi.string().email().required(),
            age: Joi.number().min(13).max(120)
          });
          
          const { error, value } = schema.validate (args);
          
          if (error) {
            throw new UserInputError('Invalid input', {
              validationErrors: error.details
            });
          }
          
          // Sanitize HTML in user input
          const sanitizedName = sanitizeHtml (value.name);
          
          return createUser({ ...value, name: sanitizedName });
        }
      }
    };
    \`\`\`
    
    **6. Query Timeout**
    
    \`\`\`javascript
    function timeoutPlugin (maxTimeout = 5000) {
      return {
        requestDidStart: () => {
          const timeout = setTimeout(() => {
            throw new Error(\`Query timeout after \${maxTimeout}ms\`);
          }, maxTimeout);
          
          return {
            willSendResponse: () => {
              clearTimeout (timeout);
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [timeoutPlugin(5000)] // 5 second timeout
    });
    \`\`\`
    
    **7. Monitoring and Alerting**
    
    \`\`\`javascript
    const Sentry = require('@sentry/node');
    
    function monitoringPlugin() {
      return {
        requestDidStart: ({ request }) => {
          const start = Date.now();
          
          return {
            didEncounterErrors: ({ errors }) => {
              errors.forEach (error => {
                // Log to Sentry
                Sentry.captureException (error, {
                  contexts: {
                    graphql: {
                      query: request.query,
                      variables: request.variables
                    }
                  }
                });
                
                // Log suspicious queries
                if (error.message.includes('depth') || 
                    error.message.includes('cost')) {
                  logger.warn('Potentially malicious query detected', {
                    query: request.query,
                    error: error.message,
                    user: request.context.user?.id
                  });
                }
              });
            },
            
            willSendResponse: ({ response }) => {
              const duration = Date.now() - start;
              
              // Alert on slow queries
              if (duration > 2000) {
                logger.warn(\`Slow query: \${duration}ms\`, {
                  query: request.query
                });
              }
              
              // Track metrics
              metrics.histogram('graphql.query.duration', duration, {
                operation: request.operationName
              });
            }
          };
        }
      };
    }
    
    const server = new ApolloServer({
      plugins: [monitoringPlugin()]
    });
    \`\`\`
    
    **8. Disable Introspection in Production**
    
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      introspection: process.env.NODE_ENV !== 'production',
      playground: process.env.NODE_ENV !== 'production'
    });
    \`\`\`
    
    **9. Persistent Query Whitelist**
    
    \`\`\`javascript
    // Only allow pre-approved queries in production
    const approvedQueries = new Map([
      ['abc123...', 'query GetUser($id: ID!) { user (id: $id) { id name } }',],
      ['def456...', 'query GetFeed { feed { id title } }',]
    ]);
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [
        (context) => ({
          Document (node) {
            if (process.env.NODE_ENV === 'production') {
              const queryHash = context.request.extensions?.persistedQuery?.sha256Hash;
              
              if (!queryHash || !approvedQueries.has (queryHash)) {
                throw new Error('Query not in whitelist');
              }
            }
          }
        })
      ]
    });
    \`\`\`
    
    **Complete Security Middleware**:
    
    \`\`\`javascript
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      
      // Authentication
      context: ({ req }) => ({
        user: getUserFromToken (req),
        ip: req.ip
      }),
      
      // Validation rules
      validationRules: [
        depthLimit(7),                    // Max depth 7
        createComplexityLimitRule(1000), // Max cost 1000
        queryTimeoutRule(5000)            // 5 second timeout
      ],
      
      // Plugins
      plugins: [
        rateLimitPlugin(),
        monitoringPlugin(),
        costBasedRateLimitPlugin()
      ],
      
      // Production settings
      introspection: process.env.NODE_ENV !== 'production',
      playground: process.env.NODE_ENV !== 'production',
      
      // Schema directives
      schemaDirectives: {
        auth: AuthDirective
      }
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **Query depth limit** prevents recursive query attacks
    2. **Query cost analysis** prevents expensive queries (complexity × multipliers)
    3. **Rate limiting** by user (requests/minute) and query cost (cost/minute)
    4. **Field-level authorization** with @auth directive or in resolvers
    5. **Input validation** with Joi or similar library
    6. **Query timeout** prevents long-running queries
    7. **Monitoring** logs suspicious queries and slow queries
    8. **Disable introspection** in production to hide schema
    9. **Persistent query whitelist** only allows pre-approved queries
    10. **Defense in depth**: layer multiple security measures
    
    **Security Checklist**:
    - ✅ Query depth limit configured
    - ✅ Query cost analysis implemented
    - ✅ Rate limiting (per-user and per-cost)
    - ✅ Field-level authorization
    - ✅ Input validation and sanitization
    - ✅ Query timeout
    - ✅ Monitoring and alerting
    - ✅ Introspection disabled in production
    - ✅ HTTPS enforced
    - ✅ JWT token validation
    - ✅ CSRF protection (if using cookies)
    - ✅ Audit logging for sensitive operations`,
    keyPoints: [
      'Implement query depth and complexity limits to prevent attacks',
      'Use field-level authorization for sensitive data',
      'Apply rate limiting per user/IP to prevent abuse',
      'Disable introspection in production',
      'Monitor query patterns for malicious behavior',
    ],
  },
  {
    id: 'graphql-performance',
    question:
      "You're experiencing performance issues with your GraphQL API. Some queries take 5-10 seconds to complete, and your database is overwhelmed with queries. Using monitoring tools, you discovered: (1) N+1 queries in multiple resolvers, (2) Inefficient pagination, (3) Missing caching. Design a comprehensive performance optimization strategy including DataLoader implementation, pagination improvements, caching at multiple levels, and monitoring. Provide specific code examples and expected performance improvements.",
    sampleAnswer: `**GraphQL Performance Optimization Strategy**
    
    **Current State** (Performance Issues):
    - Query duration: 5-10 seconds (p95)
    - Database queries per request: 500-1000 (N+1 problem)
    - Database CPU: 90% (overloaded)
    - Cache hit rate: 0% (no caching)
    
    **Target State**:
    - Query duration: <300ms (p95)
    - Database queries per request: <20
    - Database CPU: <40%
    - Cache hit rate: >70%
    
    ---
    
    **1. Solve N+1 Problem with DataLoader**
    
    **Problem**:
    \`\`\`graphql
    query {
      posts (first: 100) {     # 1 query
        id
        title
        author {              # 100 queries!
          name
        }
        comments {            # 100 queries!
          text
          author {            # N queries!
            name
          }
        }
      }
    }
    # Total: 1 + 100 + 100 + N = 200+ queries
    \`\`\`
    
    **Solution**:
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // 1. Create DataLoader for users
    function createUserLoader() {
      return new DataLoader (async (userIds) => {
        console.log(\`Batch loading \${userIds.length} users\`);
        
        // Single query for all user IDs
        const users = await db.users.findByIds (userIds);
        
        // Return in same order as requested
        const userMap = new Map (users.map (u => [u.id, u]));
        return userIds.map (id => userMap.get (id));
      });
    }
    
    // 2. Create DataLoader for comments by post
    function createCommentsByPostLoader() {
      return new DataLoader (async (postIds) => {
        console.log(\`Batch loading comments for \${postIds.length} posts\`);
        
        // Single query for all comments
        const comments = await db.comments.findByPostIds (postIds);
        
        // Group by postId
        const commentsByPost = new Map();
        postIds.forEach (id => commentsByPost.set (id, []));
        comments.forEach (comment => {
          commentsByPost.get (comment.postId).push (comment);
        });
        
        return postIds.map (id => commentsByPost.get (id) || []);
      });
    }
    
    // 3. Add loaders to context
    const server = new ApolloServer({
      context: ({ req }) => ({
        user: getUserFromToken (req),
        loaders: {
          user: createUserLoader(),
          commentsByPost: createCommentsByPostLoader()
        }
      })
    });
    
    // 4. Use in resolvers
    const resolvers = {
      Query: {
        posts: async () => {
          return await db.posts.findAll(); // 1 query
        }
      },
      
      Post: {
        author: async (post, args, { loaders }) => {
          // Batched! All posts in request combined into single query
          return await loaders.user.load (post.userId);
        },
        
        comments: async (post, args, { loaders }) => {
          // Batched! All posts in request combined into single query
          return await loaders.commentsByPost.load (post.id);
        }
      },
      
      Comment: {
        author: async (comment, args, { loaders }) => {
          // Batched!
          return await loaders.user.load (comment.userId);
        }
      }
    };
    \`\`\`
    
    **Result**:
    - **Before**: 1 + 100 + 100 + N = 200+ queries
    - **After**: 4 queries (posts, users, comments, comment authors)
    - **Improvement**: 98% reduction in queries
    
    ---
    
    **2. Advanced DataLoader Patterns**
    
    **Prime Cache After Create**:
    \`\`\`javascript
    const resolvers = {
      Mutation: {
        createUser: async (parent, args, { loaders }) => {
          const user = await db.users.create (args);
          
          // Prime the cache so future loads don't hit DB
          loaders.user.prime (user.id, user);
          
          return user;
        }
      }
    };
    \`\`\`
    
    **Clear Cache After Update**:
    \`\`\`javascript
    const resolvers = {
      Mutation: {
        updateUser: async (parent, { id, ...updates }, { loaders }) => {
          const user = await db.users.update (id, updates);
          
          // Clear old cached value
          loaders.user.clear (id);
          
          // Prime with new value
          loaders.user.prime (id, user);
          
          return user;
        }
      }
    };
    \`\`\`
    
    **Composite Keys**:
    \`\`\`javascript
    // Load posts by multiple criteria
    function createPostLoader() {
      return new DataLoader (async (keys) => {
        // keys = [{ userId: '1', status: 'published' }, ...]
        
        const posts = await db.posts.findByMultiple (keys);
        
        return keys.map (key => 
          posts.filter (p => 
            p.userId === key.userId && p.status === key.status
          )
        );
      }, {
        cacheKeyFn: (key) => \`\${key.userId}:\${key.status}\`
      });
    }
    \`\`\`
    
    ---
    
    **3. Efficient Pagination**
    
    **Problem: Offset-based pagination**:
    \`\`\`graphql
    # Slow for large offsets
    query {
      posts (limit: 20, offset: 10000) {
        # SELECT * FROM posts LIMIT 20 OFFSET 10000
        # Database must scan 10,020 rows!
      }
    }
    \`\`\`
    
    **Solution: Cursor-based pagination**:
    
    **Schema**:
    \`\`\`graphql
    type Query {
      posts (first: Int, after: String): PostConnection!
    }
    
    type PostConnection {
      edges: [PostEdge!]!
      pageInfo: PageInfo!
    }
    
    type PostEdge {
      node: Post!
      cursor: String!
    }
    
    type PageInfo {
      hasNextPage: Boolean!
      endCursor: String
    }
    \`\`\`
    
    **Resolver**:
    \`\`\`javascript
    const resolvers = {
      Query: {
        posts: async (parent, { first = 20, after }, context) => {
          let query = db.posts.query().limit (first + 1);
          
          if (after) {
            // Decode cursor (base64 encoded ID)
            const cursorId = Buffer.from (after, 'base64').toString('utf-8');
            query = query.where('id', '>', cursorId);
          }
          
          const posts = await query.orderBy('created_at', 'desc');
          const hasNextPage = posts.length > first;
          const edges = posts.slice(0, first);
          
          return {
            edges: edges.map (post => ({
              node: post,
              cursor: Buffer.from (post.id).toString('base64')
            })),
            pageInfo: {
              hasNextPage,
              endCursor: edges.length > 0 
                ? Buffer.from (edges[edges.length - 1].id).toString('base64')
                : null
            }
          };
        }
      }
    };
    \`\`\`
    
    **Benefits**:
    - No matter the page, always scans ~20 rows (not 10,020)
    - Consistent performance across all pages
    - Works well with real-time data (no skipped/duplicate items)
    
    ---
    
    **4. Multi-Level Caching**
    
    **Layer 1: DataLoader (Request-Level Cache)**:
    - Already implemented above
    - Deduplicates within single request
    - Duration: Request lifetime (~100ms)
    
    **Layer 2: Redis (Application-Level Cache)**:
    \`\`\`javascript
    const redis = require('redis');
    const redisClient = redis.createClient();
    
    // Cached resolver
    async function getCachedUser (userId) {
      const cacheKey = \`user:\${userId}\`;
      
      // Try cache first
      const cached = await redisClient.get (cacheKey);
      if (cached) {
        return JSON.parse (cached);
      }
      
      // Cache miss, load from DB
      const user = await db.users.findById (userId);
      
      // Store in cache (5 minutes)
      await redisClient.setEx (cacheKey, 300, JSON.stringify (user));
      
      return user;
    }
    
    // Integrate with DataLoader
    function createUserLoader() {
      return new DataLoader (async (userIds) => {
        // Check Redis for each ID
        const pipeline = redisClient.pipeline();
        userIds.forEach (id => pipeline.get(\`user:\${id}\`));
        const cachedResults = await pipeline.exec();
        
        // Separate hits and misses
        const hits = [];
        const misses = [];
        
        userIds.forEach((id, idx) => {
          if (cachedResults[idx][1]) {
            hits[idx] = JSON.parse (cachedResults[idx][1]);
          } else {
            misses.push({ id, idx });
          }
        });
        
        // Load misses from database
        if (misses.length > 0) {
          const missedIds = misses.map (m => m.id);
          const users = await db.users.findByIds (missedIds);
          
          // Store in Redis
          const cachePipeline = redisClient.pipeline();
          users.forEach (user => {
            cachePipeline.setEx(\`user:\${user.id}\`, 300, JSON.stringify (user));
          });
          await cachePipeline.exec();
          
          // Fill results
          misses.forEach((miss, idx) => {
            hits[miss.idx] = users[idx];
          });
        }
        
        return hits;
      });
    }
    \`\`\`
    
    **Layer 3: Apollo Response Cache**:
    \`\`\`javascript
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      plugins: [
        responseCachePlugin({
          sessionId: (context) => context.user?.id || null
        })
      ],
      cacheControl: {
        defaultMaxAge: 60 // 1 minute
      }
    });
    
    // Add cache hints to schema
    const typeDefs = gql\`
      type User @cacheControl (maxAge: 300) {
        id: ID!
        name: String!
      }
      
      type Post @cacheControl (maxAge: 60) {
        id: ID!
        title: String!
      }
      
      type Query {
        user (id: ID!): User @cacheControl (maxAge: 600)
      }
    \`;
    \`\`\`
    
    **Layer 4: CDN (Edge Cache)**:
    \`\`\`javascript
    // Use persisted queries for GET requests
    const { ApolloServer } = require('apollo-server-express');
    
    const server = new ApolloServer({
      persistedQueries: {
        cache: redis,
        ttl: 900
      }
    });
    
    // Client sends GET request with query hash
    // GET /graphql?extensions={"persistedQuery":{"version":1,"sha256Hash":"abc123..."}}
    
    // Response includes cache headers
    app.use('/graphql', (req, res, next) => {
      if (req.method === 'GET') {
        res.set('Cache-Control', 'public, max-age=60');
      }
      next();
    });
    \`\`\`
    
    ---
    
    **5. Database Query Optimization**
    
    **Add Indexes**:
    \`\`\`sql
    -- Index for post author lookups
    CREATE INDEX idx_posts_user_id ON posts (user_id);
    
    -- Index for comments by post
    CREATE INDEX idx_comments_post_id ON comments (post_id);
    
    -- Composite index for pagination
    CREATE INDEX idx_posts_created_at_id ON posts (created_at DESC, id);
    \`\`\`
    
    **Use SELECT instead of SELECT ***:
    \`\`\`javascript
    // Bad
    const users = await db.users.findByIds (userIds);
    // SELECT * FROM users WHERE id IN (...)
    
    // Good: Only select needed fields
    const users = await db.users
      .select('id', 'name', 'email', 'avatar')
      .findByIds (userIds);
    // SELECT id, name, email, avatar FROM users WHERE id IN (...)
    \`\`\`
    
    ---
    
    **6. Monitoring and Profiling**
    
    **Install Apollo Tracing**:
    \`\`\`javascript
    const { ApolloServerPluginInlineTrace } = require('apollo-server-core');
    
    const server = new ApolloServer({
      plugins: [
        ApolloServerPluginInlineTrace(),
        {
          requestDidStart() {
            const start = Date.now();
            let queryCount = 0;
            
            // Monkey-patch database query function
            const originalQuery = db.query;
            db.query = function(...args) {
              queryCount++;
              return originalQuery.apply (this, args);
            };
            
            return {
              willSendResponse({ response, context }) {
                const duration = Date.now() - start;
                
                console.log({
                  operation: context.operation?.name?.value,
                  duration: \`\${duration}ms\`,
                  queries: queryCount,
                  cacheHits: context.loaders?.user?.stats()?.cacheHits || 0
                });
                
                // Alert if slow
                if (duration > 1000) {
                  logger.warn('Slow query detected', {
                    operation: context.operation?.name?.value,
                    duration,
                    queries: queryCount
                  });
                }
                
                // Alert if too many queries
                if (queryCount > 50) {
                  logger.warn('Possible N+1 problem', {
                    operation: context.operation?.name?.value,
                    queries: queryCount
                  });
                }
              }
            };
          }
        }
      ]
    });
    \`\`\`
    
    **Dashboard Metrics**:
    \`\`\`javascript
    const metrics = {
      queryDuration: new Histogram('graphql_query_duration'),
      databaseQueries: new Histogram('graphql_database_queries'),
      cacheHitRate: new Gauge('graphql_cache_hit_rate'),
      dataLoaderBatchSize: new Histogram('graphql_dataloader_batch_size')
    };
    \`\`\`
    
    ---
    
    **Expected Performance Improvements**:
    
    | **Metric** | **Before** | **After** | **Improvement** |
    |------------|-----------|---------|----------------|
    | **p50 latency** | 3s | 150ms | 95% faster |
    | **p95 latency** | 8s | 300ms | 96% faster |
    | **p99 latency** | 12s | 500ms | 96% faster |
    | **DB queries/request** | 500-1000 | 5-20 | 98% reduction |
    | **DB CPU** | 90% | 30% | 67% reduction |
    | **Cache hit rate** | 0% | 75% | Massive improvement |
    | **Throughput** | 50 RPS | 500 RPS | 10x improvement |
    
    **Key Takeaways**:
    
    1. **DataLoader solves N+1** - batch and cache database queries
    2. **Cursor-based pagination** - consistent performance across all pages
    3. **Multi-level caching** - Request → Redis → Response → CDN
    4. **Database indexes** - essential for joins and lookups
    5. **Monitoring** - track query count, duration, cache hit rate
    6. **Prime/clear cache** - keep DataLoader in sync with mutations
    7. **SELECT only needed fields** - reduce data transfer
    8. **Query depth/cost limits** - prevent expensive queries
    9. **Persisted queries** - enable CDN caching
    10. **Continuous profiling** - identify and fix regressions`,
    keyPoints: [
      'Use DataLoader to batch and deduplicate queries, solving N+1 problem',
      'Implement cursor-based pagination instead of offset/limit for efficiency',
      'Add caching at multiple levels: CDN (persisted queries), Redis (query results), Application (DataLoader)',
      'Monitor query patterns: complexity, duration, and frequency',
      'Set query depth and cost limits to prevent expensive queries',
      'Expected improvements: 20x fewer DB queries, 15x faster response time, 70% cache hit rate',
    ],
  },
];
