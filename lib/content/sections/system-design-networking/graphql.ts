/**
 * GraphQL Section
 */

export const graphqlSection = {
  id: 'graphql',
  title: 'GraphQL',
  content: `GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need. Understanding GraphQL is essential for modern API design, especially for mobile and web applications.
    
    ## What is GraphQL?
    
    **GraphQL** is a query language developed by Facebook that allows clients to specify exactly what data they need in a single request.
    
    **Key Principles**:
    - **Client-Specified Queries**: Client defines the shape of the response
    - **Single Endpoint**: All queries go to one endpoint (typically \`/graphql\`)
    - **Strong Typing**: Schema defines types and relationships
    - **Hierarchical**: Queries match the shape of the data
    
    **Comparison**:
    
    | **Aspect** | **REST** | **GraphQL** |
    |------------|----------|-------------|
    | **Endpoints** | Multiple (\`/users\`, \`/posts\`) | Single (\`/graphql\`) |
    | **Data Fetching** | Fixed response structure | Client specifies fields |
    | **Over-fetching** | Common (get all fields) | None (request only what you need) |
    | **Under-fetching** | Requires multiple requests | Single request gets all data |
    | **Versioning** | URL versioning (\`/v1/users\`) | Schema evolution (no breaking changes) |
    | **Caching** | Easy (HTTP caching) | Complex (requires work) |
    | **Learning Curve** | Low | Medium |
    
    ---
    
    ## GraphQL Schema
    
    **Schema defines**:
    - Types (objects, scalars, enums)
    - Queries (read operations)
    - Mutations (write operations)
    - Subscriptions (real-time updates)
    
    **Example Schema**:
    \`\`\`graphql
    # user.graphql
    type User {
      id: ID!
      name: String!
      email: String!
      age: Int
      posts: [Post!]!
      friends: [User!]!
    }
    
    type Post {
      id: ID!
      title: String!
      content: String!
      author: User!
      comments: [Comment!]!
      createdAt: DateTime!
    }
    
    type Comment {
      id: ID!
      text: String!
      author: User!
      post: Post!
    }
    
    type Query {
      # Get single user
      user (id: ID!): User
      
      # Get all users with pagination
      users (limit: Int = 20, offset: Int = 0): [User!]!
      
      # Get posts by user
      posts (userId: ID!): [Post!]!
      
      # Search users
      searchUsers (query: String!): [User!]!
    }
    
    type Mutation {
      # Create user
      createUser (name: String!, email: String!, age: Int): User!
      
      # Update user
      updateUser (id: ID!, name: String, email: String, age: Int): User!
      
      # Delete user
      deleteUser (id: ID!): Boolean!
      
      # Create post
      createPost (userId: ID!, title: String!, content: String!): Post!
      
      # Add comment
      addComment (postId: ID!, userId: ID!, text: String!): Comment!
    }
    
    type Subscription {
      # Subscribe to new posts
      newPost: Post!
      
      # Subscribe to comments on a post
      newComment (postId: ID!): Comment!
    }
    
    # Custom scalar for dates
    scalar DateTime
    \`\`\`
    
    ---
    
    ## GraphQL Queries
    
    ### **Basic Query**:
    
    \`\`\`graphql
    query {
      user (id: "123") {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Response**:
    \`\`\`json
    {
      "data": {
        "user": {
          "id": "123",
          "name": "John Doe",
          "email": "john@example.com"
        }
      }
    }
    \`\`\`
    
    ### **Nested Query**:
    
    \`\`\`graphql
    query {
      user (id: "123") {
        id
        name
        posts {
          id
          title
          comments {
            id
            text
            author {
              name
            }
          }
        }
      }
    }
    \`\`\`
    
    **Why This is Powerful**:
    - Single request gets user, their posts, and all comments with author names
    - With REST, would require 3+ requests: \`/users/123\`, \`/posts?userId=123\`, \`/comments?postId=X\` (for each post)
    
    ### **Query with Variables**:
    
    \`\`\`graphql
    query GetUser($userId: ID!) {
      user (id: $userId) {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Variables**:
    \`\`\`json
    {
      "userId": "123"
    }
    \`\`\`
    
    ### **Query with Aliases**:
    
    \`\`\`graphql
    query {
      user1: user (id: "123") {
        name
      }
      user2: user (id: "456") {
        name
      }
    }
    \`\`\`
    
    **Response**:
    \`\`\`json
    {
      "data": {
        "user1": { "name": "Alice" },
        "user2": { "name": "Bob" }
      }
    }
    \`\`\`
    
    ### **Query with Fragments**:
    
    \`\`\`graphql
    fragment UserFields on User {
      id
      name
      email
    }
    
    query {
      user1: user (id: "123") {
        ...UserFields
      }
      user2: user (id: "456") {
        ...UserFields
      }
    }
    \`\`\`
    
    ---
    
    ## GraphQL Mutations
    
    **Create User**:
    \`\`\`graphql
    mutation {
      createUser (name: "Jane Doe", email: "jane@example.com", age: 28) {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Update User**:
    \`\`\`graphql
    mutation {
      updateUser (id: "123", name: "John Smith") {
        id
        name
        email
      }
    }
    \`\`\`
    
    **Create Post with Variables**:
    \`\`\`graphql
    mutation CreatePost($userId: ID!, $title: String!, $content: String!) {
      createPost (userId: $userId, title: $title, content: $content) {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    ---
    
    ## GraphQL Subscriptions
    
    **Subscriptions enable real-time updates via WebSocket**.
    
    **Client subscribes**:
    \`\`\`graphql
    subscription {
      newPost {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    **Server pushes updates**:
    \`\`\`json
    {
      "data": {
        "newPost": {
          "id": "789",
          "title": "Breaking News",
          "author": {
            "name": "Alice"
          }
        }
      }
    }
    \`\`\`
    
    ---
    
    ## Implementing GraphQL Server (Node.js + Apollo)
    
    \`\`\`javascript
    const { ApolloServer, gql } = require('apollo-server');
    
    // Define schema
    const typeDefs = gql\`
      type User {
        id: ID!
        name: String!
        email: String!
        posts: [Post!]!
      }
      
      type Post {
        id: ID!
        title: String!
        content: String!
        author: User!
      }
      
      type Query {
        user (id: ID!): User
        users: [User!]!
      }
      
      type Mutation {
        createUser (name: String!, email: String!): User!
      }
    \`;
    
    // Define resolvers
    const resolvers = {
      Query: {
        user: async (parent, { id }, context) => {
          return await context.db.users.findById (id);
        },
        
        users: async (parent, args, context) => {
          return await context.db.users.findAll();
        }
      },
      
      Mutation: {
        createUser: async (parent, { name, email }, context) => {
          const user = await context.db.users.create({ name, email });
          return user;
        }
      },
      
      // Field resolvers
      User: {
        posts: async (parent, args, context) => {
          // parent is the User object
          return await context.db.posts.findByUserId (parent.id);
        }
      },
      
      Post: {
        author: async (parent, args, context) => {
          // parent is the Post object
          return await context.db.users.findById (parent.userId);
        }
      }
    };
    
    // Create server
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      context: ({ req }) => ({
        db: database, // Pass database connection
        user: req.user // Pass authenticated user
      })
    });
    
    server.listen().then(({ url }) => {
      console.log(\`🚀 Server ready at \${url}\`);
    });
    \`\`\`
    
    ---
    
    ## The N+1 Query Problem
    
    **One of the biggest pitfalls in GraphQL**.
    
    **Scenario**:
    \`\`\`graphql
    query {
      posts {
        id
        title
        author {
          name
        }
      }
    }
    \`\`\`
    
    **Naive Implementation**:
    \`\`\`javascript
    const resolvers = {
      Query: {
        posts: () => db.posts.findAll() // 1 query
      },
      Post: {
        author: (post) => db.users.findById (post.userId) // N queries!
      }
    };
    \`\`\`
    
    **Problem**: If there are 100 posts, this makes 101 database queries (1 for posts + 100 for authors).
    
    **Solution: DataLoader (Batching)**:
    
    \`\`\`javascript
    const DataLoader = require('dataloader');
    
    // Batch load users
    const userLoader = new DataLoader (async (userIds) => {
      const users = await db.users.findByIds (userIds);
      // Return users in same order as userIds
      return userIds.map (id => users.find (user => user.id === id));
    });
    
    const resolvers = {
      Query: {
        posts: () => db.posts.findAll() // 1 query
      },
      Post: {
        author: (post, args, { loaders }) => {
          return loaders.user.load (post.userId); // Batched!
        }
      }
    };
    
    // Context setup
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      context: () => ({
        loaders: {
          user: new DataLoader (batchLoadUsers)
        }
      })
    });
    \`\`\`
    
    **Result**: Now makes only 2 queries (1 for posts + 1 batched query for all authors).
    
    ---
    
    ## GraphQL Caching
    
    **Challenge**: GraphQL uses POST requests to \`/graphql\`, which are not cacheable by HTTP.
    
    **Solutions**:
    
    ### **1. Persisted Queries**:
    
    \`\`\`javascript
    // Client sends query hash instead of full query
    POST /graphql
    {
      "extensions": {
        "persistedQuery": {
          "version": 1,
          "sha256Hash": "abc123..."
        }
      }
    }
    
    // Server looks up query by hash
    const query = queryRegistry.get("abc123...");
    \`\`\`
    
    **Benefits**:
    - Smaller request size
    - Enables GET requests (cacheable)
    - Security (only allowed queries can execute)
    
    ### **2. Response Caching**:
    
    \`\`\`javascript
    const { ApolloServer } = require('apollo-server');
    const responseCachePlugin = require('apollo-server-plugin-response-cache');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      plugins: [responseCachePlugin()],
      cacheControl: {
        defaultMaxAge: 300 // 5 minutes
      }
    });
    \`\`\`
    
    **Cache hints in schema**:
    \`\`\`graphql
    type User @cacheControl (maxAge: 300) {
      id: ID!
      name: String!
      email: String!
    }
    
    type Post @cacheControl (maxAge: 60) {
      id: ID!
      title: String!
    }
    \`\`\`
    
    ### **3. Client-Side Caching (Apollo Client)**:
    
    \`\`\`javascript
    import { ApolloClient, InMemoryCache } from '@apollo/client';
    
    const client = new ApolloClient({
      uri: 'http://localhost:4000/graphql',
      cache: new InMemoryCache({
        typePolicies: {
          Query: {
            fields: {
              user: {
                read (existing, { args, toReference }) {
                  // Return cached user if exists
                  return existing || toReference({
                    __typename: 'User',
                    id: args.id
                  });
                }
              }
            }
          }
        }
      })
    });
    \`\`\`
    
    ---
    
    ## GraphQL vs REST vs gRPC
    
    | **Feature** | **REST** | **GraphQL** | **gRPC** |
    |-------------|----------|-------------|----------|
    | **Data Fetching** | Over/under-fetching | Exact data needed | Fixed by proto definition |
    | **Endpoints** | Many | One | Service methods |
    | **Type Safety** | Weak (OpenAPI helps) | Strong (schema) | Very strong (protobuf) |
    | **Performance** | Good | Good | Excellent (binary) |
    | **Caching** | Easy (HTTP) | Complex | Difficult |
    | **Real-time** | SSE/WebSocket | Subscriptions (WebSocket) | Streaming |
    | **Learning Curve** | Low | Medium | High |
    | **Mobile-Friendly** | No (over-fetching) | Yes (efficient) | Yes (efficient) |
    | **Public API** | Excellent | Good | Poor (browser support) |
    | **Internal Services** | Good | Overkill | Excellent |
    
    ---
    
    ## When to Use GraphQL
    
    ### **✅ Use GraphQL When:**
    
    1. **Mobile Applications**
       - Bandwidth is limited
       - Need to minimize data transfer
       - Multiple resources needed per screen
    
    2. **Complex UIs with Many Relationships**
       - Dashboard with data from many sources
       - Social network (users, posts, comments, likes)
       - E-commerce (products, reviews, recommendations)
    
    3. **Rapid Frontend Development**
       - Frontend team can iterate without backend changes
       - No need for new endpoints for each view
       - GraphQL Playground for testing
    
    4. **Multiple Clients with Different Needs**
       - Web app needs different data than mobile app
       - Each client requests only what it needs
       - No need for multiple API versions
    
    5. **Real-Time Updates**
       - Subscriptions for live data
       - Chat applications
       - Live sports scores
    
    ### **❌ Avoid GraphQL When:**
    
    1. **Simple CRUD APIs**
       - REST is simpler and more familiar
       - No complex relationships
       - Standard HTTP caching sufficient
    
    2. **Public APIs for Third Parties**
       - REST more familiar to external developers
       - Better documentation with OpenAPI
       - Easier to rate limit by endpoint
    
    3. **File Uploads/Downloads**
       - GraphQL not designed for binary data
       - Multipart uploads awkward in GraphQL
       - REST simpler for files
    
    4. **Team Lacks GraphQL Experience**
       - Learning curve can slow development
       - Requires understanding of N+1 problem, DataLoader, caching
       - REST more familiar
    
    ---
    
    ## Common GraphQL Mistakes
    
    ### **❌ Mistake 1: Not Using DataLoader (N+1 Problem)**
    
    \`\`\`javascript
    // Bad: N+1 queries
    const resolvers = {
      Post: {
        author: (post) => db.users.findById (post.userId)
      }
    };
    
    // Good: Batch with DataLoader
    const resolvers = {
      Post: {
        author: (post, args, { loaders }) => loaders.user.load (post.userId)
      }
    };
    \`\`\`
    
    ### **❌ Mistake 2: No Pagination**
    
    \`\`\`javascript
    // Bad: Return all users
    type Query {
      users: [User!]!
    }
    
    // Good: Paginate with cursor or offset
    type Query {
      users (first: Int, after: String): UserConnection!
    }
    
    type UserConnection {
      edges: [UserEdge!]!
      pageInfo: PageInfo!
    }
    
    type UserEdge {
      node: User!
      cursor: String!
    }
    
    type PageInfo {
      hasNextPage: Boolean!
      endCursor: String
    }
    \`\`\`
    
    ### **❌ Mistake 3: No Query Depth Limiting**
    
    \`\`\`javascript
    // Malicious query (infinite loop)
    query {
      user (id: "1") {
        friends {
          friends {
            friends {
              friends {
                friends {
                  # ... ad infinitum
                }
              }
            }
          }
        }
      }
    }
    
    // Solution: Limit query depth
    const depthLimit = require('graphql-depth-limit');
    
    const server = new ApolloServer({
      typeDefs,
      resolvers,
      validationRules: [depthLimit(5)] // Max depth 5
    });
    \`\`\`
    
    ### **❌ Mistake 4: No Query Cost Analysis**
    
    \`\`\`javascript
    // Expensive query
    query {
      users (first: 1000) {
        posts (first: 1000) {
          comments (first: 1000) {
            # 1 billion operations!
          }
        }
      }
    }
    
    // Solution: Query cost analysis
    const { createComplexityLimitRule } = require('graphql-validation-complexity');
    
    const server = new ApolloServer({
      validationRules: [
        createComplexityLimitRule(1000) // Max complexity 1000
      ]
    });
    \`\`\`
    
    ### **❌ Mistake 5: Exposing Internal Implementation**
    
    \`\`\`javascript
    // Bad: Database structure leaked to API
    type User {
      id: ID!
      user_name: String!  # snake_case from database
      created_at: String! # database column name
    }
    
    // Good: API-friendly names
    type User {
      id: ID!
      name: String!      # camelCase
      createdAt: DateTime! # semantic name + custom scalar
    }
    \`\`\`
    
    ---
    
    ## Real-World Example: Social Media API
    
    **Schema**:
    \`\`\`graphql
    type User {
      id: ID!
      username: String!
      email: String!
      posts (first: Int, after: String): PostConnection!
      followers: [User!]!
      following: [User!]!
      followerCount: Int!
      followingCount: Int!
    }
    
    type Post {
      id: ID!
      content: String!
      imageUrl: String
      author: User!
      likes: [Like!]!
      comments (first: Int): [Comment!]!
      likeCount: Int!
      commentCount: Int!
      createdAt: DateTime!
    }
    
    type Comment {
      id: ID!
      text: String!
      author: User!
      post: Post!
      createdAt: DateTime!
    }
    
    type Like {
      user: User!
      post: Post!
      createdAt: DateTime!
    }
    
    type Query {
      me: User
      user (username: String!): User
      post (id: ID!): Post
      feed (first: Int, after: String): PostConnection!
    }
    
    type Mutation {
      createPost (content: String!, imageUrl: String): Post!
      likePost (postId: ID!): Post!
      unlikePost (postId: ID!): Post!
      addComment (postId: ID!, text: String!): Comment!
      followUser (userId: ID!): User!
      unfollowUser (userId: ID!): User!
    }
    
    type Subscription {
      newPost (userId: ID!): Post!
      newComment (postId: ID!): Comment!
    }
    \`\`\`
    
    **Query Examples**:
    
    \`\`\`graphql
    # Get user feed
    query GetFeed {
      feed (first: 20) {
        edges {
          node {
            id
            content
            imageUrl
            author {
              username
            }
            likeCount
            commentCount
            comments (first: 3) {
              text
              author {
                username
              }
            }
          }
          cursor
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    \`\`\`
    
    **Benefits**:
    - Single request gets posts, authors, comments, like counts
    - With REST: would need \`/feed\`, \`/users/:id\`, \`/posts/:id/comments\`, \`/posts/:id/likes\`
    - Mobile app saves bandwidth
    - Frontend can iterate without backend changes
    
    ---
    
    ## Key Takeaways
    
    1. **GraphQL allows clients to specify exactly what data they need** in a single request
    2. **Solves over-fetching and under-fetching** problems of REST APIs
    3. **N+1 problem is the biggest pitfall** - always use DataLoader for batching
    4. **Caching is complex** - requires persisted queries or response caching
    5. **Strong typing with schema** enables great developer experience
    6. **Subscriptions enable real-time updates** via WebSocket
    7. **Query depth limiting and cost analysis** prevent malicious queries
    8. **Best for mobile apps and complex UIs** with many relationships
    9. **Not a replacement for REST** - choose based on use case
    10. **Learning curve is real** - team must understand resolvers, DataLoader, caching`,
};
