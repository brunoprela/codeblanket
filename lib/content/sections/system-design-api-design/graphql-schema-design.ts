/**
 * GraphQL Schema Design Section
 */

export const graphqlschemadesignSection = {
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
};
