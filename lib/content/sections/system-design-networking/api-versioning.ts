/**
 * API Versioning Section
 */

export const apiversioningSection = {
  id: 'api-versioning',
  title: 'API Versioning',
  content: `API versioning is critical for maintaining backward compatibility while evolving APIs. This section covers versioning strategies, best practices, and migration patterns for distributed systems.
    
    ## Why API Versioning?
    
    **Problem**: Breaking changes break clients
    
    \`\`\`
    // Version 1
    { "name": "John Doe" }
    
    // Version 2 (BREAKING CHANGE!)
    { "firstName": "John", "lastName": "Doe" }
    
    // Existing clients break!
    const name = data.name; // undefined!
    \`\`\`
    
    **Solution**: Version your API
    
    \`\`\`
    GET /api/v1/users → { "name": "John Doe" }
    GET /api/v2/users → { "firstName": "John", "lastName": "Doe" }
    \`\`\`
    
    ---
    
    ## Versioning Strategies
    
    ### **1. URL Path Versioning** ⭐ (Most Common)
    
    **Format**: \`/api/v{version}/resource\`
    
    \`\`\`
    GET /api/v1/users
    GET /api/v2/users
    GET /api/v3/users
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    // Express.js
    const v1Router = express.Router();
    const v2Router = express.Router();
    
    // V1 routes
    v1Router.get('/users', (req, res) => {
      res.json({
        users: users.map(u => ({ name: u.fullName }))
      });
    });
    
    // V2 routes
    v2Router.get('/users', (req, res) => {
      res.json({
        users: users.map(u => ({
          firstName: u.firstName,
          lastName: u.lastName
        }))
      });
    });
    
    app.use('/api/v1', v1Router);
    app.use('/api/v2', v2Router);
    \`\`\`
    
    **Pros**:
    - Simple and explicit
    - Easy to route and cache
    - Clear deprecation path
    - Works with any client
    
    **Cons**:
    - URL changes (breaks bookmarks)
    - Multiple codebases to maintain
    
    **Best For**: Public APIs, major version changes
    
    ---
    
    ### **2. Header Versioning**
    
    **Format**: Custom header like \`API-Version\` or \`Accept\` header
    
    \`\`\`
    GET /api/users
    API-Version: 1
    
    GET /api/users
    API-Version: 2
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.use((req, res, next) => {
      const version = req.headers['api-version'] || '1';
      req.apiVersion = parseInt(version);
      next();
    });
    
    app.get('/api/users', (req, res) => {
      if (req.apiVersion === 1) {
        return res.json({
          users: users.map(u => ({ name: u.fullName }))
        });
      }
      
      if (req.apiVersion === 2) {
        return res.json({
          users: users.map(u => ({
            firstName: u.firstName,
            lastName: u.lastName
          }))
        });
      }
      
      res.status(400).json({ error: 'Unsupported API version' });
    });
    \`\`\`
    
    **Pros**:
    - Clean URLs (no version in path)
    - Single endpoint
    - Flexible versioning
    
    **Cons**:
    - Not visible in URL (harder to test)
    - Caching more complex (must include header in cache key)
    - Harder for API consumers to discover
    
    **Best For**: Internal APIs, minor version changes
    
    ---
    
    ### **3. Content Negotiation** (Accept Header)
    
    **Format**: \`Accept: application/vnd.company.v2+json\`
    
    \`\`\`
    GET /api/users
    Accept: application/vnd.myapi.v1+json
    
    GET /api/users
    Accept: application/vnd.myapi.v2+json
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.get('/api/users', (req, res) => {
      const accept = req.headers['accept'] || '';
      
      if (accept.includes('vnd.myapi.v1+json')) {
        return res
          .type('application/vnd.myapi.v1+json')
          .json({ users: users.map(u => ({ name: u.fullName })) });
      }
      
      if (accept.includes('vnd.myapi.v2+json')) {
        return res
          .type('application/vnd.myapi.v2+json')
          .json({
            users: users.map(u => ({
              firstName: u.firstName,
              lastName: u.lastName
            }))
          });
      }
      
      // Default to latest version
      return res.json({
        users: users.map(u => ({
          firstName: u.firstName,
          lastName: u.lastName
        }))
      });
    });
    \`\`\`
    
    **Pros**:
    - RESTful (proper use of HTTP)
    - Clean URLs
    
    **Cons**:
    - Complex for clients
    - Hard to test (curl requires correct headers)
    - Caching complex
    
    **Best For**: Strict RESTful APIs
    
    ---
    
    ### **4. Query Parameter Versioning**
    
    **Format**: \`/api/users?version=2\`
    
    \`\`\`
    GET /api/users?version=1
    GET /api/users?version=2
    \`\`\`
    
    **Implementation**:
    \`\`\`javascript
    app.get('/api/users', (req, res) => {
      const version = parseInt(req.query.version) || 1;
      
      if (version === 1) {
        return res.json({
          users: users.map(u => ({ name: u.fullName }))
        });
      }
      
      if (version === 2) {
        return res.json({
          users: users.map(u => ({
            firstName: u.firstName,
            lastName: u.lastName
          }))
        });
      }
      
      res.status(400).json({ error: 'Unsupported version' });
    });
    \`\`\`
    
    **Pros**:
    - Simple to implement
    - Easy to test
    - Optional (can default to latest)
    
    **Cons**:
    - Not RESTful (version is not a resource property)
    - Query params meant for filtering, not versioning
    - Pollutes query string
    
    **Best For**: Internal tools, quick prototypes
    
    ---
    
    ## Semantic Versioning for APIs
    
    **Format**: MAJOR.MINOR.PATCH (e.g., v2.1.0)
    
    - **MAJOR**: Breaking changes
    - **MINOR**: New features (backward compatible)
    - **PATCH**: Bug fixes (backward compatible)
    
    **Examples**:
    
    \`\`\`
    v1.0.0 → v1.0.1: Bug fix (backward compatible)
    v1.0.0 → v1.1.0: Added new endpoint (backward compatible)
    v1.0.0 → v2.0.0: Changed response format (BREAKING)
    \`\`\`
    
    **In Practice**:
    \`\`\`
    Only major version in URL: /api/v2/users
    Full version in response header: API-Version: 2.1.0
    \`\`\`
    
    ---
    
    ## Deprecation Strategy
    
    ### **1. Announce Deprecation**
    
    \`\`\`javascript
    app.use('/api/v1', (req, res, next) => {
      res.setHeader('Deprecation', 'true');
      res.setHeader('Sunset', 'Sat, 31 Dec 2024 23:59:59 GMT');
      res.setHeader('Link', '<https://api.example.com/docs/migration>; rel="alternate"');
      next();
    });
    \`\`\`
    
    ### **2. Monitor Usage**
    
    \`\`\`javascript
    app.use('/api/v1', (req, res, next) => {
      logger.warn('Deprecated API used', {
        endpoint: req.path,
        client: req.headers['user-agent'],
        ip: req.ip
      });
      
      metrics.deprecatedApiCalls.inc({
        version: 'v1',
        endpoint: req.path
      });
      
      next();
    });
    \`\`\`
    
    ### **3. Gradual Shutdown**
    
    **Phase 1: Soft Deprecation** (3 months)
    - Add deprecation headers
    - Email clients
    - Monitor usage
    
    **Phase 2: Hard Deprecation** (1 month)
    - Return 410 Gone for new clients
    - Allow existing clients (via whitelist)
    
    \`\`\`javascript
    const allowedClients = new Set(['client-a', 'client-b']);
    
    app.use('/api/v1', (req, res, next) => {
      const clientId = req.headers['x-client-id'];
      
      if (!allowedClients.has(clientId)) {
        return res.status(410).json({
          error: 'API version 1 is no longer supported',
          message: 'Please upgrade to v2: https://api.example.com/docs/migration'
        });
      }
      
      next();
    });
    \`\`\`
    
    **Phase 3: Complete Shutdown** (after 4 months)
    - Return 410 Gone for all clients
    - Remove v1 code
    
    ---
    
    ## Backward Compatibility Patterns
    
    ### **1. Additive Changes** (Non-Breaking)
    
    ✅ **Safe to add**:
    - New endpoints
    - New optional fields
    - New query parameters (optional)
    - New HTTP headers
    
    \`\`\`javascript
    // V1 response
    {
      "id": 1,
      "name": "John Doe"
    }
    
    // V1.1 response (backward compatible)
    {
      "id": 1,
      "name": "John Doe",
      "email": "john@example.com"  // NEW field (clients ignore if unknown)
    }
    \`\`\`
    
    ### **2. Field Transformation** (Breaking)
    
    ❌ **Breaking changes**:
    - Renaming fields
    - Changing field types
    - Removing fields
    - Changing error format
    
    **Solution: Maintain both versions**:
    \`\`\`javascript
    // Shared data layer
    function getUser(id) {
      return {
        id: id,
        firstName: 'John',
        lastName: 'Doe',
        fullName: 'John Doe', // Computed for v1
        age: 30
      };
    }
    
    // V1 endpoint
    app.get('/api/v1/users/:id', (req, res) => {
      const user = getUser(req.params.id);
      res.json({
        id: user.id,
        name: user.fullName // V1 format
      });
    });
    
    // V2 endpoint
    app.get('/api/v2/users/:id', (req, res) => {
      const user = getUser(req.params.id);
      res.json({
        id: user.id,
        firstName: user.firstName, // V2 format
        lastName: user.lastName
      });
    });
    \`\`\`
    
    ### **3. Adapter Pattern**
    
    \`\`\`javascript
    // Data model (internal)
    class User {
      constructor(id, firstName, lastName) {
        this.id = id;
        this.firstName = firstName;
        this.lastName = lastName;
      }
    }
    
    // V1 Adapter
    class UserV1Adapter {
      static toResponse(user) {
        return {
          id: user.id,
          name: \`\${user.firstName} \${user.lastName}\`
        };
      }
      
      static fromRequest(data) {
        const [firstName, ...lastNameParts] = data.name.split(' ');
        return new User(data.id, firstName, lastNameParts.join(' '));
      }
    }
    
    // V2 Adapter
    class UserV2Adapter {
      static toResponse(user) {
        return {
          id: user.id,
          firstName: user.firstName,
          lastName: user.lastName
        };
      }
      
      static fromRequest(data) {
        return new User(data.id, data.firstName, data.lastName);
      }
    }
    
    // Use in routes
    app.get('/api/v1/users/:id', async (req, res) => {
      const user = await userService.getUser(req.params.id);
      res.json(UserV1Adapter.toResponse(user));
    });
    
    app.get('/api/v2/users/:id', async (req, res) => {
      const user = await userService.getUser(req.params.id);
      res.json(UserV2Adapter.toResponse(user));
    });
    \`\`\`
    
    ---
    
    ## GraphQL Versioning
    
    **GraphQL philosophy**: No versioning, only schema evolution
    
    **Approach**: Deprecate fields, don't remove them
    
    \`\`\`graphql
    type User {
      id: ID!
      name: String! @deprecated(reason: "Use firstName and lastName instead")
      firstName: String!
      lastName: String!
    }
    \`\`\`
    
    **Clients can transition gradually**:
    \`\`\`graphql
    # Old clients
    query {
      user(id: "123") {
        name  # Still works
      }
    }
    
    # New clients
    query {
      user(id: "123") {
        firstName
        lastName
      }
    }
    \`\`\`
    
    ---
    
    ## Key Takeaways
    
    1. **URL path versioning** most common and recommended (/api/v2/users)
    2. **Semantic versioning**: Only major version in URL, full version in headers
    3. **Deprecation strategy**: Announce → Monitor → Whitelist → Shutdown (4+ months)
    4. **Additive changes are safe**: New fields, new endpoints (non-breaking)
    5. **Breaking changes**: Rename fields, change types, remove fields → new version
    6. **Adapter pattern** maintains single codebase while supporting multiple versions
    7. **GraphQL**: No versioning, deprecate fields instead
    8. **Headers**: Use Deprecation, Sunset, Link headers for deprecation
    9. **Monitor deprecated API usage** to identify clients needing migration
    10. **Keep old versions for 6-12 months** minimum before shutdown`,
};
