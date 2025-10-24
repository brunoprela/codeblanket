/**
 * Quiz questions for API Versioning section
 */

export const apiversioningQuiz = [
  {
    id: 'api-versioning-migration',
    question:
      'You need to make a breaking change to your public API used by 10,000 clients: changing the user response from {"name": "John Doe"} to {"firstName": "John", "lastName": "Doe"}. Design a complete migration strategy including: versioning approach, deprecation timeline, client communication, monitoring, handling legacy clients, and ensuring zero downtime. Provide specific implementation details.',
    sampleAnswer: `**Complete API Migration Strategy**
    
    **1. Versioning Approach: URL Path Versioning**
    
    **Why**: Public API with 10K clients needs clarity and simplicity
    
    \`\`\`
    Current: GET /api/v1/users/:id → {"id": 1, "name": "John Doe"}
    New:     GET /api/v2/users/:id → {"id": 1, "firstName": "John", "lastName": "Doe"}
    \`\`\`
    
    **2. Timeline** (6-month migration)
    
    | **Phase** | **Duration** | **Actions** |
    |-----------|--------------|-------------|
    | **Announcement** | Week 1-2 | Announce v2, deprecate v1, publish migration guide |
    | **Soft Deprecation** | Month 1-4 | V1 works fully, add deprecation headers, monitor usage |
    | **Hard Deprecation** | Month 5 | V1 rate-limited, email heavy users, whitelist critical clients |
    | **Shutdown** | Month 6 | V1 returns 410 Gone, remove code after 30 days |
    
    **3. Implementation**
    
    **Shared Data Layer**:
    \`\`\`javascript
    // models/user.js
    class User {
      constructor(data) {
        this.id = data.id;
        this.firstName = data.first_name;
        this.lastName = data.last_name;
      }
      
      // Computed property for v1
      get fullName() {
        return \`\${this.firstName} \${this.lastName}\`;
      }
      
      // V1 format
      toV1() {
        return {
          id: this.id,
          name: this.fullName
        };
      }
      
      // V2 format
      toV2() {
        return {
          id: this.id,
          firstName: this.firstName,
          lastName: this.lastName
        };
      }
    }
    
    module.exports = User;
    \`\`\`
    
    **V1 Endpoint** (deprecated):
    \`\`\`javascript
    // routes/v1/users.js
    const express = require('express');
    const router = express.Router();
    const User = require('../../models/user');
    
    // Deprecation middleware
    router.use((req, res, next) => {
      // Add deprecation headers
      res.setHeader('Deprecation', 'true');
      res.setHeader('Sunset', 'Mon, 30 Jun 2024 23:59:59 GMT');
      res.setHeader('Link', '<https://api.example.com/docs/v2-migration>; rel="alternate"');
      res.setHeader('X-API-Version', '1.0.0');
      
      // Log usage
      logger.warn('V1 API usage', {
        endpoint: req.originalUrl,
        client: req.headers['x-client-id',] || 'unknown',
        userAgent: req.headers['user-agent',],
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      // Track metrics
      metrics.apiVersionUsage.inc({
        version: 'v1',
        endpoint: req.path,
        client: req.headers['x-client-id',] || 'unknown'
      });
      
      next();
    });
    
    router.get('/users/:id', async (req, res) => {
      try {
        const user = await User.findById(req.params.id);
        if (!user) {
          return res.status(404).json({ error: 'User not found' });
        }
        
        res.json(user.toV1());
      } catch (error) {
        logger.error('V1 API error', { error, userId: req.params.id });
        res.status(500).json({ error: 'Internal server error' });
      }
    });
    
    module.exports = router;
    \`\`\`
    
    **V2 Endpoint** (new):
    \`\`\`javascript
    // routes/v2/users.js
    const express = require('express');
    const router = express.Router();
    const User = require('../../models/user');
    
    router.use((req, res, next) => {
      res.setHeader('X-API-Version', '2.0.0');
      next();
    });
    
    router.get('/users/:id', async (req, res) => {
      try {
        const user = await User.findById(req.params.id);
        if (!user) {
          return res.status(404).json({ error: 'User not found' });
        }
        
        res.json(user.toV2());
      } catch (error) {
        logger.error('V2 API error', { error, userId: req.params.id });
        res.status(500).json({ error: 'Internal server error' });
      }
    });
    
    module.exports = router;
    \`\`\`
    
    **Mount Routes**:
    \`\`\`javascript
    // app.js
    const v1Routes = require('./routes/v1/users');
    const v2Routes = require('./routes/v2/users');
    
    app.use('/api/v1', v1Routes);
    app.use('/api/v2', v2Routes);
    \`\`\`
    
    **4. Client Communication**
    
    **Week 1: Announcement Email**:
    \`\`\`
    Subject: [Action Required] API v2 Released - v1 Deprecated
    
    Dear API Consumer,
    
    We've released API v2 with improved user data structure:
    
    Breaking Change:
    - V1: {"name": "John Doe"}
    - V2: {"firstName": "John", "lastName": "Doe"}
    
    Timeline:
    - Now: V2 available, V1 fully functional
    - Month 4: V1 soft deprecated (still works)
    - Month 5: V1 rate-limited (500 req/day)
    - Month 6: V1 shutdown (returns 410 Gone)
    
    Migration Guide: https://api.example.com/docs/v2-migration
    
    Action Required:
    1. Test your integration with V2
    2. Migrate by Month 5 to avoid rate limits
    3. Contact support if you need more time
    
    Best regards,
    API Team
    \`\`\`
    
    **Month 3: Reminder Email** (to clients still on v1):
    \`\`\`
    Subject: [Urgent] API v1 Deprecation in 3 Months
    
    Dear API Consumer,
    
    Our logs show you're still using API v1:
    - Your usage: 10,000 requests/day
    - Endpoints used: GET /api/v1/users/:id
    
    V1 will be rate-limited in 2 months (500 req/day).
    Please migrate to v2 immediately.
    
    Need help? Reply to this email or visit our support page.
    \`\`\`
    
    **5. Monitoring Dashboard**
    
    **Grafana Dashboard**:
    \`\`\`yaml
    panels:
      - title: "API Version Usage"
        query: sum by (version) (rate(api_version_usage[5m]))
        type: time-series
        
      - title: "Top V1 Clients"
        query: topk(10, sum by (client) (rate(api_version_usage{version="v1"}[24h])))
        type: table
        
      - title: "V1 Usage by Endpoint"
        query: sum by (endpoint) (rate(api_version_usage{version="v1"}[1h]))
        type: bar-chart
        
      - title: "Migration Progress"
        query: |
          (sum(rate(api_version_usage{version="v2"}[1h])) / 
           (sum(rate(api_version_usage{version="v1"}[1h])) + 
            sum(rate(api_version_usage{version="v2"}[1h])))) * 100
        type: gauge
    \`\`\`
    
    **Alerts**:
    \`\`\`yaml
    - alert: V1UsageStillHigh
      expr: sum(rate(api_version_usage{version="v1"}[1h])) > 100
      for: 1h
      annotations:
        summary: "V1 API still receiving >100 req/sec after Month 4"
    
    - alert: NewV1Client
      expr: increase(api_version_usage{version="v1"}[5m]) > 0 and Month > 5
      annotations:
        summary: "New client using deprecated V1 API"
    \`\`\`
    
    **6. Hard Deprecation** (Month 5)
    
    **Rate Limiting for V1**:
    \`\`\`javascript
    // V1 rate limit: 500 requests/day
    router.use(async (req, res, next) => {
      const clientId = req.headers['x-client-id',] || req.ip;
      const key = \`v1_rate_limit:\${clientId}\`;
      
      const requests = await redis.incr(key);
      if (requests === 1) {
        await redis.expire(key, 86400); // 24 hours
      }
      
      if (requests > 500) {
        return res.status(429).json({
          error: 'V1 API rate limit exceeded',
          limit: 500,
          window: '24 hours',
          message: 'Please migrate to V2: https://api.example.com/docs/v2-migration',
          contact: 'support@example.com for extension'
        });
      }
      
      res.setHeader('X-RateLimit-Limit', '500');
      res.setHeader('X-RateLimit-Remaining', String(500 - requests));
      
      next();
    });
    \`\`\`
    
    **Whitelist Critical Clients**:
    \`\`\`javascript
    const v1Whitelist = new Set([
      'client-a', // Enterprise client, extension granted
      'client-b'  // Government client, slow approval process
    ]);
    
    router.use((req, res, next) => {
      const clientId = req.headers['x-client-id',];
      
      if (v1Whitelist.has(clientId)) {
        // Skip rate limiting
        return next();
      }
      
      // Apply rate limiting
      rateLimitV1(req, res, next);
    });
    \`\`\`
    
    **7. Complete Shutdown** (Month 6)
    
    **Return 410 Gone**:
    \`\`\`javascript
    router.use((req, res) => {
      res.status(410).json({
        error: 'API v1 has been permanently removed',
        message: 'Please use API v2',
        documentation: 'https://api.example.com/docs/v2',
        migrationGuide: 'https://api.example.com/docs/v2-migration',
        support: 'support@example.com'
      });
    });
    \`\`\`
    
    **Remove Code** (30 days after shutdown):
    \`\`\`bash
    # After confirming zero V1 traffic for 30 days
    git rm -r routes/v1/
    git commit -m "Remove deprecated API v1 code"
    \`\`\`
    
    **8. Testing Strategy**
    
    **Integration Tests**:
    \`\`\`javascript
    describe('API Versioning', () => {
      describe('V1 (deprecated)', () => {
        it('should return v1 format', async () => {
          const res = await request(app).get('/api/v1/users/123');
          
          expect(res.body).toEqual({
            id: 123,
            name: 'John Doe'
          });
          
          expect(res.headers['deprecation',]).toBe('true');
          expect(res.headers['sunset',]).toBeDefined();
        });
      });
      
      describe('V2', () => {
        it('should return v2 format', async () => {
          const res = await request(app).get('/api/v2/users/123');
          
          expect(res.body).toEqual({
            id: 123,
            firstName: 'John',
            lastName: 'Doe'
          });
          
          expect(res.headers['x-api-version',]).toBe('2.0.0');
        });
      });
    });
    \`\`\`
    
    **Key Takeaways**:
    
    1. **6-month timeline** gives clients adequate time to migrate
    2. **URL path versioning** (/api/v1, /api/v2) is clearest for public APIs
    3. **Shared data layer** with toV1()/toV2() methods maintains single codebase
    4. **Deprecation headers** (Sunset, Link) inform clients programmatically
    5. **Monitor usage** by client to identify who needs to migrate
    6. **Email communication** at announcement, 3 months, 1 month before shutdown
    7. **Rate limiting** in Month 5 encourages migration (500 req/day)
    8. **Whitelist** for critical clients needing extensions
    9. **410 Gone** after Month 6 signals permanent removal
    10. **Remove code** 30 days after confirming zero traffic`,
    keyPoints: [
      '6-month migration timeline: Announce → Monitor → Rate limit → Whitelist → Shutdown',
      'URL path versioning (/api/v1, /api/v2) most common for public APIs',
      'Share data layer with toV1()/toV2() transformation methods to maintain single codebase',
      'Use Deprecation and Sunset HTTP headers to programmatically inform clients',
      'Monitor API usage by client/version to identify who needs to migrate',
      'Rate limit deprecated version (Month 5) to encourage migration, whitelist critical clients',
    ],
  },
  {
    id: 'api-versioning-disc-2',
    question:
      'Compare URL path versioning (/api/v2/users), header versioning (Accept: application/vnd.myapi.v2+json), and query parameter versioning (/api/users?version=2) for REST APIs. For each approach, discuss pros/cons, caching implications, and when you would choose each. Provide implementation examples.',
    sampleAnswer: `**Comprehensive Comparison of API Versioning Strategies**

---

## **1. URL Path Versioning** ⭐ (Most Common)

**Format**: \`/api/v{version}/resource\`

**Example**:
\`\`\`
GET /api/v1/users/123
GET /api/v2/users/123
\`\`\`

**Implementation**:

\`\`\`typescript
// Express routing
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// v1Router
const v1Router = express.Router();
v1Router.get('/users/:id', (req, res) => {
  const user = await db.users.findById(req.params.id);
  
  // V1 format: single "name" field
  res.json({
    id: user.id,
    name: \`\${user.firstName} \${user.lastName}\`,
    email: user.email
  });
});

// v2Router
const v2Router = express.Router();
v2Router.get('/users/:id', (req, res) => {
  const user = await db.users.findById(req.params.id);
  
  // V2 format: separate firstName/lastName
  res.json({
    id: user.id,
    firstName: user.firstName,
    lastName: user.lastName,
    email: user.email
  });
});
\`\`\`

### **Pros**:

✅ **Clear and explicit** - version immediately visible in URL
✅ **Easy to test** - can use curl, Postman easily
✅ **Cacheable** - different URLs = different cache entries
✅ **Browser friendly** - can bookmark different versions
✅ **Simple routing** - standard HTTP routing works
✅ **API gateway compatible** - easy to route by path
✅ **No special client code** - just change URL

### **Cons**:

❌ **URL pollution** - version in every endpoint
❌ **Breaking changes only** - can't use for minor changes
❌ **More boilerplate** - need separate routers per version

### **Caching**:

**Excellent** - Each version is a unique URL

\`\`\`
Cache-Control: public, max-age=3600
/api/v1/users/123  →  Cached separately
/api/v2/users/123  →  Cached separately
\`\`\`

CDNs and browser caches work perfectly.

### **When to Use**:

- **Public APIs** (most common choice)
- **RESTful services**
- **Long version lifetimes** (v1, v2, v3...)
- **Breaking changes**

---

## **2. Header Versioning** (Accept Header)

**Format**: \`Accept: application/vnd.myapi.v{version}+json\`

**Example**:
\`\`\`
GET /api/users/123
Accept: application/vnd.myapi.v1+json

GET /api/users/123
Accept: application/vnd.myapi.v2+json
\`\`\`

**Implementation**:

\`\`\`typescript
app.get('/api/users/:id', (req, res) => {
  const acceptHeader = req.get('Accept') || 'application/vnd.myapi.v1+json';
  const versionMatch = acceptHeader.match(/v(\\d+)/);
  const version = versionMatch ? parseInt(versionMatch[1]) : 1;
  
  const user = await db.users.findById(req.params.id);
  
  let response;
  switch (version) {
    case 1:
      response = {
        id: user.id,
        name: \`\${user.firstName} \${user.lastName}\`,
        email: user.email
      };
      break;
    
    case 2:
      response = {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email
      };
      break;
    
    default:
      return res.status(400).json({ error: 'Unsupported API version' });
  }
  
  res.setHeader('Content-Type', \`application/vnd.myapi.v\${version}+json\`);
  res.json(response);
});
\`\`\`

### **Pros**:

✅ **RESTful** - URLs represent resources, not versions
✅ **Clean URLs** - no version pollution in path
✅ **Content negotiation** - follows HTTP standards
✅ **Flexible** - can version by media type

### **Cons**:

❌ **Hard to test** - can't use browser directly
❌ **Caching complexity** - same URL, different content
❌ **Not obvious** - version hidden in headers
❌ **Client complexity** - must set headers correctly
❌ **Debugging harder** - need to inspect headers
❌ **CDN complexity** - requires Vary: Accept header

### **Caching**:

**Complex** - Requires \`Vary: Accept\` header

\`\`\`typescript
res.setHeader('Cache-Control', 'public, max-age=3600');
res.setHeader('Vary', 'Accept');
// CDN must cache separately based on Accept header
\`\`\`

**Problem**: Not all CDNs handle \`Vary\` correctly.

**CloudFront Fix**:
\`\`\`javascript
// Whitelist Accept header
const cloudFrontConfig = {
  headers: {
    whitelist: ['Accept',]
  }
};
\`\`\`

### **When to Use**:

- **Internal APIs** (between backend services)
- **Following strict REST principles**
- **Academic/research APIs**
- **Hypermedia APIs** (HATEOAS)

---

## **3. Query Parameter Versioning**

**Format**: \`/api/resource?version=2\` or \`/api/resource?v=2\`

**Example**:
\`\`\`
GET /api/users/123?version=1
GET /api/users/123?version=2
\`\`\`

**Implementation**:

\`\`\`typescript
app.get('/api/users/:id', (req, res) => {
  const version = parseInt(req.query.version || '1');
  const user = await db.users.findById(req.params.id);
  
  let response;
  if (version === 1) {
    response = {
      id: user.id,
      name: \`\${user.firstName} \${user.lastName}\`,
      email: user.email
    };
  } else if (version === 2) {
    response = {
      id: user.id,
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email
    };
  } else {
    return res.status(400).json({ error: 'Unsupported version' });
  }
  
  res.json(response);
});
\`\`\`

### **Pros**:

✅ **Easy to test** - add ?version=2 to URL
✅ **Optional versioning** - default to latest if omitted
✅ **Browser friendly** - can bookmark
✅ **Simple client** - just change query param

### **Cons**:

❌ **Not RESTful** - query params shouldn't change resource representation
❌ **URL pollution** - messy with many params
❌ **Caching ambiguity** - \`/users/123\` and \`/users/123?version=1\` same resource?
❌ **Optional feels wrong** - version should be mandatory
❌ **Analytics harder** - query params often stripped
❌ **Routing complexity** - can't route by query param easily

### **Caching**:

**Unclear** - Do these cache separately?

\`\`\`
/api/users/123           (default v1?)
/api/users/123?version=1
/api/users/123?version=2
\`\`\`

**Solution**: Include version in Vary header or force version parameter

\`\`\`typescript
res.setHeader('Cache-Control', 'public, max-age=3600');
res.setHeader('Vary', 'version');  // Non-standard
\`\`\`

### **When to Use**:

- **Internal tools** where RESTfulness doesn't matter
- **Gradual rollouts** (e.g., A/B testing)
- **Optional features** (not core versioning)

---

## **4. Custom Header Versioning**

**Format**: \`X-API-Version: 2\` or \`API-Version: 2\`

**Example**:
\`\`\`
GET /api/users/123
X-API-Version: 2
\`\`\`

Similar to Accept header versioning but simpler.

### **Pros**:

✅ **Clean URLs**
✅ **Explicit versioning**
✅ **Simple header**

### **Cons**:

❌ **Not standard** (Accept is HTTP standard)
❌ **Caching complexity** (requires \`Vary: X-API-Version\`)
❌ **Hard to test**
❌ **Client complexity**

---

## **Comparison Table**

| Aspect | URL Path | Header (Accept) | Query Param | Custom Header |
|--------|----------|-----------------|-------------|---------------|
| **RESTful** | Debatable | ✅ Yes | ❌ No | Debatable |
| **Cacheable** | ✅ Easy | ⚠️ Complex | ⚠️ Unclear | ⚠️ Complex |
| **Testability** | ✅ Easy | ❌ Hard | ✅ Easy | ❌ Hard |
| **Visibility** | ✅ Obvious | ❌ Hidden | ✅ Visible | ❌ Hidden |
| **Client Simple** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **CDN Support** | ✅ Perfect | ⚠️ Needs Vary | ⚠️ Unclear | ⚠️ Needs Vary |
| **API Gateway** | ✅ Easy routing | ⚠️ Harder | ⚠️ Harder | ⚠️ Harder |
| **URL Pollution** | ❌ Yes | ✅ No | ❌ Yes | ✅ No |

---

## **Real-World Examples**

**URL Path** (Most Popular):
- **Stripe**: \`https://api.stripe.com/v1/charges\`
- **Twitter**: \`https://api.twitter.com/2/tweets\`
- **GitHub**: \`https://api.github.com/repos\` (path-based)
- **Twilio**: \`https://api.twilio.com/2010-04-01/Accounts\`

**Header Versioning**:
- **Azure**: \`api-version\` in query or header
- **Some GitHub APIs**: \`Accept: application/vnd.github.v3+json\`

**Query Parameter**:
- **Google Maps**: \`?v=3.exp\`
- **Azure (alternative)**: \`?api-version=2021-04-01\`

---

## **My Recommendation**

**For Public REST APIs**: **URL Path Versioning** (/api/v2/)

**Reasons**:
1. **Simplicity wins** - easy for clients to implement
2. **Caching works** - CDNs, browsers work out-of-the-box
3. **Debugging easy** - logs show version immediately
4. **Industry standard** - Stripe, Twitter, GitHub all use it
5. **Backwards compatible** - old clients keep working

**For Internal APIs**: **Header Versioning** (Accept or Custom)

**Reasons**:
1. **Cleaner URLs** - better REST semantics
2. **Service mesh friendly** - headers easier to route
3. **No URL pollution** - single endpoint
4. **Flexible** - can version independently by resource

**Avoid**: **Query Parameters** for core versioning
- Use query params for optional features, not core versioning
- Caching and semantics are unclear

---

## **Implementation Best Practices**

**1. Version Number Format**:

\`\`\`
Good:  /api/v1/, /api/v2/ (simple integers)
Okay:  /api/v1.2/, /api/v2.0/ (semantic versioning)
Bad:   /api/2023-01-15/ (dates)
\`\`\`

**2. Default Version**:

\`\`\`typescript
// Always explicit, never default to latest
app.get('/api/users/:id', (req, res) => {
  return res.status(400).json({
    error: 'API version required',
    hint: 'Use /api/v2/users/:id'
  });
});
\`\`\`

**3. Version in Response**:

\`\`\`typescript
res.setHeader('X-API-Version', '2.0.0');
res.setHeader('X-API-Deprecated', 'false');
\`\`\`

**4. Documentation**:

\`\`\`markdown
# API Versioning

We use URL path versioning:
- Current: /api/v2/
- Deprecated: /api/v1/ (sunset: 2024-06-01)
- Legacy: /api/v0/ (removed)

Version format: /api/v{major}/
\`\`\`

**Key Takeaway**: **URL path versioning wins for public APIs** due to simplicity, caching, and industry adoption. Use header versioning for internal services where clean URLs and REST semantics matter more than ease of testing.`,
    keyPoints: [
      'URL path versioning (/api/v2/): Best for public APIs (easy testing, perfect caching, industry standard)',
      'Header versioning (Accept): RESTful and clean URLs, but complex caching and hard to test',
      'Query parameter: Easy to test but not RESTful, caching ambiguous, avoid for core versioning',
      'Caching: URL path works out-of-box; headers need Vary header (CDN complexity)',
      'For public APIs: Choose URL path (Stripe, GitHub, Twitter all use it)',
      'For internal APIs: Header versioning acceptable (cleaner URLs, service mesh friendly)',
    ],
  },
  {
    id: 'api-versioning-disc-3',
    question:
      'You need to make a breaking change to your API (changing response format) but 40% of clients are still on v1. Design a strategy to minimize disruption including: versioning approach, migration timeline, client tracking, backwards compatibility layers, and rollout plan. How would you handle clients that refuse to migrate?',
    sampleAnswer: `**Breaking Change Migration Strategy**

---

## **1. Situation Analysis**

**Breaking Change**: Response format modification

**V1 Response**:
\`\`\`json
{
  "user": {
    "id": 123,
    "name": "John Doe",
    "address": "123 Main St, City, 12345"
  }
}
\`\`\`

**V2 Response** (Breaking):
\`\`\`json
{
  "id": 123,
  "firstName": "John",
  "lastName": "Doe",
  "address": {
    "street": "123 Main St",
    "city": "City",
    "zipCode": "12345"
  }
}
\`\`\`

**Current State**:
- 60% clients on v1
- 40% clients on v2
- Need to sunset v1

---

## **2. Versioning Approach**

**Use URL Path Versioning**: \`/api/v1/\` and \`/api/v2/\`

\`\`\`typescript
// Shared data layer
class UserRepository {
  async getUser(id: number): Promise<User> {
    return db.users.findById(id);
  }
}

// V1 Controller
app.get('/api/v1/users/:id', async (req, res) => {
  const user = await userRepo.getUser(req.params.id);
  
  // Transform to V1 format
  res.json({
    user: {
      id: user.id,
      name: \`\${user.firstName} \${user.lastName}\`,
      address: \`\${user.address.street}, \${user.address.city}, \${user.address.zipCode}\`
    }
  });
});

// V2 Controller
app.get('/api/v2/users/:id', async (req, res) => {
  const user = await userRepo.getUser(req.params.id);
  
  // Native V2 format
  res.json({
    id: user.id,
    firstName: user.firstName,
    lastName: user.lastName,
    address: {
      street: user.address.street,
      city: user.address.city,
      zipCode: user.address.zipCode
    }
  });
});
\`\`\`

---

## **3. Client Tracking System**

**Middleware to Track API Version Usage**:

\`\`\`typescript
interface ClientUsage {
  clientId: string;
  version: 'v1' | 'v2';
  endpoint: string;
  lastSeen: Date;
  requestCount: number;
}

// Middleware
app.use((req, res, next) => {
  const apiKey = req.headers['x-api-key',] as string;
  const version = req.path.startsWith('/api/v1/') ? 'v1' : 
                  req.path.startsWith('/api/v2/') ? 'v2' : 'unknown';
  
  // Track in Redis
  const key = \`api_usage:\${apiKey}:\${version}:\${getToday()}\`;
  redis.incr(key);
  redis.expire(key, 90 * 86400); // 90 days
  
  // Update last seen
  redis.hset(\`client:\${apiKey}\`, {
    lastVersion: version,
    lastSeen: Date.now(),
    endpoint: req.path
  });
  
  next();
});

// Dashboard query
async function getV1Clients(): Promise<ClientInfo[]> {
  const pattern = 'api_usage:*:v1:*';
  const keys = await redis.keys(pattern);
  
  const clients = await Promise.all(
    keys.map(async (key) => {
      const [_, apiKey, version, date] = key.split(':');
      const count = await redis.get(key);
      const info = await redis.hgetall(\`client:\${apiKey}\`);
      
      return {
        apiKey,
        requestsToday: parseInt(count || '0'),
        lastSeen: new Date(parseInt(info.lastSeen)),
        email: info.email,
        company: info.company
      };
    })
  );
  
  return clients.filter(c => c.requestsToday > 0);
}
\`\`\`

---

## **4. Migration Timeline (12 Months)**

### **Month 0-1: Preparation**

**Actions**:
1. **Analyze impact**:
   - Identify all affected endpoints
   - List breaking changes
   - Estimate migration effort

2. **Build v2**:
   - Implement new endpoints
   - Comprehensive testing
   - Performance benchmarks

3. **Create migration guide**:
   \`\`\`markdown
   # V1 → V2 Migration Guide
   
   ## Breaking Changes
   
   ### Response Format
   Before (v1):
   \`\`\`json
   { "user": { "name": "John Doe" } }
   \`\`\`
   
   After (v2):
   \`\`\`json
   { "firstName": "John", "lastName": "Doe" }
   \`\`\`
   
   ### Code Changes
   \`\`\`javascript
   // V1
   const name = response.user.name;
   
   // V2
   const name = \`\${response.firstName} \${response.lastName}\`;
   \`\`\`
   \`\`\`

### **Month 1-2: Soft Launch**

**Actions**:
1. **Release v2 (beta)**:
   - Opt-in only
   - No deprecation warnings yet
   - Monitor closely

2. **Contact top clients**:
   - Email 20 largest clients
   - Offer migration support
   - Schedule calls

3. **Track adoption**:
   - Dashboard showing v1 vs v2 usage
   - Identify early adopters

### **Month 2-3: Announce Deprecation**

**Email to ALL clients**:

\`\`\`
Subject: API v1 Deprecation Notice - Action Required

Hi {{name}},

We're deprecating API v1 on {{sunset_date}} (10 months from now).

## What's Changing
- Response format modernized (see docs)
- New features only in v2
- V1 will stop working on {{sunset_date}}

## Action Required
1. Review migration guide: {{migration_url}}
2. Update to /api/v2/ endpoints
3. Test in sandbox: {{sandbox_url}}

## Timeline
- Now: V2 available
- Month 6: V1 rate limited to 500 req/day
- Month 10: V1 returns 410 Gone

Questions? Reply to this email.
\`\`\`

**Add headers to v1**:

\`\`\`typescript
app.use('/api/v1/*', (req, res, next) => {
  res.setHeader('Deprecation', 'true');
  res.setHeader('Sunset', 'Sat, 01 Jan 2025 00:00:00 GMT');
  res.setHeader('Link', '</api/v2/docs>; rel="successor-version"');
  next();
});
\`\`\`

### **Month 3-6: Active Migration Period**

**Actions**:
1. **Weekly emails** to clients still on v1
2. **Office hours** for migration support
3. **Incentives**:
   - Free month for migrating early
   - Priority support
   - Beta access to new features

**Track progress**:
\`\`\`typescript
async function getMigrationProgress() {
  const v1Count = await redis.get('active_clients:v1');
  const v2Count = await redis.get('active_clients:v2');
  
  return {
    v1: parseInt(v1Count || '0'),
    v2: parseInt(v2Count || '0'),
    percentMigrated: (v2Count / (v1Count + v2Count)) * 100
  };
}
\`\`\`

### **Month 6-8: Enforcement**

**Aggressive Rate Limiting on v1**:

\`\`\`typescript
app.use('/api/v1/*', async (req, res, next) => {
  const apiKey = req.headers['x-api-key',];
  
  // Rate limit v1 to 500 requests/day
  const dailyLimit = 500;
  const count = await redis.incr(\`rate_limit:v1:\${apiKey}:\${getToday()}\`);
  
  if (count === 1) {
    await redis.expire(\`rate_limit:v1:\${apiKey}:\${getToday()}\`, 86400);
  }
  
  if (count > dailyLimit) {
    return res.status(429).json({
      error: 'Rate limit exceeded for deprecated v1 API',
      message: 'Please migrate to v2',
      migrationGuide: 'https://docs.example.com/migration',
      v2Endpoint: req.path.replace('/v1/', '/v2/')
    });
  }
  
  // Warning headers
  res.setHeader('X-RateLimit-Remaining', Math.max(0, dailyLimit - count));
  res.setHeader('X-API-Deprecated', 'true');
  
  next();
});
\`\`\`

**Email warnings**:
\`\`\`
Subject: URGENT: API v1 Rate Limited - Migrate Now

You've hit the v1 rate limit (500 req/day).

Migrate to v2 for unlimited access:
{{migration_guide_url}}

V1 shuts down in 2 months.
\`\`\`

### **Month 8-10: Final Push**

**Identify holdouts**:
\`\`\`typescript
const v1Clients = await getV1Clients();

for (const client of v1Clients) {
  // Direct phone calls for top 10 clients
  if (client.requestsPerDay > 10000) {
    console.log(\`CALL: \${client.company} - \${client.phone}\`);
  }
  
  // Final warning emails
  await sendEmail({
    to: client.email,
    subject: 'FINAL NOTICE: API v1 Shutdown in 30 Days',
    body: \`Your API keys will stop working on \${sunsetDate}\`
  });
}
\`\`\`

**Create whitelist for critical clients**:
\`\`\`typescript
const WHITELISTED_CLIENTS = [
  'api-key-enterprise-client-1', // Needs 60-day extension
  'api-key-government-client'     // Procurement delays
];

app.use('/api/v1/*', (req, res, next) => {
  const apiKey = req.headers['x-api-key',];
  
  if (WHITELISTED_CLIENTS.includes(apiKey)) {
    // Allow but log
    logger.warn(\`Whitelisted v1 access: \${apiKey}\`);
    return next();
  }
  
  // Enforce rate limit for others
  // ... (rate limiting code)
});
\`\`\`

### **Month 10: Sunset v1**

**Return 410 Gone**:

\`\`\`typescript
app.use('/api/v1/*', (req, res) => {
  const apiKey = req.headers['x-api-key',];
  
  // Check whitelist
  if (WHITELISTED_CLIENTS.includes(apiKey)) {
    // Allow for 30 more days
    if (Date.now() < WHITELIST_EXPIRY) {
      return next();
    }
  }
  
  res.status(410).json({
    error: 'Gone',
    message: 'API v1 has been permanently removed',
    migrationGuide: 'https://docs.example.com/migration',
    v2Endpoint: req.path.replace('/v1/', '/v2/'),
    supportEmail: 'support@example.com'
  });
});
\`\`\`

### **Month 11-12: Cleanup**

1. **Monitor for zero traffic**:
   \`\`\`typescript
   const v1Traffic = await redis.get(\`traffic:v1:\${getToday()}\`);
   if (v1Traffic === '0') {
     console.log('Safe to remove v1 code');
   }
   \`\`\`

2. **Remove v1 code** after 30 days of zero traffic
3. **Update documentation**
4. **Archive migration resources**

---

## **5. Handling Clients Who Refuse to Migrate**

### **Enterprise Clients (High Value)**

**Option 1: Paid Extension**:
\`\`\`
Offer 90-day extension for $5000/month
- Covers maintenance cost
- Time-limited (no indefinite extensions)
- Strict deadline enforced
\`\`\`

**Option 2: Managed Migration**:
\`\`\`
We migrate their code for them ($15,000 fixed price)
- Assigned engineer
- Complete in 2 weeks
- Guaranteed compatibility
\`\`\`

### **Small Clients (Low Value)**

**Hard Cutoff**:
- No extensions
- 410 Gone after sunset date
- "We understand this is inconvenient, but v1 has security vulnerabilities"

**Offer Alternatives**:
- Competitor APIs
- Open-source alternatives
- Refund if within contract period

### **Legal/Contractual Obligations**

**If SLA mentions "API stability"**:

1. **Argue v2 is not a breaking change to service**:
   - Same functionality, different format
   - 12-month notice exceeds industry standard

2. **Offer free migration**:
   - Absorb cost to maintain relationship
   - Document for future contracts

3. **Force majeure clause**:
   - Security vulnerabilities make v1 unmaintainable
   - Reasonable notice provided

---

## **6. Backwards Compatibility Layer** (Last Resort)

If critical clients can't migrate, build adapter:

\`\`\`typescript
// Adapter transforms v2 responses to v1 format
class V1toV2Adapter {
  transformResponse(v2Response: V2User): V1User {
    return {
      user: {
        id: v2Response.id,
        name: \`\${v2Response.firstName} \${v2Response.lastName}\`,
        address: \`\${v2Response.address.street}, \${v2Response.address.city}, \${v2Response.address.zipCode}\`
      }
    };
  }
}

// Keep v1 endpoints, but route to v2 internally
app.get('/api/v1/users/:id', async (req, res) => {
  // Call v2 internally
  const v2Response = await fetch(\`/api/v2/users/\${req.params.id}\`);
  const v2Data = await v2Response.json();
  
  // Transform to v1
  const adapter = new V1toV2Adapter();
  const v1Data = adapter.transformResponse(v2Data);
  
  res.json(v1Data);
});
\`\`\`

**Warning**: This adds technical debt. Time-limit this approach.

---

## **7. Success Metrics**

Track:
- % clients migrated: Target 95% by Month 10
- v1 traffic: Target <1% by shutdown
- Support tickets: Should decrease after Month 6
- Client churn: Minimize (<5%)

---

## **Key Takeaways**:

1. **12-month timeline** minimum for major breaking changes
2. **Track everything**: Know which clients use which versions
3. **Communicate early and often**: Email at 10, 6, 3, 1 months
4. **Incentivize migration**: Free months, priority support
5. **Enforce with rate limits**: Month 6 limits push stragglers
6. **Whitelist for critical clients**: Avoid business disruption
7. **410 Gone at sunset**: Clean break, no indefinite v1 support
8. **Hard line on refusers**: Paid extensions or hard cutoff
9. **Remove code after 30 days** of zero traffic
10. **Learn from it**: Update contracts to allow versioning`,
    keyPoints: [
      '12-month migration timeline: Announce early, rate limit mid-way, hard sunset at end',
      'Track client usage by version with middleware logging to Redis',
      'Communicate: Email at announcement, 6 months, 3 months, 1 month before shutdown',
      'Enforce migration: Rate limit v1 to 500 req/day in Month 6 to push stragglers',
      'Whitelist critical clients for temporary extensions, charge enterprise for long extensions',
      'Handle refusers: Paid extensions ($5k/month), managed migration, or hard cutoff',
      'Return 410 Gone after sunset date, remove code after 30 days of zero traffic',
    ],
  },
];
