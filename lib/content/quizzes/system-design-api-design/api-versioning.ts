/**
 * Quiz questions for API Versioning Strategies section
 */

export const apiversioningQuiz = [
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
  const fullVersion = req.headers['accept-version',] || '2.0.0';
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
];
