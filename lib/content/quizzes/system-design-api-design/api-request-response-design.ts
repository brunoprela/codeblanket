/**
 * Quiz questions for API Request/Response Design section
 */

export const apirequestresponsedesignQuiz = [
  {
    id: 'req-res-d1',
    question:
      'Design a response structure that serves both bandwidth-constrained mobile clients and data-rich web clients without maintaining two APIs.',
    sampleAnswer: `Use field selection with predefined field sets:

\`\`\`
GET /api/users/123?fields=basic    # Mobile: id, name, avatar
GET /api/users/123?fields=full     # Web: all fields
GET /api/users/123?fields=id,name,email  # Custom selection
\`\`\`

Implementation: Whitelist allowed fields, SELECT only requested columns from database, serialize only included fields.

Alternative: GraphQL allows perfect client-driven selection but requires paradigm shift.

Trade-off: Field selection is more complex backend but maintains REST principles and serves diverse clients efficiently.`,
    keyPoints: [
      'Field selection allows flexible data requirements',
      'Predefined field sets (basic, full) for common patterns',
      'GraphQL solves this elegantly but different paradigm',
      'Enable gzip compression for additional savings',
      'Trade complexity for flexibility and performance',
    ],
  },
  {
    id: 'req-res-d2',
    question:
      "Your API embeds a user's posts array inline, causing timeouts for users with thousands of posts. How to fix?",
    sampleAnswer: `Never include unbounded nested collections. Solutions:

1. **Remove nested collection, use links** (Recommended):
\`\`\`json
{
  "id": 123,
  "name": "Alice",
  "postsCount": 10000,
  "links": {"posts": "/api/users/123/posts"}
}
\`\`\`

2. **Limited preview with link**:
\`\`\`json
{
  "recentPosts": [/* 5 most recent */],
  "postsCount": 10000,
  "links": {"allPosts": "/api/users/123/posts"}
}
\`\`\`

3. **Separate endpoints**:
\`\`\`
GET /api/users/123        # User only
GET /api/posts?userId=123 # Posts paginated
\`\`\`

Real-world: GitHub, Twitter, Stripe all use separate endpoints for collections.

Rule: Always provide counts and links to paginated collections, never unbounded arrays.`,
    keyPoints: [
      'Unbounded nested collections cause performance issues',
      'Use separate endpoints for related collections',
      'Include counts and pagination links',
      'Limited previews provide convenience without risk',
      'Scalability always trumps convenience',
    ],
  },
  {
    id: 'req-res-d3',
    question:
      'Search results show inconsistent total counts (1,247 then 1,251 seconds later). Explain and propose solutions.',
    sampleAnswer: `Root cause: Real-time data changes between COUNT and SELECT queries, or eventual consistency in distributed databases.

Why exact counts are problematic:
1. COUNT(*) with filters is slow on large tables
2. Stale immediately in real-time systems
3. Requires locks for consistency (bad performance)

Solutions:

1. **Approximate counts** (Recommended):
\`\`\`json
{"approximateTotal": "~1,200"}
\`\`\`
Examples: Google ("About 1,240,000 results")

2. **Omit total** (Best for scale):
\`\`\`json
{"hasMore": true, "nextCursor": "..."}
\`\`\`
Examples: Twitter, Instagram feeds

3. **Cache with TTL**: Fast but still potentially inconsistent

Only use exact counts for:
- Small datasets (<10K records)
- Admin dashboards
- Static data

This isn't a bug—it's a fundamental trade-off between consistency, performance, and scale in distributed systems.`,
    keyPoints: [
      'Exact counts expensive and stale in real-time systems',
      'Approximate counts faster and honest about uncertainty',
      'Omitting counts scales best (infinite scroll pattern)',
      'Cursor pagination works without total counts',
      'Trade accuracy for performance at scale',
    ],
  },
];
