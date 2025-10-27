/**
 * Quiz questions for SQL vs NoSQL Decision Framework section
 */

export const sqlvsnosqldecisionQuiz = [
  {
    id: 'sql-vs-nosql-disc-q1',
    question:
      'You are designing a social media platform like Twitter. Some engineers argue for using PostgreSQL for everything to keep it simple, while others want MongoDB for flexible tweet data and Cassandra for the timeline. What approach would you take and why? Discuss the trade-offs.',
    sampleAnswer: `I would use a **polyglot persistence** approach with multiple databases, each serving a specific purpose optimized for different requirements.

**My Architecture:**1. **PostgreSQL for Users, Relationships, and Core Data**
   - User profiles (id, username, email, password_hash)
   - Follow relationships (follower_id, followee_id)
   - Account settings and authentication
   
   **Why**: This data requires strong consistency and relational integrity. When user A follows user B, that relationship must be immediately consistent. PostgreSQL's ACID properties and foreign keys ensure data integrity.

2. **Cassandra for Tweets and Timelines**
   - Store tweets with partition key on user_id
   - Store timelines (user's feed) with partition key on user_id, clustered by timestamp
   - Access pattern: "Get recent tweets for user X" - single partition read
   
   **Why**: Tweets are the highest volume data (millions per minute) and need horizontal scalability. Each tweet is independent (no JOINs needed). Cassandra excels at write-heavy workloads and time-series data. Eventual consistency is acceptable for tweets appearing in feeds slightly delayed.

3. **Redis for Feed Cache**
   - Cache hot users' timelines (celebrities, trending accounts)
   - Session storage
   - Rate limiting counters
   
   **Why**: Sub-millisecond reads for frequently accessed data, reducing Cassandra load.

4. **Elasticsearch for Search**
   - Full-text search on tweets, hashtags, users
   
   **Why**: Specialized for full-text search with relevance ranking.

**Trade-offs:**

**PostgreSQL-Only Approach:**
- ✅ **Pros**: Simple, single system, easy transactions, excellent for < 1M users
- ❌ **Cons**: Won't scale to Twitter\'s billions of tweets, sharding PostgreSQL is complex

**Polyglot Approach (My Recommendation):**
- ✅ **Pros**: Each database optimized for its use case, scales to billions of users
- ❌ **Cons**: Operational complexity (4 systems to monitor, maintain, back up), no cross-database transactions, potential consistency issues between systems

**Why This Trade-off is Worth It:**
At Twitter's scale (500M tweets/day, 300M users), a single PostgreSQL instance cannot handle the write throughput or storage. The operational complexity of multiple databases is offset by performance and scalability gains. However, I'd start with PostgreSQL for an MVP and migrate to polyglot as scale demands it.

**Key Insight**: There\'s no single "correct" answer. For a small startup (< 100K users), PostgreSQL for everything is pragmatic. For Twitter scale, polyglot persistence is necessary despite added complexity. The decision depends on current scale, growth trajectory, and team expertise.`,
    keyPoints: [
      'Use PostgreSQL for transactional data requiring strong consistency (users, follows)',
      'Use Cassandra for high-volume, write-heavy data with simple access patterns (tweets, timelines)',
      'Polyglot persistence adds operational complexity but enables scale',
      'Start simple (PostgreSQL) and migrate to specialized databases as scale requires',
      'At Twitter scale, the complexity of multiple databases is justified by performance gains',
    ],
  },
  {
    id: 'sql-vs-nosql-disc-q2',
    question:
      'A startup wants to build their entire application on NoSQL (MongoDB) because they heard "NoSQL scales better" and they want to avoid SQL migrations as their schema evolves. Is this a good decision? What advice would you give them?',
    sampleAnswer: `This is generally a **poor decision** based on hype rather than actual requirements. I would advise the startup to **start with PostgreSQL** unless they have specific, justified reasons for NoSQL.

**Why Starting with PostgreSQL is Better:**1. **ACID Transactions**
   Most applications need transactions at some point (payments, reservations, inventory). Adding transactional guarantees to NoSQL architecture later is extremely difficult. PostgreSQL gives you this from day one.

2. **Flexible Queries**
   In a startup, requirements change constantly. You'll need to answer questions you didn't anticipate: "Show me users who signed up last month but never completed their profile." SQL makes ad-hoc queries easy. NoSQL requires you to anticipate all query patterns upfront.

3. **Mature Tooling**
   PostgreSQL has 30+ years of mature tools: ORMs (TypeORM, Prisma, SQLAlchemy), admin panels (pgAdmin, DBeaver), migration tools (Alembic, Knex), backup/restore tools. MongoDB's ecosystem is less mature.

4. **Easier to Hire**
   More developers know SQL than MongoDB-specific query language. Faster onboarding.

5. **Relational Data**
   Most applications have relationships: users → orders → products. SQL is designed for this. NoSQL requires denormalization and data duplication, leading to consistency issues.

**When MongoDB Would Be Justified:**

- **True flexible schema need**: If your application is a CMS where each content type has truly different attributes (blog posts vs videos vs podcasts)
- **Prototyping phase**: For rapid prototyping where you're still figuring out the data model
- **Document-centric**: If your app naturally stores independent documents (logs, events, content articles)

**Addressing Their Concerns:**

**"NoSQL scales better"**
- **Reality**: PostgreSQL scales fine for 99% of startups. Most never exceed 1TB of data or 10K QPS, well within PostgreSQL's capacity.
- **When to migrate**: If you actually reach scale limits (billions of records, 100K+ QPS), migrate to NoSQL then. Don't prematurely optimize.

**"Avoid migrations as schema evolves"**
- **Reality**: Schema changes in PostgreSQL are straightforward with migration tools. NoSQL doesn't eliminate the problem - you still need to handle different document versions in your code.
- **Modern tools**: Tools like Prisma, TypeORM make SQL migrations painless.

**My Recommendation:**1. **Start with PostgreSQL** for the core application
2. **Use PostgreSQL's JSONB** for truly flexible fields if needed
3. **Add Redis** for caching and session storage (simple, proven win)
4. **Add specialized databases** only when you have measured performance problems that justify the operational complexity

**Example Architecture:**
\`\`\`
PostgreSQL:
                    - Users, orders, products, inventory
  - Use JSONB fields for flexible metadata

Redis:
    - Session storage
        - Frequently accessed data cache

            (Add MongoDB / Cassandra / etc.only if proven necessary)
\`\`\`

**Key Message**: Choose databases based on actual requirements, not hype. PostgreSQL is battle-tested, handles 99% of startup needs, and has a rich ecosystem. NoSQL is a tool for specific problems (massive scale, truly flexible schema, specialized access patterns), not a universal upgrade.`,
    keyPoints: [
      'Start with PostgreSQL unless you have specific, justified needs for NoSQL',
      'PostgreSQL handles > 99% of startup scale requirements',
      'SQL provides ACID, flexible queries, mature tooling, easier hiring',
      'NoSQL "scaling" advantage only matters at massive scale (billions of records)',
      'PostgreSQL JSONB fields provide schema flexibility when needed',
      'Add specialized databases (Redis, MongoDB) only for specific, measured problems',
    ],
  },
  {
    id: 'sql-vs-nosql-disc-q3',
    question:
      'Your company currently uses PostgreSQL for everything. The application is slowing down with 500M records in the users_events table (click tracking). Queries are timing out. Some engineers propose migrating everything to Cassandra. Is this the right approach? What would you do?',
    sampleAnswer: `Migrating **everything** to Cassandra would be a **mistake**. The problem is not PostgreSQL itself but using the wrong database for the wrong job. The solution is **polyglot persistence**: keep PostgreSQL for transactional data and move events to a specialized system.

**Problem Analysis:**

The issue is clear: **event logging** (high volume, append-only, time-series data) is a poor fit for PostgreSQL. 500M events in a relational table causes:
- Slow queries (JOINs and indexes degrade with table size)
- Bloated indexes
- Expensive vacuuming
- Storage overhead

However, this doesn't mean **all** your data should move to Cassandra. Your transactional data (users, orders, payments) likely works fine in PostgreSQL.

**My Solution:**

**Keep PostgreSQL for Transactional Data:**
- Users table
- Orders table
- Payments table
- Products table
- Anything requiring ACID transactions or complex JOINs

**Move Events to Specialized System:**

**Option 1: Cassandra (Best for Scale)**
\`\`\`
    Table: user_events
Partition Key: (user_id, date)  // Partition by user and day
Clustering Key: timestamp       // Sort by time within partition

Access pattern: "Get events for user X on date Y"
        \`\`\`

**Why Cassandra**: Handles billions of events, optimized for time-series, fast writes, horizontal scaling.

**Option 2: ClickHouse (Best for Analytics)**
- Columnar database optimized for analytical queries
- Compress data 10x compared to PostgreSQL
- Fast aggregations on billions of rows

**Option 3: Elasticsearch (Best for Search)**
- If you need full-text search on events
- Time-based indices for automatic rollover

**Option 4: BigQuery/Redshift (Best for Data Warehouse)**
- If events primarily used for analytics/BI
- Serverless, pay per query

**Implementation Strategy:**

**Phase 1: Stop the Bleeding**
\`\`\`
    1. Add Redis cache for hot queries
2. Partition PostgreSQL table by date (recent data in hot partition)
    3. Archive old events to cold storage(S3)
        \`\`\`

**Phase 2: Migrate Events**
\`\`\`
    1. Set up Cassandra cluster
    2. Dual - write: Write events to both PostgreSQL and Cassandra
    3. Backfill historical events in batches
    4. Switch reads to Cassandra
    5. Stop writing to PostgreSQL, decommission events table
        \`\`\`

**Phase 3: Keep PostgreSQL for Core Data**
\`\`\`
    PostgreSQL:
    - users(10M records) ✅ Perfect fit
        - orders(50M records) ✅ Perfect fit
            - products(1M records) ✅ Perfect fit

    Cassandra:
    - user_events(500M records) ✅ Designed for this

Redis:
        - Cache hot user data
            - Session storage
                \`\`\`

**Why This Hybrid Approach is Better:**

✅ **Keep ACID where needed**: Payments, orders stay in PostgreSQL with transactions
✅ **Optimize for scale**: Events move to Cassandra designed for billions of records
✅ **Preserve existing code**: Most application code unchanged
✅ **Minimize risk**: Gradual migration, one table at a time
✅ **Lower operational burden**: Only add Cassandra for specific problem

**Why Migrating Everything to Cassandra is Wrong:**

❌ **Lose ACID transactions**: Payments, orders need atomicity
❌ **Lose JOINs**: Cassandra has no JOINs; you'd denormalize everything
❌ **Rewrite all queries**: Application code requires massive rewrite
❌ **Operational complexity**: Learning curve, new tooling, new expertise
❌ **Premature**: Your transactional data (users, orders) isn't the problem

**Cost Comparison:**

**Migrate Everything to Cassandra:**
- Months of development rewriting queries
- Training team on Cassandra
- Risk of losing transactional integrity
- Complexity managing denormalized data

**Hybrid Approach:**
- 2-3 weeks to move events table
- Keep existing application mostly unchanged
- Solve the actual problem (events table)

**Key Insight**: The database slowdown is from one specific table (events) with a specific access pattern (time-series, high volume). Solve that specific problem with a specialized database, not a wholesale migration that introduces more problems than it solves.`,
    keyPoints: [
      'Do not migrate everything - identify the specific problem table/use case',
      'Use polyglot persistence: right database for each job',
      'Keep PostgreSQL for transactional data (users, orders, payments)',
      'Move high-volume time-series data (events) to Cassandra or ClickHouse',
      'Implement gradual migration: dual-write, backfill, switch reads, decommission',
      'Wholesale database migrations are high-risk and often unnecessary',
    ],
  },
];
