import { MultipleChoiceQuestion } from '@/lib/types';

export const databaseFundamentalsQuiz = [
  {
    id: 'sql-db-fund-q-1',
    question:
      'You are building a financial application that transfers money between user accounts. Explain how you would implement this using database transactions. Include: (1) why transactions are necessary, (2) what ACID properties protect against, (3) implementation in both raw SQL and SQLAlchemy, (4) handling of potential race conditions, (5) error handling and rollback strategy. What could go wrong without proper transaction handling?',
    sampleAnswer: `Financial transfer implementation: (1) Why transactions: Money transfers are atomic operations—both debit and credit must succeed together or fail together. Without transactions, a crash between debit and credit would lose money. (2) ACID protection: Atomicity ensures both operations complete or neither does. Consistency maintains business rules (no negative balances). Isolation prevents concurrent transfers from interfering (read uncommitted, read committed, repeatable read, serializable levels). Durability ensures committed transfers survive crashes. (3) Raw SQL implementation: BEGIN; UPDATE accounts SET balance = balance - 100 WHERE id = 1; UPDATE accounts SET balance = balance + 100 WHERE id = 2; COMMIT; Check both UPDATEs affected 1 row, otherwise ROLLBACK. (4) SQLAlchemy implementation: Use session as transaction boundary: session.begin(), account1.balance -= 100, account2.balance += 100, session.commit(). Session automatically rolls back on exception. (5) Race conditions: Two concurrent transfers from same account could overdraw. Solutions: (a) SELECT FOR UPDATE to lock rows, (b) Optimistic locking with version column, (c) CHECK constraint: balance >= 0 at database level. (6) Error handling: Wrap in try/except, call session.rollback() on exception, re-raise to caller. Log failures for reconciliation. (7) Without transactions: Money could be debited but not credited (lost money), concurrent transfers could overdraw account, crashes leave inconsistent state, no audit trail of failures.`,
    keyPoints: [
      'Transactions ensure atomicity: both operations succeed or both fail',
      'ACID properties protect data integrity: consistency, isolation, durability',
      'Use SELECT FOR UPDATE or optimistic locking to prevent race conditions',
      'Always rollback on exceptions, use CHECK constraints for business rules',
      'Without transactions: money loss, inconsistent state, race conditions',
    ],
  },
  {
    id: 'sql-db-fund-q-2',
    question:
      'Design a schema for a blogging platform with users, posts, comments, and tags. Address: (1) primary key strategy (auto-increment vs UUID, and why), (2) foreign key relationships and cascade behavior, (3) indexes needed for common queries, (4) how to model many-to-many relationship (posts-tags), (5) timestamp tracking, (6) soft delete strategy. Include SQL table definitions and explain trade-offs.',
    sampleAnswer: `Blogging platform schema design: (1) Primary key strategy: Use UUID for distributed systems (can generate IDs independently, no coordination), auto-increment for single-database systems (smaller, faster joins). Recommendation: UUID for modern apps (id UUID PRIMARY KEY DEFAULT gen_random_uuid()). Trade-off: UUIDs are 16 bytes vs 4 bytes for integer, but eliminate coordination bottleneck. (2) Foreign keys: CREATE TABLE posts (id UUID PRIMARY KEY, user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE, title TEXT, content TEXT, created_at TIMESTAMP DEFAULT NOW(), updated_at TIMESTAMP, deleted_at TIMESTAMP); CASCADE deletes posts when user deleted, SET NULL preserves posts. Choose based on business logic. (3) Indexes: CREATE INDEX idx_posts_user_id ON posts(user_id) - for user's posts. CREATE INDEX idx_posts_created ON posts(created_at DESC) - for recent posts. CREATE INDEX idx_comments_post_user ON comments(post_id, user_id) - for post comments. CREATE INDEX idx_posts_deleted ON posts(deleted_at) WHERE deleted_at IS NULL - partial index for active posts. Index foreign keys always. Composite indexes for common query combinations. (4) Many-to-many: Junction table: CREATE TABLE post_tags (post_id UUID REFERENCES posts(id) ON DELETE CASCADE, tag_id UUID REFERENCES tags(id) ON DELETE CASCADE, PRIMARY KEY (post_id, tag_id)). Composite primary key prevents duplicates. CASCADE deletes associations when post/tag deleted. (5) Timestamps: created_at (insertion time), updated_at (last modification), deleted_at (soft delete). Use triggers for automatic updated_at: CREATE TRIGGER update_posts_updated_at BEFORE UPDATE ON posts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(). (6) Soft delete: Add deleted_at TIMESTAMP column, filter WHERE deleted_at IS NULL in queries. Benefits: data recovery, audit trail. Drawbacks: all queries must filter, unique constraints complicated. Alternative: archive table for deleted records.`,
    keyPoints: [
      'UUID for distributed systems, auto-increment for single database',
      'Foreign keys with CASCADE/SET NULL based on business logic',
      'Index foreign keys, composite indexes for common query patterns',
      'Many-to-many: junction table with composite primary key',
      'Soft delete: deleted_at column, requires filtering in all queries',
    ],
  },
  {
    id: 'sql-db-fund-q-3',
    question:
      'Explain when you should use an ORM vs raw SQL in production applications. Provide: (1) specific scenarios where ORM is superior, (2) specific scenarios where raw SQL is necessary, (3) how to combine both approaches, (4) performance implications, (5) testing and debugging considerations. Include code examples for a hybrid approach.',
    sampleAnswer: `ORM vs Raw SQL decision framework: (1) Use ORM for: (a) CRUD operations: user = User(email="test@example.com"); session.add(user); session.commit() - clean, Pythonic, type-safe. (b) Simple queries: users = session.query(User).filter(User.is_active == True).all() - readable, composable, IDE autocomplete. (c) Relationship traversal: for post in user.posts: print(post.title) - automatic joins, lazy/eager loading control. (d) Database abstraction: switch PostgreSQL to MySQL with minimal changes. (e) Business logic: ORM models contain validation, computed properties, methods. (2) Use raw SQL for: (a) Complex analytics: WITH recursive CTEs, window functions, multiple aggregations. SQLAlchemy can express these but raw SQL clearer. (b) Bulk operations: INSERT with 10,000 rows, use COPY (PostgreSQL) or LOAD DATA (MySQL) - 10-100x faster than ORM. (c) Database-specific features: full-text search (to_tsvector), JSON operators (->>, ?), specialized indexes (GIN, GiST). (d) Performance-critical paths: hand-optimized queries, specific indexes, query plans. (e) Schema migrations: complex DDL, concurrent index creation, data migrations. (3) Hybrid approach: Use repository pattern: class UserRepository: def create(self, email): user = User(email=email); session.add(user); session.commit(); return user. def analytics(self): return session.execute(text("SELECT DATE(created_at), COUNT(*) FROM users GROUP BY DATE(created_at)")).fetchall(). Encapsulates data access, uses ORM for simple, raw SQL for complex. (4) Performance: ORM adds 5-20% overhead for simple queries (object creation, attribute access). Negligible for I/O-bound operations (network, disk). Raw SQL faster for bulk operations, complex queries. Measure with profiling, optimize hot paths. (5) Testing: ORM easier to mock: Mock session.query. Raw SQL harder: Mock connection, cursor. Solution: Repository pattern testable with dependency injection. Debugging: ORM use echo=True to see SQL. Raw SQL direct inspection. Use SQLAlchemy's compiled queries for production debugging: str(query.statement.compile(compile_kwargs={"literal_binds": True})). Hybrid approach: 80% ORM (productivity), 20% raw SQL (performance, features).`,
    keyPoints: [
      'ORM for CRUD, simple queries, relationships - productivity and type safety',
      'Raw SQL for analytics, bulk ops, database-specific features - performance',
      'Hybrid: repository pattern, ORM for business logic, raw SQL for complex',
      'ORM adds 5-20% overhead, negligible for I/O-bound, critical for bulk ops',
      'Testing: repository pattern with dependency injection, echo=True for debugging',
    ],
  },
];
