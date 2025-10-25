import { MultipleChoiceQuestion } from '@/lib/types';

export const databaseIntegrationSqlalchemyMultipleChoice = [
  {
    id: 'fastapi-db-mc-1',
    question:
      'What is the purpose of pool_pre_ping in SQLAlchemy engine configuration?',
    options: [
      'To test database performance before queries',
      'To verify connection health before using it from the pool',
      'To reduce latency by keeping connections warm',
      'To automatically retry failed queries',
    ],
    correctAnswer: 1,
    explanation:
      'pool_pre_ping=True makes SQLAlchemy verify connection health before using it from the pool. It executes a simple "SELECT 1" query to check if the connection is still valid. This prevents errors from stale connections (e.g., database restarted, connection timed out, network issues). Without pool_pre_ping, you might get errors like "connection already closed" when trying to use a stale connection. Small performance cost (~1ms per connection checkout) but prevents frustrating production errors. Always use pool_pre_ping=True in production!',
  },
  {
    id: 'fastapi-db-mc-2',
    question:
      'What is the N+1 query problem and how do you fix it in SQLAlchemy?',
    options: [
      'Having N+1 database connections; fix with connection pooling',
      "Executing 1 query for a list + N queries for each item's relationships; fix with joinedload/selectinload",
      'Having N+1 tables; fix by normalizing database schema',
      'Making N+1 API calls; fix with batch requests',
    ],
    correctAnswer: 1,
    explanation:
      'N+1 problem: Query 1 fetches N items, then N additional queries fetch related data for each item. Example: posts = db.query(Post).all(); for post in posts: print(post.author.username) → 1 query for posts + N queries for authors = N+1 total! Fix with joinedload (JOIN): db.query(Post).options (joinedload(Post.author)).all() → 1 query total. Or selectinload (SELECT IN): options (selectinload(Post.author)) → 2 queries total (posts, then all authors in one query). Can cause severe performance issues (100 posts = 101 queries = 1000ms instead of 1-2 queries = 50ms). Always profile and fix N+1 issues!',
  },
  {
    id: 'fastapi-db-mc-3',
    question: 'What does orm_mode = True do in Pydantic Config?',
    options: [
      'Enables ORM query optimization',
      'Allows Pydantic to create models from ORM objects using attribute access',
      'Automatically creates database tables',
      'Enables transaction management',
    ],
    correctAnswer: 1,
    explanation:
      'orm_mode = True allows Pydantic models to be created from ORM objects (SQLAlchemy models) that use attribute access (user.username) instead of dict access (user["username"]). Without orm_mode: UserResponse(**db_user.__dict__) or manual mapping. With orm_mode: UserResponse.from_orm (db_user) works automatically. Essential for FastAPI + SQLAlchemy integration. In FastAPI endpoints with response_model: @app.get("/users/{id}", response_model=UserResponse); return db_user → FastAPI calls from_orm internally. Bridges SQLAlchemy models and Pydantic schemas elegantly.',
  },
  {
    id: 'fastapi-db-mc-4',
    question:
      'When should you use joinedload vs selectinload for relationship loading?',
    options: [
      "Always use joinedload because it's faster",
      'joinedload for few related records (JOIN), selectinload for many (SELECT IN)',
      'selectinload for all cases to avoid JOINs',
      'They are identical in performance',
    ],
    correctAnswer: 1,
    explanation:
      'joinedload uses JOIN: Single query, but can create cartesian product with many related records. Use for: one-to-one relationships, one-to-many with few children (<100), when you need all related data. selectinload uses SELECT IN: 2 queries (parent, then SELECT related WHERE id IN (...)). Use for: one-to-many with many children (>100), many-to-many relationships, when JOIN would create huge result set. Example: 100 posts, each with 50 comments. joinedload: 1 query, 5000 rows returned (100×50). selectinload: 2 queries, 100+5000=5100 rows total, but no cartesian product. Generally: selectinload safer for large datasets.',
  },
  {
    id: 'fastapi-db-mc-5',
    question:
      'What happens if you forget to commit a transaction in SQLAlchemy?',
    options: [
      'Changes are automatically saved',
      'Changes are lost when session closes (rollback)',
      'Database throws an error',
      'Changes are queued for later commit',
    ],
    correctAnswer: 1,
    explanation:
      'Without db.commit(), changes are lost when the session closes. SQLAlchemy sessions use transactions—changes are staged but not persisted until commit. Example: db.add (user); db.close() without commit → user not saved! Session closure triggers automatic rollback of uncommitted changes. This is a safety feature preventing accidental data corruption. Always explicitly commit: db.add (user); db.commit(); db.refresh (user). Or use autocommit=True (not recommended—lose transaction control). In FastAPI with get_db dependency, commit in endpoint, not in dependency (keeps transactions explicit and testable).',
  },
];
