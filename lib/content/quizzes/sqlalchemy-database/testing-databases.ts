import { MultipleChoiceQuestion } from '@/lib/types';

export const testingDatabasesQuiz = [
  {
    id: 'sql-test-q-1',
    question:
      'Design a complete testing strategy for a SQLAlchemy application with unit tests, integration tests, and CI/CD. Address: (1) test database setup (SQLite vs PostgreSQL), (2) fixture strategy, (3) factory pattern implementation, (4) when to use mocks vs real database, (5) CI/CD pipeline configuration. Include complete code examples.',
    sampleAnswer:
      'Comprehensive testing strategy: (1) Test database setup: Use SQLite in-memory for unit tests (fast, isolated) - engine = create_engine("sqlite:///:memory:"). Use PostgreSQL for integration tests (tests real DB behavior, constraints, indexes). Setup: @pytest.fixture with transaction rollback per test. (2) Fixture strategy: Session-scoped test_engine (create once), function-scoped test_db (transaction per test). Code: transaction = connection.begin(); yield session; transaction.rollback(). Each test gets clean database. (3) Factory pattern: Use Factory Boy for test data. class UserFactory(SQLAlchemyModelFactory): email = factory.Sequence(lambda n: f"user{n}@test.com"). Benefits: DRY test data, realistic data with Faker, relationships handled. (4) Mock vs real DB: Unit tests (business logic): Mock repositories - fast, no DB dependency. Integration tests (queries, relationships): Real database - tests actual SQL. E2E tests: Full stack with real DB. Rule: Mock external dependencies, use real DB for data layer. (5) CI/CD: GitHub Actions with PostgreSQL service. Steps: checkout, install deps, run migrations (alembic upgrade head), run tests with coverage (pytest --cov), upload coverage report. Environment: DATABASE_URL=postgresql://postgres:postgres@localhost/test_db. Run on every push/PR. (6) Coverage: Aim for 80%+ on repositories, services. 100% on critical business logic. Use pytest-cov: pytest --cov=myapp --cov-report=html. (7) Test organization: tests/test_models.py (model tests), tests/test_repositories.py (query tests), tests/test_services.py (business logic with mocks), tests/test_api.py (endpoint integration tests). Result: Fast unit tests (< 1s), thorough integration tests (< 10s), automated CI/CD catching bugs before production.',
    keyPoints: [
      'SQLite for unit (fast), PostgreSQL for integration (real DB behavior)',
      'Transaction rollback fixture: clean DB per test, no cleanup needed',
      'Factory Boy: DRY test data with relationships, Faker for realistic data',
      'Mock business logic (unit), real DB for queries (integration)',
      'CI/CD: PostgreSQL service, migrations, coverage reports, automated on push',
    ],
  },
  {
    id: 'sql-test-q-2',
    question:
      'Explain how to test database migrations thoroughly. Include: (1) testing upgrade/downgrade, (2) testing idempotency, (3) testing data migrations, (4) handling migration failures, (5) CI/CD integration for migrations. Provide complete test implementation.',
    sampleAnswer:
      'Migration testing strategy: (1) Upgrade/downgrade tests: def test_migration_up_down(alembic_config): upgrade(alembic_config, "head"); downgrade(alembic_config, "-1"); upgrade(alembic_config, "head"). Verifies migrations reversible. (2) Idempotency test: def test_migration_idempotent(alembic_config): upgrade(alembic_config, "head"); upgrade(alembic_config, "head"). Should not error when run twice. (3) Data migration tests: Test data transformations. Before migration: insert test data. After migration: verify data transformed correctly. Example: name split migration - verify first_name/last_name populated from name column. (4) Failure handling: Test rollback on error. Wrap migration in transaction, simulate failure (unique constraint violation), verify rollback leaves DB in consistent state. (5) CI/CD integration: Separate job for migration tests. Steps: fresh DB, run all migrations from scratch, test upgrade/downgrade cycles, verify schema matches models (alembic check). Run before merging PRs. (6) Production simulation: Test on copy of production data (anonymized). Catch edge cases missed in dev. Verify performance (large tables). (7) Verification: After migration, run alembic check to verify schema matches models. Query information_schema to verify indexes, constraints created. Example test: conn.execute("SELECT * FROM information_schema.indexes WHERE table_name = \'users\'") - verify index exists. Result: Catch migration issues before production, safe deployments, confidence in rollback procedures.',
    keyPoints: [
      'Test upgrade/downgrade cycle: up→down→up, verifies reversibility',
      'Idempotency: run upgrade twice, should not error',
      'Data migrations: verify transformations, test edge cases',
      'Failure handling: transaction rollback, verify consistent state',
      'CI/CD: Fresh DB, run all migrations, alembic check, before merge',
    ],
  },
  {
    id: 'sql-test-q-3',
    question:
      'You need to test a complex query with joins, aggregations, and filtering. Explain: (1) whether to use mocks or real database, (2) how to set up test data efficiently, (3) what to assert, (4) how to test performance, (5) testing query optimization (indexes). Include complete test implementation.',
    sampleAnswer:
      'Complex query testing: (1) Real database required: Complex queries test SQL generation, joins, aggregations - can\'t mock effectively. Use PostgreSQL test database (not SQLite - different SQL dialects). (2) Test data setup: Use Factory Boy for efficient creation. user = UserFactory.create(); posts = PostFactory.create_batch(10, user=user); comments = CommentFactory.create_batch(50, post=posts[0]). Fixtures for common scenarios: @pytest.fixture def user_with_activity(test_db, factories) returns user with posts and comments. (3) Assertions: Test result correctness: assert len(results) == expected_count. Test data shape: assert all(hasattr(r, "post_count") for r in results). Test ordering: assert results[0].created_at > results[1].created_at. Test aggregations: assert results[0].comment_count == 50. Edge cases: empty results, NULL handling, division by zero in aggregations. (4) Performance testing: Use time.time() to measure query duration. threshold = 0.1  # 100ms; start = time.time(); query_result; duration = time.time() - start; assert duration < threshold. Test with realistic data volume (1000+ rows). (5) Index testing: Test with EXPLAIN ANALYZE. result = session.execute(text("EXPLAIN ANALYZE " + query_sql)). Verify uses indexes: assert "Index Scan" in str(result), not "Seq Scan". Test before/after index creation: measure performance improvement. (6) Test query composition: Verify query builds correctly with different filters. Test with filter A, filter B, filter A+B - ensure AND/OR logic correct. (7) Regression tests: Save query results as snapshot. On changes, compare new results to snapshot - catch unexpected changes. Example test: def test_user_post_stats(test_db, user_with_activity): stmt = select(User, func.count(Post.id)).join(User.posts).group_by(User.id); result = test_db.execute(stmt).one(); assert result[1] == 10  # post count. Use EXPLAIN: verify Index Scan on posts.user_id. Result: Confidence in query correctness, performance, and optimization.',
    keyPoints: [
      "Real DB required: Tests actual SQL, joins, aggregations (can't mock)",
      'Factory Boy: Efficient test data creation, batch operations',
      'Assert: count, shape, ordering, aggregations, edge cases (NULL, empty)',
      'Performance: time.time(), assert < 100ms, test realistic volume',
      'EXPLAIN ANALYZE: verify Index Scan not Seq Scan, test index effectiveness',
    ],
  },
];

export const testingDatabasesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-test-mc-1',
    question: 'When should you use in-memory SQLite vs PostgreSQL for testing?',
    options: [
      'Always use SQLite (faster)',
      'SQLite for unit tests, PostgreSQL for integration tests',
      'Always use PostgreSQL',
      'No difference',
    ],
    correctAnswer: 1,
    explanation:
      'Use SQLite in-memory for unit tests: Fast (< 1s), isolated, no setup. ' +
      'Use PostgreSQL for integration tests: Tests real DB behavior, constraints, indexes, full-text search, JSON operators. ' +
      'SQLite limitations: Limited ALTER TABLE, no full-text search, different SQL dialect. ' +
      'Strategy: SQLite for model tests, PostgreSQL for query tests and integration. ' +
      'CI/CD: Both - SQLite for speed, PostgreSQL for realism.',
  },
  {
    id: 'sql-test-mc-2',
    question:
      'What is the best way to ensure a clean database state for each test?',
    options: [
      'Drop and recreate tables',
      'Transaction rollback per test',
      'Manual cleanup in teardown',
      'Truncate all tables',
    ],
    correctAnswer: 1,
    explanation:
      'Transaction rollback is the best approach: Start transaction before test, run test, rollback after. ' +
      'Fast (no DROP/CREATE), clean state guaranteed, isolation between tests. ' +
      'Implementation: connection = test_engine.connect(); transaction = connection.begin(); yield session; transaction.rollback(). ' +
      "Alternative (truncate) is slower and doesn't reset sequences. Recreating tables is very slow.",
  },
  {
    id: 'sql-test-mc-3',
    question: 'What is the purpose of Factory Boy in database testing?',
    options: [
      'Creates database connections',
      'Generates realistic test data with relationships',
      'Runs migrations',
      'Mocks database',
    ],
    correctAnswer: 1,
    explanation:
      'Factory Boy generates realistic test data efficiently. ' +
      'Benefits: (1) DRY - define data structure once, (2) Faker integration for realistic data, (3) Handles relationships automatically, (4) Sequences for unique values, (5) Override specific fields per test. ' +
      'Example: UserFactory.create() generates user with realistic name, email. ' +
      'UserFactory.create(email="specific@example.com") overrides email. ' +
      'PostFactory.create(user=user) creates post with relationship.',
  },
  {
    id: 'sql-test-mc-4',
    question: 'When should you mock the database in tests?',
    options: [
      'Always',
      'Never',
      'For unit testing business logic, use real DB for data layer tests',
      'Only in production',
    ],
    correctAnswer: 2,
    explanation:
      'Mock database for unit testing business logic (services, use cases). ' +
      'Real database for data layer (repositories, queries, relationships). ' +
      'Reasoning: Business logic tests should be fast and isolated - mock repository dependencies. ' +
      'Data layer tests must verify actual SQL, joins, constraints - require real database. ' +
      'Example: UserService test: mock UserRepository (unit test). ' +
      'UserRepository test: real database (integration test).',
  },
  {
    id: 'sql-test-mc-5',
    question: 'What does transaction rollback ensure in database tests?',
    options: [
      'Faster queries',
      'Clean database state - changes reverted after each test',
      'Better performance',
      'Data persistence',
    ],
    correctAnswer: 1,
    explanation:
      "Transaction rollback ensures clean state: All changes made during test (INSERT, UPDATE, DELETE) are reverted after test completes. Benefits: (1) No cleanup code needed, (2) Tests don't interfere with each other, (3) Fast (rollback is instant), (4) Works even if test fails/errors. Implementation: Start transaction before test, run test, rollback transaction in finally block. Critical for test isolation and reliability.",
  },
];
