import { MultipleChoiceQuestion } from '@/lib/types';

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
      'Use SQLite in-memory for unit tests: Fast (< 1s), isolated, no setup. Use PostgreSQL for integration tests: Tests real DB behavior, constraints, indexes, full-text search, JSON operators. SQLite limitations: Limited ALTER TABLE, no full-text search, different SQL dialect. Strategy: SQLite for model tests, PostgreSQL for query tests and integration. CI/CD: Both - SQLite for speed, PostgreSQL for realism.',
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
      "Transaction rollback is the best approach: Start transaction before test, run test, rollback after. Fast (no DROP/CREATE), clean state guaranteed, isolation between tests. Implementation: connection = test_engine.connect(); transaction = connection.begin(); yield session; transaction.rollback(). Alternative (truncate) is slower and doesn't reset sequences. Recreating tables is very slow.",
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
      'Factory Boy generates realistic test data efficiently. Benefits: (1) DRY - define data structure once, (2) Faker integration for realistic data, (3) Handles relationships automatically, (4) Sequences for unique values, (5) Override specific fields per test. Example: UserFactory.create() generates user with realistic name, email. UserFactory.create(email="specific@example.com") overrides email. PostFactory.create(user=user) creates post with relationship.',
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
      'Mock database for unit testing business logic (services, use cases). Real database for data layer (repositories, queries, relationships). Reasoning: Business logic tests should be fast and isolated - mock repository dependencies. Data layer tests must verify actual SQL, joins, constraints - require real database. Example: UserService test: mock UserRepository (unit test). UserRepository test: real database (integration test).',
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
