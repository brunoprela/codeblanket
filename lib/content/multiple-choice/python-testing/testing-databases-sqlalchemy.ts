import { MultipleChoiceQuestion } from '@/lib/types';

export const testingDatabasesSqlalchemyMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tds-mc-1',
      question:
        'What is the main advantage of using SQLite in-memory database for tests?',
      options: [
        'It provides better compatibility with production PostgreSQL',
        'It is 100× faster than disk-based databases and requires no cleanup',
        'It automatically mocks all database operations',
        'It supports more advanced SQL features than PostgreSQL',
      ],
      correctAnswer: 1,
      explanation:
        'SQLite in-memory (:memory:) is 100× faster: No disk I/O, database created/destroyed in RAM, perfect for unit tests. Example: 1000 tests PostgreSQL 10 min, SQLite :memory: 6 seconds. Trade-off: PostgreSQL-specific features may not work. Strategy: SQLite locally (speed), PostgreSQL in CI (production parity). Not for: mocking (real DB), advanced features (fewer than PostgreSQL). Use: create_engine("sqlite:///:memory:").',
    },
    {
      id: 'tds-mc-2',
      question:
        'Why use transaction rollback instead of truncating tables after tests?',
      options: [
        'Rollback is slower but more reliable',
        'Rollback is much faster and guarantees clean state',
        'Truncate is not supported by SQLAlchemy',
        'Rollback works with foreign key constraints better',
      ],
      correctAnswer: 1,
      explanation:
        'Transaction rollback is faster (1ms vs 100ms for truncate): One transaction.rollback() vs DELETE FROM each table. Clean state guaranteed: Rollback undoes all changes automatically. Pattern: Begin transaction → test → rollback. Truncate issues: Must disable foreign keys, delete in correct order, slower. Rollback: connection.begin(); session = Session(bind=connection); yield session; transaction.rollback(). Result: 100× faster cleanup, automatic dependency handling.',
    },
    {
      id: 'tds-mc-3',
      question: 'What is Factory Boy used for in database testing?',
      options: [
        'Creating database schemas automatically',
        'Generating realistic test data for models with minimal code',
        'Optimizing database query performance',
        'Managing database connections and sessions',
      ],
      correctAnswer: 1,
      explanation:
        'Factory Boy generates test data efficiently: Define once, use everywhere. Example: class UserFactory(factory.alchemy.SQLAlchemyModelFactory): username = factory.Sequence(lambda n: f"user{n}"); email = factory.Faker("email"). Usage: user = UserFactory.create() → generates realistic user with Faker data. Benefits: DRY (don\'t repeat data creation), realistic data (Faker), handles relationships (SubFactory). Not for: schema creation (Alembic), query optimization, or connection management. Essential for maintainable test suites.',
    },
    {
      id: 'tds-mc-4',
      question:
        'What is the recommended fixture scope for a database engine in pytest?',
      options: [
        'function (created for each test)',
        'class (created for each test class)',
        'module (created for each test file)',
        'session (created once for entire test run)',
      ],
      correctAnswer: 3,
      explanation:
        'Session scope for engine (expensive to create): @pytest.fixture(scope="session") def engine(): create_engine(...); Base.metadata.create_all(engine); yield; drop_all. Created once at start, reused across all tests (fast). Combine with function-scoped session for isolation: @pytest.fixture def session(engine): connection.begin(); session = Session(); yield; rollback(). Pattern: Expensive setup (engine) → session scope. Cheap, needs isolation (session) → function scope. Result: Fast tests with proper isolation.',
    },
    {
      id: 'tds-mc-5',
      question:
        'How do you test database migrations in SQLAlchemy with Alembic?',
      options: [
        'Migrations cannot be tested, only manually verified',
        'Create test that runs alembic upgrade, verifies schema, runs downgrade',
        'Use pytest-alembic plugin exclusively',
        'Migrations are tested automatically by SQLAlchemy',
      ],
      correctAnswer: 1,
      explanation:
        'Test migrations up and down: def test_migration(): alembic upgrade head → verify schema changes (check tables/columns exist), alembic downgrade -1 → verify rollback (tables removed). Example: assert "users" in db.table_names() after upgrade, assert "users" not in db.table_names() after downgrade. Critical: Ensures migrations are reversible, detects schema errors before production, validates data transformations. Run on separate test DB (not in-memory). Essential for production database changes.',
    },
  ];
