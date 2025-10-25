import { MultipleChoiceQuestion } from '@/lib/types';

export const testingFastapiMultipleChoice = [
  {
    id: 1,
    question:
      "What is the primary advantage of using FastAPI's TestClient for testing?",
    options: [
      'TestClient allows testing endpoints without starting a server, making tests fast and eliminating network overhead',
      'TestClient automatically generates test cases from your code',
      'TestClient only works with async functions',
      'TestClient requires Docker to run',
    ],
    correctAnswer: 0,
    explanation:
      'TestClient runs your FastAPI application in-process without starting an actual server or making network requests. This makes tests extremely fast (no TCP overhead), reliable (no network flakiness), and easy to debug. Tests execute synchronously even for async endpoints. Pattern: client = TestClient (app); response = client.get("/endpoint"). No need to run uvicorn, no port conflicts, tests run in milliseconds.',
  },
  {
    id: 2,
    question:
      'How do you override FastAPI dependencies for testing (e.g., to bypass authentication)?',
    options: [
      'Use app.dependency_overrides[original_dependency] = mock_dependency to replace dependencies during tests',
      'Dependencies cannot be overridden in tests',
      'Edit the original dependency function',
      'Use environment variables to disable dependencies',
    ],
    correctAnswer: 0,
    explanation:
      'FastAPI provides app.dependency_overrides dictionary for testing. Pattern: app.dependency_overrides[get_current_user] = lambda: test_user. This replaces the real dependency with a mock for all test requests. Remember to clear overrides after tests: app.dependency_overrides.clear(). This enables testing protected endpoints without real authentication, mocking database connections, and replacing external service calls.',
  },
  {
    id: 3,
    question:
      'What is the best strategy for database testing in FastAPI to ensure test isolation?',
    options: [
      'Use transaction rollback: wrap each test in a transaction that rolls back after the test completes, ensuring clean state',
      'Delete all data before each test',
      'Use a separate database for each test',
      'Restart the database server between tests',
    ],
    correctAnswer: 0,
    explanation:
      "Transaction rollback provides fast, reliable test isolation. Pattern: Start transaction before test → run test (all DB operations) → rollback transaction after test. Benefits: Fast (no actual commits), isolated (changes don't affect other tests), clean state automatically restored. Implementation with pytest fixture: @pytest.fixture def db(): connection = engine.connect(); transaction = connection.begin(); session = Session (bind=connection); yield session; transaction.rollback(). Deleting data (option 2) is slow, separate databases per test (option 3) don't scale, restarting DB (option 4) is extremely slow.",
  },
  {
    id: 4,
    question: 'When testing async FastAPI endpoints, what should you use?',
    options: [
      'TestClient works for both sync and async endpoints automatically, or use httpx.AsyncClient with pytest.mark.asyncio for true async tests',
      'Async endpoints cannot be tested',
      'Only asyncio.run() can test async endpoints',
      'Convert all endpoints to sync for testing',
    ],
    correctAnswer: 0,
    explanation:
      'TestClient handles async endpoints automatically - it runs them synchronously in tests for simplicity. For testing concurrent async behavior, use httpx.AsyncClient: async with httpx.AsyncClient (app=app) as client: response = await client.get("/endpoint"). Mark test with @pytest.mark.asyncio. TestClient is simpler for most tests, AsyncClient when testing actual async behavior (concurrent requests, timeouts, async context managers). Never convert production async to sync just for testing (option 4).',
  },
  {
    id: 5,
    question: 'What is the purpose of pytest fixtures in FastAPI testing?',
    options: [
      'Fixtures provide reusable test setup and teardown, like creating test database sessions, mock users, or configured test clients',
      'Fixtures are decorators that make tests run faster',
      'Fixtures automatically fix bugs in your code',
      'Fixtures are only used for mocking',
    ],
    correctAnswer: 0,
    explanation:
      'Pytest fixtures enable DRY testing by providing reusable setup/teardown. Common fixtures: @pytest.fixture def client(): return TestClient (app) (test client), @pytest.fixture def db(): (database session with rollback), @pytest.fixture def authenticated_client(): (client with auth token). Fixtures can depend on other fixtures, have different scopes (function, session), and automatically handle cleanup with yield. Benefits: Reduce boilerplate, ensure consistent test setup, automatic cleanup, composition through fixture dependencies.',
  },
].map(({ id, ...q }, idx) => ({ id: `fastapi-mc-${idx + 1}`, ...q }));
