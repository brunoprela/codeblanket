import { MultipleChoiceQuestion } from '@/lib/types';

export const fixturesDeepDiveMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fdd-mc-1',
    question: 'What is the default scope for pytest fixtures?',
    options: [
      'session (created once for entire test run)',
      'module (created once per test file)',
      'class (created once per test class)',
      'function (created once per test function)',
    ],
    correctAnswer: 3,
    explanation:
      'Default fixture scope is function—fixture created and destroyed for EACH test function: Ensures test isolation (clean state per test). No shared state between tests. Can be parallelized safely. Example: @pytest.fixture or @pytest.fixture(scope="function") are equivalent. Override with scope parameter: scope="class", scope="module", scope="session". Why function default? Favors correctness (isolation) over performance. Most tests need isolated data. Explicit opt-in for shared fixtures (module/session scope).',
  },
  {
    id: 'fdd-mc-2',
    question:
      'When using yield in fixtures, when does the cleanup code (after yield) execute?',
    options: [
      'Immediately after the fixture is created, before any tests run',
      'After each test that uses the fixture, even if the test fails',
      'Only if the test passes successfully',
      'At the end of the entire test session, regardless of when fixture was created',
    ],
    correctAnswer: 1,
    explanation:
      'Cleanup code after yield runs after each test that uses the fixture, EVEN IF TEST FAILS: Guaranteed cleanup: yield ensures teardown runs. Failure safe: Runs even if test raises exception. Timing: Executes based on fixture scope (function scope → after each test, session scope → after all tests). Example: @pytest.fixture def db(): db = connect(); yield db; db.close() → db.close() runs after test, even if test fails. Alternative: request.addfinalizer() also guarantees cleanup. Critical for resources: Prevents leaks (connections, files, processes).',
  },
  {
    id: 'fdd-mc-3',
    question:
      'What happens when a function-scoped fixture depends on a session-scoped fixture?',
    options: [
      'pytest raises ScopeMismatch error—function scope cannot depend on session scope',
      'Both fixtures become session-scoped automatically',
      'It works fine—lower scope (function) can depend on higher scope (session)',
      'The session-scoped fixture is recreated for each test',
    ],
    correctAnswer: 2,
    explanation:
      'Lower scope can depend on higher scope—function-scoped fixture CAN depend on session-scoped: Works fine: Each test gets new function fixture, which uses same session fixture. Example: @pytest.fixture(scope="session") def app(): return App(); @pytest.fixture def client(app): return app.test_client() → client recreated per test, app created once. ScopeMismatch occurs in REVERSE: Session fixture depending on function fixture fails (session created once, but function fixture created per-test—conflict). Rule: Higher/equal scope can depend on higher/equal scope. Lower scope can depend on ANY scope. Benefit: Reuse expensive setup (session) while maintaining test isolation (function).',
  },
  {
    id: 'fdd-mc-4',
    question: 'What does the autouse=True parameter do in a fixture?',
    options: [
      'Makes the fixture run automatically for every test without explicitly requesting it',
      'Automatically detects when the fixture is needed based on test code',
      'Automatically runs setup and teardown in parallel for faster execution',
      'Automatically retries the fixture if it fails to initialize',
    ],
    correctAnswer: 0,
    explanation:
      "autouse=True makes fixture run automatically for EVERY test without explicit request: No parameter needed: Tests don't need to add fixture as parameter. Runs for all tests: In scope (function-scoped autouse runs for all tests in file, session-scoped for all tests in session). Use cases: Database cleanup before each test, logging setup, environment variable configuration. Example: @pytest.fixture(autouse=True) def clean_db(): truncate_all_tables() → runs before every test. Warning: Use sparingly—makes it less clear what fixtures each test uses. Not for detection/parallelization/retries.",
  },
  {
    id: 'fdd-mc-5',
    question:
      'What is the primary benefit of using fixture factories (fixtures that return factory functions)?',
    options: [
      'Fixture factories run faster than regular fixtures',
      'Fixture factories can create multiple instances with different parameters within a single test',
      'Fixture factories automatically clean up all created resources',
      'Fixture factories can be used across different test files without conftest.py',
    ],
    correctAnswer: 1,
    explanation:
      'Fixture factories allow creating MULTIPLE INSTANCES with different parameters in one test: Multiple objects: Factory can be called multiple times with different args. Flexibility: Default parameters with override capability. Example: @pytest.fixture def user_factory(db): def make_user(name="User", age=30): return User(name, age); return make_user → test_multiple_users(user_factory): alice = user_factory("Alice", 25); bob = user_factory("Bob", 35) → create 2 users with different data. Compare to regular fixture: Only returns one instance per test. Factory factories don\'t inherently run faster or clean up automatically (though you can implement cleanup tracking). Still need conftest.py for sharing across files.',
  },
];
