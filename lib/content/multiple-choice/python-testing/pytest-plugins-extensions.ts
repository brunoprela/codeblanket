import { MultipleChoiceQuestion } from '@/lib/types';

export const pytestPluginsExtensionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ppe-mc-1',
    question: 'What does pytest-xdist do?',
    options: [
      'Distributes tests across multiple machines',
      'Runs tests in parallel using multiple CPU cores',
      'Creates test distributions for different Python versions',
      'Extends pytest with additional assertions',
    ],
    correctAnswer: 1,
    explanation:
      'pytest-xdist runs tests in parallel across multiple CPU cores: pytest -n 4 uses 4 cores, pytest -n auto auto-detects CPU count. Speedup: 8 cores = ~6-8× faster (not perfect 8× due to overhead). Example: 1000 tests × 0.5s = 500s serial, ~70s with 8 cores. Can distribute across machines with --dist, but primary use is local parallelization. Not for Python versions (use tox) or assertions. Essential for large test suites.',
  },
  {
    id: 'ppe-mc-2',
    question: 'What is the purpose of --cov-fail-under in pytest-cov?',
    options: [
      'Fails tests if coverage drops below threshold',
      'Only runs tests with coverage below threshold',
      'Generates coverage report only if above threshold',
      'Sets the minimum Python version for coverage',
    ],
    correctAnswer: 0,
    explanation:
      '--cov-fail-under=80 fails pytest if coverage < 80%: pytest --cov=src --cov-fail-under=80 → exit code 1 if coverage 79%, exit code 0 if 80%+. Use in CI to enforce coverage requirements: Prevent merges that reduce coverage, maintain quality standards. Example: [pytest] addopts = --cov-fail-under=80 in pytest.ini. Not for: running specific tests, conditional reports, or Python versions. Essential for maintaining test coverage in CI/CD.',
  },
  {
    id: 'ppe-mc-3',
    question: 'What does @pytest.mark.timeout(5) do?',
    options: [
      'Waits 5 seconds before starting the test',
      'Fails the test if it takes longer than 5 seconds',
      'Runs the test 5 times',
      'Sets the test priority to 5',
    ],
    correctAnswer: 1,
    explanation:
      '@pytest.mark.timeout(5) from pytest-timeout fails test if it exceeds 5 seconds: Prevents hanging tests from blocking CI, catches infinite loops/deadlocks, enforces performance requirements. Example: @pytest.mark.timeout(5) def test_fast(): slow_operation() # Fails if >5s. Global default: timeout = 30 in pytest.ini. Disable for specific test: @pytest.mark.timeout(0). Not for: delays, repetition, or priority. Essential for reliable CI pipelines.',
  },
  {
    id: 'ppe-mc-4',
    question: 'What is the mocker fixture provided by pytest-mock?',
    options: [
      'A fixture that creates mock HTTP servers',
      'A cleaner interface for unittest.mock patching',
      'A fixture that automatically mocks all imports',
      'A mock database connection',
    ],
    correctAnswer: 1,
    explanation:
      'mocker fixture provides cleaner mocking interface: def test_function (mocker): mock = mocker.patch("module.Service") vs @patch("module.Service") def test_function (mock_service). Benefits: No decorator needed, cleaner syntax, integrates with pytest fixtures. Example: mocker.patch(), mocker.Mock(), mocker.MagicMock(). Not for: HTTP servers (use pytest-httpserver), auto-mocking imports, or database connections. Makes mocking more Pythonic and maintainable.',
  },
  {
    id: 'ppe-mc-5',
    question: 'What does pytest-asyncio enable?',
    options: [
      'Automatically converts synchronous tests to async',
      'Provides @pytest.mark.asyncio decorator for testing async/await code',
      'Makes all fixtures async by default',
      'Runs tests asynchronously in parallel',
    ],
    correctAnswer: 1,
    explanation:
      'pytest-asyncio enables testing async/await code with @pytest.mark.asyncio: @pytest.mark.asyncio async def test_async(): result = await async_func(); assert result == expected. Also supports: Async fixtures (@pytest.fixture async def), async context managers, asyncio event loops. Not for: converting sync to async (must write async tests), making all fixtures async (opt-in), or parallelization (use pytest-xdist). Essential for testing FastAPI, aiohttp, asyncpg applications.',
  },
];
