import { MultipleChoiceQuestion } from '@/lib/types';

export const testingAsyncCodeMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'tac-mc-1',
    question: 'What is required to test an async function with pytest?',
    options: [
      'No special requirements, pytest handles async automatically',
      'Install pytest-asyncio and use @pytest.mark.asyncio decorator',
      'Convert async functions to sync before testing',
      'Use unittest.TestCase with async methods',
    ],
    correctAnswer: 1,
    explanation:
      "pytest-asyncio required for async testing: pip install pytest-asyncio, then @pytest.mark.asyncio async def test_async(): result = await async_func(); assert result. Without decorator, pytest treats async function as regular function (doesn't await). Config: asyncio_mode = auto in pytest.ini auto-detects async tests. Not automatic, can't convert to sync (loses concurrency), unittest.TestCase doesn't support async well. Essential for FastAPI, aiohttp, asyncpg testing.",
  },
  {
    id: 'tac-mc-2',
    question: 'How do you mock an async function?',
    options: [
      'Use regular Mock(), same as synchronous functions',
      'Use AsyncMock from unittest.mock',
      'Async functions cannot be mocked',
      'Use @patch decorator with async=True parameter',
    ],
    correctAnswer: 1,
    explanation:
      'AsyncMock for async functions: from unittest.mock import AsyncMock; mock = AsyncMock(); mock.return_value = "result"; await mock(). Regular Mock doesn\'t work: await mock() fails. With mocker: mock_func = mocker.patch("module.async_func", new_callable=AsyncMock). Example: @pytest.mark.asyncio async def test(): mock_api = AsyncMock(); mock_api.return_value = {"data": "test"}; result = await fetch_with_api (mock_api). AsyncMock essential for testing async integrations.',
  },
  {
    id: 'tac-mc-3',
    question: 'What does asyncio.gather() do in tests?',
    options: [
      'Waits for all async operations to complete sequentially',
      'Runs multiple async operations concurrently and collects results',
      'Cancels all pending async operations',
      'Converts async operations to synchronous',
    ],
    correctAnswer: 1,
    explanation:
      "asyncio.gather() runs async operations concurrently: results = await asyncio.gather (fetch(1), fetch(2), fetch(3)) → runs all 3 fetches in parallel, returns [result1, result2, result3]. Use in tests: Test concurrent behavior, verify race conditions, test parallelism. Example: @pytest.mark.asyncio async def test_concurrent(): results = await asyncio.gather(*[api_call() for _ in range(10)]); assert len (results) == 10. Not sequential (that's for loops), not cancellation (asyncio.cancel), not conversion. Essential for testing async performance.",
  },
  {
    id: 'tac-mc-4',
    question: 'How do you create an async fixture in pytest?',
    options: [
      '@pytest.fixture def fixture(): return async_value',
      '@pytest.fixture async def fixture(): return await async_value',
      'Async fixtures are not supported in pytest',
      '@pytest.async_fixture def fixture(): return async_value',
    ],
    correctAnswer: 1,
    explanation:
      "Async fixtures use async def: @pytest.fixture async def async_client(): async with aiohttp.ClientSession() as session: yield session. Usage: @pytest.mark.asyncio async def test (async_client): result = await async_client.get(...). Can await inside fixture, yield for cleanup, compose with other async fixtures. Not sync def (can't await), not @pytest.async_fixture (doesn't exist). Pattern: @pytest.fixture async def with yield for cleanup. Essential for async test setup.",
  },
  {
    id: 'tac-mc-5',
    question: 'What is the purpose of asyncio.wait_for() in tests?',
    options: [
      'Waits indefinitely for an async operation to complete',
      'Sets a timeout and raises TimeoutError if exceeded',
      'Waits for multiple operations to complete',
      'Converts async operations to blocking calls',
    ],
    correctAnswer: 1,
    explanation:
      "asyncio.wait_for() enforces timeout: with pytest.raises (asyncio.TimeoutError): await asyncio.wait_for (slow_operation(), timeout=1.0) → raises if >1s. Use in tests: Verify timeout handling, test slow operations fail fast, ensure responsiveness. Example: result = await asyncio.wait_for (api_call(), 5.0) → max 5 seconds. Not indefinite (that's await alone), not multiple ops (that's gather), not blocking (still async). Essential for testing timeout behavior.",
  },
];
