export const testingAsyncCodeQuiz = [
  {
    id: 'tac-q-1',
    question:
      'Design testing strategy for FastAPI application with async database (asyncpg) and external API calls (aiohttp). Address: async fixtures, mocking async operations, testing concurrent requests, error handling, and performance.',
    sampleAnswer:
      'FastAPI async testing: (1) Async fixtures: @pytest.fixture async def async_db(): async with create_async_engine() as engine: yield engine. @pytest.fixture async def async_client (app): async with AsyncClient (app=app, base_url="http://test") as client: yield client. (2) Mocking: Use AsyncMock for external APIs. mock_api = mocker.patch("myapp.fetch_external", new_callable=AsyncMock); mock_api.return_value = {"data": "mocked"}. (3) Concurrent requests: @pytest.mark.asyncio async def test_concurrent(): results = await asyncio.gather(*[client.get("/api") for _ in range(10)]); assert all (r.status_code == 200 for r in results). (4) Errors: Test timeouts, connection errors, retries. with pytest.raises (asyncio.TimeoutError): await asyncio.wait_for (slow_operation(), 1.0). (5) Performance: Use pytest-benchmark with async: async def test_perf (benchmark): await benchmark.pedantic (async_operation, rounds=100).',
    keyPoints: [
      'Async fixtures: @pytest.fixture async def, async context managers, yield in async',
      'Mocking: AsyncMock for async functions, new_callable=AsyncMock in mocker.patch',
      'Concurrent: asyncio.gather for multiple requests, test parallelism and race conditions',
      'Errors: Test timeouts (asyncio.wait_for), exceptions, connection failures',
      'Performance: pytest-benchmark with async, measure concurrent throughput',
    ],
  },
];
