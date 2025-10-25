export const pytestPluginsExtensionsQuiz = [
  {
    id: 'ppe-q-1',
    question:
      'Design a CI/CD testing strategy using pytest plugins for a Django application with 5,000 tests. Address: (1) coverage requirements and enforcement, (2) parallel execution optimization, (3) timeout configuration for different test types, (4) integration with GitHub Actions, (5) performance benchmarking. Include specific plugin configurations and expected execution times.',
    sampleAnswer:
      'CI/CD pytest plugin strategy: (1) Coverage with pytest-cov: pytest.ini: [pytest] addopts = --cov=myapp --cov-fail-under=80 --cov-report=xml --cov-report=term-missing. Enforce 80% minimum coverage, fail CI if below. Branch coverage: --cov-branch. Exclude: .coveragerc to exclude migrations, tests, __init__.py. (2) Parallel with pytest-xdist: pytest -n auto splits tests across CPU cores. 5,000 tests serial: 50 minutes (0.6s/test). Parallel 8 cores: 7 minutes (8× speedup). Config: addopts = -n auto --dist loadscope. (3) Timeouts with pytest-timeout: Unit tests: @pytest.mark.timeout(5), Integration: @pytest.mark.timeout(30), E2E: @pytest.mark.timeout(300). Default: timeout = 60 in pytest.ini. (4) GitHub Actions: .github/workflows/test.yml: pytest --cov=src --cov-report=xml --junitxml=junit.xml -n auto. Upload coverage to Codecov. (5) Benchmarking with pytest-benchmark: Track performance regressions. pytest --benchmark-only for performance tests.',
    keyPoints: [
      'Coverage: --cov-fail-under=80, --cov-branch, exclude migrations/.coveragerc',
      'Parallel: -n auto, 8× speedup (50min → 7min for 5K tests)',
      'Timeouts: Unit 5s, Integration 30s, E2E 300s, default 60s',
      'CI: XML report for Codecov, junitxml for test results, parallel execution',
      'Benchmarking: pytest-benchmark for performance regression detection',
    ],
  },
  {
    id: 'ppe-q-2',
    question:
      'Compare pytest plugin options for async testing: pytest-asyncio vs pytest-trio vs pytest-aiohttp. Address: use cases, performance, fixture support, integration complexity, and when to choose each. Provide code examples.',
    sampleAnswer:
      'Async testing plugin comparison: (1) pytest-asyncio (most common): For asyncio-based code. @pytest.mark.asyncio for async tests. Async fixtures supported. Good for FastAPI, aiohttp, asyncpg. Example: @pytest.mark.asyncio async def test_api (async_client): response = await async_client.get("/users"); assert response.status == 200. (2) pytest-trio: For Trio async framework. @pytest.mark.trio decorator. Better structured concurrency than asyncio. Use when: App uses Trio, need nursery pattern. (3) pytest-aiohttp: Specifically for aiohttp apps. Provides aiohttp_client fixture. Example: async def test_endpoint (aiohttp_client): client = await aiohttp_client (app); resp = await client.get("/api"). Integration: aiohttp-specific features (middlewares, routes). Choose: pytest-asyncio for general async (FastAPI, asyncpg), pytest-aiohttp for aiohttp apps, pytest-trio for Trio framework.',
    keyPoints: [
      'pytest-asyncio: General asyncio, FastAPI/aiohttp/asyncpg, @pytest.mark.asyncio',
      'pytest-trio: Trio framework, structured concurrency, nursery pattern',
      'pytest-aiohttp: aiohttp-specific, aiohttp_client fixture, middleware testing',
      'Choose based on framework: asyncio → pytest-asyncio, Trio → pytest-trio, aiohttp → pytest-aiohttp',
      'Performance similar, differ in framework integration and fixture support',
    ],
  },
  {
    id: 'ppe-q-3',
    question:
      'Debug performance issues in a test suite using pytest plugins: tests take 45 minutes, goal is <10 minutes. Use pytest-xdist, pytest-cov, pytest-benchmark, and pytest-profiling. Address: identifying bottlenecks, parallelization strategy, coverage overhead, and optimization. Provide specific commands and expected improvements.',
    sampleAnswer:
      'Test performance optimization: (1) Identify bottlenecks with pytest-profiling: pytest --profile. Analyze: Database setup: 10 minutes (22%), Slow tests: 20 minutes (44%), Coverage: 5 minutes (11%), Actual execution: 10 minutes (22%). (2) Parallelize with pytest-xdist: pytest -n 8 (8 cores). Expected: 45 min / 8 = 5.6 min (but coverage overhead). Actual: 8 minutes (5.6× speedup). (3) Reduce coverage overhead: --cov only in CI: Local: pytest -n auto (no coverage, 6 min). CI: pytest --cov=src -n auto (coverage, 10 min). Or: pytest-cov-split for parallel coverage. (4) Optimize fixtures: session-scoped DB: @pytest.fixture (scope="session") for DB schema. Save 8 minutes (setup once vs per-test). (5) Mark slow tests: @pytest.mark.slow, run separately: pytest -m "not slow" (fast, 5 min daily). pytest -m slow (slow, 5 min nightly). Total: 45 min → 5 min daily + 5 min nightly (10× improvement).',
    keyPoints: [
      'Bottlenecks: pytest --profile shows DB setup 22%, slow tests 44%, coverage 11%',
      'Parallelize: -n 8 gives 5.6× speedup (45min → 8min)',
      'Coverage: Skip locally (6min), run in CI only (10min), or pytest-cov-split',
      'Fixtures: Session-scoped DB saves 8min (setup once vs per-test)',
      'Split tests: Fast (-m "not slow", 5min daily) + Slow (nightly 5min) = 10× faster',
    ],
  },
];
