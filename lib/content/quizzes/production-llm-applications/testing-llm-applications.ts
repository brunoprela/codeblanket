export const testingLlmApplicationsQuiz = [
  {
    id: 'pllm-q-11-1',
    question:
      'Design a comprehensive testing strategy for LLM applications covering unit tests, integration tests, load tests, and quality evaluation. How do you test non-deterministic outputs without excessive costs?',
    sampleAnswer:
      'Unit tests (mock LLMs): Use unittest.mock or pytest fixtures to mock OpenAI responses, test business logic without API calls (auth, rate limiting, caching), test error handling paths, snapshot testing for prompt construction. Cost: $0. Integration tests (real API, limited): Test critical paths with real LLM calls, use temperature=0 for reproducibility, limit to 10-20 test cases, run on PR merge only, cache responses for subsequent runs. Cost: ~$1 per run. Load tests: Use Locust/k6 with mocked LLMs to simulate traffic patterns, test rate limiting, queue handling, database performance, identify bottlenecks. Run weekly. Cost: $0 (mocked). Quality evaluation: Eval harness with 100 diverse prompts, compare outputs using LLM-as-judge (GPT-4 rates relevance/quality 1-5), track metrics over time (average score, p95 latency, cost per eval), regression testing on prompt changes. Run daily on staging. Cost: ~$5/day. Implementation: @pytest.fixture mock_openai with configurable responses, pytest.mark.slow for expensive tests, pytest.mark.integration for real API, separate test/prod API keys, CI/CD runs unit always, integration on merge, load weekly. Monitor test costs in dashboard, alert if >$50/day.',
    keyPoints: [
      'Layered testing: unit (mocked, frequent), integration (real API, selective), load (mocked, weekly)',
      'Temperature=0 and caching for reproducible integration tests',
      'LLM-as-judge for quality evaluation with cost monitoring',
    ],
  },
  {
    id: 'pllm-q-11-2',
    question:
      'How would you test rate limiting, error handling, and retry logic in an LLM application? Provide specific test cases and explain how to simulate various failure scenarios.',
    sampleAnswer:
      'Rate limiting tests: 1) Make requests at exact limit (should succeed), 2) Exceed limit (should get 429), 3) Wait for reset (should succeed again), 4) Burst test (send 100 requests rapidly, verify burst handling), 5) Multiple users (ensure isolation), 6) Verify headers (X-RateLimit-*). Mock time.time() to simulate time passing. Error handling tests: 1) Mock openai.error.RateLimitError (verify retry with backoff), 2) Mock openai.error.APIError (verify retry), 3) Mock openai.error.InvalidRequestError (verify no retry), 4) Mock timeout (verify fallback), 5) Mock OpenAI completely down (verify circuit breaker). Use @patch("openai.ChatCompletion.create") to inject failures. Retry logic tests: 1) First call fails, second succeeds (verify retried once), 2) All calls fail (verify max retries then fails), 3) Measure backoff timing (verify exponential), 4) Verify jitter (random variation), 5) Check idempotency (retries dont duplicate side effects). Simulate failures: Create test fixtures with configurable failure rates, use pytest.mark.parametrize for multiple scenarios, integration tests with chaos engineering (randomly inject failures), load tests with deliberate overload. Example: @pytest.fixture def failing_openai (failures=2): call_count = 0; def side_effect(*args): nonlocal call_count; call_count += 1; if call_count <= failures: raise RateLimitError(); return success_response; mock.side_effect = side_effect.',
    keyPoints: [
      'Comprehensive rate limit testing including burst and isolation',
      'Mock various exception types to test error handling paths',
      'Measure retry timing and verify exponential backoff with jitter',
    ],
  },
  {
    id: 'pllm-q-11-3',
    question:
      'Explain load testing strategies for LLM applications. What metrics would you measure, how would you simulate realistic user behavior, and how do you identify bottlenecks?',
    sampleAnswer:
      'Load testing with Locust: Simulate user behavior (login, start conversation, send 5 messages with 2-5s think time, rate responses), gradually ramp up from 10 to 1000 users over 10min, maintain peak for 10min, measure throughout. Metrics: Requests per second (target: >100), p50/p95/p99 latency (target: <2s/<5s/<10s), error rate (target: <1%), throughput (MB/s), database connections used, queue depth, cache hit rate, CPU/memory utilization. Realistic simulation: Multiple user types (free: 1 req/min, pro: 10 req/min), varied message lengths (50-500 tokens), include think time between requests, concurrent conversations, mix of cache hits and misses. Identify bottlenecks: Profile during load test, check which services have highest latency, monitor database slow queries, track memory leaks (increasing RAM over time), measure cache effectiveness, analyze queue buildup. Tools: Locust for load generation, Prometheus for metrics, Grafana for visualization, profiling with py-spy. Scenarios: Normal load (steady 50 req/s), peak load (bursts to 200 req/s), sustained high load (100 req/s for 1hr), failure scenarios (Redis down, database slow). Fix bottlenecks: Scale horizontally (more workers), optimize database queries, increase cache, add rate limiting, upgrade instance sizes.',
    keyPoints: [
      'Realistic user behavior simulation with gradual ramp-up and think time',
      'Comprehensive metrics: latency percentiles, error rate, resource utilization',
      'Systematic bottleneck identification through profiling and metrics analysis',
    ],
  },
];
