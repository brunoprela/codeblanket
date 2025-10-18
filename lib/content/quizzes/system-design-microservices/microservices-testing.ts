/**
 * Quiz questions for Microservices Testing section
 */

export const microservicestestingQuiz = [
  {
    id: 'q1-testing',
    question:
      'Why are contract tests important in microservices? How do they differ from integration tests?',
    sampleAnswer:
      'Contract tests verify service boundaries without running both services together. Consumer (Order Service) defines expected contract (what it needs from Payment Service), provider (Payment Service) verifies it can fulfill that contract. Benefits: (1) Catch breaking changes before deployment, (2) No need to run both services (faster), (3) Consumer-driven (knows what it needs), (4) Automated boundary testing. Integration tests test service WITH its dependencies running (database, message queue). Contract tests test service boundaries (API contracts) WITHOUT running the other service. Example: Pact lets Order Service define "when I call POST /charge with {amount: 50}, I expect {status: SUCCESS}". Payment Service verifies it fulfills this. If Payment changes response format, contract test fails.',
    keyPoints: [
      'Contract tests verify service API boundaries',
      'Consumer defines contract, provider verifies',
      'Catches breaking changes before deployment',
      "Doesn't require running both services (vs integration)",
      'Tool: Pact for consumer-driven contract testing',
    ],
  },
  {
    id: 'q2-testing',
    question:
      'How do you test asynchronous event-driven communication between microservices?',
    sampleAnswer:
      'Testing async communication requires polling/waiting for eventual consistency. Approach: (1) Publish event to message bus, (2) Poll consumer service for expected state change with timeout, (3) Verify outcome. Example: Order Service publishes OrderCreated event → wait for Email Service to send email → verify email sent. Use helper function waitFor() that polls condition every 100ms up to 5s timeout. Challenges: (1) Timing - need generous timeouts, (2) Flakiness - network delays, (3) Test isolation - clean up events between tests. Alternative: Mock message bus for faster testing, but sacrifice realism. For integration tests, use real message bus (RabbitMQ/Kafka in Docker) with test queues.',
    keyPoints: [
      'Async testing requires polling for eventual consistency',
      'Publish event → wait for outcome → verify',
      'Use waitFor() helper with timeout (e.g., 5 seconds)',
      'Trade-off: real message bus (slow, realistic) vs mocks (fast, less realistic)',
      'Clean up events/queues between tests for isolation',
    ],
  },
  {
    id: 'q3-testing',
    question:
      'What is the testing pyramid for microservices? Why keep E2E tests minimal?',
    sampleAnswer:
      'Testing pyramid (bottom to top): (1) Unit tests (most) - fast, isolated, cheap, high coverage, (2) Integration tests (more) - service + dependencies (DB, cache), slower but still manageable, (3) Contract tests (some) - service boundaries, faster than E2E, (4) E2E tests (few) - full system, only critical paths. Keep E2E minimal because: (1) Slow - seconds to minutes per test (vs milliseconds for unit), (2) Flaky - network issues, timing problems, (3) Expensive - require all services running, (4) Hard to debug - failure could be anywhere in system. Instead, push testing down: more unit tests, integration tests for each service, contract tests for boundaries. E2E only for critical happy paths (checkout flow) and major failure scenarios.',
    keyPoints: [
      'Pyramid: many unit tests, fewer integration, some contract, minimal E2E',
      'E2E tests are slow, flaky, expensive, hard to debug',
      'Push testing down the pyramid for faster feedback',
      'E2E only for critical paths and major failures',
      'Contract tests reduce need for E2E (verify boundaries)',
    ],
  },
];
