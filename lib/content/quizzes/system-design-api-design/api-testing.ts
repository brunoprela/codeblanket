/**
 * Quiz questions for API Testing section
 */

export const apitestingQuiz = [
  {
    id: 'testing-d1',
    question:
      'Design a comprehensive testing strategy for a payment API. Include unit, integration, contract, security, and load tests.',
    sampleAnswer: `Complete testing strategy for payment API:

**1. Unit Tests** (Jest): Validation, hashing, token generation
**2. Integration Tests** (Supertest): API endpoints with mocked payment gateway
**3. Contract Tests** (Pact): Verify OpenAPI spec compliance
**4. Security Tests**: SQL injection, auth bypass, rate limiting
**5. Load Tests** (k6): 1000 req/s with p95 < 500ms
**6. E2E Tests** (Stripe test mode): Full payment flow
**7. Monitoring**: Real-user monitoring in production`,
    keyPoints: [
      'Unit tests for business logic (validation, calculations)',
      'Integration tests with mocked external services',
      'Security tests for injection, auth, rate limiting',
      'Load tests to verify performance under stress',
      'E2E tests in sandbox environment with real gateway',
    ],
  },
  {
    id: 'testing-d2',
    question:
      'Your CI/CD pipeline takes 45 minutes to run tests. How would you optimize it?',
    sampleAnswer: `Test optimization strategy:

**1. Parallelize**: Run tests in parallel (10 workers)
**2. Split by type**: Unit (2 min) → Integration (10 min) → E2E (30 min)
**3. Fail fast**: Run unit tests first, stop if failing
**4. Cache dependencies**: Docker layer caching, npm cache
**5. Selective testing**: Only test changed code in PR
**6. Database snapshots**: Reuse DB state between tests
**7. Mock external services**: Reduce network calls
**8. Remove flaky tests**: Fix or delete unstable tests

Result: 45 min → 10 min pipeline`,
    keyPoints: [
      'Parallelize tests across multiple workers',
      'Run fast tests first to fail early',
      'Cache dependencies and database snapshots',
      'Mock external services to eliminate network delays',
      'Use selective testing for PRs (only test changed code)',
    ],
  },
  {
    id: 'testing-d3',
    question:
      'Compare unit vs integration vs E2E tests for a REST API. What percentage of each should you have?',
    sampleAnswer: `Test distribution for REST API:

**Unit Tests (70%)**:
- Fast (ms), cheap, many
- Test: validation, business logic, utilities
- Example: validateEmail(), calculateTax()

**Integration Tests (20%)**:
- Medium speed (seconds), some
- Test: API endpoints, database queries
- Example: POST /users returns 201

**E2E Tests (10%)**:
- Slow (minutes), expensive, few
- Test: Critical user flows
- Example: Complete checkout flow

**Rationale**:
- Unit tests catch most bugs quickly
- Integration tests verify components work together
- E2E tests ensure critical flows work end-to-end
- Balance speed (unit) vs confidence (E2E)

**Real Example (Stripe)**:
- 10,000+ unit tests (seconds)
- 500+ integration tests (minutes)
- 50+ E2E tests (hours)

Don't invert pyramid (too many E2E tests = slow CI).`,
    keyPoints: [
      'Unit tests (70%): Fast, many, test individual functions',
      'Integration tests (20%): Medium, some, test API endpoints',
      'E2E tests (10%): Slow, few, test critical user flows',
      'Balance speed (unit) with confidence (E2E)',
      'Inverted pyramid (many E2E) causes slow, brittle CI',
    ],
  },
];
