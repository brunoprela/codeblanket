export const integrationTestingQuiz = [
  {
    id: 'it-q-1',
    question:
      'Design integration testing strategy for microservices: Service A (FastAPI) → Service B (Django) → Database. Address: service startup, cross-service communication testing, database state management, Docker orchestration, and CI/CD integration.',
    sampleAnswer:
      'Microservices integration testing: (1) Docker Compose startup: docker-compose.test.yml with all services (service_a, service_b, postgres, redis). Health checks: wait_for_healthy before tests. (2) Cross-service communication: Test Service A → B: response = service_a_client.post("/process", json={...}); verify Service B received request via database or logs. Use real HTTP calls, not mocks. (3) Database state: Shared test database, transaction rollback per test or truncate tables. Fixtures: @pytest.fixture(scope="session") def test_db() for schema, function-scoped for data. (4) Docker orchestration: pytest.fixture starts docker-compose, waits for health, runs tests, docker-compose down. (5) CI/CD: GitHub Actions runs docker-compose up, pytest --maxfail=1, docker-compose down. Parallel: Run service tests separately (unit), integration sequentially.',
    keyPoints: [
      'Docker Compose: All services in docker-compose.test.yml, health checks before tests',
      'Cross-service: Real HTTP calls between services, verify via database/logs',
      'Database: Shared test DB, transaction rollback or truncate per test',
      'Orchestration: Fixture starts docker-compose, wait_for_healthy, cleanup',
      'CI/CD: docker-compose up → pytest → down, sequential integration tests',
    ],
  },
];
