import { MultipleChoiceQuestion } from '@/lib/types';

export const integrationTestingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'it-mc-1',
    question: 'What is the main difference between unit and integration tests?',
    options: [
      'Unit tests are faster because they mock all dependencies',
      'Integration tests verify multiple components work together with real dependencies',
      'Unit tests require database, integration tests do not',
      'Integration tests cannot use fixtures',
    ],
    correctAnswer: 1,
    explanation:
      'Integration tests use real dependencies and test component interactions: Unit: Mock database, test single function (10ms). Integration: Real database, test API + DB (100ms-1s). Example: Unit tests add() function in isolation. Integration tests POST /users endpoint → inserts to real database → verifies data. Purpose: Unit catches logic bugs, integration catches interaction bugs. Integration tests are slower (real I/O) but catch issues unit tests miss. Both use fixtures.',
  },
  {
    id: 'it-mc-2',
    question: 'Why use Docker Compose for integration tests?',
    options: [
      'Docker Compose makes tests run faster than native installations',
      'Docker Compose provides isolated, reproducible test environments',
      'Docker Compose automatically generates test data',
      'Docker Compose is required by pytest',
    ],
    correctAnswer: 1,
    explanation:
      'Docker Compose ensures consistent, isolated environments: docker-compose.test.yml defines postgres, redis, services. Benefits: Same environment locally and CI, clean state per run, no "works on my machine". Example: docker-compose up → tests run → docker-compose down. Isolation prevents tests from affecting development databases. Reproducibility: Same postgres version, configuration across all environments. Not faster (overhead from Docker), doesn\'t generate data, not required (can use local services).',
  },
  {
    id: 'it-mc-3',
    question:
      'How should integration tests handle database state between tests?',
    options: [
      'Share database state across all tests for speed',
      'Use transaction rollback or truncate tables to ensure clean state',
      'Create new database for each test',
      'Database state does not matter in integration tests',
    ],
    correctAnswer: 1,
    explanation:
      'Clean state between tests essential: Transaction rollback (fast): Begin transaction → test → rollback. Truncate (slower): Delete all rows between tests. Example: @pytest.fixture def db_session(engine): connection.begin(); yield session; transaction.rollback(). Ensures test isolation: Test 1 creates user "alice", doesn\'t affect test 2. New database per test too slow (10s per test). Shared state causes flaky tests (test order matters). Database state critical—dirty state = false positives/negatives.',
  },
  {
    id: 'it-mc-4',
    question:
      'What marker is commonly used to tag integration tests separately from unit tests?',
    options: [
      '@pytest.mark.slow',
      '@pytest.mark.integration',
      '@pytest.mark.e2e',
      '@pytest.mark.docker',
    ],
    correctAnswer: 1,
    explanation:
      '@pytest.mark.integration standard for marking integration tests: @pytest.mark.integration def test_api_with_db(): .... Run selectively: pytest -m integration (only integration), pytest -m "not integration" (only unit). Why separate: Unit tests fast (2 min), integration slow (10 min). Run unit on every commit, integration before merge. Could use @pytest.mark.slow but less semantically clear. @pytest.mark.e2e for end-to-end (subset of integration). Not docker-specific marker.',
  },
  {
    id: 'it-mc-5',
    question:
      'When testing external API integrations, what is VCR.py used for?',
    options: [
      'Recording and replaying HTTP interactions for reproducible tests',
      'Mocking all HTTP requests automatically',
      'Measuring API response times',
      'Validating API schemas',
    ],
    correctAnswer: 0,
    explanation:
      'VCR.py records/replays HTTP interactions: First run: Makes real API call, records response to cassette file. Subsequent runs: Replays recorded response (no actual API call). Example: @vcr.use_cassette("github_api.yaml") def test_github(): response = requests.get("https://api.github.com/users/octocat"). Benefits: Fast (no actual API calls), reproducible (same response), no rate limits, works offline. Not full mocking (records real responses), not for timing (uses recorded), not schema validation. Essential for reliable external API testing.',
  },
];
