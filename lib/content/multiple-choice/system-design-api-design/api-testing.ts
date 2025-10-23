/**
 * Multiple choice questions for API Testing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apitestingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'testing-q1',
    question: 'What is the purpose of contract testing for APIs?',
    options: [
      'To test database performance',
      'To verify the API matches its OpenAPI specification',
      'To test user authentication',
      'To measure API latency',
    ],
    correctAnswer: 1,
    explanation:
      'Contract testing ensures API responses match the documented OpenAPI spec (correct status codes, schema, required fields). Catches when implementation drift from spec. Tools like Pact or jest-openapi automate this.',
  },
  {
    id: 'testing-q2',
    question:
      'Why should integration tests mock external dependencies (payment gateways, email services)?',
    options: [
      'To make tests faster and more reliable',
      'To reduce server costs',
      "External services don't support testing",
      'Mocking is required by testing frameworks',
    ],
    correctAnswer: 0,
    explanation:
      'Mocking external dependencies makes tests: (1) Fast (no network calls), (2) Reliable (no external failures), (3) Repeatable (no side effects like charging cards). Test YOUR code, not third-party services. Use real services in E2E tests only.',
  },
  {
    id: 'testing-q3',
    question: 'In the test pyramid, why should unit tests outnumber E2E tests?',
    options: [
      'Unit tests are easier to write',
      'Unit tests are fast, cheap, and isolate failures; E2E tests are slow and expensive',
      'E2E tests are less important',
      'Unit tests provide better coverage',
    ],
    correctAnswer: 1,
    explanation:
      'Test pyramid: Many unit tests (fast, cheap, pinpoint failures) → Some integration tests (medium) → Few E2E tests (slow, expensive, brittle). Unit tests run in ms, E2E tests in minutes. Balance speed and confidence.',
  },
  {
    id: 'testing-q4',
    question: 'What should be the threshold for "successful" in a load test?',
    options: [
      'All requests succeed with 0% errors',
      'p95 latency < target AND error rate < acceptable threshold (e.g., 1%)',
      'Average response time is low',
      'Server CPU stays below 100%',
    ],
    correctAnswer: 1,
    explanation:
      "Load test thresholds: p95 latency < target (e.g., 500ms) AND error rate < 1-5%. Some errors acceptable under load. Don't just track averages (misleading). p95/p99 show real user experience.",
  },
  {
    id: 'testing-q5',
    question:
      'Why test both successful requests AND error cases (400, 500 responses)?',
    options: [
      'To increase test coverage percentage',
      'To ensure API handles errors gracefully and returns proper error responses',
      'Because testing frameworks require it',
      'To slow down the build process',
    ],
    correctAnswer: 1,
    explanation:
      "Testing error cases ensures: (1) Proper status codes (400 vs 500), (2) Clear error messages, (3) No crashes, (4) Security (don't leak sensitive info). Happy path tests are insufficient. Error handling is critical for UX.",
  },
];
