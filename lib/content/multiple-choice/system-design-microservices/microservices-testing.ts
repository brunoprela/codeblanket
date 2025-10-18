/**
 * Multiple choice questions for Microservices Testing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicestestingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-testing-1',
    question: 'What is the main purpose of contract testing in microservices?',
    options: [
      'To test database performance',
      'To verify service API boundaries without running both services together',
      'To replace all integration tests',
      'To test UI components',
    ],
    correctAnswer: 1,
    explanation:
      "Contract testing (e.g., Pact) verifies that a provider service fulfills the API contract that a consumer service expects, without actually running both services together. Consumer defines expectations, generates contract file, provider verifies it meets the contract. This catches breaking changes before deployment and is faster than integration tests requiring both services. Option 1 is unrelated. Option 3 is wrong (contract tests complement, don't replace integration tests). Option 4 is wrong (that's UI testing).",
  },
  {
    id: 'mc-testing-2',
    question:
      'Why should you minimize end-to-end (E2E) tests in microservices?',
    options: [
      'E2E tests are not important',
      'E2E tests are slow, flaky, expensive to maintain, and hard to debug',
      "E2E tests don't test the full system",
      'E2E tests are only for monoliths',
    ],
    correctAnswer: 1,
    explanation:
      "E2E tests require all services running, making them slow (seconds to minutes), flaky (network issues, timing problems), expensive (infrastructure costs, maintenance), and hard to debug (failure could be anywhere). Instead, follow the testing pyramid: push most testing down to unit/integration/contract tests, use E2E only for critical happy paths. Option 1 is wrong (E2E tests ARE important, just use sparingly). Option 3 is wrong (they do test full system, that's why they're slow). Option 4 is wrong (E2E tests apply to both).",
  },
  {
    id: 'mc-testing-3',
    question: 'How do you test asynchronous event-driven communication?',
    options: [
      "You can't test async communication",
      'Only with manual testing',
      'Publish event, poll for expected outcome with timeout',
      'Synchronous tests work fine',
    ],
    correctAnswer: 2,
    explanation:
      "Testing async communication requires: (1) Publish event to message bus, (2) Poll consumer service for expected state change (e.g., every 100ms), (3) Verify outcome within timeout (e.g., 5 seconds). This accounts for eventual consistency. Example: publish OrderCreated → wait for email sent → verify email. Use helper functions like waitFor() with polling and timeout. Option 1 is defeatist. Option 2 ignores automation. Option 4 doesn't work (need to wait for eventual consistency).",
  },
  {
    id: 'mc-testing-4',
    question: 'What is a component test in microservices?',
    options: [
      'Testing UI components',
      'Testing a single function',
      'Testing an entire service in isolation with mocked external dependencies',
      'Testing hardware components',
    ],
    correctAnswer: 2,
    explanation:
      'Component testing tests an entire microservice (all endpoints, business logic) in isolation with external services mocked. Start the service, mock dependencies (other services, external APIs), send requests via API, verify responses. This is faster than E2E (no need to start all services) but more comprehensive than unit tests (tests full service). Example: start Order Service, mock Payment/Inventory services, test order creation flow. Option 1 is UI testing. Option 2 is unit testing. Option 4 is hardware testing.',
  },
  {
    id: 'mc-testing-5',
    question: 'What is chaos testing (chaos engineering)?',
    options: [
      'Testing without any plan',
      'Intentionally injecting failures (kill services, add latency) to verify system resilience',
      'Testing during chaotic deployments',
      'Random testing without assertions',
    ],
    correctAnswer: 1,
    explanation:
      'Chaos testing (chaos engineering) intentionally injects failures into the system to verify resilience: kill service instances, inject network latency, fill disk space, corrupt data. Verify system degrades gracefully, recovers automatically, and alerts work. Tools: Chaos Monkey (Netflix), Gremlin. Example: kill Payment Service → Order Service should still work (create orders as PENDING) → Payment Service restarts → orders process. Option 1 is wrong (chaos testing is very planned). Options 3 and 4 misunderstand the concept.',
  },
];
