/**
 * Multiple choice questions for Circuit Breaker Pattern section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const circuitbreakerMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-circuit-1',
    question: 'What is the primary purpose of a circuit breaker pattern?',
    options: [
      'To retry failed requests automatically',
      'To prevent cascading failures by failing fast when a service is down',
      'To load balance requests across multiple service instances',
      'To cache responses from slow services',
    ],
    correctAnswer: 1,
    explanation:
      "Circuit breaker prevents cascading failures by detecting when a service is failing and failing fast (returning error immediately) instead of waiting for timeouts. This prevents thread pool exhaustion in the calling service. When a downstream service fails repeatedly, the circuit \"opens\" and stops calling it, allowing the calling service to stay healthy. Option 1 is wrong (that's retry pattern). Option 3 is wrong (that's load balancer). Option 4 is wrong (that's caching layer). Circuit breaker is specifically about preventing cascading failures.",
  },
  {
    id: 'mc-circuit-2',
    question:
      'A circuit breaker is in OPEN state. What happens when a new request arrives?',
    options: [
      'Request is queued until service recovers',
      'Request is retried with exponential backoff',
      'Request fails immediately without calling the service',
      'Request is sent to service normally',
    ],
    correctAnswer: 2,
    explanation:
      "When circuit is OPEN, requests fail immediately (fail fast) without calling the downstream service. This prevents wasting resources (threads, connections) on calls that will likely fail. The circuit stays open for a configured timeout period, then transitions to HALF_OPEN to test if service recovered. Option 1 is not automatic (you could implement this as a fallback). Option 2 is wrong (no retries when circuit is open). Option 4 is wrong (that's CLOSED state).",
  },
  {
    id: 'mc-circuit-3',
    question:
      'Your Order Service calls Payment Service (critical) and Recommendation Service (non-critical). How should you configure circuit breakers?',
    options: [
      'Use one circuit breaker for both services',
      "Don't use circuit breakers for critical services",
      'Use separate circuit breakers: lenient for Payment (higher threshold, retry sooner), stricter for Recommendations (lower threshold, wait longer)',
      'Only use circuit breaker for Payment Service',
    ],
    correctAnswer: 2,
    explanation:
      'Use separate circuit breakers per dependency (bulkheading pattern). Configure based on criticality: Payment Service gets more lenient settings (higher failure threshold, shorter reset timeout) because we want to give it more chances before opening. Recommendations get stricter settings because failures are less critical. Option 1 is wrong (email failure would open circuit for payment too). Option 2 is wrong (critical services especially need circuit breakers). Option 4 is incomplete (recommendations also benefit from circuit breaker).',
  },
  {
    id: 'mc-circuit-4',
    question:
      'What is the difference between circuit breaker HALF_OPEN and CLOSED states?',
    options: [
      'No difference, they are the same',
      'HALF_OPEN allows limited test requests to check recovery; CLOSED allows all requests normally',
      'HALF_OPEN blocks all requests; CLOSED allows all requests',
      'HALF_OPEN is faster than CLOSED',
    ],
    correctAnswer: 1,
    explanation:
      'HALF_OPEN is a testing state after circuit has been OPEN. It allows limited test requests through to check if the downstream service has recovered. If tests succeed, circuit transitions to CLOSED (normal operation, all requests allowed). If tests fail, circuit goes back to OPEN. This prevents thundering herd problem (all requests hitting recovering service at once). CLOSED is normal operation where all requests pass through while monitoring for failures. Option 1 is wrong (very different states). Option 3 confuses HALF_OPEN with OPEN. Option 4 makes no sense.',
  },
  {
    id: 'mc-circuit-5',
    question: 'When should you NOT use a circuit breaker?',
    options: [
      'When calling external payment gateway (critical service)',
      'When calling internal recommendation service across network',
      'For in-process function calls within the same service',
      'When calling notification service',
    ],
    correctAnswer: 2,
    explanation:
      "Don't use circuit breakers for in-process function calls within the same service. Circuit breakers are for network calls between services to prevent cascading failures due to timeouts and resource exhaustion. Local function calls don't have these issues. Options 1, 2, and 4 are all valid use cases for circuit breakers (external services, internal services across network, and non-critical services all benefit from circuit breakers). Only skip circuit breakers for local calls.",
  },
];
