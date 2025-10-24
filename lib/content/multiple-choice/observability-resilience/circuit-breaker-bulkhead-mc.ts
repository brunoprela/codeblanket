/**
 * Multiple choice questions for Circuit Breaker & Bulkhead Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const circuitBreakerBulkheadMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the primary purpose of the Circuit Breaker pattern?',
    options: [
      'To break electrical circuits',
      'To prevent cascading failures by failing fast when a dependency is unhealthy',
      'To route traffic between services',
      'To encrypt network traffic',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit Breaker prevents cascading failures by failing fast when a dependency is down, instead of waiting for timeouts that exhaust resources. Example: Database down → Circuit breaker opens after 5 failures → New requests fail immediately (50ms) instead of waiting 30s timeout → Threads freed immediately → API stays up → Auto-tests recovery every 30s. Without circuit breaker: All threads blocked waiting → Thread pool exhausted → API crashes → Complete outage.',
  },
  {
    id: 'mc2',
    question: 'What are the three states of a circuit breaker?',
    options: [
      'Open, Closed, Broken',
      'Closed (normal), Open (failing fast), Half-Open (testing recovery)',
      'Active, Inactive, Standby',
      'Running, Stopped, Paused',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breaker has three states: CLOSED (normal operation, requests pass through, count failures), OPEN (fail immediately without calling dependency, wait for timeout), HALF-OPEN (test if dependency recovered by allowing limited requests through - if success → close, if failure → re-open). State transition: Closed → (failures > threshold) → Open → (timeout expires) → Half-Open → (success) → Closed OR (failure) → Open.',
  },
  {
    id: 'mc3',
    question: 'What is the Bulkhead pattern?',
    options: [
      'A pattern for database partitioning',
      "Isolating resources into separate pools so failure in one doesn't exhaust resources for others",
      'A load balancing strategy',
      'A caching technique',
    ],
    correctAnswer: 1,
    explanation:
      'Bulkhead pattern isolates resources (thread pools, connections) into separate compartments, like ship bulkheads prevent entire ship from flooding. Example: Without bulkheads: Shared 50-thread pool → Payment API (slow) uses 45 threads → Search API starved (0 threads left) → Both fail. With bulkheads: Payment pool: 20 threads, Search pool: 10 threads → Payment slowness isolated to its pool → Search unaffected. Prevents one failing service from exhausting resources for all services.',
  },
  {
    id: 'mc4',
    question: 'When should liveness checks include database connectivity?',
    options: [
      'Always include database in liveness',
      'Never include database in liveness (use readiness instead)',
      'Only for read-only databases',
      'Only during deployments',
    ],
    correctAnswer: 1,
    explanation:
      "NEVER check database in liveness probe because: Liveness failure → Restart container. Database down → Liveness fails → Container restarts → Database still down → Infinite restart loop! Database should be checked in READINESS probe: Database down → Readiness fails → Remove from load balancer (don't restart) → Database recovers → Readiness passes → Add back to LB. Liveness should be minimal (can app respond?), readiness checks dependencies.",
  },
  {
    id: 'mc5',
    question: 'Why use both Circuit Breaker AND Bulkhead patterns together?',
    options: [
      'They solve the same problem, so use one or the other',
      'Circuit Breaker fails fast, Bulkhead isolates failures - complementary protection',
      'Bulkhead is only for Kubernetes',
      'Circuit Breaker is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'Use both because they provide complementary protection: Bulkhead ISOLATES failures (Payment service can only use 20 threads, protecting other services). Circuit Breaker FAILS FAST (After 5 failures, circuit opens, requests fail in 50ms instead of 30s timeout). Together: Bulkhead limits blast radius (only 20 threads affected), Circuit breaker quickly detects and stops calling failing service (prevents wasting even those 20 threads). Example: Payment service down → Bulkhead limits to 20 threads → Circuit breaker opens after 5 failures → Minimal impact, fast recovery.',
  },
];
