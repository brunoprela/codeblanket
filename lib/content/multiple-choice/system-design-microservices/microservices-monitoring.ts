/**
 * Multiple choice questions for Microservices Monitoring section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicesmonitoringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-monitoring-1',
    question: 'What are the three pillars of observability?',
    options: [
      'CPU, Memory, Disk',
      'Metrics, Logs, Distributed Tracing',
      'Frontend, Backend, Database',
      'Unit tests, Integration tests, E2E tests',
    ],
    correctAnswer: 1,
    explanation:
      'The three pillars of observability are: (1) Metrics - numeric measurements over time (request rate, latency, errors), (2) Logs - text records of events for debugging, (3) Distributed Tracing - tracking requests across multiple services. Together they provide complete visibility into distributed systems. Option 1 lists infrastructure metrics (part of observability but not the pillars). Options 3 and 4 are unrelated concepts.',
  },
  {
    id: 'mc-monitoring-2',
    question: 'What are RED metrics and why are they important?',
    options: [
      'Redis, Elasticsearch, Docker',
      'Rate, Errors, Duration - measure user-facing service health',
      'Read, Execute, Debug',
      'Replica, Endpoint, Deployment',
    ],
    correctAnswer: 1,
    explanation:
      'RED metrics measure user-facing health: Rate (requests/second), Errors (error percentage), Duration (latency percentiles p50/p95/p99). These are symptoms that directly impact users, making them better than cause-based metrics like CPU/memory. Monitor RED metrics for every microservice. Options 1, 3, and 4 are not related to monitoring metrics.',
  },
  {
    id: 'mc-monitoring-3',
    question: 'What is the purpose of a correlation ID in microservices?',
    options: [
      'To generate unique database IDs',
      'To track a single request as it flows through multiple services',
      'To correlate CPU and memory usage',
      'To link services together',
    ],
    correctAnswer: 1,
    explanation:
      'Correlation ID is a unique identifier (UUID) attached to each request, passed through all services via HTTP headers. It allows you to trace a single request across multiple services in logs and traces, making debugging much easier. Example: query logs by correlation ID to see full request flow. Option 1 confuses correlation ID with database ID. Option 3 is nonsensical. Option 4 misunderstands the concept.',
  },
  {
    id: 'mc-monitoring-4',
    question: 'When should you trigger an alert in microservices monitoring?',
    options: [
      'When CPU usage exceeds 80%',
      'When user-facing symptoms occur (high error rate > 5%, P95 latency > 1s)',
      'For every single request failure',
      'When disk space reaches 50%',
    ],
    correctAnswer: 1,
    explanation:
      "Alert on user-facing symptoms: error rate > threshold (e.g., 5% for 5 minutes), latency exceeding SLO (P95 > 1s), or service completely down. Don't alert on causes (CPU, disk) unless they directly impact users. Don't alert on single failures (too noisy). Option 1 - high CPU doesn't always affect users. Option 3 - single failures create alert fatigue. Option 4 - 50% disk isn't urgent (alert at 90% and predict when it will fill).",
  },
  {
    id: 'mc-monitoring-5',
    question: 'What is an SLO (Service Level Objective)?',
    options: [
      'A slow service that needs optimization',
      'A target for reliability metrics (e.g., 99.9% availability, P95 < 500ms)',
      'A service level agreement with legal penalties',
      'A type of Kubernetes object',
    ],
    correctAnswer: 1,
    explanation:
      'SLO is a target for service reliability: "99.9% of requests succeed" (availability SLO) or "95% of requests < 500ms" (latency SLO). SLOs define acceptable service performance and error budgets (0.1% of requests can fail). When budget exhausted, focus on reliability instead of features. Option 3 confuses SLO with SLA (SLA is contract with penalties, SLO is internal target). Options 1 and 4 are unrelated.',
  },
];
