/**
 * Multiple choice questions for API Monitoring & Analytics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apimonitoringMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'monitoring-q1',
    question:
      'What are the "Golden Signals" for monitoring according to Google SRE?',
    options: [
      'CPU, Memory, Disk, Network',
      'Latency, Traffic, Errors, Saturation',
      'Uptime, Throughput, Response Time, Availability',
      'Load Average, Queue Depth, Thread Count, Connection Pool',
    ],
    correctAnswer: 1,
    explanation:
      'Google SRE defines Golden Signals as: Latency (request duration), Traffic (requests/sec), Errors (error rate), and Saturation (resource usage). These four metrics provide comprehensive view of system health. Other metrics are useful but these four are foundational.',
    difficulty: 'easy',
  },
  {
    id: 'monitoring-q2',
    question: 'Why track p95 and p99 latency in addition to average latency?',
    options: [
      'They are easier to calculate than average',
      'They show tail latency that affects user experience but is hidden by averages',
      'They use less memory than tracking all request times',
      'They are required for compliance',
    ],
    correctAnswer: 1,
    explanation:
      'Percentiles (p95, p99) reveal tail latency: the slowest requests that affect real users. Average can be misleading (e.g., average 100ms but p99 is 5s means 1% of users have terrible experience). Always track percentiles, not just averages.',
    difficulty: 'medium',
  },
  {
    id: 'monitoring-q3',
    question:
      'What is the purpose of request ID tracking in distributed systems?',
    options: [
      'To encrypt requests for security',
      'To trace a single user request across multiple services',
      'To count total number of requests',
      'To generate unique database primary keys',
    ],
    correctAnswer: 1,
    explanation:
      'Request IDs (X-Request-ID header) allow tracing a single user request as it flows through multiple services. Essential for debugging: you can find all logs related to one request across API gateway, service A, service B, database, etc.',
    difficulty: 'easy',
  },
  {
    id: 'monitoring-q4',
    question:
      'What is distributed tracing and why is it important for microservices?',
    options: [
      'A way to deploy services across multiple data centers',
      'Tracking individual requests across multiple services to understand latency bottlenecks',
      'A method for distributing database queries',
      'A security technique for encrypting inter-service communication',
    ],
    correctAnswer: 1,
    explanation:
      'Distributed tracing (e.g., Jaeger, Zipkin) tracks requests across services, showing: API Gateway (10ms) → Service A (50ms) → Service B (200ms) ← bottleneck! Crucial for debugging microservices where a single request touches many services.',
    difficulty: 'medium',
  },
  {
    id: 'monitoring-q5',
    question:
      'What is the difference between synthetic monitoring and real user monitoring (RUM)?',
    options: [
      'Synthetic uses fake data, RUM uses real data',
      'Synthetic runs automated tests from specific locations, RUM tracks actual user interactions',
      'Synthetic is for frontend, RUM is for backend',
      'Synthetic is cheaper than RUM',
    ],
    correctAnswer: 1,
    explanation:
      "Synthetic monitoring: automated tests from specific locations (e.g., AWS health checks every 1 min). RUM: tracks real users' actual API calls with timing data. Both are valuable: synthetic catches outages, RUM shows real user experience.",
    difficulty: 'easy',
  },
];
