/**
 * Multiple choice questions for API Gateway section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apigatewayMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary purpose of an API Gateway in a microservices architecture?',
    options: [
      'To scale backend services horizontally',
      'To provide a single entry point and handle cross-cutting concerns like authentication and rate limiting',
      'To store data for microservices',
      'To replace all backend services with a monolith',
    ],
    correctAnswer: 1,
    explanation:
      "API Gateway primary purpose: Single entry point for clients + handle cross-cutting concerns (auth, rate limiting, logging, request routing). Not for scaling (that's load balancer), data storage (that's database), or replacing services (gateway routes to services, doesn't replace them).",
  },
  {
    id: 'mc2',
    question:
      'Your API Gateway validates JWT tokens on every request (50ms per validation). Requests are slow. What optimization would have the biggest impact?',
    options: [
      'Use a faster JWT library',
      'Cache authentication results in Redis (validate once, cache for 5 minutes)',
      'Skip authentication for some endpoints',
      'Use HTTP instead of HTTPS',
    ],
    correctAnswer: 1,
    explanation:
      'Caching auth results has biggest impact: Validate JWT once (50ms), cache result in Redis. Subsequent requests: Check cache (1ms) instead of validating (50ms). Cache hit rate: 95%+ (users make multiple requests). Result: 50ms → 2.5ms (20× faster). Option 1 might save 5-10ms. Option 3 is security risk. Option 4 is security risk (never skip HTTPS).',
  },
  {
    id: 'mc3',
    question: 'When should you use BOTH an API Gateway and Load Balancer?',
    options: [
      'Never, they serve the same purpose',
      'When you have a monolithic application with multiple instances',
      'When you have microservices with multiple instances per service',
      'Only when you have more than 100 users',
    ],
    correctAnswer: 2,
    explanation:
      'Use both when: Microservices (multiple services) + multiple instances per service. Architecture: Client → Gateway (routes by path: /api/users, /api/orders) → Load Balancer per service (distributes across instances) → Service instances. Gateway handles routing/auth. Load Balancer handles distribution. This is industry standard for production microservices.',
  },
  {
    id: 'mc4',
    question: 'What is the main risk of using an API Gateway?',
    options: [
      'It makes clients more complex',
      'It eliminates the need for authentication',
      'It becomes a single point of failure if not properly configured',
      'It prevents horizontal scaling',
    ],
    correctAnswer: 2,
    explanation:
      'Main risk: Single point of failure. If gateway down, ALL services unavailable (even if services healthy). Mitigation: (1) Deploy multiple gateway instances behind load balancer. (2) Health checks and auto-scaling. (3) Circuit breaker patterns. Gateway actually simplifies clients (option 1 wrong), enables auth (option 2 wrong), and enables scaling (option 3 wrong).',
  },
  {
    id: 'mc5',
    question:
      'Your API Gateway needs to call User Service, Order Service, and Recommendation Service for a single client request. What is the BEST approach?',
    options: [
      'Call services sequentially (User → Order → Recommendation)',
      'Call services in parallel using async I/O',
      'Cache all responses permanently to avoid calling services',
      'Have the client call each service directly',
    ],
    correctAnswer: 1,
    explanation:
      'Best approach: Parallel async calls. Sequential: 100ms + 150ms + 100ms = 350ms total. Parallel: max(100ms, 150ms, 100ms) = 150ms total (2.3× faster). Use Promise.all or similar. Option 3 (permanent cache) stale data. Option 4 defeats purpose of gateway (client complexity).',
  },
];
