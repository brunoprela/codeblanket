/**
 * Multiple choice questions for API Gateway Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apigatewaypatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'gateway-q1',
    question:
      'What is the primary benefit of using the Backend for Frontend (BFF) pattern?',
    options: [
      'Reduces server costs by sharing infrastructure',
      'Allows customizing API responses for different client types (web, mobile, IoT)',
      'Automatically translates REST to GraphQL',
      'Eliminates the need for authentication',
    ],
    correctAnswer: 1,
    explanation:
      "BFF pattern creates separate gateway per client type, allowing customization: mobile gets minimal data (bandwidth), web gets full HTML, IoT gets compact binary. This optimizes each client without compromising others. It doesn't share infrastructure (opposite), doesn't translate protocols, or handle auth automatically.",
  },
  {
    id: 'gateway-q2',
    question: 'Why should business logic NOT be implemented in an API gateway?',
    options: [
      'Gateways are too slow to execute complex logic',
      'It violates separation of concerns and makes the gateway harder to maintain',
      'Gateways cannot access databases',
      "It's impossible to test business logic in gateways",
    ],
    correctAnswer: 1,
    explanation:
      "API gateways should handle cross-cutting concerns (auth, routing, rate limiting), not business logic. Business logic belongs in backend services for maintainability, testability, and separation of concerns. Gateways aren't inherently slow, can access DBs (but shouldn't), and can be tested.",
  },
  {
    id: 'gateway-q3',
    question: 'What is an Aggregator Gateway pattern and when is it useful?',
    options: [
      'A gateway that compresses responses; useful for slow networks',
      'A gateway that combines multiple backend calls into single response; useful for mobile apps',
      'A gateway that aggregates logs; useful for monitoring',
      'A gateway that pools database connections; useful for high traffic',
    ],
    correctAnswer: 1,
    explanation:
      'Aggregator Gateway makes parallel calls to multiple services and combines responses into one. This reduces client requests from N to 1, critical for mobile apps with limited bandwidth and battery. Example: dashboard fetching user + orders + recommendations in one request instead of three.',
  },
  {
    id: 'gateway-q4',
    question:
      'How does circuit breaking at the API gateway improve system reliability?',
    options: [
      'It encrypts traffic between services',
      'It prevents cascading failures by failing fast when a service is down',
      'It balances load across multiple gateway instances',
      'It caches responses to reduce backend calls',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breaker monitors failures and "opens" (stops sending requests) when threshold exceeded, preventing cascading failures and giving backend time to recover. After timeout, it tries again (half-open). This is different from load balancing, caching, or encryption.',
  },
  {
    id: 'gateway-q5',
    question: 'What is a potential drawback of using an API gateway?',
    options: [
      'It makes microservices more tightly coupled',
      'It can become a single point of failure and performance bottleneck',
      'It prevents using multiple programming languages',
      'It requires rewriting all backend services',
    ],
    correctAnswer: 1,
    explanation:
      "API gateway is a single entry point, making it a potential single point of failure (mitigate with HA/redundancy) and bottleneck (all traffic goes through it). It actually reduces coupling (not increases), doesn't affect backend languages, and doesn't require rewriting services.",
  },
];
