/**
 * Multiple choice questions for Monolith vs Microservices section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const monolithvsmicroservicesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main advantage of a monolithic architecture?',
    options: [
      'Better scalability',
      'Independent team autonomy',
      'Simplicity in development, deployment, and debugging',
      'Fault isolation',
    ],
    correctAnswer: 2,
    explanation:
      'The main advantage of a monolithic architecture is simplicity. Everything is in one codebase, one deployment, one set of logs. This makes development faster initially, deployment simpler, and debugging easier (single call stack). Microservices offer better scalability and team autonomy but at the cost of higher complexity.',
  },
  {
    id: 'mc2',
    question:
      'When should you consider migrating from monolith to microservices?',
    options: [
      'Immediately when starting any new project',
      'When you have 3-5 developers',
      'When team grows beyond 50 developers or you need independent scaling of components',
      'Never, monoliths are always better',
    ],
    correctAnswer: 2,
    explanation:
      'Migrate to microservices when team grows beyond 50 developers (coordination in monolith becomes bottleneck) or when you need independent scaling (one component needs 10x more resources). For small teams (< 20 devs) or early-stage startups, the complexity of microservices outweighs the benefits. Start with monolith, extract services when needed.',
  },
  {
    id: 'mc3',
    question: 'What is the Strangler Fig pattern?',
    options: [
      'A pattern for building microservices from scratch',
      'A pattern for incrementally migrating from monolith to microservices by extracting services one at a time',
      'A pattern for scaling monolithic applications',
      'A pattern for database sharding',
    ],
    correctAnswer: 1,
    explanation:
      'The Strangler Fig pattern is an incremental migration approach where you extract microservices from a monolith one at a time, gradually "strangling" the monolith. This is much safer than a big bang rewrite. You create new services alongside the monolith, route traffic to them, and remove code from the monolith incrementally.',
  },
  {
    id: 'mc4',
    question: 'What is the Saga pattern used for in microservices?',
    options: [
      'Load balancing between services',
      'Managing distributed transactions across microservices with eventual consistency',
      'Service discovery',
      'API gateway routing',
    ],
    correctAnswer: 1,
    explanation:
      "The Saga pattern manages distributed transactions across microservices. Since you can't have ACID transactions spanning multiple services/databases, Saga implements a sequence of local transactions with compensating transactions for rollback. For example: create order → charge payment → reduce inventory. If payment fails, cancel order (compensating transaction).",
  },
  {
    id: 'mc5',
    question: 'What is a common anti-pattern when implementing microservices?',
    options: [
      'Using different programming languages for different services',
      'Multiple microservices sharing the same database',
      'Independent deployment of services',
      'Service-to-service communication via APIs',
    ],
    correctAnswer: 1,
    explanation:
      "Sharing a database between microservices is an anti-pattern because it creates tight coupling. Schema changes affect multiple services, you can't deploy services independently, and it defeats the purpose of microservices (isolation). Each microservice should have its own database (database per service pattern).",
  },
];
