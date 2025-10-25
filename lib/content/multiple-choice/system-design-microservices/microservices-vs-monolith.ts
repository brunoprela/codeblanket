/**
 * Multiple choice questions for Microservices vs Monolith section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicesvsmonolithMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-microservices-1',
    question:
      'Your startup has 8 engineers and is still finding product-market fit. Users are growing 50% month-over-month. What architecture should you choose?',
    options: [
      'Microservices from day 1 to prepare for scale',
      'Well-designed monolith with clear module boundaries',
      'Serverless functions for everything',
      'Distributed monolith with shared database',
    ],
    correctAnswer: 1,
    explanation:
      'With a small team (8 engineers) and uncertain product direction (still finding PMF), a well-designed modular monolith is ideal. It allows fast iteration, avoids operational complexity of microservices, and can be extracted into services later using the Strangler Fig pattern. Premature microservices would slow down development and add unnecessary complexity. The 50% growth, while impressive, can be handled by vertical/horizontal scaling of a monolith.',
  },
  {
    id: 'mc-microservices-2',
    question:
      'Which of the following is NOT a benefit of microservices architecture?',
    options: [
      'Independent scalability of different components',
      'Lower operational complexity than monoliths',
      'Technology flexibility across services',
      'Fault isolation between services',
    ],
    correctAnswer: 1,
    explanation:
      "Microservices have HIGHER operational complexity than monoliths, not lower. You must manage service discovery, distributed tracing, multiple deployments, network failures, and eventual consistency. The other options are genuine benefits: independent scaling (scale only what needs it), technology flexibility (use Go for performance, Python for ML), and fault isolation (one service failure doesn't crash everything).",
  },
  {
    id: 'mc-microservices-3',
    question: 'What is a "distributed monolith" and why is it problematic?',
    options: [
      'A monolith deployed across multiple regions for low latency',
      'Microservices architecture where services share a database and are tightly coupled',
      'A monolith that uses distributed caching for performance',
      'A microservice that handles multiple domains',
    ],
    correctAnswer: 1,
    explanation:
      "A distributed monolith is the worst of both worlds: you have the operational complexity of microservices (network calls, multiple deployments, distributed system challenges) but without the benefits (services can't deploy independently due to tight coupling, shared database creates bottleneck). It\'s an anti-pattern that often results from poorly designed microservices migration. Signs include: shared database, synchronous coupling chains, inability to deploy services independently.",
  },
  {
    id: 'mc-microservices-4',
    question:
      'Your company has 100 engineers across 15 teams. Deployment coordination is painful, with teams blocked waiting for others. Different components have vastly different scaling needs. What should you do?',
    options: [
      'Keep monolith but improve coordination processes',
      'Rewrite entire monolith to microservices in 6 months',
      'Use Strangler Fig pattern to gradually extract services',
      'Create a distributed monolith with shared database',
    ],
    correctAnswer: 2,
    explanation:
      'The Strangler Fig pattern is the safe approach: incrementally extract services starting with low-risk components (like notifications), validate the approach, then continue. This avoids the high-risk "big bang" rewrite while allowing teams to gain microservices experience gradually. With 100 engineers and clear pain points (deployment blocking, different scaling needs), you have the organizational maturity for microservices. Don\'t improve coordination (scales poorly) or create distributed monolith (anti-pattern). Never do big rewrites (high failure rate).',
  },
  {
    id: 'mc-microservices-5',
    question:
      'Which company successfully operates at massive scale with a monolithic architecture?',
    options: [
      'Netflix (700+ microservices)',
      'Shopify (Rails monolith powering millions of stores)',
      'Uber (2,000+ microservices)',
      'Amazon (two-pizza team microservices)',
    ],
    correctAnswer: 1,
    explanation:
      "Shopify is a famous example of a successful monolith at scale. Despite powering millions of stores, they maintain a well-architected Rails monolith with clear module boundaries, strong ownership, and good testing. They extract services only when truly needed. This proves that monoliths CAN scale with good architecture. Netflix, Uber, and Amazon all use extensive microservices. The lesson: architecture quality matters more than whether it's monolith or microservices.",
  },
];
