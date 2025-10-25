/**
 * Multiple choice questions for Microservices Anti-Patterns section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicesantipatternsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc-anti-1',
      question: 'What is a distributed monolith?',
      options: [
        'A monolith deployed on multiple servers',
        'Microservices that are tightly coupled and must be deployed together',
        'A large microservice',
        'A monolith with distributed caching',
      ],
      correctAnswer: 1,
      explanation:
        'Distributed monolith refers to microservices that are so tightly coupled they must be deployed together, negating microservices benefits. Symptoms: shared database, synchronous chains, breaking changes ripple across services. This is worst of both worlds - all complexity of microservices without benefits (independent deployment, scaling). Fix: database per service, async communication, API versioning. Option 1 is just horizontal scaling. Option 3 is a nanoservice concern. Option 4 is unrelated.',
    },
    {
      id: 'mc-anti-2',
      question:
        'Why should you NOT start a greenfield project with microservices?',
      options: [
        'Microservices are outdated',
        "You don't know the correct service boundaries yet; start with modular monolith",
        'Microservices are only for large companies',
        'Microservices are slower',
      ],
      correctAnswer: 1,
      explanation:
        "Don't start greenfield with microservices because you don't know correct boundaries yet. You'll guess wrong and face costly refactoring. Start with modular monolith (clear boundaries, easy to split later), extract services gradually when pain points emerge (Strangler Fig). Almost every successful microservices story (Amazon, Netflix, Uber) started as monolith. Option 1 is wrong (microservices are current). Option 3 is wrong (size matters but not the only factor). Option 4 is wrong (microservices can be faster at scale).",
    },
    {
      id: 'mc-anti-3',
      question:
        'What is the problem with long synchronous call chains in microservices?',
      options: [
        'They are too fast',
        'High latency (sum of all) and low availability (product of all)',
        'They use too much memory',
        'They are easy to debug',
      ],
      correctAnswer: 1,
      explanation:
        'Long synchronous chains: A → B → C → D have high latency (50ms + 100ms + 150ms = 300ms) and low availability (99.9% × 99.9% × 99.9% = 99.7%). One service failure breaks entire chain. Solution: async communication where possible, parallelize calls, circuit breakers, caching. Example: 6 services with 99.9% uptime each = 99.4% combined (43 min/month downtime → 4.3 hrs/month). Option 1 is wrong (slower, not faster). Option 3 is not the main issue. Option 4 is wrong (hard to debug, not easy).',
    },
    {
      id: 'mc-anti-4',
      question: "What is Conway\'s Law?",
      options: [
        'A law about API design',
        'Organizations design systems that mirror their communication structure',
        'A law about database schemas',
        'A law about deployment frequency',
      ],
      correctAnswer: 1,
      explanation:
        "Conway\'s Law: \"Organizations design systems that mirror their communication structure.\" For microservices: team structure must align with architecture. Each team should own their services end-to-end (frontend, backend, DB) to enable independent deployment. Don't have functional teams (Frontend Team, Backend Team) each touching all services (coordination nightmare). Options 1, 3, and 4 are unrelated concepts. Conway's Law is about team structure affecting system design.",
    },
    {
      id: 'mc-anti-5',
      question: 'Why is a shared database an anti-pattern in microservices?',
      options: [
        'Databases are expensive',
        "It creates tight coupling; services can't evolve or scale independently",
        "It\'s too slow",
        "It's a security risk",
      ],
      correctAnswer: 1,
      explanation:
        "Shared database creates tight coupling: schema change breaks multiple services, can't scale databases independently, can't choose different database types (SQL vs NoSQL), unclear ownership, transaction boundaries unclear. Defeats the purpose of microservices (independence). Solution: database per service pattern. Services communicate via APIs or events. Data duplication is acceptable trade-off for independence. Options 1, 3, and 4 may be concerns but aren't the main reason shared database is an anti-pattern.",
    },
  ];
