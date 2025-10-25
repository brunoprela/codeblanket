/**
 * Quiz questions for Microservices Anti-Patterns section
 */

export const microservicesantipatternsQuiz = [
  {
    id: 'q1-anti',
    question:
      'What is a distributed monolith? Why is it considered the worst of both worlds?',
    sampleAnswer:
      "Distributed monolith: microservices that are tightly coupled, must be deployed together, share database, or have synchronous call chains. Worst of both worlds because: (1) All complexity of microservices - distributed system, network calls, eventual consistency, multiple deployments, (2) None of benefits - can't deploy independently, can't scale independently, can't choose different tech stacks, changes ripple through all services. Example: Services that share database or must deploy together to avoid breaking changes. Better to have actual monolith (simpler) or proper microservices (independent). How to fix: database per service, async communication, API versioning, backward-compatible changes.",
    keyPoints: [
      'Distributed monolith: microservices that are tightly coupled',
      'All complexity of microservices (distributed, network)',
      'None of benefits (independent deployment, scaling)',
      'Often caused by: shared database, synchronous chains, no API versioning',
      'Fix: database per service, async communication, independence',
    ],
  },
  {
    id: 'q2-anti',
    question:
      'Why should you NOT start a greenfield project with microservices? What should you do instead?',
    sampleAnswer:
      "Don't start greenfield with microservices because: (1) Don't know correct boundaries yet - will get wrong initially, costly to refactor, (2) Over-engineering - premature optimization, (3) Team may lack expertise - microservices require mature operations, (4) Operational complexity - distributed systems are hard. Instead: Start with modular monolith - clear module boundaries within monolith, easy to split later. Then: Extract most painful part first (e.g., report generation), gradually extract more (Strangler Fig pattern), eventually reach mature microservices. Almost every successful microservices story started as monolith. Example: Amazon, Netflix, Uber all started as monoliths. Only split when monolith became too large.",
    keyPoints: [
      "Don't start greenfield with microservices (don't know boundaries)",
      'Start with modular monolith (clear boundaries, easy to split)',
      'Extract gradually when pain points emerge (Strangler Fig)',
      'Microservices require operational maturity',
      'Most successful microservices stories started as monoliths',
    ],
  },
  {
    id: 'q3-anti',
    question:
      "What is Conway\'s Law and why does it matter for microservices architecture?",
    sampleAnswer:
      'Conway\'s Law: "Organizations design systems that mirror their communication structure." Matters because: Architecture and team structure must align. Bad: Order Service, Payment Service, Shipping Service BUT teams organized as Frontend Team, Backend Team, DBA Team → every service touched by all teams → coordination nightmare, slow releases. Good: Team 1 owns Order Service end-to-end (frontend, backend, DB), Team 2 owns Payment Service end-to-end → teams can deploy independently, no coordination needed. Solution: Align team ownership with service boundaries. Each team should own their services completely. This enables independent deployment and velocity. Inverse Conway: Design architecture first, then structure teams to match.',
    keyPoints: [
      "Conway\'s Law: systems mirror communication structure",
      'Architecture and team structure must align',
      'Bad: functional teams (frontend/backend) touching all services',
      'Good: each team owns services end-to-end',
      'Enables independent deployment and fast iteration',
    ],
  },
];
