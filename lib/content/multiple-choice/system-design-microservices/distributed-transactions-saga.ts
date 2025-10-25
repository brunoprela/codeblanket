/**
 * Multiple choice questions for Distributed Transactions & Saga Pattern section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const distributedtransactionssagaMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc-saga-1',
      question:
        "Why doesn't Two-Phase Commit (2PC) work well for microservices?",
      options: [
        "It\'s too simple and doesn't provide enough features",
        "It's blocking, reduces availability, and many modern databases don't support it",
        'It only works with SQL databases',
        "It\'s faster than Saga pattern",
      ],
      correctAnswer: 1,
      explanation:
        "2PC is blocking (services hold locks while waiting for coordinator), which reduces throughput and availability. If coordinator crashes, all services are stuck. Modern NoSQL and cloud databases often don't support 2PC. In distributed systems, availability is usually more important than strong consistency (CAP theorem). Saga pattern provides better availability with eventual consistency. Option 1 is wrong (2PC is actually complex). Option 3 is partially true but not the main reason. Option 4 is wrong (2PC is slower due to synchronous coordination).",
    },
    {
      id: 'mc-saga-2',
      question: 'In a Saga, what is a compensating transaction?',
      options: [
        'A faster version of a transaction that compensates for slow performance',
        'An undo operation that reverses the effects of a previous transaction',
        'A transaction that adds extra features to compensate for missing ones',
        'A payment to developers for extra work',
      ],
      correctAnswer: 1,
      explanation:
        'A compensating transaction is an undo operation that semantically reverses a previous transaction. Example: if the forward transaction is "Reserve Inventory", the compensating transaction is "Release Inventory". If "Charge Card", then "Refund Card". This is how Sagas handle failures - by rolling back completed steps using compensations. Compensations should be idempotent (safe to execute multiple times) and use semantic rollback (mark as CANCELLED) rather than physical rollback (DELETE record). Option 1 is nonsense. Option 3 confuses "compensating" with "complementing". Option 4 is a joke.',
    },
    {
      id: 'mc-saga-3',
      question:
        "What\'s the main difference between choreography-based and orchestration-based sagas?",
      options: [
        'Choreography is faster than orchestration',
        'Choreography is decentralized (event-driven), orchestration has a central coordinator',
        'Choreography only works with 2 services',
        "Orchestration can't handle failures",
      ],
      correctAnswer: 1,
      explanation:
        'Choreography is decentralized: services communicate via events, each service knows its next step. No central coordinator. Orchestration is centralized: an orchestrator tells each service what to do, coordinates the flow, and handles failures. Choreography is simpler for basic flows but harder to understand as complexity grows. Orchestration is easier to understand, debug, and modify, but requires a coordinator service (potential single point of failure). Option 1 is debatable (both can be fast). Option 3 is wrong (choreography works with any number). Option 4 is wrong (orchestration handles failures very well).',
    },
    {
      id: 'mc-saga-4',
      question:
        'Your e-commerce saga: CreateOrder → ReserveInventory → ChargePayment → Ship. Payment fails. What happens?',
      options: [
        'System crashes and needs manual intervention',
        'Execute compensating transactions: ReleaseInventory → CancelOrder',
        'Retry payment indefinitely until it succeeds',
        'Keep the order as-is and notify the admin',
      ],
      correctAnswer: 1,
      explanation:
        'When a saga step fails, execute compensating transactions in reverse order for all completed steps. PaymentFailed → execute compensation for ReserveInventory (ReleaseInventory) → execute compensation for CreateOrder (CancelOrder). This leaves the system in a consistent state. The order is marked CANCELLED, inventory is released, and customer is notified. Option 1 is wrong (sagas handle failures automatically). Option 3 is wrong (payment failure might be non-retryable like "insufficient funds"). Option 4 leaves inconsistent state (order without payment).',
    },
    {
      id: 'mc-saga-5',
      question: 'What type of consistency does the Saga pattern provide?',
      options: [
        'Strong consistency (ACID)',
        'Eventual consistency',
        'No consistency',
        'Immediate consistency',
      ],
      correctAnswer: 1,
      explanation:
        "Saga pattern provides eventual consistency. During saga execution, the system is temporarily in an inconsistent state (e.g., order created but payment not yet processed). Eventually, when the saga completes (either successfully or via compensations), the system reaches a consistent state. This is a trade-off: we sacrifice immediate consistency for better availability and scalability. Option 1 is wrong (sagas don't provide ACID guarantees). Option 3 is wrong (eventual consistency is still a form of consistency). Option 4 is wrong (there's a delay before consistency is achieved).",
    },
  ];
