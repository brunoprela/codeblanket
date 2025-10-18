/**
 * Quiz questions for Distributed Transactions & Saga Pattern section
 */

export const distributedtransactionssagaQuiz = [
  {
    id: 'q1-saga',
    question:
      "Explain why you can't use traditional database transactions across microservices. What problems would you encounter?",
    sampleAnswer:
      "Traditional database transactions (ACID) don't work across microservices because each service has its own database. You can't have a BEGIN TRANSACTION that spans multiple databases across the network. Problems: (1) Database locks can't span services, (2) Two-phase commit is blocking and reduces availability (if one service is down, all are blocked), (3) Many modern databases (NoSQL, cloud) don't support distributed transactions, (4) CAP theorem - must choose between consistency and availability; microservices prefer availability. Instead, use Saga pattern with eventual consistency and compensating transactions.",
    keyPoints: [
      'Each microservice has its own database (database per service pattern)',
      'Cannot use single transaction across network boundaries',
      '2PC is blocking and reduces system availability',
      'CAP theorem: microservices prefer availability over strong consistency',
      'Solution: Saga pattern with eventual consistency',
    ],
  },
  {
    id: 'q2-saga',
    question:
      'When would you use choreography vs orchestration for a saga? Give a specific example for each.',
    sampleAnswer:
      'Use choreography for simple, linear flows with few steps where services are already event-driven. Example: User registration - User Service creates user → publishes UserCreated → Email Service sends welcome email → Analytics Service tracks signup. Only 3 services, linear flow. Use orchestration for complex flows with conditional logic, multiple branches, or when you need centralized monitoring. Example: E-commerce checkout - Order Service orchestrator coordinates: create order → check inventory (if out of stock, cancel) → authorize payment (if fails, release inventory) → calculate shipping → capture payment → send confirmation. 6+ services with complex failure handling. Orchestration makes this easier to understand and debug.',
    keyPoints: [
      'Choreography: Simple, linear flows with few steps',
      'Choreography: Already event-driven architecture',
      'Orchestration: Complex flows with conditional logic',
      'Orchestration: Need centralized monitoring and debugging',
      'Orchestration: Easier to modify flow later',
    ],
  },
  {
    id: 'q3-saga',
    question:
      'Your saga fails at step 3 of 5. How do you handle rollback? What challenges might you face with compensating transactions?',
    sampleAnswer:
      "Execute compensating transactions in reverse order for completed steps: Step 2 compensation → Step 1 compensation. Store saga state after each step to know what to rollback. Challenges: (1) Compensations must be idempotent (if rollback crashes halfway, need to restart safely), (2) Some operations can't be fully compensated (email sent - can send \"sorry, cancelled\" but can't unsend), (3) Compensation might fail (payment refund fails - need retry with backoff), (4) Race conditions (user trying to use inventory that's being released), (5) Semantic vs physical compensation (mark order CANCELLED vs deleting order record). Store full audit trail, make compensations idempotent, implement retry logic, and prefer semantic compensation.",
    keyPoints: [
      'Execute compensating transactions in reverse order',
      'Store saga state to know what needs compensation',
      'Compensations must be idempotent (safe to retry)',
      "Some operations can't be fully compensated",
      'Prefer semantic compensation (status change) over physical (delete)',
    ],
  },
];
