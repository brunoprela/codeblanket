/**
 * Multiple Choice Questions for Event-Driven Architecture
 */

import { MultipleChoiceQuestion } from '../../../types';

export const eventDrivenArchitectureMC: MultipleChoiceQuestion[] = [
  {
    id: 'event-driven-architecture-mc-1',
    question:
      'What is the difference between event notification and event-carried state transfer?',
    options: [
      'Event notification includes full data, event-carried state transfer includes only IDs',
      'Event notification includes minimal data (just notification), event-carried state transfer includes full state data',
      'They are the same',
      'Event notification is faster than event-carried state transfer',
    ],
    correctAnswer: 1,
    explanation:
      'Event notification contains minimal information (e.g., "OrderPlaced: order_123"), requiring consumers to query for details. Event-carried state transfer includes full state data in the event itself (e.g., complete order details: customer, items, amount). Event notification is loosely coupled but chatty (requires queries). Event-carried state transfer is more coupled (to schema) but efficient (no queries needed). Choose based on trade-off: coupling vs. performance.',
  },
  {
    id: 'event-driven-architecture-mc-2',
    question: 'What is event sourcing?',
    options: [
      'Storing only the current state of entities in a database',
      'Storing all events that led to the current state as the source of truth',
      'Caching events for faster access',
      'Compressing events to save storage',
    ],
    correctAnswer: 1,
    explanation:
      'Event sourcing stores all state changes as a sequence of events, which becomes the source of truth. Instead of storing "AccountBalance: $100", you store events "AccountOpened($0), Deposited($50), Deposited($50)". Current state is derived by replaying events. Benefits: complete audit trail, temporal queries ("what was balance at time X?"), replay for debugging, event-driven reactions. Drawbacks: complexity, replay overhead, storage for all history. Use for domains requiring audit (finance, healthcare).',
  },
  {
    id: 'event-driven-architecture-mc-3',
    question: 'What is CQRS (Command Query Responsibility Segregation)?',
    options: [
      'Using the same model for reads and writes',
      'Separating the write model (commands) from the read model (queries)',
      'Caching database queries for faster access',
      'Replicating data across multiple databases',
    ],
    correctAnswer: 1,
    explanation:
      'CQRS separates the write model (handling commands like CreateOrder) from the read model (handling queries like GetOrderDetails). Commands modify state and emit events. Events update read models (materialized views optimized for queries). Benefits: independent scaling of reads/writes, optimized read models for different views, eventual consistency. For example, write model validates business rules and stores events; read model maintains denormalized views (order feed, user timeline) updated via events.',
  },
  {
    id: 'event-driven-architecture-mc-4',
    question: 'What is a saga pattern in event-driven architectures?',
    options: [
      'A pattern for compressing events',
      'A pattern for managing distributed transactions across multiple services through choreography or orchestration',
      'A pattern for caching frequently accessed events',
      'A pattern for encrypting events',
    ],
    correctAnswer: 1,
    explanation:
      'A saga is a pattern for managing distributed transactions across multiple services. Instead of a single ACID transaction, a saga coordinates a sequence of local transactions through events. If one step fails, compensating transactions rollback previous steps. Two types: Choreography (services react to events, no central coordinator) and Orchestration (central coordinator manages workflow). Example: Order saga: CreateOrder → ReserveInventory → ProcessPayment. If payment fails, publish RefundPayment and ReleaseInventory events to rollback.',
  },
  {
    id: 'event-driven-architecture-mc-5',
    question:
      'What challenge does eventual consistency introduce in event-driven architectures?',
    options: [
      'Events are lost',
      'Read models may be temporarily stale (not immediately reflecting latest writes)',
      'Events cannot be replayed',
      'Systems become too slow',
    ],
    correctAnswer: 1,
    explanation:
      'Eventual consistency means read models are updated asynchronously after writes, causing temporary staleness (50-200ms typically). For example, after creating an order (write), immediately querying the order feed (read model) might not show the new order yet. Strategies to handle: (1) Optimistic UI (show new order immediately, refresh later), (2) Version-based consistency (wait for specific version), (3) Read-your-writes (read from write model temporarily). Trade-off: consistency vs. scalability.',
  },
];
