/**
 * Multiple choice questions for Event-Driven Microservices section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const eventdrivenmicroservicesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc-events-1',
      question:
        'What is the main advantage of event-driven architecture over request-response?',
      options: [
        "It\'s always faster",
        "Loose coupling - services don't need to know about each other",
        'It eliminates the need for databases',
        "It\'s easier to debug",
      ],
      correctAnswer: 1,
      explanation:
        'Event-driven architecture provides loose coupling: services publish events without knowing who subscribes. New subscribers can be added without changing publisher. Services can fail independently. Option 1 is wrong (not always faster - eventual consistency means delay). Option 3 is nonsense (still need databases). Option 4 is wrong (event-driven is actually harder to debug due to distributed nature). The main benefit is loose coupling and independent scaling.',
    },
    {
      id: 'mc-events-2',
      question: 'What is idempotency in event processing?',
      options: [
        'Processing events as fast as possible',
        'Processing the same event multiple times produces the same result',
        'Processing events in order',
        'Processing events without errors',
      ],
      correctAnswer: 1,
      explanation:
        "Idempotency means processing the same event multiple times has the same effect as processing it once. This is crucial because message brokers often deliver events at-least-once (may deliver multiple times). Without idempotency, you'd have duplicate charges, double decrements, etc. Implementation: track processed event IDs, check before processing, use transactions. Options 1, 3, and 4 describe different concepts (performance, ordering, reliability) but not idempotency.",
    },
    {
      id: 'mc-events-3',
      question: 'What is event sourcing?',
      options: [
        'Getting events from external sources',
        'Storing all changes as a sequence of events; event log is source of truth',
        'Finding the source of bugs in event handlers',
        'A way to compress events',
      ],
      correctAnswer: 1,
      explanation:
        'Event sourcing stores all state changes as a sequence of immutable events. Instead of storing current state in database (status="SHIPPED"), store events (OrderCreated, PaymentReceived, OrderShipped). Current state is derived by replaying events. Benefits: full audit trail, time travel debugging, can create new views by replaying events. Drawback: complexity, query performance. Options 1, 3, and 4 misunderstand the concept.',
    },
    {
      id: 'mc-events-4',
      question: 'What is a dead letter queue (DLQ)?',
      options: [
        'A queue for events from terminated services',
        'A queue for events that failed to process after multiple retries',
        'A queue for events that are no longer needed',
        'A backup queue',
      ],
      correctAnswer: 1,
      explanation:
        'Dead Letter Queue stores events that failed to process after multiple retries (e.g., 3 attempts with exponential backoff). This prevents poison messages from blocking the queue forever. Events in DLQ can be analyzed, fixed, and reprocessed manually. Alerts should fire when events land in DLQ. Option 1 confuses terminology. Option 3 is wrong (events aren\'t "deleted"). Option 4 is wrong (it\'s for failures, not backup).',
    },
    {
      id: 'mc-events-5',
      question:
        'When should you use event-driven architecture vs request-response?',
      options: [
        "Always use events (they're better)",
        'Events for background tasks/notifications; request-response when immediate feedback needed',
        'Always use request-response (events are too complex)',
        "They're interchangeable",
      ],
      correctAnswer: 1,
      explanation:
        'Use events for: background tasks (emails, analytics), multiple interested parties, loose coupling needed, eventual consistency acceptable. Use request-response for: operations needing immediate feedback (login, search, checkout confirmation), simple workflows, strong consistency required. Example: Charge payment with request-response (need to know if succeeded), notify email service with event (fire and forget). Options 1 and 3 are extreme positions. Option 4 is wrong (they have different characteristics and trade-offs).',
    },
  ];
