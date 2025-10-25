/**
 * Multiple Choice Questions for Message Queue Fundamentals
 */

import { MultipleChoiceQuestion } from '../../../types';

export const messageQueueFundamentalsMC: MultipleChoiceQuestion[] = [
  {
    id: 'message-queue-fundamentals-mc-1',
    question:
      'What is the primary advantage of using a message queue in a distributed system?',
    options: [
      'Faster data processing than direct API calls',
      'Decoupling producers and consumers, enabling asynchronous communication',
      'Automatic data encryption for all messages',
      'Eliminating the need for databases',
    ],
    correctAnswer: 1,
    explanation:
      'The primary advantage of message queues is decoupling producers and consumers. Producers can send messages without waiting for consumers to process them, and consumers can process messages at their own pace. This enables asynchronous communication, improves system reliability (if consumer fails, messages are not lost), and allows independent scaling of producers and consumers.',
  },
  {
    id: 'message-queue-fundamentals-mc-2',
    question:
      'In a message queue system, what does "at-least-once delivery" guarantee mean?',
    options: [
      'Each message is delivered exactly once, with no duplicates',
      'Messages are delivered in the exact order they were sent',
      'Each message is delivered one or more times, duplicates are possible',
      'Messages are never lost, but delivery order is random',
    ],
    correctAnswer: 2,
    explanation:
      'At-least-once delivery guarantees that each message will be delivered to the consumer at least once, but duplicates are possible (e.g., if the consumer crashes after processing but before acknowledging). This is stronger than at-most-once (no duplicates, but loss possible) but weaker than exactly-once (no duplicates, no loss). Systems requiring at-least-once must implement idempotent processing to handle duplicates.',
  },
  {
    id: 'message-queue-fundamentals-mc-3',
    question:
      'What is the key difference between a queue and a topic in messaging systems?',
    options: [
      'Queues are faster than topics',
      'Queues support point-to-point (one consumer), topics support publish-subscribe (multiple consumers)',
      'Queues store messages permanently, topics delete messages immediately',
      'Queues use TCP, topics use UDP',
    ],
    correctAnswer: 1,
    explanation:
      'The fundamental difference is the consumption model: Queues implement point-to-point messaging where each message is consumed by exactly one consumer (competing consumers pattern). Topics implement publish-subscribe where each message is delivered to all subscribers. For example, a task queue (one worker processes each task) uses a queue, while event broadcasting (all services receive order event) uses a topic.',
  },
  {
    id: 'message-queue-fundamentals-mc-4',
    question:
      'In a message queue, what is the purpose of a Dead Letter Queue (DLQ)?',
    options: [
      'To permanently delete messages that are older than 30 days',
      'To store messages that cannot be processed after multiple retry attempts',
      'To improve message delivery speed by caching frequently accessed messages',
      'To replicate messages across multiple data centers',
    ],
    correctAnswer: 1,
    explanation:
      'A Dead Letter Queue (DLQ) stores messages that have failed processing after multiple retry attempts. This prevents "poison messages" (messages with bugs or invalid data) from blocking the main queue indefinitely. Messages in the DLQ can be inspected, debugged, and potentially reprocessed after fixing the underlying issue. Without a DLQ, a single bad message could block the entire queue.',
  },
  {
    id: 'message-queue-fundamentals-mc-5',
    question:
      'Which delivery guarantee is most difficult to implement and has the highest performance overhead?',
    options: [
      'At-most-once delivery',
      'At-least-once delivery',
      'Exactly-once delivery',
      'Best-effort delivery',
    ],
    correctAnswer: 2,
    explanation:
      'Exactly-once delivery is the most difficult to implement and has the highest overhead. It requires distributed transactions, idempotency keys, or deduplication mechanisms to ensure no message is lost or duplicated. Implementation typically involves: (1) Transactional writes (read-process-write atomically), (2) Idempotency (store processed message IDs), (3) Distributed consensus. Systems like Kafka use transactions and idempotent producers to achieve this, but with latency and complexity trade-offs.',
  },
];
