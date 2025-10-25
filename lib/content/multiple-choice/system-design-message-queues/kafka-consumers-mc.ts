/**
 * Multiple Choice Questions for Kafka Consumers
 */

import { MultipleChoiceQuestion } from '../../../types';

export const kafkaConsumersMC: MultipleChoiceQuestion[] = [
  {
    id: 'kafka-consumers-mc-1',
    question: 'What happens when a Kafka consumer in a consumer group fails?',
    options: [
      'All messages are lost',
      'The consumer group is disbanded',
      "Rebalancing occurs, reassigning the failed consumer's partitions to remaining consumers",
      'Messages are automatically routed to a dead letter queue',
    ],
    correctAnswer: 2,
    explanation:
      'When a consumer fails, Kafka detects this through missed heartbeats and triggers rebalancing. The partitions assigned to the failed consumer are redistributed among the remaining consumers in the group. For example, if Consumer 1 (processing partitions 0-3) fails in a group of 3 consumers with 12 partitions, its partitions are reassigned to Consumers 2 and 3. This ensures fault tolerance and automatic recovery without data loss.',
  },
  {
    id: 'kafka-consumers-mc-2',
    question:
      'What is the difference between commitSync() and commitAsync() in Kafka consumers?',
    options: [
      'commitSync() is faster than commitAsync()',
      'commitSync() blocks until the commit succeeds, commitAsync() returns immediately without waiting',
      'commitSync() commits to all brokers, commitAsync() commits to one broker only',
      'commitSync() is used for manual commits, commitAsync() is automatic',
    ],
    correctAnswer: 1,
    explanation:
      "commitSync() blocks until the offset commit is acknowledged by Kafka, guaranteeing the commit succeeded before continuing. commitAsync() sends the commit request and immediately returns without waiting, using a callback for success/failure notification. commitSync() is safer (guaranteed commit) but slower (blocking). commitAsync() is faster (non-blocking) but doesn't guarantee the commit succeeded before moving on. Use commitSync() for critical data, commitAsync() for high throughput.",
  },
  {
    id: 'kafka-consumers-mc-3',
    question:
      'Why is it important for Kafka consumer processing to be idempotent?',
    options: [
      'To improve processing speed',
      'To handle duplicate message delivery gracefully without side effects',
      'To reduce memory usage',
      'To enable automatic retries',
    ],
    correctAnswer: 1,
    explanation:
      "Idempotent processing ensures that processing a message multiple times has the same effect as processing it once. This is crucial because Kafka\'s at-least-once delivery can result in duplicates (e.g., if a consumer crashes after processing but before committing). For example, a payment processor should check if a payment ID was already processed (using Redis or a database) before charging a card again. Without idempotency, duplicate messages could lead to double charges or inconsistent state.",
  },
  {
    id: 'kafka-consumers-mc-4',
    question: 'What is consumer lag in Kafka?',
    options: [
      'The time delay between message production and consumption',
      "The difference between the latest offset in a partition and the consumer's committed offset",
      'The number of consumers that are offline',
      'The time taken for rebalancing to complete',
    ],
    correctAnswer: 1,
    explanation:
      "Consumer lag is the difference between the latest offset (high water mark) in a partition and the consumer's committed offset. For example, if the latest offset is 10,000 and the consumer is at offset 9,500, the lag is 500 messages. High lag indicates the consumer is falling behind, which could mean it's processing too slowly, the consumer is down, or there's a spike in message volume. Monitoring lag is critical for identifying performance issues.",
  },
  {
    id: 'kafka-consumers-mc-5',
    question:
      'What is the purpose of the "max.poll.interval.ms" configuration in Kafka consumers?',
    options: [
      'Maximum time to wait for new messages in a poll',
      'Maximum time between poll() calls before the consumer is considered dead',
      'Maximum time to wait for offset commit',
      'Maximum time for rebalancing',
    ],
    correctAnswer: 1,
    explanation:
      "max.poll.interval.ms sets the maximum time allowed between poll() calls. If this time is exceeded, the consumer is considered dead and removed from the consumer group, triggering rebalancing. This prevents a consumer from holding onto partitions indefinitely if it's stuck processing a batch of messages. For example, if max.poll.interval.ms=300000 (5 min) and processing takes 6 minutes, the consumer is kicked out. Set this based on your maximum processing time per batch.",
  },
];
