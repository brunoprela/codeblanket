/**
 * Multiple choice questions for Message Queues & Async Processing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const messagequeuesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Your web application sends 10K requests/sec to a payment processing service that can only handle 1K requests/sec. What will happen without a message queue?',
    options: [
      'The payment service will process all 10K requests/sec successfully',
      '9K requests/sec will fail or timeout (90% failure rate)',
      'The requests will be automatically queued by the network layer',
      'The web application will slow down to match the payment service',
    ],
    correctAnswer: 1,
    explanation:
      '9K requests/sec will fail or timeout (90% failure rate). Without a message queue, the payment service is overwhelmed: Can only process 1K/sec, remaining 9K/sec rejected or timeout. Users see errors, transactions lost. Solution: Add message queue between web app and payment service. Queue buffers requests, payment service consumes at its own pace (1K/sec). All 10K requests eventually processed (may take 10 seconds to drain queue).',
  },
  {
    id: 'mc2',
    question:
      'What is the main difference between "at-most-once" and "at-least-once" message delivery?',
    options: [
      'At-most-once is faster but may lose messages; at-least-once is slower but guarantees delivery',
      'At-most-once delivers to one consumer; at-least-once delivers to all consumers',
      'At-most-once uses queues; at-least-once uses topics',
      'At-most-once requires acknowledgments; at-least-once does not',
    ],
    correctAnswer: 0,
    explanation:
      'At-most-once: Fast, fire-and-forget, but message may be lost (no retries). At-least-once: Slower, with acknowledgments and retries, guarantees delivery but may deliver duplicates. At-least-once is most common in production (reliability > speed). Consumer must be idempotent (handle duplicates). At-most-once used for non-critical data (metrics, logs).',
  },
  {
    id: 'mc3',
    question: 'What is a Dead Letter Queue (DLQ), and when is it used?',
    options: [
      'A queue for messages that are processed successfully',
      'A queue for high-priority messages that need immediate processing',
      'A queue for messages that fail processing repeatedly after multiple retries',
      'A queue for messages that expired due to TTL',
    ],
    correctAnswer: 2,
    explanation:
      'Dead Letter Queue (DLQ) holds messages that fail processing after multiple retries. Example: Consumer tries to process message, fails 3 times (bad data, bug, external service down). After 3 failures, message moved to DLQ (stop retrying). Operations team inspects DLQ, fixes issue, reprocesses messages. Benefit: Prevents poison pill messages from blocking queue forever.',
  },
  {
    id: 'mc4',
    question: 'When should you use a Topic (pub/sub) instead of a Queue?',
    options: [
      'When you want each message processed by exactly one consumer (task distribution)',
      'When you want each message delivered to multiple subscribers (event broadcasting)',
      'When you need high throughput and low latency',
      'When messages need to be processed in order',
    ],
    correctAnswer: 1,
    explanation:
      'Use Topic (pub/sub) when one message needs to be delivered to multiple subscribers (event broadcasting). Example: "PaymentCompleted" event delivered to Email service, Analytics service, Inventory service. Each subscriber processes message independently. Use Queue when message should be processed by exactly one consumer (task distribution). Example: "Process video encoding" task consumed by one worker.',
  },
  {
    id: 'mc5',
    question: 'Your message consumer must be idempotent. What does this mean?',
    options: [
      'The consumer must process messages in order',
      'The consumer must process messages very fast',
      'Processing the same message multiple times has the same effect as processing it once',
      'The consumer must acknowledge messages immediately',
    ],
    correctAnswer: 2,
    explanation:
      'Idempotent: Processing same message multiple times has same effect as processing once. Why needed: At-least-once delivery may deliver duplicates (if consumer crashes after processing but before acknowledging). Example: Payment consumer tracks processed message IDs. On duplicate, skips processing (payment already made). Non-idempotent example: Incrementing counter (duplicate message → counter incremented twice, wrong result).',
  },
];
