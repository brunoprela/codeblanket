/**
 * Multiple Choice Questions for AWS SQS & SNS
 */

import { MultipleChoiceQuestion } from '../../../types';

export const awsSqsSnsMC: MultipleChoiceQuestion[] = [
  {
    id: 'aws-sqs-sns-mc-1',
    question:
      'What is the key difference between AWS SQS Standard and FIFO queues?',
    options: [
      'Standard is faster, FIFO guarantees message ordering',
      'Standard is cheaper, FIFO is more expensive',
      'Standard queues guarantee ordering, FIFO queues do not',
      'Standard queues have unlimited throughput and best-effort ordering, FIFO queues guarantee strict ordering but limited to 3000 messages/sec',
    ],
    correctAnswer: 3,
    explanation:
      'SQS Standard queues provide unlimited throughput with best-effort ordering (messages may arrive out of order) and at-least-once delivery (duplicates possible). FIFO queues guarantee strict ordering and exactly-once processing, but are limited to 3000 messages/second (300 with batching). Use Standard for high-throughput, non-critical ordering (e.g., log processing). Use FIFO when order matters (e.g., payment transactions, order processing per customer).',
  },
  {
    id: 'aws-sqs-sns-mc-2',
    question: 'What is visibility timeout in SQS?',
    options: [
      'Time before a message is automatically deleted',
      'Time during which a message is invisible to other consumers after being received',
      'Maximum time a message can stay in the queue',
      'Time between message receives',
    ],
    correctAnswer: 1,
    explanation:
      'Visibility timeout is the duration during which a message is invisible to other consumers after being received by one consumer. This prevents multiple consumers from processing the same message simultaneously. For example, with a 30-second visibility timeout, after Consumer A receives a message, it has 30 seconds to process and delete it before the message becomes visible again (and potentially received by Consumer B). Set visibility timeout to at least your maximum expected processing time.',
  },
  {
    id: 'aws-sqs-sns-mc-3',
    question: 'How does SNS fanout pattern work with SQS?',
    options: [
      'SNS sends each message to one randomly selected SQS queue',
      'SNS broadcasts each message to all subscribed SQS queues simultaneously',
      'SNS compresses messages before sending to SQS',
      'SNS routes messages based on message content',
    ],
    correctAnswer: 1,
    explanation:
      'The SNS fanout pattern broadcasts each message published to an SNS topic to all subscribed SQS queues simultaneously. This enables parallel processing by multiple independent services. For example, publish an "order placed" event to SNS, which then delivers to payment-queue, inventory-queue, and email-queue. Each service processes the event independently. This decouples services and enables easy addition of new subscribers without changing the publisher.',
  },
  {
    id: 'aws-sqs-sns-mc-4',
    question: 'What is the purpose of a Dead Letter Queue (DLQ) in SQS?',
    options: [
      'To permanently archive all messages',
      'To store messages that fail processing after a specified number of attempts',
      'To speed up message delivery',
      'To replicate messages across regions',
    ],
    correctAnswer: 1,
    explanation:
      'A DLQ stores messages that cannot be successfully processed after a specified number of receive attempts (maxReceiveCount). For example, if maxReceiveCount=3, a message that is received 3 times but not deleted (indicating processing failures) is automatically moved to the DLQ. This prevents "poison messages" from blocking the queue indefinitely. Messages in the DLQ can be inspected, debugged, and potentially reprocessed after fixing the underlying issue.',
  },
  {
    id: 'aws-sqs-sns-mc-5',
    question: 'What is long polling in SQS and why is it beneficial?',
    options: [
      'Receiving messages in larger batches',
      'Waiting up to 20 seconds for messages to arrive instead of returning immediately if queue is empty',
      'Keeping messages in the queue longer',
      'Polling multiple queues simultaneously',
    ],
    correctAnswer: 1,
    explanation:
      'Long polling (WaitTimeSeconds up to 20 seconds) waits for messages to arrive if the queue is empty, instead of immediately returning an empty response. This reduces the number of empty receives, lowering costs and improving efficiency. For example, with short polling, you might make 100 empty requests/second ($0.40 per million = significant cost). With long polling (20 sec), you make only 3 empty requests/minute, dramatically reducing API calls and costs while maintaining responsiveness.',
  },
];
