/**
 * Multiple Choice Questions for RabbitMQ
 */

import { MultipleChoiceQuestion } from '../../../types';

export const rabbitmqMC: MultipleChoiceQuestion[] = [
  {
    id: 'rabbitmq-mc-1',
    question: 'What is the role of an exchange in RabbitMQ?',
    options: [
      'To store messages permanently',
      'To receive messages from producers and route them to queues based on routing rules',
      'To consume messages from queues',
      'To compress messages for storage',
    ],
    correctAnswer: 1,
    explanation:
      'An exchange receives messages from producers and routes them to one or more queues based on routing rules (bindings and routing keys). The producer sends to an exchange, not directly to queues. RabbitMQ provides different exchange types: direct (exact routing key match), topic (pattern matching), fanout (broadcast to all queues), and headers (based on message headers). The exchange is the message routing logic layer in RabbitMQ.',
  },
  {
    id: 'rabbitmq-mc-2',
    question:
      'Which RabbitMQ exchange type broadcasts messages to all bound queues, ignoring routing keys?',
    options: [
      'Direct exchange',
      'Topic exchange',
      'Fanout exchange',
      'Headers exchange',
    ],
    correctAnswer: 2,
    explanation:
      'A fanout exchange broadcasts messages to all bound queues, completely ignoring the routing key. This is useful for event broadcasting where multiple services need to receive the same event. For example, when an order is placed, you might want to notify payment service, inventory service, and email service simultaneously. Simply bind all three queues to the fanout exchange, and every message sent to the exchange is delivered to all queues.',
  },
  {
    id: 'rabbitmq-mc-3',
    question:
      'What does manual acknowledgment (manual ack) in RabbitMQ ensure?',
    options: [
      'Messages are processed faster',
      'Messages are not removed from the queue until the consumer explicitly acknowledges successful processing',
      'Messages are automatically retried',
      'Messages are compressed',
    ],
    correctAnswer: 1,
    explanation:
      'Manual acknowledgment ensures messages are not removed from the queue until the consumer explicitly sends an ack (acknowledgment) after successfully processing. If the consumer crashes before acking, RabbitMQ redelivers the message to another consumer. This provides at-least-once delivery guarantee. With auto-ack (default), messages are immediately removed upon delivery, risking data loss if the consumer crashes before processing. For critical data, always use manual ack.',
  },
  {
    id: 'rabbitmq-mc-4',
    question: 'What is a quorum queue in RabbitMQ?',
    options: [
      'A queue that requires manual acknowledgments',
      'A replicated queue type providing high availability through distributed consensus',
      'A queue that prioritizes messages',
      'A queue for storing dead letters',
    ],
    correctAnswer: 1,
    explanation:
      'Quorum queues are a replicated queue type that uses Raft consensus algorithm for high availability and data safety. Unlike classic mirrored queues, quorum queues provide stronger guarantees: data is replicated to a majority of nodes (quorum) before confirming writes. If the leader node fails, a follower is automatically elected. Quorum queues are the recommended HA option in RabbitMQ 3.8+, replacing classic mirrored queues. Trade-off: slightly higher latency for stronger durability.',
  },
  {
    id: 'rabbitmq-mc-5',
    question:
      'In RabbitMQ topic exchanges, what do the wildcards "*" and "#" mean?',
    options: [
      '"*" matches any string, "#" matches zero or more words separated by dots',
      '"*" matches one word, "#" matches zero or more words separated by dots',
      '"*" matches numbers only, "#" matches letters only',
      '"*" and "#" are the same',
    ],
    correctAnswer: 1,
    explanation:
      'In topic exchanges, "*" matches exactly one word, and "#" matches zero or more words. Words are separated by dots. For example, routing key "user.created" matches patterns "user.*" and "#", but not "user.*.update". Routing key "order.payment.completed" matches "order.#" (zero or more words) and "order.*.completed" (exactly one word between). This enables flexible pub/sub patterns like subscribing to all ERROR logs from any service: "*.ERROR".',
  },
];
