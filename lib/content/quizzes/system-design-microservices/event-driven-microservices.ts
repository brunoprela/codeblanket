/**
 * Quiz questions for Event-Driven Microservices section
 */

export const eventdrivenmicroservicesQuiz = [
  {
    id: 'q1-events',
    question:
      'What is the difference between events and commands in microservices? When would you use each?',
    sampleAnswer:
      'Commands tell service to DO something (ChargePayment, ReserveInventory). Sent to specific service. Expect success/failure response. Synchronous. Events announce something HAPPENED (OrderCreated, PaymentCompleted). Published to anyone interested (pub/sub). No response expected. Asynchronous. Use commands for: operations needing immediate feedback (user clicks "checkout" → charge payment → show confirmation). Use events for: background tasks (send email, update analytics), multiple subscribers (OrderCreated → email service, analytics service, warehouse service), loose coupling (new services can subscribe without changing publisher). Example: Use command to charge payment (need to know if it succeeded), use event to notify email service (fire and forget).',
    keyPoints: [
      'Commands: tell service to do something, specific recipient, expect response',
      'Events: announce something happened, pub/sub, no response expected',
      'Commands: synchronous, immediate feedback needed',
      'Events: asynchronous, background tasks, multiple subscribers',
      'Trade-off: Commands=tight coupling but immediate, Events=loose coupling but eventual',
    ],
  },
  {
    id: 'q2-events',
    question:
      'What is idempotency in event-driven systems? Why is it important and how do you implement it?',
    sampleAnswer:
      'Idempotency: processing same event multiple times has same effect as processing once. Important because: message brokers may deliver event multiple times (at-least-once delivery), retries on failures, network issues. Without idempotency: inventory decremented twice, email sent twice, payment charged twice. Implementation: (1) Track processed events in database with event ID, (2) Before processing, check if event ID already processed, (3) Use database transaction to update state AND mark as processed atomically. Example: INSERT INTO processed_events (eventId) before decrementing inventory. If event processed twice, second time fails on unique constraint. Alternative: use database unique constraints (orderId+status) to make operations naturally idempotent.',
    keyPoints: [
      'Idempotency: same event processed multiple times = same result',
      'Important because: at-least-once delivery, retries, network issues',
      'Without it: double charges, duplicate emails, incorrect state',
      'Implementation: track processed event IDs in database',
      'Use transactions to update state AND mark as processed atomically',
    ],
  },
  {
    id: 'q3-events',
    question:
      'Compare Kafka vs RabbitMQ for event-driven microservices. When would you choose each?',
    sampleAnswer:
      'Kafka: Distributed commit log, high throughput (millions msg/sec), persistence (events stored permanently), partitioning for parallelism, event replay (consumers can replay from any offset), consumer groups. Best for: event streaming, event sourcing, high throughput, need event history. RabbitMQ: Traditional message queue, flexible routing (exchanges, bindings), multiple protocols (AMQP, MQTT), simpler, lower throughput. Best for: task queues, RPC, complex routing, need message acknowledgment. Choose Kafka for: analytics pipeline, event sourcing, need to replay events, very high scale. Choose RabbitMQ for: simpler requirements, need flexible routing, task distribution, request-reply pattern. Both support pub/sub, but Kafka better for streaming, RabbitMQ better for routing.',
    keyPoints: [
      'Kafka: distributed log, high throughput, persistence, replay events',
      'RabbitMQ: message queue, flexible routing, simpler, lower throughput',
      'Kafka best for: event streaming, event sourcing, high scale',
      'RabbitMQ best for: task queues, routing, RPC',
      'Choose based on: throughput needs, need for event replay, complexity tolerance',
    ],
  },
];
