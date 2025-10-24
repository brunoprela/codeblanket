/**
 * Multiple Choice Questions for Kafka Producers
 */

import { MultipleChoiceQuestion } from '../../../types';

export const kafkaProducersMC: MultipleChoiceQuestion[] = [
  {
    id: 'kafka-producers-mc-1',
    question:
      'What is the purpose of the "acks" configuration in Kafka producers?',
    options: [
      {
        id: 'a',
        text: 'To specify how many acknowledgments the producer requires before considering a request complete',
      },
      { id: 'b', text: 'To set the number of messages to batch together' },
      { id: 'c', text: 'To control message compression level' },
      { id: 'd', text: 'To determine partition assignment strategy' },
    ],
    correctAnswer: 'a',
    explanation:
      'The "acks" configuration specifies how many acknowledgments the producer requires from brokers before considering a request complete. acks=0 (no ack, fastest but least durable), acks=1 (leader ack only, balanced), acks=all (leader + all in-sync replicas ack, slowest but most durable). For critical data like financial transactions, use acks=all to ensure durability. For logs or metrics, acks=1 may suffice.',
  },
  {
    id: 'kafka-producers-mc-2',
    question:
      'Which Kafka producer configuration helps reduce network bandwidth by sending multiple messages together?',
    options: [
      { id: 'a', text: 'compression.type' },
      { id: 'b', text: 'batch.size and linger.ms' },
      { id: 'c', text: 'acks' },
      { id: 'd', text: 'retries' },
    ],
    correctAnswer: 'b',
    explanation:
      'The batch.size and linger.ms configurations enable batching, which sends multiple messages together in one request, reducing network overhead. batch.size sets the maximum batch size in bytes, while linger.ms adds a delay (e.g., 10ms) to wait for more messages before sending. For example, linger.ms=10 waits up to 10ms to accumulate messages into batches, significantly reducing the number of requests at the cost of slight latency increase.',
  },
  {
    id: 'kafka-producers-mc-3',
    question:
      'What does enabling "enable.idempotence=true" in a Kafka producer guarantee?',
    options: [
      { id: 'a', text: 'Messages are sent faster' },
      {
        id: 'b',
        text: 'No duplicate messages are written to Kafka even if the producer retries',
      },
      { id: 'c', text: 'Messages are automatically compressed' },
      { id: 'd', text: 'Messages are replicated across all brokers' },
    ],
    correctAnswer: 'b',
    explanation:
      'Enabling idempotence (enable.idempotence=true) ensures that even if the producer retries sending a message (e.g., due to network timeout), Kafka will not write duplicates. The producer assigns a sequence number to each message, and the broker uses this to detect and reject duplicates. This is essential for exactly-once semantics. Note that idempotence automatically sets acks=all, retries=MAX_INT, and max.in.flight.requests.per.connection=5.',
  },
  {
    id: 'kafka-producers-mc-4',
    question:
      'Which partitioning strategy in Kafka ensures messages with the same key always go to the same partition?',
    options: [
      { id: 'a', text: 'Round-robin partitioning' },
      { id: 'b', text: 'Random partitioning' },
      { id: 'c', text: 'Hash-based partitioning on key' },
      { id: 'd', text: 'Sticky partitioning' },
    ],
    correctAnswer: 'c',
    explanation:
      'Hash-based partitioning on key ensures messages with the same key always go to the same partition, preserving order for that key. Kafka computes hash(key) % num_partitions to determine the partition. For example, all orders for customer "user_123" will go to the same partition, ensuring they are processed in order. This is critical for use cases like order processing, user activity tracking, and maintaining state per entity.',
  },
  {
    id: 'kafka-producers-mc-5',
    question:
      'What is the benefit of using compression in Kafka producers (e.g., compression.type=lz4)?',
    options: [
      { id: 'a', text: 'Improves message encryption' },
      {
        id: 'b',
        text: 'Reduces network bandwidth and broker storage by compressing messages',
      },
      { id: 'c', text: 'Increases message sending speed' },
      { id: 'd', text: 'Guarantees message ordering' },
    ],
    correctAnswer: 'b',
    explanation:
      'Compression reduces network bandwidth and broker storage by compressing messages before sending. LZ4 is fast with moderate compression, gzip offers higher compression but slower, snappy balances both. For example, JSON logs might compress 5-10×, significantly reducing costs. However, compression adds CPU overhead on producers and consumers (for decompression). Trade-off: smaller messages and lower network/storage costs vs. higher CPU usage.',
  },
];
