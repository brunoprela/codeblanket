/**
 * Multiple Choice Questions for Apache Kafka Architecture
 */

import { MultipleChoiceQuestion } from '../../../types';

export const apacheKafkaArchitectureMC: MultipleChoiceQuestion[] = [
  {
    id: 'apache-kafka-architecture-mc-1',
    question: 'In Kafka, what is the purpose of partitions within a topic?',
    options: [
      'To encrypt messages for security',
      'To enable parallel processing and horizontal scalability',
      'To automatically compress messages',
      'To replicate messages across data centers',
    ],
    correctAnswer: 1,
    explanation:
      'Partitions enable parallel processing and horizontal scalability in Kafka. Each partition is an ordered, immutable sequence of messages that can be processed independently. Multiple consumers can read from different partitions simultaneously, enabling high throughput. For example, a topic with 10 partitions can be consumed by up to 10 consumers in parallel, each processing one partition.',
  },
  {
    id: 'apache-kafka-architecture-mc-2',
    question: 'What is the role of the replication factor in Kafka?',
    options: [
      'It determines how many times a message is sent to consumers',
      'It specifies the number of copies of each partition maintained across brokers for fault tolerance',
      'It controls the compression level for messages',
      'It sets the number of consumer groups that can read from a topic',
    ],
    correctAnswer: 1,
    explanation:
      'The replication factor specifies how many copies of each partition are maintained across different brokers for fault tolerance. For example, a replication factor of 3 means each partition has one leader and two followers. If a broker fails, one of the followers can be promoted to leader, ensuring no data loss. A higher replication factor increases durability but requires more storage.',
  },
  {
    id: 'apache-kafka-architecture-mc-3',
    question: 'In Kafka, what is the role of a consumer group?',
    options: [
      'To compress messages before storing them',
      'To enable load balancing among multiple consumers of the same application',
      'To replicate messages across brokers',
      'To manage producer connections to brokers',
    ],
    correctAnswer: 1,
    explanation:
      'A consumer group enables load balancing among multiple consumers of the same application. Each partition is assigned to exactly one consumer within a group, allowing parallel processing. For example, if a topic has 12 partitions and a consumer group has 3 consumers, each consumer processes 4 partitions. This enables horizontal scaling of consumers while ensuring each message is processed once per consumer group.',
  },
  {
    id: 'apache-kafka-architecture-mc-4',
    question: 'What happens to messages in Kafka after they are consumed?',
    options: [
      'They are immediately deleted',
      'They remain in the log for a configured retention period, regardless of consumption',
      'They are moved to a separate archive topic',
      'They are compressed to save storage',
    ],
    correctAnswer: 1,
    explanation:
      'Unlike traditional message queues, Kafka retains messages for a configured retention period (e.g., 7 days) regardless of whether they have been consumed. This allows multiple consumers to read the same messages, enables replay for debugging or reprocessing, and supports stream processing applications that may need to re-read historical data. Messages are eventually deleted based on retention policy (time-based or size-based).',
  },
  {
    id: 'apache-kafka-architecture-mc-5',
    question:
      'What is the maximum number of consumers that can efficiently read from a single Kafka topic with 8 partitions within one consumer group?',
    options: [
      'Unlimited consumers',
      '1 consumer',
      '8 consumers',
      '16 consumers',
    ],
    correctAnswer: 2,
    explanation:
      'The maximum number of consumers that can efficiently read from a topic is equal to the number of partitions. In this case, with 8 partitions, you can have up to 8 consumers in a consumer group. Each consumer will be assigned one partition. If you have more than 8 consumers (e.g., 12), the extra 4 consumers will be idle with no partitions assigned. This 1:1 mapping between partitions and consumers is a key Kafka design principle.',
  },
];
