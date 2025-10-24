/**
 * Multiple Choice Questions for Kafka Streams
 */

import { MultipleChoiceQuestion } from '../../../types';

export const kafkaStreamsMC: MultipleChoiceQuestion[] = [
  {
    id: 'kafka-streams-mc-1',
    question:
      'What is the key difference between KStream and KTable in Kafka Streams?',
    options: [
      { id: 'a', text: 'KStream is faster than KTable' },
      {
        id: 'b',
        text: 'KStream represents a stream of events (inserts), KTable represents a changelog (upserts)',
      },
      { id: 'c', text: 'KStream stores data in memory, KTable stores on disk' },
      { id: 'd', text: 'KStream is for producers, KTable is for consumers' },
    ],
    correctAnswer: 'b',
    explanation:
      'KStream represents an unbounded stream of immutable events (facts), where each record is a new event. KTable represents a changelog of mutable state, where each record is an update (upsert) to the current value for a key. For example, user clicks are a KStream (each click is an event), while user profiles are a KTable (updates to current state). KTable can be queried for the latest value of a key, while KStream contains all historical events.',
  },
  {
    id: 'kafka-streams-mc-2',
    question:
      'What does "exactly-once semantics" (EOS) in Kafka Streams guarantee?',
    options: [
      {
        id: 'a',
        text: 'Messages are processed in exactly the order they were sent',
      },
      {
        id: 'b',
        text: 'Each input record is processed exactly once, with no duplicates in output',
      },
      { id: 'c', text: 'Each consumer reads each message exactly once' },
      { id: 'd', text: 'Messages are never lost' },
    ],
    correctAnswer: 'b',
    explanation:
      'Exactly-once semantics (EOS) in Kafka Streams guarantees that each input record affects the final results exactly once, even if processing fails and retries occur. This means no duplicate outputs or lost messages. Kafka Streams achieves this through idempotent producers and transactional writes. For example, if processing crashes after writing to output topic but before committing input offset, upon restart, the same input is processed again but the duplicate output is prevented. Set processing.guarantee=exactly_once_v2 to enable.',
  },
  {
    id: 'kafka-streams-mc-3',
    question: 'Where is state stored in a Kafka Streams application?',
    options: [
      { id: 'a', text: 'Only in memory' },
      { id: 'b', text: 'In an external database like PostgreSQL' },
      {
        id: 'c',
        text: 'Locally in RocksDB, with a changelog topic in Kafka for fault tolerance',
      },
      { id: 'd', text: 'In Kafka topics only' },
    ],
    correctAnswer: 'c',
    explanation:
      'Kafka Streams stores state locally in embedded RocksDB databases for fast access, and backs it up to Kafka changelog topics for fault tolerance. When performing aggregations or joins, state is written to RocksDB and mirrored to a changelog topic. If an instance fails, the new instance can restore state from the changelog. This design provides fast local queries while ensuring durability through Kafka\'s replication. For example, a word count state is stored in RocksDB and backed up to "word-count-changelog" topic.',
  },
  {
    id: 'kafka-streams-mc-4',
    question: 'What is a tumbling window in Kafka Streams?',
    options: [
      { id: 'a', text: 'A window that overlaps with adjacent windows' },
      { id: 'b', text: 'A fixed-size, non-overlapping time window' },
      { id: 'c', text: 'A window based on session gaps' },
      { id: 'd', text: 'A window that adapts size based on message volume' },
    ],
    correctAnswer: 'b',
    explanation:
      'A tumbling window is a fixed-size, non-overlapping time window. For example, 5-minute tumbling windows are [00:00-00:05), [00:05-00:10), [00:10-00:15), etc. Each event belongs to exactly one window. Use tumbling windows for periodic aggregations like "page views every 5 minutes" or "sales per hour". This is different from hopping windows (which overlap) or session windows (which are based on inactivity gaps).',
  },
  {
    id: 'kafka-streams-mc-5',
    question: 'How do you scale a Kafka Streams application?',
    options: [
      { id: 'a', text: 'Increase memory allocation' },
      {
        id: 'b',
        text: 'Deploy more application instances (horizontal scaling)',
      },
      { id: 'c', text: 'Increase the number of brokers' },
      { id: 'd', text: 'Use a faster CPU' },
    ],
    correctAnswer: 'b',
    explanation:
      "Kafka Streams applications scale horizontally by deploying more instances. Each instance is assigned a subset of partitions from the input topics, and they coordinate automatically through Kafka's consumer group protocol. For example, with 12 partitions and 3 instances, each instance processes 4 partitions. Adding a 4th instance redistributes to 3 partitions each. Maximum parallelism equals the number of partitions in the input topic. No cluster management or coordination service is needed—just start more instances.",
  },
];
