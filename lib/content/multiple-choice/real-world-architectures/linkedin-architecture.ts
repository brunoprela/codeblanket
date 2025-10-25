/**
 * Multiple choice questions for LinkedIn Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const linkedinarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: "What is LinkedIn's Espresso database built on top of?",
    options: [
      'PostgreSQL with custom distributed layer',
      'MySQL storage engine with distributed coordination',
      'Cassandra with additional indexing layer',
      'Custom storage engine from scratch',
    ],
    correctAnswer: 1,
    explanation:
      "Espresso is built on top of MySQL storage engine with a custom distributed layer. This approach leverages MySQL's proven durability and ACID properties while adding distributed capabilities like horizontal sharding, cross-datacenter replication, and document model with secondary indexes. Espresso provides timeline consistency and handles 1000+ nodes with 100TB+ data serving millions of QPS for LinkedIn's social graph and feed data.",
  },
  {
    id: 'mc2',
    question:
      'Which LinkedIn tool is used for cross-datacenter data replication?',
    options: [
      'Espresso replication',
      'Kafka MirrorMaker',
      'Brooklin',
      'Samza streaming',
    ],
    correctAnswer: 2,
    explanation:
      "Brooklin is LinkedIn's tool for cross-datacenter data replication. It provides reliable, scalable replication of data between datacenters with configurable consistency models. Brooklin works with various data sources (Espresso, Kafka, databases) and ensures that data changes are propagated across LinkedIn's global infrastructure. It complements Kafka for event propagation and Espresso for storage replication.",
  },
  {
    id: 'mc3',
    question: "What ML evolution did LinkedIn's feed ranking undergo?",
    options: [
      'Started with deep learning, moved to simpler logistic regression',
      'Always used deep neural networks',
      'Started with gradient boosted trees, evolved to deep neural networks',
      'Uses rule-based ranking without ML',
    ],
    correctAnswer: 2,
    explanation:
      "LinkedIn's feed ranking evolved from gradient boosted trees to deep neural networks. Initially, they used gradient boosted trees (GBT) for ranking, which worked well with hand-crafted features. As computational resources and data volumes grew, LinkedIn migrated to deep neural networks with embeddings for members and posts. This allows the model to learn feature representations automatically and resulted in 10%+ increase in feed engagement.",
  },
  {
    id: 'mc4',
    question:
      'Which consistency model does LinkedIn use for most read operations like viewing profiles and feeds?',
    options: [
      'Strong consistency with synchronous cross-DC replication',
      'Timeline consistency with read-local, async cross-DC replication',
      'Eventual consistency with conflict resolution',
      'Sequential consistency with causal ordering',
    ],
    correctAnswer: 1,
    explanation:
      'LinkedIn uses timeline consistency for most read operations. Users read from their local datacenter (low latency <10ms) with guarantees that reads reflect recent writes from the same datacenter. Cross-datacenter replication happens asynchronously, typically consistent within seconds. This balances low latency with acceptable consistency for social network workloads. Strong consistency is reserved for critical operations like accepting connections or posting jobs.',
  },
  {
    id: 'mc5',
    question:
      'Which streaming technology does LinkedIn use for real-time feed updates?',
    options: [
      'Apache Flink for stream processing',
      'Kafka and Samza for stream processing',
      'AWS Kinesis with Lambda',
      'Spark Streaming with microbatching',
    ],
    correctAnswer: 1,
    explanation:
      "LinkedIn uses Kafka for event streaming and Samza for stream processing. When users post content or interact with feeds, events are published to Kafka. Samza consumes these streams in real-time, updates member feeds, triggers notifications, and updates analytics. This architecture was developed at LinkedIn (both Kafka and Samza are LinkedIn open-source projects) and enables real-time updates across LinkedIn's ecosystem.",
  },
];
