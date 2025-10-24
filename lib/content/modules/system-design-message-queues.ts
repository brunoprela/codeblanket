/**
 * Message Queues & Event Streaming Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { messagequeuefundamentalsSection } from '../sections/system-design-message-queues/message-queue-fundamentals';
import { apachekafkaarchitectureSection } from '../sections/system-design-message-queues/apache-kafka-architecture';
import { kafkaproducersSection } from '../sections/system-design-message-queues/kafka-producers';
import { kafkaconsumersSection } from '../sections/system-design-message-queues/kafka-consumers';
import { kafkastreamsSection } from '../sections/system-design-message-queues/kafka-streams';
import { rabbitmqSection } from '../sections/system-design-message-queues/rabbitmq';
import { awssqssnsSection } from '../sections/system-design-message-queues/aws-sqs-sns';
import { eventdrivenarchitectureSection } from '../sections/system-design-message-queues/event-driven-architecture';
import { streamprocessingSection } from '../sections/system-design-message-queues/stream-processing';
import { messageschemaevolutionSection } from '../sections/system-design-message-queues/message-schema-evolution';

// Import quizzes
import { messagequeuefundamentalsQuiz } from '../quizzes/system-design-message-queues/message-queue-fundamentals';
import { apachekafkaarchitectureQuiz } from '../quizzes/system-design-message-queues/apache-kafka-architecture';
import { kafkaproducersQuiz } from '../quizzes/system-design-message-queues/kafka-producers';
import { kafkaconsumersQuiz } from '../quizzes/system-design-message-queues/kafka-consumers';
import { kafkastreamsQuiz } from '../quizzes/system-design-message-queues/kafka-streams';
import { rabbitmqQuiz } from '../quizzes/system-design-message-queues/rabbitmq';
import { awssqssnsQuiz } from '../quizzes/system-design-message-queues/aws-sqs-sns';
import { eventdrivenarchitectureQuiz } from '../quizzes/system-design-message-queues/event-driven-architecture';
import { streamprocessingQuiz } from '../quizzes/system-design-message-queues/stream-processing';
import { messageschemaevolutionQuiz } from '../quizzes/system-design-message-queues/message-schema-evolution';

// Import multiple choice
import { messageQueueFundamentalsMC } from '../multiple-choice/system-design-message-queues/message-queue-fundamentals-mc';
import { apacheKafkaArchitectureMC } from '../multiple-choice/system-design-message-queues/apache-kafka-architecture-mc';
import { kafkaProducersMC } from '../multiple-choice/system-design-message-queues/kafka-producers-mc';
import { kafkaConsumersMC } from '../multiple-choice/system-design-message-queues/kafka-consumers-mc';
import { kafkaStreamsMC } from '../multiple-choice/system-design-message-queues/kafka-streams-mc';
import { rabbitmqMC } from '../multiple-choice/system-design-message-queues/rabbitmq-mc';
import { awsSqsSnsMC } from '../multiple-choice/system-design-message-queues/aws-sqs-sns-mc';
import { eventDrivenArchitectureMC } from '../multiple-choice/system-design-message-queues/event-driven-architecture-mc';
import { streamProcessingMC } from '../multiple-choice/system-design-message-queues/stream-processing-mc';
import { messageSchemaEvolutionMC } from '../multiple-choice/system-design-message-queues/message-schema-evolution-mc';

export const systemDesignMessageQueuesModule: Module = {
  id: 'system-design-message-queues',
  title: 'Message Queues & Event Streaming',
  description:
    'Master async communication, event-driven architecture, and stream processing with Kafka, RabbitMQ, and AWS messaging services',
  category: 'system-design',
  difficulty: 'advanced',
  estimatedTime: '12-15 hours',
  prerequisites: [
    'system-design-fundamentals',
    'system-design-database-design',
  ],
  icon: '📨',
  keyTakeaways: [
    'Message queues enable asynchronous, decoupled communication in distributed systems',
    'Queue (point-to-point) for task distribution; Topic (pub/sub) for event broadcasting',
    'Delivery guarantees: at-most-once (fast), at-least-once (reliable), exactly-once (critical data)',
    'Apache Kafka: Distributed commit log with high throughput, replay capability, and event streaming',
    'Kafka partitioning enables horizontal scalability and ordering per key',
    'Producer idempotence prevents duplicates (PID + sequence numbers)',
    'Consumer groups enable parallel processing with automatic load balancing',
    'RabbitMQ: Flexible routing, lower latency, perfect for task queues and RPC patterns',
    'AWS SQS/SNS: Fully managed, zero ops, cost-effective for AWS workloads',
    'Event-driven architecture: Loose coupling, scalability, and extensibility through events',
    'Event sourcing: Immutable event log provides complete audit trail and time travel',
    'Stream processing: Real-time data processing with windowing, stateful operations, and exactly-once semantics',
    'Schema evolution: Backward/forward compatibility critical for independent service deployment',
    'Choose technology based on requirements: Kafka (streaming), RabbitMQ (routing), SQS (managed)',
  ],
  learningObjectives: [
    'Understand message queue fundamentals and when to use queues vs topics',
    'Master Kafka architecture: topics, partitions, replication, ISR, and fault tolerance',
    'Configure Kafka producers for reliability: idempotence, acks, retries, batching',
    'Design Kafka consumers with proper offset management, rebalancing, and error handling',
    'Build real-time stream processing applications with Kafka Streams',
    'Implement RabbitMQ exchanges, queues, and routing patterns',
    'Architect event-driven systems with proper event design and saga patterns',
    'Handle schema evolution with backward/forward compatibility',
    'Choose appropriate messaging technology for different use cases',
    'Design production-ready messaging systems with monitoring and alerting',
  ],
  sections: [
    {
      ...messagequeuefundamentalsSection,
      quiz: messagequeuefundamentalsQuiz,
      multipleChoice: messageQueueFundamentalsMC,
    },
    {
      ...apachekafkaarchitectureSection,
      quiz: apachekafkaarchitectureQuiz,
      multipleChoice: apacheKafkaArchitectureMC,
    },
    {
      ...kafkaproducersSection,
      quiz: kafkaproducersQuiz,
      multipleChoice: kafkaProducersMC,
    },
    {
      ...kafkaconsumersSection,
      quiz: kafkaconsumersQuiz,
      multipleChoice: kafkaConsumersMC,
    },
    {
      ...kafkastreamsSection,
      quiz: kafkastreamsQuiz,
      multipleChoice: kafkaStreamsMC,
    },
    {
      ...rabbitmqSection,
      quiz: rabbitmqQuiz,
      multipleChoice: rabbitmqMC,
    },
    {
      ...awssqssnsSection,
      quiz: awssqssnsQuiz,
      multipleChoice: awsSqsSnsMC,
    },
    {
      ...eventdrivenarchitectureSection,
      quiz: eventdrivenarchitectureQuiz,
      multipleChoice: eventDrivenArchitectureMC,
    },
    {
      ...streamprocessingSection,
      quiz: streamprocessingQuiz,
      multipleChoice: streamProcessingMC,
    },
    {
      ...messageschemaevolutionSection,
      quiz: messageschemaevolutionQuiz,
      multipleChoice: messageSchemaEvolutionMC,
    },
  ],
};
