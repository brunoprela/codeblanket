/**
 * System Design Trade-offs Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { consistencyvsavailabilitySection } from '../sections/system-design-tradeoffs/consistency-vs-availability';
import { latencyvsthroughputSection } from '../sections/system-design-tradeoffs/latency-vs-throughput';
import { strongvseventualconsistencySection } from '../sections/system-design-tradeoffs/strong-vs-eventual-consistency';
import { syncvsasynccommunicationSection } from '../sections/system-design-tradeoffs/sync-vs-async-communication';
import { normalizationvsdenormalizationSection } from '../sections/system-design-tradeoffs/normalization-vs-denormalization';
import { verticalvshorizontalscalingSection } from '../sections/system-design-tradeoffs/vertical-vs-horizontal-scaling';
import { sqlvsnosqlSection } from '../sections/system-design-tradeoffs/sql-vs-nosql';
import { monolithvsmicroservicesSection } from '../sections/system-design-tradeoffs/monolith-vs-microservices';
import { pushvspullmodelsSection } from '../sections/system-design-tradeoffs/push-vs-pull-models';
import { inmemoryvspersistentstorageSection } from '../sections/system-design-tradeoffs/in-memory-vs-persistent-storage';
import { batchvsstreamprocessingSection } from '../sections/system-design-tradeoffs/batch-vs-stream-processing';

// Import quizzes
import { consistencyvsavailabilityQuiz } from '../quizzes/system-design-tradeoffs/consistency-vs-availability';
import { latencyvsthroughputQuiz } from '../quizzes/system-design-tradeoffs/latency-vs-throughput';
import { strongvseventualconsistencyQuiz } from '../quizzes/system-design-tradeoffs/strong-vs-eventual-consistency';
import { syncvsasynccommunicationQuiz } from '../quizzes/system-design-tradeoffs/sync-vs-async-communication';
import { normalizationvsdenormalizationQuiz } from '../quizzes/system-design-tradeoffs/normalization-vs-denormalization';
import { verticalvshorizontalscalingQuiz } from '../quizzes/system-design-tradeoffs/vertical-vs-horizontal-scaling';
import { sqlvsnosqlQuiz } from '../quizzes/system-design-tradeoffs/sql-vs-nosql';
import { monolithvsmicroservicesQuiz } from '../quizzes/system-design-tradeoffs/monolith-vs-microservices';
import { pushvspullmodelsQuiz } from '../quizzes/system-design-tradeoffs/push-vs-pull-models';
import { inmemoryvspersistentstorageQuiz } from '../quizzes/system-design-tradeoffs/in-memory-vs-persistent-storage';
import { batchvsstreamprocessingQuiz } from '../quizzes/system-design-tradeoffs/batch-vs-stream-processing';

// Import multiple choice
import { consistencyvsavailabilityMultipleChoice } from '../multiple-choice/system-design-tradeoffs/consistency-vs-availability';
import { latencyvsthroughputMultipleChoice } from '../multiple-choice/system-design-tradeoffs/latency-vs-throughput';
import { strongvseventualconsistencyMultipleChoice } from '../multiple-choice/system-design-tradeoffs/strong-vs-eventual-consistency';
import { syncvsasynccommunicationMultipleChoice } from '../multiple-choice/system-design-tradeoffs/sync-vs-async-communication';
import { normalizationvsdenormalizationMultipleChoice } from '../multiple-choice/system-design-tradeoffs/normalization-vs-denormalization';
import { verticalvshorizontalscalingMultipleChoice } from '../multiple-choice/system-design-tradeoffs/vertical-vs-horizontal-scaling';
import { sqlvsnosqlMultipleChoice } from '../multiple-choice/system-design-tradeoffs/sql-vs-nosql';
import { monolithvsmicroservicesMultipleChoice } from '../multiple-choice/system-design-tradeoffs/monolith-vs-microservices';
import { pushvspullmodelsMultipleChoice } from '../multiple-choice/system-design-tradeoffs/push-vs-pull-models';
import { inmemoryvspersistentstorageMultipleChoice } from '../multiple-choice/system-design-tradeoffs/in-memory-vs-persistent-storage';
import { batchvsstreamprocessingMultipleChoice } from '../multiple-choice/system-design-tradeoffs/batch-vs-stream-processing';

export const systemDesignTradeoffsModule: Module = {
  id: 'system-design-tradeoffs',
  title: 'System Design Trade-offs',
  description:
    'Master the art of making architectural decisions and discussing trade-offs in system design interviews',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '⚖️',
  keyTakeaways: [
    'CAP theorem: During network partitions, choose Consistency (CP) or Availability (AP)',
    'Consistency (CP): Banking, payments, inventory - accuracy critical, sacrifice availability',
    'Availability (AP): Social media, catalogs - UX critical, sacrifice perfect consistency',
    'Latency: Time per operation (user-facing) vs Throughput: Total operations (batch jobs)',
    'Strong consistency: All reads see latest write (higher latency, lower availability)',
    'Eventual consistency: Reads may be stale temporarily (lower latency, higher availability)',
    'Most production systems use hybrid approaches: Different consistency for different data',
    'Synchronous: Simple, immediate feedback, tight coupling vs Asynchronous: Scalable, resilient, complex',
    'Normalization: Data integrity, less redundancy vs Denormalization: Query performance, more redundancy',
    'Vertical scaling: Simple but limited vs Horizontal scaling: Complex but unlimited',
    'SQL: ACID, structure, complex queries vs NoSQL: BASE, flexibility, scale',
    'Monolith: Simple, fast development vs Microservices: Scalable, complex operations',
    'Push: Real-time, low latency, persistent connections vs Pull: On-demand, stateless, easier to scale',
    'In-memory: 100x faster, volatile, high cost/GB vs Persistent: Durable, large capacity, low cost/GB',
    'Batch: High throughput, high latency (hours) vs Stream: Low latency (seconds), complex, higher cost',
    'Batching increases throughput at cost of latency (good for background jobs)',
    'Use percentiles (P95, P99) not averages to measure latency',
    "Little's Law: Throughput = Concurrency / Latency",
    'Read-your-writes consistency: Critical for UX in eventually consistent systems',
    'Twitter uses hybrid feed generation: Push for regular users, pull for celebrities',
    'Cache hit rate target: >90% for well-configured caches',
    'Lambda Architecture: Combine batch (accuracy) + stream (latency) for best of both worlds',
  ],
  learningObjectives: [
    'Understand CAP theorem and when to prioritize consistency vs availability',
    'Explain the difference between latency and throughput',
    'Choose appropriate consistency models based on use case requirements',
    'Analyze trade-offs between synchronous and asynchronous communication',
    'Decide when to normalize vs denormalize database schemas',
    'Compare vertical and horizontal scaling approaches',
    'Understand when to choose SQL vs NoSQL databases',
    'Evaluate monolith vs microservices architecture decisions',
    'Determine when to use push vs pull models for data delivery',
    'Design hybrid storage strategies combining in-memory and persistent storage',
    'Choose between batch and stream processing based on latency requirements',
    "Apply Little's Law to capacity planning",
    'Design hybrid consistency models for different data types',
    'Implement conflict resolution strategies for eventual consistency',
    'Justify architectural decisions with clear trade-off analysis',
    'Design caching strategies with appropriate TTLs and eviction policies',
  ],
  sections: [
    {
      ...consistencyvsavailabilitySection,
      quiz: consistencyvsavailabilityQuiz,
      multipleChoice: consistencyvsavailabilityMultipleChoice,
    },
    {
      ...latencyvsthroughputSection,
      quiz: latencyvsthroughputQuiz,
      multipleChoice: latencyvsthroughputMultipleChoice,
    },
    {
      ...strongvseventualconsistencySection,
      quiz: strongvseventualconsistencyQuiz,
      multipleChoice: strongvseventualconsistencyMultipleChoice,
    },
    {
      ...syncvsasynccommunicationSection,
      quiz: syncvsasynccommunicationQuiz,
      multipleChoice: syncvsasynccommunicationMultipleChoice,
    },
    {
      ...normalizationvsdenormalizationSection,
      quiz: normalizationvsdenormalizationQuiz,
      multipleChoice: normalizationvsdenormalizationMultipleChoice,
    },
    {
      ...verticalvshorizontalscalingSection,
      quiz: verticalvshorizontalscalingQuiz,
      multipleChoice: verticalvshorizontalscalingMultipleChoice,
    },
    {
      ...sqlvsnosqlSection,
      quiz: sqlvsnosqlQuiz,
      multipleChoice: sqlvsnosqlMultipleChoice,
    },
    {
      ...monolithvsmicroservicesSection,
      quiz: monolithvsmicroservicesQuiz,
      multipleChoice: monolithvsmicroservicesMultipleChoice,
    },
    {
      ...pushvspullmodelsSection,
      quiz: pushvspullmodelsQuiz,
      multipleChoice: pushvspullmodelsMultipleChoice,
    },
    {
      ...inmemoryvspersistentstorageSection,
      quiz: inmemoryvspersistentstorageQuiz,
      multipleChoice: inmemoryvspersistentstorageMultipleChoice,
    },
    {
      ...batchvsstreamprocessingSection,
      quiz: batchvsstreamprocessingQuiz,
      multipleChoice: batchvsstreamprocessingMultipleChoice,
    },
  ],
};
