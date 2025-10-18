/**
 * Database Design & Theory Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { sqlvsnosqldecisionSection } from '../sections/system-design-database-design/sql-vs-nosql-decision';
import { captheoremSection } from '../sections/system-design-database-design/cap-theorem';
import { pacelctheoremSection } from '../sections/system-design-database-design/pacelc-theorem';
import { consistencymodelsSection } from '../sections/system-design-database-design/consistency-models';
import { acidvsbaseSection } from '../sections/system-design-database-design/acid-vs-base';
import { databaseindexingSection } from '../sections/system-design-database-design/database-indexing';
import { normalizationdenormalizationSection } from '../sections/system-design-database-design/normalization-denormalization';
import { databasetransactionslockingSection } from '../sections/system-design-database-design/database-transactions-locking';
import { databaseconnectionpoolingSection } from '../sections/system-design-database-design/database-connection-pooling';
import { timeseriesspecializedSection } from '../sections/system-design-database-design/timeseries-specialized';

// Import quizzes
import { sqlvsnosqldecisionQuiz } from '../quizzes/system-design-database-design/sql-vs-nosql-decision';
import { captheoremQuiz } from '../quizzes/system-design-database-design/cap-theorem';
import { pacelctheoremQuiz } from '../quizzes/system-design-database-design/pacelc-theorem';
import { consistencymodelsQuiz } from '../quizzes/system-design-database-design/consistency-models';
import { acidvsbaseQuiz } from '../quizzes/system-design-database-design/acid-vs-base';
import { databaseindexingQuiz } from '../quizzes/system-design-database-design/database-indexing';
import { normalizationdenormalizationQuiz } from '../quizzes/system-design-database-design/normalization-denormalization';
import { databasetransactionslockingQuiz } from '../quizzes/system-design-database-design/database-transactions-locking';
import { databaseconnectionpoolingQuiz } from '../quizzes/system-design-database-design/database-connection-pooling';
import { timeseriesspecializedQuiz } from '../quizzes/system-design-database-design/timeseries-specialized';

// Import multiple choice
import { sqlvsnosqldecisionMultipleChoice } from '../multiple-choice/system-design-database-design/sql-vs-nosql-decision';
import { captheoremMultipleChoice } from '../multiple-choice/system-design-database-design/cap-theorem';
import { pacelctheoremMultipleChoice } from '../multiple-choice/system-design-database-design/pacelc-theorem';
import { consistencymodelsMultipleChoice } from '../multiple-choice/system-design-database-design/consistency-models';
import { acidvsbaseMultipleChoice } from '../multiple-choice/system-design-database-design/acid-vs-base';
import { databaseindexingMultipleChoice } from '../multiple-choice/system-design-database-design/database-indexing';
import { normalizationdenormalizationMultipleChoice } from '../multiple-choice/system-design-database-design/normalization-denormalization';
import { databasetransactionslockingMultipleChoice } from '../multiple-choice/system-design-database-design/database-transactions-locking';
import { databaseconnectionpoolingMultipleChoice } from '../multiple-choice/system-design-database-design/database-connection-pooling';
import { timeseriesspecializedMultipleChoice } from '../multiple-choice/system-design-database-design/timeseries-specialized';

export const systemDesignDatabaseDesignModule: Module = {
  id: 'system-design-database-design',
  title: 'Database Design & Theory',
  description:
    'Deep dive into database selection, design patterns, scaling strategies, and understanding when to use SQL vs NoSQL',
  category: 'undefined',
  difficulty: 'medium',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🗄️',
  keyTakeaways: [
    'SQL databases provide ACID transactions, complex JOINs, and strong consistency',
    'NoSQL databases provide horizontal scalability, flexible schemas, and specific optimizations',
    'Choose SQL for: ACID, complex relationships, stable schemas, business intelligence',
    'Choose NoSQL for: Massive scale, flexible schemas, simple access patterns, high throughput',
    'Polyglot persistence (multiple databases) is common in production',
    'Start with SQL; add NoSQL for specific needs',
    'CAP theorem: Choose 2 of 3 (Consistency, Availability, Partition Tolerance) during partitions',
    'PACELC extends CAP: Also consider Latency vs Consistency during normal operation',
    'PA/EL systems (Cassandra, DynamoDB) optimize for availability and low latency',
    'PC/EC systems (HBase, Spanner) optimize for strong consistency',
    'Design NoSQL data models based on access patterns, not normalized forms',
    'Database indexing speeds reads but slows writes - use strategically',
    'Normalization reduces redundancy; denormalization improves read performance',
    'Consider operational complexity when choosing multiple database systems',
    'In interviews, discuss trade-offs and justify choices based on requirements',
  ],
  learningObjectives: [
    'Understand the fundamental differences between SQL and NoSQL databases',
    'Know when to choose SQL vs NoSQL based on requirements',
    'Master the four NoSQL categories: Document, Key-Value, Column-Family, Graph',
    'Understand ACID transactions and when they are critical',
    'Learn to design data models for specific access patterns',
    'Understand polyglot persistence and when to use multiple databases',
    'Learn to discuss database trade-offs in system design interviews',
    'Understand CAP theorem and consistency models',
    'Know real-world examples of SQL and NoSQL usage at scale',
    'Master the decision framework for database selection',
  ],
  sections: [
    {
      ...sqlvsnosqldecisionSection,
      quiz: sqlvsnosqldecisionQuiz,
      multipleChoice: sqlvsnosqldecisionMultipleChoice,
    },
    {
      ...captheoremSection,
      quiz: captheoremQuiz,
      multipleChoice: captheoremMultipleChoice,
    },
    {
      ...pacelctheoremSection,
      quiz: pacelctheoremQuiz,
      multipleChoice: pacelctheoremMultipleChoice,
    },
    {
      ...consistencymodelsSection,
      quiz: consistencymodelsQuiz,
      multipleChoice: consistencymodelsMultipleChoice,
    },
    {
      ...acidvsbaseSection,
      quiz: acidvsbaseQuiz,
      multipleChoice: acidvsbaseMultipleChoice,
    },
    {
      ...databaseindexingSection,
      quiz: databaseindexingQuiz,
      multipleChoice: databaseindexingMultipleChoice,
    },
    {
      ...normalizationdenormalizationSection,
      quiz: normalizationdenormalizationQuiz,
      multipleChoice: normalizationdenormalizationMultipleChoice,
    },
    {
      ...databasetransactionslockingSection,
      quiz: databasetransactionslockingQuiz,
      multipleChoice: databasetransactionslockingMultipleChoice,
    },
    {
      ...databaseconnectionpoolingSection,
      quiz: databaseconnectionpoolingQuiz,
      multipleChoice: databaseconnectionpoolingMultipleChoice,
    },
    {
      ...timeseriesspecializedSection,
      quiz: timeseriesspecializedQuiz,
      multipleChoice: timeseriesspecializedMultipleChoice,
    },
  ],
};
