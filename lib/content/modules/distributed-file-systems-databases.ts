/**
 * System Design: Distributed File Systems & Databases Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { gfsSection } from '../sections/distributed-file-systems-databases/google-file-system';
import { hdfsSection } from '../sections/distributed-file-systems-databases/hdfs';
import { s3ArchitectureSection } from '../sections/distributed-file-systems-databases/s3-architecture';
import { blobStorageSection } from '../sections/distributed-file-systems-databases/blob-storage-patterns';
import { distributedObjectStorageSection } from '../sections/distributed-file-systems-databases/distributed-object-storage';
import { bigtableSection } from '../sections/distributed-file-systems-databases/google-bigtable';
import { cassandraSection } from '../sections/distributed-file-systems-databases/apache-cassandra';
import { dynamodbSection } from '../sections/distributed-file-systems-databases/dynamodb';
import { mongodbSection } from '../sections/distributed-file-systems-databases/mongodb';
import { redisSection } from '../sections/distributed-file-systems-databases/redis-deep-dive';
import { hbaseSection } from '../sections/distributed-file-systems-databases/apache-hbase';
import { distributedTransactionsSection } from '../sections/distributed-file-systems-databases/distributed-transactions';

// Import quizzes
import { gfsQuiz } from '../quizzes/distributed-file-systems-databases/google-file-system-discussion';
import { hdfsQuiz } from '../quizzes/distributed-file-systems-databases/hdfs-discussion';
import { s3ArchitectureQuiz } from '../quizzes/distributed-file-systems-databases/s3-architecture-discussion';
import { blobStorageQuiz } from '../quizzes/distributed-file-systems-databases/blob-storage-patterns-discussion';
import { distributedObjectStorageQuiz } from '../quizzes/distributed-file-systems-databases/distributed-object-storage-discussion';
import { bigtableQuiz } from '../quizzes/distributed-file-systems-databases/google-bigtable-discussion';
import { cassandraQuiz } from '../quizzes/distributed-file-systems-databases/apache-cassandra-discussion';
import { dynamodbQuiz } from '../quizzes/distributed-file-systems-databases/dynamodb-discussion';
import { mongodbQuiz } from '../quizzes/distributed-file-systems-databases/mongodb-discussion';
import { redisQuiz } from '../quizzes/distributed-file-systems-databases/redis-deep-dive-discussion';
import { hbaseQuiz } from '../quizzes/distributed-file-systems-databases/apache-hbase-discussion';
import { distributedTransactionsQuiz } from '../quizzes/distributed-file-systems-databases/distributed-transactions-discussion';

// Import multiple choice
import { gfsMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/google-file-system-mc';
import { hdfsMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/hdfs-mc';
import { s3ArchitectureMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/s3-architecture-mc';
import { blobStorageMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/blob-storage-patterns-mc';
import { distributedObjectStorageMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/distributed-object-storage-mc';
import { bigtableMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/google-bigtable-mc';
import { cassandraMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/apache-cassandra-mc';
import { dynamodbMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/dynamodb-mc';
import { mongodbMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/mongodb-mc';
import { redisMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/redis-deep-dive-mc';
import { hbaseMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/apache-hbase-mc';
import { distributedTransactionsMultipleChoice } from '../multiple-choice/distributed-file-systems-databases/distributed-transactions-mc';

export const distributedFileSystemsDatabasesModule: Module = {
  id: 'distributed-file-systems-databases',
  title: 'System Design: Distributed File Systems & Databases',
  description:
    'Master large-scale storage systems, distributed file systems, and specialized databases including GFS, HDFS, S3, BigTable, Cassandra, DynamoDB, MongoDB, Redis, and HBase',
  category: 'System Design',
  difficulty: 'Advanced',
  estimatedTime: '6-8 hours',
  prerequisites: [
    'system-design-fundamentals',
    'system-design-database-design',
  ],
  icon: '💾',
  keyTakeaways: [
    'GFS pioneered distributed file systems with single master architecture and chunk-based storage',
    'HDFS brings GFS concepts to Hadoop ecosystem with NameNode/DataNode architecture',
    'Amazon S3 provides highly available object storage with eventual consistency (now strong consistency)',
    'Object storage differs from file systems: no hierarchy, metadata-rich, HTTP access',
    'BigTable pioneered wide-column stores with SSTable and LSM-tree architecture',
    'Cassandra provides masterless, highly available distributed database with tunable consistency',
    'DynamoDB offers managed NoSQL with partition/sort keys and eventual consistency by default',
    'MongoDB provides document-oriented storage with flexible schema and rich query language',
    'Redis excels at in-memory caching with diverse data structures and pub/sub capabilities',
    'HBase provides strong consistency on HDFS with column-family storage',
    'Two-phase commit (2PC) provides distributed transactions but blocks on coordinator failure',
    'Paxos and Raft achieve consensus in distributed systems through majority voting',
    'Distributed transactions trade performance for consistency - use sparingly',
  ],
  learningObjectives: [
    'Understand GFS architecture and design decisions',
    'Explain HDFS NameNode/DataNode model and replication strategy',
    'Design systems using Amazon S3 and understand consistency models',
    'Compare object storage vs file systems vs block storage',
    "Understand BigTable's SSTable and LSM-tree architecture",
    'Design highly available systems with Cassandra',
    'Implement efficient data models in DynamoDB with partition/sort keys',
    'Choose between SQL and NoSQL databases for different use cases',
    'Optimize Redis for caching, session storage, and real-time features',
    'Understand when to use specialized databases (graph, time-series, document)',
    'Implement distributed transactions with 2PC and understand limitations',
    'Explain Paxos and Raft consensus algorithms',
  ],
  sections: [
    {
      ...gfsSection,
      quiz: gfsQuiz,
      multipleChoice: gfsMultipleChoice,
    },
    {
      ...hdfsSection,
      quiz: hdfsQuiz,
      multipleChoice: hdfsMultipleChoice,
    },
    {
      ...s3ArchitectureSection,
      quiz: s3ArchitectureQuiz,
      multipleChoice: s3ArchitectureMultipleChoice,
    },
    {
      ...blobStorageSection,
      quiz: blobStorageQuiz,
      multipleChoice: blobStorageMultipleChoice,
    },
    {
      ...distributedObjectStorageSection,
      quiz: distributedObjectStorageQuiz,
      multipleChoice: distributedObjectStorageMultipleChoice,
    },
    {
      ...bigtableSection,
      quiz: bigtableQuiz,
      multipleChoice: bigtableMultipleChoice,
    },
    {
      ...cassandraSection,
      quiz: cassandraQuiz,
      multipleChoice: cassandraMultipleChoice,
    },
    {
      ...dynamodbSection,
      quiz: dynamodbQuiz,
      multipleChoice: dynamodbMultipleChoice,
    },
    {
      ...mongodbSection,
      quiz: mongodbQuiz,
      multipleChoice: mongodbMultipleChoice,
    },
    {
      ...redisSection,
      quiz: redisQuiz,
      multipleChoice: redisMultipleChoice,
    },
    {
      ...hbaseSection,
      quiz: hbaseQuiz,
      multipleChoice: hbaseMultipleChoice,
    },
    {
      ...distributedTransactionsSection,
      quiz: distributedTransactionsQuiz,
      multipleChoice: distributedTransactionsMultipleChoice,
    },
  ],
};
