/**
 * Multiple choice questions for SQL vs NoSQL Decision Framework section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const sqlvsnosqldecisionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-vs-nosql-q1',
    question:
      'You are designing a banking system that handles money transfers between accounts. Which database type is most appropriate and why?',
    options: [
      'MongoDB because it scales horizontally and has flexible schema',
      'Cassandra because it can handle high write throughput for transactions',
      'PostgreSQL because it provides ACID transactions ensuring transfers are atomic',
      'Redis because it provides fast in-memory operations for quick transfers',
    ],
    correctAnswer: 2,
    explanation:
      'PostgreSQL (SQL) is the correct choice because banking systems require ACID transactions to ensure atomicity. When transferring money, you must guarantee that if $100 is deducted from Account A, it is definitely added to Account B - either both operations succeed or both fail. This atomicity is critical for financial accuracy and is a core strength of SQL databases. While MongoDB and Cassandra scale well, they sacrifice strong consistency. Redis is for caching, not primary transactional data.',
    difficulty: 'medium',
  },
  {
    id: 'sql-vs-nosql-q2',
    question:
      'You are building a product catalog for an e-commerce site where different product types (books, electronics, clothing) have vastly different attributes. Books have ISBN and page count, electronics have specifications, clothing has sizes and materials. What database approach makes most sense?',
    options: [
      'SQL with separate tables for each product type (books, electronics, clothing)',
      'SQL with a single products table and many nullable columns for all possible attributes',
      'NoSQL document database (MongoDB) where each product document can have different fields',
      'NoSQL key-value store (Redis) with JSON-encoded product information',
    ],
    correctAnswer: 2,
    explanation:
      'MongoDB (NoSQL document database) is ideal for this use case because each product can have a completely different schema stored in its document. Books can have {isbn, pageCount}, electronics can have {brand, specifications}, and clothing can have {size, material} without forcing all products into the same rigid table structure. Option A creates maintenance overhead with multiple tables, Option B results in sparse tables with many null values wasting space, and Option D (Redis) is for caching not primary storage.',
    difficulty: 'medium',
  },
  {
    id: 'sql-vs-nosql-q3',
    question:
      'Your application needs to execute the following analytics query: "Show me the top 5 customers by total order value in the last 6 months, grouped by region, including the average order value and product categories purchased." Which database type is better suited?',
    options: [
      'MongoDB because it has aggregation pipelines for complex queries',
      'Cassandra because it handles large-scale distributed data',
      'PostgreSQL because SQL excels at complex joins, aggregations, and grouping',
      'DynamoDB because it provides fast queries at scale',
    ],
    correctAnswer: 2,
    explanation:
      "PostgreSQL (SQL) is the best choice for this complex analytical query. SQL databases excel at JOINs across multiple tables (customers, orders, products, regions), aggregations (SUM, AVG, GROUP BY), and filtering (last 6 months, top 5). The query likely requires joining 3-4 tables and SQL's declarative language makes this straightforward. While MongoDB has aggregation pipelines, they're more complex for multi-collection operations. Cassandra and DynamoDB are optimized for simple, predefined access patterns, not ad-hoc complex analytics.",
    difficulty: 'hard',
  },
  {
    id: 'sql-vs-nosql-q4',
    question:
      'You are designing Instagram\'s photo storage system which needs to store metadata for billions of photos. Access pattern: "Get all photos for user X" and "Get photo by ID". The system must handle millions of writes per day and scale globally. What database strategy is most appropriate?',
    options: [
      'MySQL with master-slave replication to handle read traffic',
      'PostgreSQL with table partitioning by user_id',
      'Cassandra with partition key on user_id and clustering key on timestamp',
      'MongoDB with sharding on user_id',
    ],
    correctAnswer: 2,
    explanation:
      'Cassandra is the best choice for Instagram-scale photo metadata storage. With partition key on user_id, all photos for a user are stored together enabling fast "get all photos for user X" queries. Clustering by timestamp provides time-ordered results. Cassandra excels at write-heavy workloads (millions of photo uploads), scales horizontally to billions of records, and supports multi-datacenter replication for global distribution. MySQL master-slave struggles at this scale, PostgreSQL partitioning is complex to manage at billions of records, and while MongoDB could work, Cassandra is proven at Instagram scale and designed specifically for this access pattern.',
    difficulty: 'hard',
  },
  {
    id: 'sql-vs-nosql-q5',
    question:
      'Which of the following scenarios is the BEST use case for a SQL database instead of NoSQL?',
    options: [
      'Storing IoT sensor data with 10 million writes per second',
      'Caching user session data with TTL expiration',
      'Storing user profiles in a chat application with flexible attributes',
      'Managing inventory for an e-commerce site where stock levels must be accurate to prevent overselling',
    ],
    correctAnswer: 3,
    explanation:
      'Inventory management is the best SQL use case because it requires ACID transactions to prevent overselling. When a customer buys an item, you need to atomically: (1) check stock level, (2) decrement inventory, (3) create order record. This must be strongly consistent - you cannot oversell items due to eventual consistency. SQL ensures immediate consistency across these operations. Option A (IoT writes) is perfect for Cassandra/InfluxDB, Option B (session caching) is ideal for Redis, Option C (flexible profiles) fits MongoDB well.',
    difficulty: 'medium',
  },
];
