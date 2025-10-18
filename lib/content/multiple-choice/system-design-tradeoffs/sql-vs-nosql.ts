/**
 * Multiple choice questions for SQL vs NoSQL section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const sqlvsnosqlMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What does ACID stand for in SQL databases?',
    options: [
      'Asynchronous, Consistent, Isolated, Durable',
      'Atomicity, Consistency, Isolation, Durability',
      'Atomic, Concurrent, Integrated, Distributed',
      'Available, Consistent, Isolated, Distributed',
    ],
    correctAnswer: 1,
    explanation:
      "ACID stands for Atomicity (transactions are all-or-nothing), Consistency (database remains in valid state), Isolation (concurrent transactions don't interfere), and Durability (committed data persists). These properties make SQL databases ideal for financial transactions and critical data.",
  },
  {
    id: 'mc2',
    question: 'Which scenario is best suited for NoSQL over SQL?',
    options: [
      'Banking transactions requiring exact balances',
      'Social media feeds with billions of posts requiring massive horizontal scaling',
      'Inventory management where overselling must be prevented',
      'Complex reporting with JOINs across multiple tables',
    ],
    correctAnswer: 1,
    explanation:
      "Social media feeds with billions of posts are perfect for NoSQL (like Cassandra) because: (1) Need massive horizontal scaling, (2) Eventual consistency is acceptable (slight delay in feed is fine), (3) Access patterns are simple (get posts by user_id), (4) Need to denormalize for performance. Banking, inventory, and complex reports require SQL's ACID and JOINs.",
  },
  {
    id: 'mc3',
    question: 'What is the main advantage of NoSQL databases for scaling?',
    options: [
      'Better security',
      'Built-in horizontal scaling with automatic sharding',
      'Faster single-server performance',
      'Better data compression',
    ],
    correctAnswer: 1,
    explanation:
      'NoSQL databases like Cassandra and MongoDB are designed for horizontal scaling from day one. Adding more nodes automatically distributes data and increases capacity. SQL databases require complex manual sharding for horizontal write scaling. This makes NoSQL much easier to scale to billions of records.',
  },
  {
    id: 'mc4',
    question: 'What is polyglot persistence?',
    options: [
      'Using multiple programming languages in one application',
      'Using multiple databases (SQL and NoSQL) for different data needs in the same system',
      'Storing data in multiple data centers',
      'Supporting multiple languages in your user interface',
    ],
    correctAnswer: 1,
    explanation:
      'Polyglot persistence means using different databases for different needs in the same system. For example: PostgreSQL for transactions, MongoDB for product catalogs, Redis for caching. This "use the right tool for the job" approach is common in modern systems like Netflix and Amazon.',
  },
  {
    id: 'mc5',
    question: 'Why do NoSQL databases often require denormalization?',
    options: [
      'To save disk space',
      "Because they don't support or have expensive JOIN operations",
      'To improve security',
      'Because they are schema-less',
    ],
    correctAnswer: 1,
    explanation:
      "NoSQL databases like Cassandra and DynamoDB don't support JOIN operations (or make them very expensive). To avoid application-level JOINs, you must denormalize by duplicating data. For example, embed author information in each post rather than joining to a users table. This trades storage and write complexity for fast reads.",
  },
];
