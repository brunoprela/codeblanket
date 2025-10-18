/**
 * Multiple choice questions for Data Management in Microservices section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const datamanagementmicroservicesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc-data-1',
      question:
        'What is the main advantage of the "database per service" pattern?',
      options: [
        'Makes queries easier with JOINs across services',
        'Provides loose coupling - services can evolve databases independently',
        'Eliminates the need for backups',
        'Guarantees strong consistency across services',
      ],
      correctAnswer: 1,
      explanation:
        'Database per service provides loose coupling - each service owns its data and can evolve its schema independently without breaking other services. Services can also choose different database technologies (SQL, NoSQL, graph) based on their needs. Option 1 is wrong (database per service makes JOINs impossible). Option 3 is wrong (still need backups). Option 4 is wrong (database per service provides eventual consistency, not strong consistency).',
    },
    {
      id: 'mc-data-2',
      question:
        'You need to display orders with product details. Each service has its own database. What approach should you use?',
      options: [
        'Use SQL JOIN across both databases',
        'Share the database between Order and Product services',
        'Use API composition (get orders, then get products) or data duplication (store product details in Order Service)',
        'Give Order Service direct access to Product database',
      ],
      correctAnswer: 2,
      explanation:
        'Use API composition (Order Service calls Product Service to get details) for simple cases, or data duplication (Order Service stores product name/price when order created) for better performance. Update duplicated data via events when products change. Option 1 is impossible (different databases). Option 2 breaks database per service pattern. Option 4 violates encapsulation and creates tight coupling. API composition or data duplication are the correct microservices patterns.',
    },
    {
      id: 'mc-data-3',
      question: 'What is CQRS in the context of microservices data management?',
      options: [
        'A type of database that supports microservices',
        'Command Query Responsibility Segregation - separate read and write models',
        'A security pattern for encrypting data',
        'A caching layer for databases',
      ],
      correctAnswer: 1,
      explanation:
        'CQRS (Command Query Responsibility Segregation) separates read and write models. Write side: services write to their own databases. Read side: dedicated read models (denormalized databases) optimized for queries, updated via events from write side. Example: OrderViewService subscribes to OrderCreated, ProductUpdated events and maintains a denormalized view combining order + product + user data for fast queries. Solves the "no JOIN" problem in microservices. Option 1 is wrong (it\'s a pattern, not a database). Options 3 and 4 are unrelated concepts.',
    },
    {
      id: 'mc-data-4',
      question:
        'Which database type would you choose for a Product Catalog service with flexible attributes that vary by product type?',
      options: [
        'Relational (PostgreSQL) - for strong schema',
        'Document (MongoDB) - for flexible schema',
        'Graph (Neo4j) - for relationships',
        'Time-series (InfluxDB) - for metrics',
      ],
      correctAnswer: 1,
      explanation:
        'Document databases like MongoDB are perfect for flexible schemas where attributes vary significantly. Example: electronics have (processor, RAM), clothing has (size, color, material), books have (author, ISBN). With MongoDB, each product can have different attributes without schema changes. PostgreSQL would require either EAV pattern (slow) or JSON columns (less structured). Graph databases are for relationship-heavy data. Time-series is for temporal data. Choose the database that matches your data model.',
    },
    {
      id: 'mc-data-5',
      question: 'What is the Strangler Fig pattern in data migration?',
      options: [
        'Delete old data after migrating to new database',
        'Gradually extract services while initially sharing database, then split databases incrementally',
        'Replicate all data to new database immediately',
        'Use database triggers to sync data',
      ],
      correctAnswer: 1,
      explanation:
        "Strangler Fig is a gradual migration pattern: Phase 1 - extract service but keep shared DB, Phase 2 - replicate data to service's own DB, Phase 3 - service writes to own DB and syncs back, Phase 4 - cut over completely. This allows safe, incremental migration from monolith to microservices without big-bang rewrites. Named after strangler fig vines that gradually replace host trees. Option 1 loses data. Option 3 is risky (big bang). Option 4 creates tight coupling.",
    },
  ];
