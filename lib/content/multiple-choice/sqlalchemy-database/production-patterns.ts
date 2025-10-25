import { MultipleChoiceQuestion } from '@/lib/types';

export const productionPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-prod-mc-1',
    question: 'What is the purpose of pool_pre_ping=True?',
    options: [
      'Faster queries',
      'Test connection before checkout to catch disconnects',
      'Increase pool size',
      'Enable caching',
    ],
    correctAnswer: 1,
    explanation:
      'pool_pre_ping=True tests connection before checkout. Problem: Connections can die (network issue, database restart, idle timeout). Without pre_ping: Application gets dead connection, query fails with error. With pre_ping: Engine tests connection (SELECT 1), if dead, creates new connection, application gets working connection. Small overhead (extra ping), but prevents errors. Critical for production reliability. Use always in production.',
  },
  {
    id: 'sql-prod-mc-2',
    question: 'How should you size the connection pool?',
    options: [
      'pool_size=1000 (as large as possible)',
      'pool_size = number_of_application_threads',
      'pool_size=5 (keep it small)',
      'pool_size = number_of_CPU_cores',
    ],
    correctAnswer: 1,
    explanation:
      'pool_size = number_of_application_threads. Reasoning: Each thread needs one connection when executing query. Example: 10 gunicorn workers * 4 threads = 40 threads → pool_size=40. max_overflow = 2 * pool_size for burst capacity. Too large: Wastes DB resources, connections idle. Too small: Threads wait for connections, pool_timeout errors. Formula: pool_size + max_overflow < database_max_connections. PostgreSQL default max_connections=100, so pool_size=40 + max_overflow=40 = 80 (safe).',
  },
  {
    id: 'sql-prod-mc-3',
    question: 'When should you cache query results?',
    options: [
      'Always',
      'Never',
      'For frequently read, rarely changed data',
      'Only in development',
    ],
    correctAnswer: 2,
    explanation:
      "Cache frequently read, rarely changed data. Examples: User profiles (read 1000x/sec, updated 1x/day), product catalog, blog posts. Don't cache: Real-time data (stock prices), frequently changing data (shopping cart), user-specific data (unless scoped by user). Strategy: Redis cache with 5-15min TTL. Invalidate on write: delete cache key after update. Benefits: 5-10x faster response, reduced DB load. Costs: Stale data (eventual consistency), complexity (cache invalidation hard). Production: 70-90% cache hit rate typical.",
  },
  {
    id: 'sql-prod-mc-4',
    question: 'What does the Repository pattern provide?',
    options: [
      'Faster queries',
      'Abstract data access layer - hide SQLAlchemy from business logic',
      'Automatic caching',
      'Connection pooling',
    ],
    correctAnswer: 1,
    explanation:
      'Repository pattern abstracts data access. Benefits: (1) Testable - mock repositories in unit tests, no database needed. (2) Swappable - change database (SQLAlchemy → MongoDB) without changing business logic. (3) Organized - all User queries in UserRepository. (4) Clean architecture - business logic depends on interface (UserRepository), not implementation (SQLAlchemy). Example: UserService depends on UserRepository interface. Unit test: mock repository. Integration test: real SQLAlchemy repository. Production: Use for large applications, domain-driven design.',
  },
  {
    id: 'sql-prod-mc-5',
    question: 'What should a database health check endpoint test?',
    options: [
      'Only HTTP server',
      'Database connectivity (SELECT 1) and connection pool status',
      'Only application code',
      'User authentication',
    ],
    correctAnswer: 1,
    explanation:
      'Health check tests: (1) Database connectivity - execute SELECT 1, if fails, DB down. (2) Connection pool status - checked_out vs size, if > 80% pool nearly exhausted (degraded). Return: 200 OK if healthy, 503 Service Unavailable if unhealthy. Load balancer uses: routes traffic only to healthy instances. Example: Instance A - DB down, health check returns 503, LB stops sending requests. Instance B - healthy, health check returns 200, LB sends all requests to B. Critical for zero-downtime deployments and automatic failover.',
  },
];
