/**
 * Multiple choice questions for Database Connection Pooling section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const databaseconnectionpoolingMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pool-1',
      question: 'Why is creating a new database connection expensive?',
      options: [
        'Database queries are slow and require multiple round trips',
        'TCP handshake, authentication, session init, and SSL negotiation add 10-20ms overhead',
        'Connection pooling libraries are poorly optimized',
        'Databases limit the rate of new connections to 1 per second',
      ],
      correctAnswer: 1,
      explanation:
        "Option B is correct. Creating a new database connection involves multiple steps: TCP 3-way handshake, authentication (username/password), session initialization (memory allocation), and SSL/TLS negotiation if encryption is enabled. This typically adds 10-20ms overhead for same-datacenter connections, which is significantly more than query execution time (often 1-2ms). Connection pooling eliminates this overhead by reusing connections. Option A confuses connection setup with query execution. Option C is incorrect. Option D is false; databases don't have such limits.",
    },
    {
      id: 'pool-2',
      question:
        'You have 5 application instances and a database with max_connections=200. What is a safe maximum pool size per instance?',
      options: [
        '200 connections (use full database capacity)',
        '40 connections (200 / 5)',
        '32 connections ((200 × 0.8) / 5)',
        '100 connections (leave room for manual connections)',
      ],
      correctAnswer: 2,
      explanation:
        "Option C is correct. Best practice is to use 80% of database max_connections to leave headroom for admin connections, monitoring, and unexpected spikes: (200 × 0.8) / 5 = 32 connections per instance. Option A would exhaust the database (no room for anything else). Option B doesn't leave safety margin. Option D is arbitrary and could still exhaust connections (5 × 100 = 500 > 200). The 80% rule ensures the database doesn't run out of connections while maximizing application concurrency.",
    },
    {
      id: 'pool-3',
      question: 'What is a connection leak and how do you prevent it?',
      options: [
        'A connection leak is when the pool creates too many connections; prevent by setting maxconn',
        'A connection leak is when a connection is checked out but never returned; prevent with try/finally blocks',
        'A connection leak is when idle connections are not closed; prevent with idle_timeout',
        'A connection leak is when stale connections remain in the pool; prevent with health checks',
      ],
      correctAnswer: 1,
      explanation:
        'Option B is correct. A connection leak occurs when a connection is checked out from the pool but never returned, usually due to exceptions or early returns. This gradually exhausts the pool. Prevention: always return connections in a finally block or use context managers (with statements). Option A describes pool exhaustion, not leaks. Option C describes idle connection buildup (different issue). Option D describes stale connections, not leaks. Connection leaks are the #1 cause of pool problems.',
    },
    {
      id: 'pool-4',
      question:
        'What is the optimal connection pool size for a web application?',
      options: [
        'As large as possible to handle maximum concurrent requests',
        'Equal to the number of CPU cores on the application server',
        'Approximately (core_count × 2) where core_count is the database server cores',
        'Equal to the expected maximum concurrent users',
      ],
      correctAnswer: 2,
      explanation:
        "Option C is correct. The formula (core_count × 2) where core_count refers to the database server CPU cores provides optimal throughput. This balances parallelism with resource contention. Beyond this, adding more connections causes context switching and lock contention, degrading performance. Option A is wrong - too many connections cause contention. Option B refers to wrong server (app not database). Option D would be massive overkill (thousands of users, but they don't all query simultaneously). HikariCP's extensive testing validated this formula.",
    },
    {
      id: 'pool-5',
      question:
        'Which configuration parameter prevents stale connections from accumulating in the pool?',
      options: [
        'min_connections - sets minimum connections to maintain',
        'max_lifetime - closes connections after maximum age',
        'timeout - maximum time to wait for available connection',
        'max_connections - sets maximum pool size',
      ],
      correctAnswer: 1,
      explanation:
        'Option B (max_lifetime) is correct. This parameter closes and recreates connections after they reach a certain age (e.g., 1 hour), preventing stale connections from database restarts, network issues, or credential rotations. This ensures connections are periodically refreshed. Option A (min_connections) maintains minimum pool size. Option C (timeout) is for acquiring connections. Option D (max_connections) is pool capacity. Additionally, idle_timeout also helps by closing connections idle for too long, but max_lifetime is more comprehensive as it handles connections regardless of activity.',
    },
  ];
