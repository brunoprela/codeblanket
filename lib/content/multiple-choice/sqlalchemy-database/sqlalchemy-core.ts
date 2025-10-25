import { MultipleChoiceQuestion } from '@/lib/types';

export const sqlalchemyCoreMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-core-mc-1',
    question:
      'What is the recommended way to create a database engine in SQLAlchemy for a production application?',
    options: [
      'Create a new engine for each database operation',
      'Create a single engine instance and reuse it across the application',
      'Create one engine per user session',
      'Create engines on-demand and dispose immediately',
    ],
    correctAnswer: 1,
    explanation:
      'The engine should be created once as a singleton and reused across the entire application. The engine manages a connection pool internally, and creating multiple engines defeats connection pooling, wastes resources, and hurts performance. Each engine maintains its own pool—creating engines repeatedly would create many pools with connections sitting idle. Create once at application startup, store globally or as application state.',
  },
  {
    id: 'sql-core-mc-2',
    question: 'What does pool_pre_ping=True do in engine configuration?',
    options: [
      'Tests connections before checkout to ensure they are alive',
      'Pre-creates all connections in the pool at startup',
      'Pings the database server to check if it is online',
      'Reduces latency by keeping connections warm',
    ],
    correctAnswer: 0,
    explanation:
      'pool_pre_ping=True tests each connection with a simple query (SELECT 1) before handing it to the application. This ensures stale connections (database restart, network timeout) are detected and recycled before causing errors. Without pre_ping, your application would receive a dead connection and error on first query. Critical for production: databases restart for maintenance, networks have timeouts. Small overhead (1-2ms) worth the reliability.',
  },
  {
    id: 'sql-core-mc-3',
    question:
      'In SQLAlchemy 2.0, what is the recommended way to execute a SELECT query using the ORM?',
    options: [
      'session.query(User).filter(User.email == "test").all()',
      'session.execute(select(User).where(User.email == "test")).scalars().all()',
      'session.select(User).where(User.email == "test")',
      'User.query.filter(User.email == "test").all()',
    ],
    correctAnswer: 1,
    explanation:
      'SQLAlchemy 2.0 uses session.execute() with select() construct: session.execute(select(User).where(User.email == "test")).scalars().all(). The query() API (option 1) is legacy 1.x style, still works but not recommended. scalars() extracts the entity from Row tuples. This unified API works for both sync and async code. Options 3 and 4 are invalid—session has no select method, User has no query attribute.',
  },
  {
    id: 'sql-core-mc-4',
    question: 'What happens when you call session.commit()?',
    options: [
      'Only saves changes to the database',
      'Flushes changes to database and commits the transaction',
      'Closes the session',
      'Refreshes all objects from the database',
    ],
    correctAnswer: 1,
    explanation:
      'session.commit() performs two operations: (1) flush() - sends pending changes to database (INSERTs, UPDATEs, DELETEs become SQL), (2) commit() - commits the database transaction, making changes permanent. After commit, objects are expired by default (expire_on_commit=True), requiring database hit to access attributes. session.commit() does not close the session—you must call session.close() explicitly or use context manager.',
  },
  {
    id: 'sql-core-mc-5',
    question: 'What is the purpose of sessionmaker in SQLAlchemy?',
    options: [
      'Creates database tables',
      'Factory for creating configured Session instances',
      'Manages database migrations',
      'Handles connection pooling',
    ],
    correctAnswer: 1,
    explanation:
      'sessionmaker is a factory that creates Session instances with pre-configured settings. Instead of: Session(bind=engine, autocommit=False, ...) every time, you create SessionLocal = sessionmaker(bind=engine, ...) once, then SessionLocal() to get configured sessions. This centralizes session configuration and follows the factory pattern. The engine handles connection pooling, not sessionmaker. Alembic handles migrations, not sessionmaker.',
  },
];
