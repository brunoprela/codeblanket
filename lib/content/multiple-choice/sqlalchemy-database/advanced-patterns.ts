import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-patterns-mc-1',
    question: 'What is the primary benefit of the Repository Pattern?',
    options: [
      'Faster queries',
      'Separation of business logic from data access',
      'Automatic caching',
      'Better indexing',
    ],
    correctAnswer: 1,
    explanation:
      'Repository Pattern separates business logic from data access layer. ' +
      'Benefits: (1) Business logic does not depend on SQLAlchemy/database, (2) Centralized query logic, (3) Easier testing (mock repository), (4) Can swap implementations (SQL to NoSQL). ' +
      'Example: UserRepository provides get (id), find_by_email (email), add (user). ' +
      'Business layer uses repository methods, never writes queries directly. ' +
      'Testing: Mock repository for unit tests, real repository for integration tests.',
  },
  {
    id: 'sql-patterns-mc-2',
    question: 'What does the Unit of Work pattern manage?',
    options: [
      'Query optimization',
      'Database transaction lifecycle and commits',
      'Connection pooling',
      'Index creation',
    ],
    correctAnswer: 1,
    explanation:
      'Unit of Work manages database transaction lifecycle: Starts transaction, tracks changes, commits all together, rolls back on error. ' +
      'Implementation: Context manager (__enter__ creates session, __exit__ commits/rolls back). ' +
      'Coordinates multiple repositories: with UnitOfWork() as uow: uow.users.add (user); uow.posts.add (post). ' +
      'Single commit for all changes (atomic). Auto-rollback if exception occurs. ' +
      'Benefits: Centralized transaction management, automatic rollback, clean code.',
  },
  {
    id: 'sql-patterns-mc-3',
    question: 'What is the difference between Active Record and Data Mapper?',
    options: [
      'Active Record is faster',
      'Active Record: model knows persistence; Data Mapper: separated',
      'Data Mapper requires more indexes',
      'No difference',
    ],
    correctAnswer: 1,
    explanation:
      'Active Record: Model contains persistence logic (user.save(), user.delete()). Simple, good for CRUD apps. Example: Django ORM. ' +
      'Data Mapper: Model is pure Python (no DB knowledge), mapper handles persistence. Complex, good for domain-driven design. Example: SQLAlchemy. ' +
      'Active Record pros: Simple, intuitive. Cons: Domain model coupled to database. ' +
      'Data Mapper pros: Domain model pure, testable, flexible. Cons: More complex, requires mapper layer. ' +
      'SQLAlchemy is Data Mapper (declarative models are mapped separately).',
  },
  {
    id: 'sql-patterns-mc-4',
    question: 'What is the Query Object pattern used for?',
    options: [
      'Optimizing indexes',
      'Encapsulating query logic in reusable classes',
      'Managing connections',
      'Caching results',
    ],
    correctAnswer: 1,
    explanation:
      'Query Object encapsulates query logic in reusable class. ' +
      'Example: class ActiveUsersQuery: def execute (self, session): return session.execute (select(User).where(User.active==True)).scalars(). ' +
      'Benefits: (1) Reusable across codebase, (2) Named (self-documenting), (3) Testable in isolation, (4) Can add methods (with_posts(), ordered_by_name()). ' +
      'Usage: users = ActiveUsersQuery (session).execute(). ' +
      'Alternative to raw queries scattered throughout code.',
  },
  {
    id: 'sql-patterns-mc-5',
    question: 'What does the Specification pattern enable?',
    options: [
      'Faster queries',
      'Composable, reusable query filters',
      'Automatic migrations',
      'Connection pooling',
    ],
    correctAnswer: 1,
    explanation:
      'Specification pattern: Composable, reusable filters. ' +
      'Example: class ActiveUserSpec: def to_sqlalchemy (self): return User.active==True. ' +
      'Compose with operators: spec = ActiveUserSpec() & EmailVerifiedSpec(). ' +
      'Use in query: select(User).where (spec.to_sqlalchemy()). ' +
      'Benefits: (1) DRY - define filter once, use everywhere, (2) Composable with &/|/~ (AND/OR/NOT), (3) Testable, (4) Business rules as objects. ' +
      'Example composition: (Active & Verified) | AdminSpec = active verified users OR admins.',
  },
];
