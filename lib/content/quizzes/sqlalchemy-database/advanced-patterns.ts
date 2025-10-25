import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedPatternsQuiz = [
  {
    id: 'sql-patterns-q-1',
    question:
      'Implement the Repository Pattern for SQLAlchemy. Explain: (1) why use Repository Pattern, (2) how it abstracts database access, (3) implementation with generics, (4) integration with Unit of Work, (5) testing benefits. Provide complete implementation.',
    sampleAnswer:
      'Repository Pattern implementation: (1) Why: Separates business logic from data access. Centralizes queries in one place. Easier testing (mock repository). Swap implementations (SQL to NoSQL). Domain-driven design. (2) Abstraction: Repository provides collection-like interface. Methods: get(id), add(entity), remove(entity), find_by_email(email). Hides SQLAlchemy details from business layer. (3) Implementation: class UserRepository: def __init__(self, session): self.session = session. def get(self, user_id): return self.session.get(User, user_id). def add(self, user): self.session.add(user). def find_by_email(self, email): return self.session.execute(select(User).where(User.email==email)).scalar_one_or_none(). Generic: class Repository[T]: def __init__(self, model: Type[T], session): self.model = model; self.session = session. (4) Unit of Work integration: UoW manages session lifecycle and commits. with UnitOfWork() as uow: user = uow.users.find_by_email(email); user.name = "New Name"; uow.commit(). UoW commits all repositories together (transaction). (5) Testing: Mock repository for unit tests: mock_repo = Mock(UserRepository); mock_repo.get.return_value = user. Test business logic without database. Integration tests use real repository. Result: Clean separation of concerns, testable code, flexibility to change persistence.',
    keyPoints: [
      'Repository Pattern: Abstracts data access, collection-like interface',
      'Benefits: Centralized queries, testable, swappable implementations',
      'Implementation: get(id), add(), find_by_*(), hides SQLAlchemy',
      'Unit of Work: Manages session/transaction, commits all repos together',
      'Testing: Mock for unit tests, real DB for integration',
    ],
  },
  {
    id: 'sql-patterns-q-2',
    question:
      'Design a Unit of Work pattern for managing database transactions in SQLAlchemy. Address: (1) purpose of UoW, (2) implementation as context manager, (3) integration with repositories, (4) nested transactions, (5) error handling and rollback. Include complete code.',
    sampleAnswer:
      'Unit of Work pattern: (1) Purpose: Manages database transaction lifecycle. Tracks changes to entities. Commits all changes together (atomic). Automatic rollback on error. Coordinates multiple repositories. (2) Context manager implementation: class UnitOfWork: def __enter__(self): self.session = Session(); self.users = UserRepository(self.session); self.posts = PostRepository(self.session); return self. def __exit__(self, exc_type, exc_val, exc_tb): if exc_type: self.session.rollback(); else: self.session.commit(); self.session.close(). Usage: with UnitOfWork() as uow: user = uow.users.get(123); post = uow.posts.create(user); (auto commits). (3) Repository integration: Repositories share same session. Changes tracked automatically by SQLAlchemy. Single commit commits all changes across repositories. (4) Nested transactions: Use SAVEPOINT. uow.session.begin_nested(); try: risky_operation(); uow.session.commit(); except: uow.session.rollback(). Outer transaction unaffected if inner rolls back. (5) Error handling: __exit__ checks exc_type. If exception, rollback and re-raise. Log errors for debugging. Cleanup resources in finally block. Result: Transaction management centralized, automatic rollback, clean code.',
    keyPoints: [
      'Unit of Work: Manages transaction lifecycle, tracks changes, atomic commit',
      'Context manager: __enter__ creates session, __exit__ commits or rolls back',
      'Repository integration: Shared session, single commit for all repos',
      'Nested transactions: SAVEPOINT for rollback of partial work',
      'Error handling: Auto-rollback on exception, cleanup in __exit__',
    ],
  },
  {
    id: 'sql-patterns-q-3',
    question:
      'Explain advanced query patterns in SQLAlchemy: (1) Query Object pattern, (2) Specification pattern for composable filters, (3) Data Mapper pattern, (4) Active Record vs Data Mapper. Provide implementations and use cases.',
    sampleAnswer:
      'Advanced query patterns: (1) Query Object: Encapsulates query logic in reusable class. class ActiveUsersQuery: def __init__(self, session): self.session = session. def execute(self): return self.session.execute(select(User).where(User.active==True)).scalars(). Benefits: Reusable, testable, named. Usage: ActiveUsersQuery(session).execute(). (2) Specification pattern: Composable filters. class Specification: def is_satisfied_by(self, item): pass. class ActiveUserSpec(Specification): def to_sqlalchemy(self): return User.active==True. Compose: spec = ActiveUserSpec() & EmailVerifiedSpec(). Query: select(User).where(spec.to_sqlalchemy()). Benefits: DRY, composable with &/|/~. (3) Data Mapper: Separates domain model from database model. Domain: class User: def __init__(self, name): self.name = name (no DB knowledge). Mapper: maps domain to DB. session.add(mapper.to_db(user)). Benefits: Domain model pure Python, independent of ORM. (4) Active Record vs Data Mapper: Active Record: Model knows about persistence (user.save()). Simple, good for CRUD apps. Data Mapper: Model pure Python, mapper handles persistence. Complex, good for domain-driven design. SQLAlchemy is Data Mapper by default (models are declarative but mapped separately). Result: Flexible querying, reusable logic, clean architecture.',
    keyPoints: [
      'Query Object: Encapsulates query in reusable class, named and testable',
      'Specification: Composable filters with &/|/~, DRY',
      'Data Mapper: Domain model independent of database/ORM',
      'Active Record: Model knows persistence (simple). Data Mapper: Separated (flexible)',
      'Use patterns for complex domains, DRY queries, testability',
    ],
  },
];
