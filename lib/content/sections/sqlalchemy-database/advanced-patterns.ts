export const advancedPatterns = {
  title: 'Advanced Patterns & Techniques',
  id: 'advanced-patterns',
  content: `
# Advanced Patterns & Techniques

## Repository Pattern

\`\`\`python
class UserRepository:
    def __init__(self, session):
        self.session = session
    
    def create(self, email: str) -> User:
        user = User(email=email)
        self.session.add(user)
        self.session.commit()
        return user
    
    def get_by_id(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)
    
    def find_active(self) -> list[User]:
        return self.session.execute(
            select(User).where(User.is_active == True)
        ).scalars().all()
\`\`\`

## Unit of Work Pattern

\`\`\`python
class UnitOfWork:
    def __init__(self):
        self.session = SessionLocal()
        self.users = UserRepository(self.session)
        self.posts = PostRepository(self.session)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.session.close()
    
    def commit(self):
        self.session.commit()
    
    def rollback(self):
        self.session.rollback()

# Usage
with UnitOfWork() as uow:
    user = uow.users.create("test@example.com")
    post = uow.posts.create(user.id, "Title")
    uow.commit()
\`\`\`

## Summary

✅ Repository pattern for data access abstraction  
✅ Unit of Work for transaction management  
✅ Domain-Driven Design integration  
✅ Clean architecture with SQLAlchemy
`,
};

export const advancedPatternsQuiz = [
  {
    id: 'sql-patterns-q-1',
    question:
      'Explain the Repository pattern and its benefits for SQLAlchemy applications.',
    sampleAnswer:
      'Repository pattern: Encapsulates data access logic in dedicated classes. Benefits: (1) Abstracts database details from business logic, (2) Testable - can mock repository, (3) Centralized queries, (4) Easier to switch ORMs. Implementation: UserRepository with methods create(), find_by_id(), find_active(). Business logic depends on repository interface, not SQLAlchemy directly.',
    keyPoints: [
      'Encapsulates data access',
      'Testable with mocking',
      'Abstract database details',
      'Centralized queries',
      'Switch ORMs easily',
    ],
  },
  {
    id: 'sql-patterns-q-2',
    question:
      'Design a Unit of Work pattern that manages multiple repositories and transactions.',
    sampleAnswer:
      'Unit of Work: Coordinates multiple repositories in single transaction. Implementation: UnitOfWork class with __enter__/__exit__, contains repository instances, provides commit()/rollback(). Usage: with UnitOfWork() as uow: create user and post, single commit. Benefits: Transaction spans multiple operations, consistent state, easier testing.',
    keyPoints: [
      'Coordinates multiple repositories',
      'Single transaction boundary',
      'Context manager for cleanup',
      'commit()/rollback() methods',
      'Easier testing',
    ],
  },
  {
    id: 'sql-patterns-q-3',
    question:
      'How do you implement the Active Record pattern using SQLAlchemy models?',
    sampleAnswer:
      'Active Record: Business logic in model class. Implementation: Add methods to model - User.authenticate(), User.send_email(). Query methods: User.find_by_email(email) as classmethod. Benefits: Simple, intuitive. Drawbacks: Tight coupling to database, harder to test, violates single responsibility. Recommendation: Use for simple apps, prefer Repository for complex systems.',
    keyPoints: [
      'Business logic in model',
      'Query methods as classmethods',
      'Simple and intuitive',
      'Tight coupling drawback',
      'Use for simple apps',
    ],
  },
];

export const advancedPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-patterns-mc-1',
    question: 'What is the main benefit of the Repository pattern?',
    options: [
      'Faster queries',
      'Abstracts data access, improves testability and flexibility',
      'Reduces code',
      'Automatic caching',
    ],
    correctAnswer: 1,
    explanation:
      "Repository pattern abstracts data access logic into dedicated classes. Benefits: (1) Business logic doesn't depend on SQLAlchemy directly, (2) Easy to mock repositories for testing, (3) Centralized query logic, (4) Can swap ORMs without changing business code. Trade-off: More code/classes, but worth it for maintainability in production systems.",
  },
  {
    id: 'sql-patterns-mc-2',
    question: 'What does the Unit of Work pattern manage?',
    options: [
      'Worker threads',
      'Transaction boundaries across multiple repositories',
      'User sessions',
      'Cache invalidation',
    ],
    correctAnswer: 1,
    explanation:
      'Unit of Work manages transaction boundaries across multiple repositories/operations. Instead of committing after each operation, UoW batches changes and commits once. Example: create user + create posts + assign roles in single transaction. If any fails, all rollback. Provides: uow.commit(), uow.rollback(). Essential for maintaining consistency in complex operations.',
  },
  {
    id: 'sql-patterns-mc-3',
    question: 'Why is the Active Record pattern sometimes discouraged?',
    options: [
      'Too slow',
      'Tight coupling between domain logic and database, harder to test',
      'Not supported by SQLAlchemy',
      'Requires more code',
    ],
    correctAnswer: 1,
    explanation:
      "Active Record (business logic in models) creates tight coupling between domain and database. Issues: (1) Hard to test - can't easily mock database, (2) Violates single responsibility (model does too much), (3) Database details leak into business logic. However, it's simple and works well for small apps. For production systems, prefer Repository pattern for better separation and testability.",
  },
  {
    id: 'sql-patterns-mc-4',
    question:
      'What is dependency injection in the context of SQLAlchemy sessions?',
    options: [
      'Injecting SQL',
      'Passing session as function parameter instead of global',
      'Database connection',
      'ORM feature',
    ],
    correctAnswer: 1,
    explanation:
      'Dependency injection: Pass session as parameter instead of using global. Example: def create_user(email, session) vs using global session. Benefits: (1) Testable - can pass mock session, (2) Explicit dependencies, (3) Thread-safe, (4) Clear lifecycle. FastAPI uses this: def endpoint(db: Session = Depends(get_db)). Avoid global sessions - causes thread safety and testing issues.',
  },
  {
    id: 'sql-patterns-mc-5',
    question:
      'What pattern helps separate database models from API response models?',
    options: [
      'Factory pattern',
      'DTO (Data Transfer Object) / Pydantic models',
      'Singleton pattern',
      'Observer pattern',
    ],
    correctAnswer: 1,
    explanation:
      "DTOs/Pydantic models separate database from API layer. ORM models for database, Pydantic for API responses. Benefits: (1) Control API shape independently, (2) No DetachedInstanceError (DTOs have no session), (3) Validation and serialization, (4) Security - don't expose internal structure. Example: class UserResponse(BaseModel): id: int; email: str. Convert: UserResponse.from_orm(user).",
  },
];
