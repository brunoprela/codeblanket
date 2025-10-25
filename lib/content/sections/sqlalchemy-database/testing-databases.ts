export const testingWithDatabases = {
  title: 'Testing with Databases',
  id: 'testing-databases',
  content: `
# Testing with Databases

## Introduction

Testing database code is essential but challenging. This section covers test database setup, fixtures, factories, mocking strategies, and integration testing patterns for SQLAlchemy applications.

In this section, you'll master:
- Test database setup and teardown
- pytest fixtures for databases
- Factory pattern for test data
- Testing queries and relationships
- Mocking database operations
- Integration vs unit testing strategies
- CI/CD integration
- Testing migrations

### Why Database Testing Matters

**Production reality**: Untested database code causes: data corruption, constraint violations, N+1 problems in production, and migration failures. Comprehensive database testing is non-negotiable for production applications.

---

## Test Database Setup

### In-Memory SQLite for Tests

\`\`\`python
"""
Fast In-Memory Database for Tests
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.models import Base

@pytest.fixture(scope="function")
def test_db():
    """Create in-memory SQLite database for each test"""
    # In-memory database (fastest)
    engine = create_engine("sqlite:///:memory:")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)

# Usage
def test_create_user(test_db):
    user = User(email="test@example.com")
    test_db.add(user)
    test_db.commit()
    
    assert user.id is not None
    assert user.email == "test@example.com"
\`\`\`

### PostgreSQL Test Database

\`\`\`python
"""
Real PostgreSQL for Integration Tests
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.models import Base

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine (once per test session)"""
    # Use separate test database
    engine = create_engine("postgresql://localhost/test_db")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Drop all tables after tests
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create clean database session for each test"""
    connection = test_engine.connect()
    transaction = connection.begin()
    
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    yield session
    
    # Rollback transaction (undo all changes)
    session.close()
    transaction.rollback()
    connection.close()

# Each test runs in transaction that rolls back
def test_user_creation(test_db):
    user = User(email="test@example.com")
    test_db.add(user)
    test_db.commit()
    
    # Changes rolled back after test
\`\`\`

---

## Factory Pattern for Test Data

### Factory Boy Integration

\`\`\`python
"""
Factory Boy for Test Data Generation
"""

import factory
from factory.alchemy import SQLAlchemyModelFactory
from myapp.models import User, Post

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = None  # Set per test
    
    email = factory.Sequence(lambda n: f"user{n}@example.com")
    username = factory.Faker('user_name')
    is_active = True
    created_at = factory.Faker('date_time')

class PostFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Post
        sqlalchemy_session = None
    
    title = factory.Faker('sentence')
    content = factory.Faker('text')
    user = factory.SubFactory(UserFactory)

# Usage in tests
@pytest.fixture
def factories(test_db):
    """Configure factories with test session"""
    UserFactory._meta.sqlalchemy_session = test_db
    PostFactory._meta.sqlalchemy_session = test_db
    return {"user": UserFactory, "post": PostFactory}

def test_user_posts(test_db, factories):
    # Create test data easily
    user = factories["user"].create(email="specific@example.com")
    post1 = factories["post"].create(user=user, title="First Post")
    post2 = factories["post"].create(user=user, title="Second Post")
    
    assert len(user.posts) == 2
    assert user.posts[0].title == "First Post"
\`\`\`

### Manual Fixtures

\`\`\`python
"""
Manual Test Fixtures
"""

@pytest.fixture
def sample_user(test_db):
    """Create sample user for tests"""
    user = User(
        email="test@example.com",
        username="testuser",
        is_active=True
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user

@pytest.fixture
def user_with_posts(test_db, sample_user):
    """User with posts"""
    posts = [
        Post(title=f"Post {i}", content=f"Content {i}", user=sample_user)
        for i in range(3)
    ]
    test_db.add_all(posts)
    test_db.commit()
    return sample_user

def test_user_posts_relationship(user_with_posts):
    assert len(user_with_posts.posts) == 3
    assert all(post.user_id == user_with_posts.id for post in user_with_posts.posts)
\`\`\`

---

## Testing Queries

### Query Result Testing

\`\`\`python
"""
Testing Query Results
"""

def test_find_active_users(test_db, factories):
    # Create test data
    active = factories["user"].create(is_active=True)
    inactive = factories["user"].create(is_active=False)
    
    # Query active users
    result = test_db.execute(
        select(User).where(User.is_active == True)
    ).scalars().all()
    
    # Assertions
    assert len(result) == 1
    assert result[0].id == active.id
    assert inactive not in result

def test_user_search_by_email(test_db, factories):
    user = factories["user"].create(email="search@example.com")
    
    result = test_db.execute(
        select(User).where(User.email == "search@example.com")
    ).scalar_one_or_none()
    
    assert result is not None
    assert result.id == user.id
\`\`\`

### Testing Relationships

\`\`\`python
"""
Testing Model Relationships
"""

def test_one_to_many_relationship(test_db, factories):
    """Test User -> Posts relationship"""
    user = factories["user"].create()
    posts = [factories["post"].create(user=user) for _ in range(3)]
    
    # Test forward relationship
    assert len(user.posts) == 3
    assert all(isinstance(post, Post) for post in user.posts)
    
    # Test backward relationship
    for post in posts:
        assert post.user.id == user.id

def test_many_to_many_relationship(test_db):
    """Test Post -> Tags relationship"""
    post = Post(title="Test")
    tag1 = Tag(name="Python")
    tag2 = Tag(name="SQL")
    
    post.tags.extend([tag1, tag2])
    test_db.add(post)
    test_db.commit()
    
    # Test relationship
    assert len(post.tags) == 2
    assert tag1 in post.tags
    assert post in tag1.posts
\`\`\`

---

## Testing with Mocks

### Mocking Database Sessions

\`\`\`python
"""
Mocking Database Operations
"""

from unittest.mock import Mock, MagicMock
import pytest

def test_user_service_with_mock():
    """Test business logic without real database"""
    # Mock session
    mock_session = Mock()
    mock_user = User(id=1, email="test@example.com")
    
    # Configure mock
    mock_session.execute.return_value.scalar_one_or_none.return_value = mock_user
    
    # Test service
    service = UserService(mock_session)
    result = service.get_user_by_email("test@example.com")
    
    # Assertions
    assert result.id == 1
    assert result.email == "test@example.com"
    mock_session.execute.assert_called_once()
\`\`\`

### Mocking Repositories

\`\`\`python
"""
Mock Repository for Testing Business Logic
"""

@pytest.fixture
def mock_user_repository():
    """Mock UserRepository"""
    repo = Mock(spec=UserRepository)
    
    # Configure default behavior
    repo.find_by_id.return_value = User(id=1, email="test@example.com")
    repo.find_by_email.return_value = User(id=1, email="test@example.com")
    repo.create.return_value = User(id=2, email="new@example.com")
    
    return repo

def test_user_service(mock_user_repository):
    service = UserService(mock_user_repository)
    
    user = service.register_user("new@example.com")
    
    # Verify repository called correctly
    mock_user_repository.create.assert_called_once_with("new@example.com")
\`\`\`

---

## Integration Testing

### Testing Complete Workflows

\`\`\`python
"""
Integration Test: Complete User Registration Flow
"""

def test_user_registration_workflow(test_db):
    """Test complete registration: create user, send email, create profile"""
    
    # Step 1: Create user
    user = User(email="new@example.com", username="newuser")
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    
    # Step 2: Create profile
    profile = UserProfile(user_id=user.id, bio="Test bio")
    test_db.add(profile)
    test_db.commit()
    
    # Step 3: Verify relationships
    assert user.profile is not None
    assert user.profile.bio == "Test bio"
    
    # Step 4: Query user with profile
    result = test_db.execute(
        select(User)
        .options(selectinload(User.profile))
        .where(User.email == "new@example.com")
    ).scalar_one()
    
    assert result.profile.bio == "Test bio"
\`\`\`

---

## Testing Migrations

### Migration Testing

\`\`\`python
"""
Test Alembic Migrations
"""

import pytest
from alembic.command import upgrade, downgrade
from alembic.config import Config

@pytest.fixture
def alembic_config():
    """Alembic configuration for tests"""
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", "postgresql://localhost/test_db")
    return config

def test_migration_up_down(alembic_config):
    """Test migration can apply and revert"""
    # Upgrade to head
    upgrade(alembic_config, "head")
    
    # Downgrade one revision
    downgrade(alembic_config, "-1")
    
    # Re-upgrade
    upgrade(alembic_config, "head")

def test_migration_idempotent(alembic_config):
    """Test migration can run multiple times"""
    # Run twice - should not error
    upgrade(alembic_config, "head")
    upgrade(alembic_config, "head")
\`\`\`

---

## CI/CD Integration

### GitHub Actions

\`\`\`yaml
# .github/workflows/test.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run migrations
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: |
          alembic upgrade head
      
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test_db
        run: |
          pytest --cov=myapp --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
\`\`\`

---

## Best Practices

### Testing Pyramid

\`\`\`
        /\\
       /  \\  E2E Tests (few, slow, expensive)
      /    \\
     /      \\
    / Integration Tests (some, medium speed)
   /          \\
  /            \\
 / Unit Tests   \\  (many, fast, cheap)
/________________\\
\`\`\`

### Test Organization

\`\`\`python
"""
Organized Test Structure
"""

# tests/
#   conftest.py           # Shared fixtures
#   test_models.py        # Model tests
#   test_repositories.py  # Repository tests
#   test_services.py      # Business logic tests
#   test_api.py           # API endpoint tests
#   test_migrations.py    # Migration tests

# conftest.py
@pytest.fixture(scope="session")
def test_engine():
    \"\"\"Shared test engine\"\"\"
    ...

@pytest.fixture(scope="function")
def test_db(test_engine):
    \"\"\"Clean database per test\"\"\"
    ...

@pytest.fixture
def factories(test_db):
    \"\"\"Configure factories\"\"\"
    ...
\`\`\`

---

## Summary

### Key Takeaways

✅ **In-memory SQLite**: Fast for unit tests  
✅ **PostgreSQL**: For integration tests (tests real DB behavior)  
✅ **Transaction rollback**: Clean database per test  
✅ **Factory Boy**: Generate test data easily  
✅ **Mock**: Test business logic without database  
✅ **Integration tests**: Test complete workflows  
✅ **CI/CD**: Automated testing with real database  
✅ **Test migrations**: Up, down, and idempotency

### Testing Strategy

✅ **Unit tests**: Models, business logic (fast, many)  
✅ **Integration tests**: Queries, relationships (medium)  
✅ **E2E tests**: Complete workflows (slow, few)  
✅ **Migration tests**: Up/down/idempotent  
✅ **Coverage**: Aim for 80%+ on critical code

### Production Checklist

✅ Test database setup with fixtures  
✅ Use factories for test data generation  
✅ Test all queries and relationships  
✅ Mock for unit tests, real DB for integration  
✅ Test migrations before production  
✅ CI/CD with automated testing  
✅ Coverage reports and monitoring

### Next Steps

In the next section, we'll explore **Multi-Database & Sharding**: horizontal scaling, read replicas, and distributed database patterns.
`,
};
