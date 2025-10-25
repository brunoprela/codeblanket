export const testingDatabasesSqlalchemy = {
  title: 'Testing Databases & SQLAlchemy',
  id: 'testing-databases-sqlalchemy',
  content: `
# Testing Databases & SQLAlchemy

## Introduction

**Database testing is one of the most challenging aspects of software testing**—tests must be fast, isolated, and reliable while working with stateful, persistent systems. Poor database testing leads to flaky tests, slow CI/CD pipelines, and bugs that only appear in production. Professional database testing requires understanding transactions, fixtures, factories, and the trade-offs between speed and production parity.

This section covers professional patterns for testing SQLAlchemy applications, from basic model tests to complex multi-tenant scenarios.

---

## The Challenge of Database Testing

### Problems with Naive Approaches

**Anti-pattern 1**: Using production database for tests
\`\`\`python
# DON'T DO THIS
engine = create_engine("postgresql://prod:password@prod-db/production")
\`\`\`
**Problems**: Destroys production data, tests affect real users, security risk.

**Anti-pattern 2**: Shared test database without cleanup
\`\`\`python
def test_create_user():
    user = User(username="alice")
    db.session.add(user)
    db.session.commit()
    # Never cleaned up—affects next test
\`\`\`
**Problems**: Tests depend on order, flaky failures, "works on my machine."

**Anti-pattern 3**: Manually deleting data after each test
\`\`\`python
def teardown():
    db.session.query(User).delete()
    db.session.query(Post).delete()
    db.session.commit()
\`\`\`
**Problems**: Slow (multiple DELETE queries), must remember all tables, foreign key issues.

### Professional Approach: Isolated, Fast, Reliable

**Requirements**:
1. **Isolated**: Each test starts with clean database state
2. **Fast**: Tests run in milliseconds, not seconds
3. **Reliable**: No flaky failures from shared state
4. **Production-like**: Catches real database issues

**Strategy**: Use in-memory SQLite for speed, PostgreSQL for production parity, transaction rollback for isolation.

---

## Test Database Setup

### Pattern 1: In-Memory SQLite (Development)

**Best for**: Unit tests, rapid feedback, local development

\`\`\`python
# conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from myapp.models import Base

@pytest.fixture(scope="session")
def engine():
    """
    Create in-memory SQLite database for entire test session.
    
    Benefits:
    - 100× faster than PostgreSQL (no disk I/O)
    - No cleanup needed (destroyed when process exits)
    - Perfect for unit tests
    
    Limitations:
    - Lacks PostgreSQL-specific features (JSONB, arrays, full-text search)
    - Different SQL dialect (may hide production issues)
    """
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,  # Set True for SQL debugging
        connect_args={"check_same_thread": False}  # Allow multi-threaded access
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Provide clean database session for each test.
    
    Uses transaction rollback for isolation:
    1. Begin transaction
    2. Run test
    3. Rollback (clean state for next test)
    """
    connection = engine.connect()
    transaction = connection.begin()
    
    # Create session bound to this connection
    Session = scoped_session(sessionmaker(bind=connection))
    session = Session()
    
    yield session
    
    # Cleanup
    session.close()
    transaction.rollback()  # Undo all changes
    connection.close()
    Session.remove()
\`\`\`

**Usage**:
\`\`\`python
def test_create_user(db_session):
    user = User(username="alice", email="alice@example.com")
    db_session.add(user)
    db_session.commit()
    
    assert user.id is not None
    assert user.username == "alice"
    # Rolled back after test—next test has clean database
\`\`\`

### Pattern 2: PostgreSQL Test Database (CI/CD)

**Best for**: Integration tests, CI/CD, catching PostgreSQL-specific issues

\`\`\`python
# conftest.py
import pytest
from sqlalchemy import create_engine
from myapp.models import Base

@pytest.fixture(scope="session")
def engine():
    """
    Create PostgreSQL test database.
    
    Benefits:
    - Production parity (same database engine)
    - Tests PostgreSQL-specific features
    - Catches real database issues
    
    Limitations:
    - Slower than SQLite (disk I/O, connection overhead)
    - Requires PostgreSQL server
    """
    # Use test database (never production!)
    db_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
    engine = create_engine(db_url, echo=False)
    
    # Drop all tables (clean slate)
    Base.metadata.drop_all(engine)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup after all tests
    Base.metadata.drop_all(engine)
    engine.dispose()

@pytest.fixture(scope="function")
def db_session(engine):
    """Same transaction rollback pattern as SQLite"""
    connection = engine.connect()
    transaction = connection.begin()
    
    Session = scoped_session(sessionmaker(bind=connection))
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
    Session.remove()
\`\`\`

### Pattern 3: Docker PostgreSQL (Local Development)

**Best for**: Local testing with production parity, no manual PostgreSQL setup

\`\`\`yaml
# docker-compose.test.yml
version: '3.8'
services:
  test_db:
    image: postgres:15
    environment:
      POSTGRES_DB: test_db
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
    ports:
      - "5433:5432"  # Avoid conflict with local PostgreSQL
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user"]
      interval: 5s
      timeout: 5s
      retries: 5
\`\`\`

\`\`\`python
# conftest.py
import pytest
import subprocess
import time

@pytest.fixture(scope="session", autouse=True)
def docker_db():
    """Start PostgreSQL in Docker for tests"""
    # Start containers
    subprocess.run(["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"], check=True)
    
    # Wait for health check
    time.sleep(3)
    
    yield
    
    # Cleanup
    subprocess.run(["docker-compose", "-f", "docker-compose.test.yml", "down"], check=True)
\`\`\`

### Choosing the Right Strategy

| Strategy | Speed | Production Parity | Setup Complexity | Use Case |
|----------|-------|-------------------|------------------|----------|
| **SQLite :memory:** | ⚡️⚡️⚡️ | ⭐️ | ⚡️ | Unit tests, local dev |
| **PostgreSQL Local** | ⚡️ | ⚡️⚡️⚡️ | ⚡️⚡️ | Integration tests, CI |
| **Docker PostgreSQL** | ⚡️⚡️ | ⚡️⚡️⚡️ | ⚡️⚡️⚡️ | Local dev, no PostgreSQL install |

**Professional recommendation**: SQLite for local unit tests (speed), PostgreSQL in CI (production parity).

---

## Transaction Rollback Pattern (Essential)

**The transaction rollback pattern is the foundation of fast, isolated database tests.**

### How It Works

\`\`\`python
@pytest.fixture(scope="function")
def db_session(engine):
    # 1. Create connection
    connection = engine.connect()
    
    # 2. Begin transaction (savepoint)
    transaction = connection.begin()
    
    # 3. Create session bound to this transaction
    session = Session(bind=connection)
    
    # 4. Run test (all changes in transaction)
    yield session
    
    # 5. Rollback transaction (undo all changes)
    session.close()
    transaction.rollback()  # ← Key: Returns to clean state
    connection.close()
\`\`\`

### Benefits

1. **Fast**: Rollback is 100× faster than DELETE queries
   - DELETE: 100ms for 1000 rows
   - Rollback: <1ms (single transaction undo)

2. **Automatic cleanup**: No need to track what was created

3. **Handles relationships**: Foreign keys automatically handled

4. **Perfect isolation**: Each test gets pristine database

### Comparison: Rollback vs Truncate vs Drop/Create

\`\`\`python
# Approach 1: Rollback (BEST)
def cleanup_rollback(session):
    session.rollback()  # 1ms
    
# Approach 2: Truncate (SLOW)
def cleanup_truncate(session):
    session.execute("TRUNCATE users, posts, comments CASCADE")  # 100ms
    
# Approach 3: Drop/Create (VERY SLOW)
def cleanup_drop_create(engine):
    Base.metadata.drop_all(engine)   # 500ms
    Base.metadata.create_all(engine)  # 500ms
\`\`\`

**Performance for 100 tests**:
- Rollback: 0.1 seconds
- Truncate: 10 seconds
- Drop/Create: 100 seconds

---

## Testing Models

### Basic Model Tests

\`\`\`python
from myapp.models import User, Post

def test_user_creation(db_session):
    """Test creating user model"""
    user = User(
        username="alice",
        email="alice@example.com",
        age=30
    )
    db_session.add(user)
    db_session.commit()
    
    # Verify auto-generated ID
    assert user.id is not None
    
    # Verify fields
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.age == 30
    
    # Verify timestamps (if using created_at)
    assert user.created_at is not None

def test_user_repr(db_session):
    """Test string representation"""
    user = User(username="alice")
    assert "alice" in repr(user)

def test_user_validation(db_session):
    """Test model validation"""
    user = User(username="a", email="invalid")  # Too short, invalid email
    db_session.add(user)
    
    with pytest.raises(ValueError):
        db_session.commit()
\`\`\`

### Testing Relationships

\`\`\`python
def test_one_to_many_relationship(db_session):
    """Test user → posts relationship"""
    # Create user
    user = User(username="alice")
    db_session.add(user)
    db_session.commit()
    
    # Create posts
    post1 = Post(title="First Post", content="Hello", author=user)
    post2 = Post(title="Second Post", content="World", author=user)
    db_session.add_all([post1, post2])
    db_session.commit()
    
    # Test relationship access
    assert len(user.posts) == 2
    assert user.posts[0].title == "First Post"
    assert user.posts[1].title == "Second Post"
    
    # Test back reference
    assert post1.author == user
    assert post2.author == user

def test_many_to_many_relationship(db_session):
    """Test users ↔ groups relationship"""
    # Create users
    alice = User(username="alice")
    bob = User(username="bob")
    
    # Create groups
    admins = Group(name="Admins")
    users = Group(name="Users")
    
    # Setup relationships
    alice.groups.extend([admins, users])
    bob.groups.append(users)
    
    db_session.add_all([alice, bob, admins, users])
    db_session.commit()
    
    # Test relationships
    assert len(alice.groups) == 2
    assert len(bob.groups) == 1
    assert len(admins.users) == 1
    assert len(users.users) == 2

def test_cascade_delete(db_session):
    """Test cascade deletion"""
    user = User(username="alice")
    post = Post(title="Test", author=user)
    
    db_session.add_all([user, post])
    db_session.commit()
    
    post_id = post.id
    
    # Delete user (should cascade to posts)
    db_session.delete(user)
    db_session.commit()
    
    # Verify post was deleted
    assert db_session.query(Post).filter_by(id=post_id).first() is None
\`\`\`

### Testing Queries

\`\`\`python
def test_simple_query(db_session):
    """Test basic querying"""
    # Create test data
    alice = User(username="alice", age=30)
    bob = User(username="bob", age=25)
    charlie = User(username="charlie", age=35)
    
    db_session.add_all([alice, bob, charlie])
    db_session.commit()
    
    # Query all
    users = db_session.query(User).all()
    assert len(users) == 3
    
    # Query with filter
    young_users = db_session.query(User).filter(User.age < 30).all()
    assert len(young_users) == 1
    assert young_users[0].username == "bob"
    
    # Query with order
    ordered = db_session.query(User).order_by(User.age.desc()).all()
    assert ordered[0].username == "charlie"
    assert ordered[1].username == "alice"
    assert ordered[2].username == "bob"

def test_complex_query(db_session):
    """Test joins and aggregations"""
    # Setup data
    alice = User(username="alice")
    bob = User(username="bob")
    
    Post(title="Alice Post 1", author=alice)
    Post(title="Alice Post 2", author=alice)
    Post(title="Bob Post 1", author=bob)
    
    db_session.add_all([alice, bob])
    db_session.commit()
    
    # Query with join
    from sqlalchemy import func
    result = db_session.query(
        User.username,
        func.count(Post.id).label("post_count")
    ).join(Post).group_by(User.id).all()
    
    assert len(result) == 2
    assert dict(result)[("alice",)] == 2
    assert dict(result)[("bob",)] == 1

def test_query_performance(db_session):
    """Test N+1 query problem"""
    # Create data with relationships
    users = [User(username=f"user{i}") for i in range(10)]
    db_session.add_all(users)
    db_session.commit()
    
    for user in users:
        db_session.add(Post(title=f"Post by {user.username}", author=user))
    db_session.commit()
    
    # Bad: N+1 queries (1 for users + 10 for each user's posts)
    users = db_session.query(User).all()
    for user in users:
        _ = user.posts  # Triggers separate query
    
    # Good: Eager loading (1 query with join)
    from sqlalchemy.orm import joinedload
    users = db_session.query(User).options(joinedload(User.posts)).all()
    for user in users:
        _ = user.posts  # No additional query
\`\`\`

---

## Factory Pattern with Factory Boy

**Factory Boy generates realistic test data efficiently**—eliminating repetitive test setup.

### Installation

\`\`\`bash
pip install factory-boy
\`\`\`

### Basic Factory

\`\`\`python
# tests/factories.py
import factory
from factory.alchemy import SQLAlchemyModelFactory
from myapp.models import User, Post

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = None  # Set per test
        sqlalchemy_session_persistence = "commit"
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    age = factory.Faker("random_int", min=18, max=80)
    created_at = factory.Faker("date_time_this_year")

class PostFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Post
        sqlalchemy_session = None
        sqlalchemy_session_persistence = "commit"
    
    title = factory.Faker("sentence", nb_words=6)
    content = factory.Faker("paragraph", nb_sentences=5)
    author = factory.SubFactory(UserFactory)  # Auto-creates related user
\`\`\`

### Using Factories in Tests

\`\`\`python
def test_with_factory(db_session):
    """Clean, concise test data creation"""
    # Set session for factories
    UserFactory._meta.sqlalchemy_session = db_session
    PostFactory._meta.sqlalchemy_session = db_session
    
    # Create single user
    user = UserFactory.create()
    assert user.id is not None
    assert "@example.com" in user.email
    
    # Create batch of users
    users = UserFactory.create_batch(5)
    assert len(users) == 5
    
    # Create with overrides
    admin = UserFactory.create(username="admin", age=40)
    assert admin.username == "admin"
    assert admin.age == 40
    
    # Create with relationships
    post = PostFactory.create(author=user)
    assert post.author == user

def test_factory_traits(db_session):
    """Use traits for variations"""
    class UserFactory(SQLAlchemyModelFactory):
        class Meta:
            model = User
            sqlalchemy_session = db_session
        
        username = factory.Sequence(lambda n: f"user{n}")
        age = 30
        is_active = True
        
        class Params:
            admin = factory.Trait(
                username="admin",
                is_admin=True
            )
            inactive = factory.Trait(
                is_active=False
            )
    
    # Normal user
    user = UserFactory.create()
    assert not user.is_admin
    
    # Admin user
    admin = UserFactory.create(admin=True)
    assert admin.is_admin
    
    # Inactive admin
    inactive_admin = UserFactory.create(admin=True, inactive=True)
    assert inactive_admin.is_admin
    assert not inactive_admin.is_active
\`\`\`

### Factory Fixture Pattern

\`\`\`python
# conftest.py
@pytest.fixture
def factories(db_session):
    """Configure all factories with session"""
    UserFactory._meta.sqlalchemy_session = db_session
    PostFactory._meta.sqlalchemy_session = db_session
    
    return {
        "User": UserFactory,
        "Post": PostFactory,
    }

# Usage in tests
def test_with_factory_fixture(factories):
    user = factories["User"].create(username="alice")
    post = factories["Post"].create(author=user)
    assert post.author == user
\`\`\`

---

## Advanced Patterns

### Testing Database Migrations

\`\`\`python
import pytest
from alembic.config import Config
from alembic import command

def test_migration_up_down(engine):
    """Test migration is reversible"""
    alembic_cfg = Config("alembic.ini")
    
    # Start from base
    command.downgrade(alembic_cfg, "base")
    
    # Migrate up
    command.upgrade(alembic_cfg, "head")
    
    # Verify migration applied
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "posts" in tables
    
    # Migrate down
    command.downgrade(alembic_cfg, "-1")
    
    # Verify rollback
    tables = inspector.get_table_names()
    assert "posts" not in tables

def test_migration_data_transformation(engine):
    """Test data is preserved during migration"""
    # Create test data with old schema
    # Run migration
    # Verify data transformed correctly
    pass
\`\`\`

### Testing Multi-Tenancy

\`\`\`python
def test_tenant_isolation(db_session):
    """Test tenant data isolation"""
    # Create tenant 1 data
    tenant1_user = User(username="alice", tenant_id=1)
    db_session.add(tenant1_user)
    db_session.commit()
    
    # Create tenant 2 data
    tenant2_user = User(username="bob", tenant_id=2)
    db_session.add(tenant2_user)
    db_session.commit()
    
    # Query with tenant filter
    tenant1_users = db_session.query(User).filter_by(tenant_id=1).all()
    assert len(tenant1_users) == 1
    assert tenant1_users[0].username == "alice"
    
    tenant2_users = db_session.query(User).filter_by(tenant_id=2).all()
    assert len(tenant2_users) == 1
    assert tenant2_users[0].username == "bob"
\`\`\`

---

## Best Practices

1. **Use transaction rollback** for isolation (100× faster than truncate)
2. **SQLite for speed** (local dev), **PostgreSQL for CI** (production parity)
3. **Factory Boy for test data** (DRY, realistic, maintainable)
4. **Session-scoped engine**, function-scoped session
5. **Test relationships** and cascade behavior
6. **Avoid N+1 queries** (use eager loading)
7. **Test migrations** are reversible
8. **Docker for dependencies** (reproducible environments)

---

## Performance Comparison

| Approach | 100 Tests | 1000 Tests |
|----------|-----------|------------|
| **SQLite + Rollback** | 2 sec | 20 sec |
| **PostgreSQL + Rollback** | 10 sec | 100 sec |
| **PostgreSQL + Truncate** | 30 sec | 300 sec |
| **PostgreSQL + Drop/Create** | 100 sec | 1000 sec |

**Recommendation**: SQLite + rollback for development (fast feedback), PostgreSQL + rollback for CI (production parity).

---

## Summary

**Professional database testing**:
- **Fast**: In-memory SQLite (100× faster than PostgreSQL)
- **Isolated**: Transaction rollback per test (1ms cleanup)
- **Realistic**: Factory Boy for test data
- **Production-like**: PostgreSQL in CI
- **Maintainable**: Clean fixtures, no manual cleanup

Master these patterns for **reliable, fast database tests** that catch bugs before production.
`,
};
