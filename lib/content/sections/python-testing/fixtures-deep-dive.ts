export const fixturesDeepDive = {
  title: 'Fixtures Deep Dive',
  id: 'fixtures-deep-dive',
  content: `
# Fixtures Deep Dive

## Introduction

**Fixtures are pytest's killer feature**—they're what make pytest vastly superior to unittest's setUp/tearDown. Fixtures provide a powerful dependency injection system that eliminates code duplication, improves test isolation, and makes tests more maintainable.

### The Problem Fixtures Solve

Consider a test suite without fixtures:

\`\`\`python
def test_create_user():
    # Duplicate setup code
    db = create_database_connection()
    db.begin_transaction()
    
    user = User (name="Alice")
    db.add (user)
    db.commit()
    
    assert user.id is not None
    
    # Duplicate cleanup
    db.rollback()
    db.close()

def test_update_user():
    # Same duplicate setup!
    db = create_database_connection()
    db.begin_transaction()
    
    user = User (name="Bob")
    db.add (user)
    user.name = "Robert"
    db.commit()
    
    assert user.name == "Robert"
    
    # Same duplicate cleanup!
    db.rollback()
    db.close()

# Repeated 100 times across test suite...
\`\`\`

**Problem**: 100 tests × 10 lines setup/cleanup = 1,000 lines of duplicate code.

**Fixtures solve this**:

\`\`\`python
@pytest.fixture
def db_session():
    """Reusable database fixture"""
    db = create_database_connection()
    db.begin_transaction()
    yield db
    db.rollback()
    db.close()

def test_create_user (db_session):
    """db_session injected automatically"""
    user = User (name="Alice")
    db_session.add (user)
    db_session.commit()
    assert user.id is not None

def test_update_user (db_session):
    """Same fixture, zero duplication"""
    user = User (name="Bob")
    db_session.add (user)
    user.name = "Robert"
    db_session.commit()
    assert user.name == "Robert"
\`\`\`

**Result**: 100 tests use 1 fixture (8 lines) instead of 1,000 lines of duplicate code. **99.2% reduction in duplication.**

---

## Basic Fixtures

### Defining Fixtures

Use \`@pytest.fixture\` decorator:

\`\`\`python
import pytest

@pytest.fixture
def sample_data():
    """Fixture that returns sample data"""
    return {"name": "Alice", "age": 30}

def test_user_name (sample_data):
    """Use fixture by naming it as parameter"""
    assert sample_data["name"] == "Alice"

def test_user_age (sample_data):
    """Each test gets fresh copy"""
    assert sample_data["age"] == 30
\`\`\`

**Key concepts**:
- Fixtures are functions decorated with \`@pytest.fixture\`
- Tests request fixtures by parameter name
- Fixtures can return data, objects, or resources
- Each test gets its own fixture instance (by default)

### Setup and Teardown with yield

Fixtures use \`yield\` for setup/teardown:

\`\`\`python
@pytest.fixture
def database_connection():
    """Fixture with setup and teardown"""
    # Setup: runs before test
    print("Connecting to database...")
    db = DatabaseConnection()
    db.connect()
    
    # Provide resource to test
    yield db
    
    # Teardown: runs after test (even if test fails)
    print("Disconnecting from database...")
    db.disconnect()

def test_query (database_connection):
    """Uses database, guaranteed cleanup"""
    result = database_connection.query("SELECT * FROM users")
    assert len (result) > 0
    # db.disconnect() called automatically after test
\`\`\`

**Execution order**:
1. Setup code (before yield)
2. Test runs (uses yielded value)
3. Teardown code (after yield)
4. Teardown guaranteed even if test fails

### Fixtures Without Return Values

Fixtures don't have to return anything—they can perform side effects:

\`\`\`python
@pytest.fixture
def clean_database():
    """Fixture that cleans database (no return value)"""
    database.truncate_all_tables()
    # No yield/return needed if no value to provide

def test_user_creation (clean_database):
    """Database cleaned before test runs"""
    user = create_user("Alice")
    assert user.id == 1  # First user ID (clean database)
\`\`\`

---

## Fixture Scope

**Scope** determines how often fixture is created and destroyed.

### Function Scope (Default)

**Created once per test function** (default behavior):

\`\`\`python
@pytest.fixture (scope="function")  # Default, can omit
def user():
    """New user for each test"""
    print("Creating user...")
    return User (name="Alice")

def test_user_name (user):
    print("Test 1")
    assert user.name == "Alice"

def test_user_age (user):
    print("Test 2")
    user.age = 30
    assert user.age == 30

# Output:
# Creating user...
# Test 1
# Creating user...
# Test 2
# (User created twice—once per test)
\`\`\`

**Use when**: Test needs isolated data (default choice).

### Class Scope

**Created once per test class**:

\`\`\`python
@pytest.fixture (scope="class")
def api_client():
    """One client for entire test class"""
    print("Creating API client...")
    client = APIClient()
    yield client
    print("Closing API client...")
    client.close()

class TestUserAPI:
    def test_create_user (self, api_client):
        print("Test 1")
        response = api_client.post("/users", {"name": "Alice"})
        assert response.status_code == 201
    
    def test_get_user (self, api_client):
        print("Test 2")
        response = api_client.get("/users/1")
        assert response.status_code == 200

# Output:
# Creating API client...
# Test 1
# Test 2
# Closing API client...
# (Client created once for both tests)
\`\`\`

**Use when**: Expensive setup shared across related tests.

### Module Scope

**Created once per test module (file)**:

\`\`\`python
@pytest.fixture (scope="module")
def database():
    """One database for entire test file"""
    print("Setting up database...")
    db = create_test_database()
    yield db
    print("Tearing down database...")
    db.drop_all_tables()
    db.close()

def test_query_1(database):
    result = database.query("SELECT * FROM users")
    assert isinstance (result, list)

def test_query_2(database):
    result = database.query("SELECT * FROM products")
    assert isinstance (result, list)

# Database created once when first test runs
# Same database used for both tests
# Database destroyed after last test in file
\`\`\`

**Use when**: Expensive resource shared across all tests in file.

### Session Scope

**Created once per entire test session**:

\`\`\`python
@pytest.fixture (scope="session")
def browser():
    """One browser for entire test session"""
    print("Launching browser...")
    browser = webdriver.Chrome()
    yield browser
    print("Closing browser...")
    browser.quit()

# In test_login.py
def test_login (browser):
    browser.get("http://example.com/login")
    # ...

# In test_checkout.py
def test_checkout (browser):
    browser.get("http://example.com/checkout")
    # ...

# Browser launched once at start of test session
# Same browser used across ALL test files
# Browser closed at end of session
\`\`\`

**Use when**: Very expensive setup (database schema creation, browser launch).

**⚠️ Warning**: Session-scoped fixtures can cause test interdependence. Use carefully.

### Scope Comparison

| Scope    | Created | Destroyed | Use Case                      |
|----------|---------|-----------|-------------------------------|
| function | Per test | After each test | Isolated test data (default) |
| class    | Per test class | After last test in class | Shared expensive setup for related tests |
| module   | Per file | After last test in file | Database connection per file |
| session  | Once per test run | At end of test run | Database schema, browser launch |

**Performance impact**:

\`\`\`python
# 1000 tests with function-scoped database (slow)
@pytest.fixture
def db():
    return create_database()  # Created 1000 times

# 1000 tests with session-scoped database (fast)
@pytest.fixture (scope="session")
def db():
    return create_database()  # Created once

# Speedup: 50 seconds → 5 seconds (10× faster)
\`\`\`

---

## Fixture Dependencies

Fixtures can depend on other fixtures (composition):

\`\`\`python
@pytest.fixture
def database():
    """Base database fixture"""
    db = Database()
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture
def user_repository (database):
    """Depends on database fixture"""
    return UserRepository (database)

@pytest.fixture
def user_service (user_repository):
    """Depends on user_repository (which depends on database)"""
    return UserService (user_repository)

def test_create_user (user_service):
    """Uses user_service (which uses user_repository, which uses database)"""
    user = user_service.create_user("Alice")
    assert user.id is not None

# Execution order:
# 1. database fixture created
# 2. user_repository fixture created (receives database)
# 3. user_service fixture created (receives user_repository)
# 4. Test runs (receives user_service)
# 5. Teardown in reverse order
\`\`\`

**Benefits**:
- **Modularity**: Each fixture has single responsibility
- **Reusability**: Mix and match fixtures
- **Clarity**: Explicit dependencies

### Complex Dependency Example

\`\`\`python
@pytest.fixture
def db_engine():
    """Database engine"""
    engine = create_engine("postgresql://...")
    yield engine
    engine.dispose()

@pytest.fixture
def db_session (db_engine):
    """Database session (depends on engine)"""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker (bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def user (db_session):
    """Test user (depends on session)"""
    user = User (name="Alice", email="alice@example.com")
    db_session.add (user)
    db_session.commit()
    return user

@pytest.fixture
def authenticated_client (user):
    """API client with auth (depends on user)"""
    client = APIClient()
    token = generate_token (user)
    client.set_header("Authorization", f"Bearer {token}")
    return client

def test_get_profile (authenticated_client):
    """Uses entire chain: db_engine → db_session → user → authenticated_client"""
    response = authenticated_client.get("/me")
    assert response.status_code == 200
    assert response.json["name"] == "Alice"
\`\`\`

---

## Autouse Fixtures

Fixtures that run automatically for every test:

\`\`\`python
@pytest.fixture (autouse=True)
def clean_database():
    """Runs before EVERY test automatically"""
    print("Cleaning database...")
    database.truncate_all_tables()

def test_create_user():
    """Database automatically cleaned before this test"""
    user = User (name="Alice")
    assert user.id == 1  # First user (clean DB)

def test_create_product():
    """Database automatically cleaned before this test too"""
    product = Product (name="Widget")
    assert product.id == 1  # First product (clean DB)

# No need to add clean_database parameter to tests
# Fixture runs automatically
\`\`\`

**Use cases for autouse**:
- **Database cleanup**: Clean DB before every test
- **Logging setup**: Configure logging for every test
- **Environment setup**: Set environment variables
- **Mocking**: Auto-mock external services

**⚠️ Warning**: Use autouse sparingly—makes it less clear what fixtures each test uses.

### Autouse with Scope

\`\`\`python
@pytest.fixture (scope="session", autouse=True)
def setup_test_environment():
    """Runs once at start of test session"""
    print("Setting up test environment...")
    os.environ["ENV"] = "test"
    os.environ["DEBUG"] = "true"
    yield
    print("Cleaning up test environment...")
    del os.environ["ENV"]
    del os.environ["DEBUG"]

# Runs automatically for entire test session
# No tests need to request this fixture
\`\`\`

---

## Fixture Factories

Fixtures that return factory functions for creating multiple instances:

\`\`\`python
@pytest.fixture
def user_factory (db_session):
    """Returns factory function to create users"""
    def make_user (name="Test User", email=None):
        if email is None:
            email = f"{name.replace(' ', '')}@example.com"
        user = User (name=name, email=email)
        db_session.add (user)
        db_session.commit()
        return user
    
    return make_user

def test_multiple_users (user_factory):
    """Create multiple users with factory"""
    alice = user_factory (name="Alice")
    bob = user_factory (name="Bob")
    charlie = user_factory (name="Charlie")
    
    assert alice.email == "Alice@example.com"
    assert bob.email == "Bob@example.com"
    assert charlie.email == "Charlie@example.com"
\`\`\`

**Benefits**:
- **Flexibility**: Create multiple instances with different parameters
- **Convenience**: Defaults provided, override as needed
- **Reusability**: Factory used across many tests

### Advanced Factory Pattern

\`\`\`python
@pytest.fixture
def payment_factory (db_session):
    """Factory with cleanup tracking"""
    created_payments = []
    
    def make_payment (amount, currency="USD", status="pending"):
        payment = Payment (amount=amount, currency=currency, status=status)
        db_session.add (payment)
        db_session.commit()
        created_payments.append (payment)
        return payment
    
    yield make_payment
    
    # Cleanup: delete all created payments
    for payment in created_payments:
        db_session.delete (payment)
    db_session.commit()

def test_payment_processing (payment_factory):
    """Create multiple payments, automatically cleaned up"""
    payment1 = payment_factory (amount=100.0)
    payment2 = payment_factory (amount=200.0)
    payment3 = payment_factory (amount=50.0, currency="EUR")
    
    assert len([payment1, payment2, payment3]) == 3
    # All payments deleted automatically after test
\`\`\`

---

## Parametrizing Fixtures

Create multiple versions of same fixture with different parameters:

\`\`\`python
@pytest.fixture (params=["sqlite", "postgresql", "mysql"])
def database (request):
    """Fixture parametrized by database type"""
    db_type = request.param
    
    if db_type == "sqlite":
        db = SQLiteDatabase(":memory:")
    elif db_type == "postgresql":
        db = PostgreSQLDatabase("localhost", "testdb")
    elif db_type == "mysql":
        db = MySQLDatabase("localhost", "testdb")
    
    db.connect()
    yield db
    db.disconnect()

def test_query (database):
    """Test runs 3 times (once per database)"""
    result = database.query("SELECT 1")
    assert result is not None

# Output:
# test_query[sqlite] PASSED
# test_query[postgresql] PASSED
# test_query[mysql] PASSED
\`\`\`

**Use case**: Test same functionality across different configurations.

### Parametrized Fixture with IDs

\`\`\`python
@pytest.fixture(
    params=[
        ("USD", 100.0),
        ("EUR", 85.0),
        ("GBP", 75.0)
    ],
    ids=["USD", "EUR", "GBP"]
)
def currency_amount (request):
    """Parametrized fixture with readable IDs"""
    currency, amount = request.param
    return {"currency": currency, "amount": amount}

def test_payment (currency_amount):
    """Test runs 3 times with clear output"""
    assert currency_amount["amount"] > 0

# Output:
# test_payment[USD] PASSED
# test_payment[EUR] PASSED
# test_payment[GBP] PASSED
\`\`\`

---

## Built-in Fixtures

pytest provides useful built-in fixtures:

### tmp_path: Temporary Directory

\`\`\`python
def test_file_operations (tmp_path):
    """tmp_path provides unique temporary directory"""
    # Create test file
    test_file = tmp_path / "data.txt"
    test_file.write_text("Hello, World!")
    
    # Read file
    content = test_file.read_text()
    assert content == "Hello, World!"
    
    # Directory automatically cleaned up after test
    # No need to manually delete files
\`\`\`

### capsys: Capture stdout/stderr

\`\`\`python
def test_print_output (capsys):
    """capsys captures print statements"""
    print("Hello, World!")
    print("Testing output", file=sys.stderr)
    
    captured = capsys.readouterr()
    assert "Hello, World!" in captured.out
    assert "Testing output" in captured.err
\`\`\`

### monkeypatch: Mock/Patch

\`\`\`python
def test_environment_variable (monkeypatch):
    """monkeypatch temporarily modifies environment"""
    monkeypatch.setenv("API_KEY", "test_key_123")
    
    assert os.environ["API_KEY"] == "test_key_123"
    # Automatically restored after test
\`\`\`

### request: Fixture Metadata

\`\`\`python
@pytest.fixture
def database (request):
    """Access fixture metadata with request"""
    # Get test function that requested this fixture
    test_name = request.node.name
    print(f"Setting up database for {test_name}")
    
    db = Database()
    
    # Add finalizer (alternative to yield)
    request.addfinalizer (db.close)
    
    return db
\`\`\`

---

## Real-World Example: Complete Test Suite

\`\`\`python
"""
Complete test suite with fixtures
Testing a payment processing service
"""
import pytest
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from myapp.models import Base, User, Payment
from myapp.services import PaymentService


# ==================== Database Fixtures ====================

@pytest.fixture (scope="session")
def db_engine():
    """Session-scoped database engine"""
    engine = create_engine("postgresql://test:test@localhost/testdb")
    
    # Create all tables once
    Base.metadata.create_all (engine)
    
    yield engine
    
    # Drop all tables at end
    Base.metadata.drop_all (engine)
    engine.dispose()


@pytest.fixture
def db_session (db_engine):
    """Function-scoped clean database session"""
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker (bind=connection)
    session = Session()
    
    yield session
    
    # Rollback transaction (clean state for next test)
    session.close()
    transaction.rollback()
    connection.close()


# ==================== Model Factories ====================

@pytest.fixture
def user_factory (db_session):
    """Factory for creating test users"""
    def make_user (name="Test User", email=None, balance=Decimal("1000.00")):
        if email is None:
            email = f"{name.replace(' ', '')}@example.com"
        
        user = User (name=name, email=email, balance=balance)
        db_session.add (user)
        db_session.commit()
        return user
    
    return make_user


@pytest.fixture
def payment_factory (db_session):
    """Factory for creating test payments"""
    def make_payment (user, amount, currency="USD", status="pending"):
        payment = Payment(
            user=user,
            amount=amount,
            currency=currency,
            status=status
        )
        db_session.add (payment)
        db_session.commit()
        return payment
    
    return make_payment


# ==================== Service Fixtures ====================

@pytest.fixture
def payment_service (db_session):
    """Payment service instance"""
    return PaymentService (db_session)


# ==================== Common Test Data ====================

@pytest.fixture
def alice (user_factory):
    """Standard test user Alice"""
    return user_factory (name="Alice", balance=Decimal("500.00"))


@pytest.fixture
def bob (user_factory):
    """Standard test user Bob"""
    return user_factory (name="Bob", balance=Decimal("1000.00"))


# ==================== Tests ====================

class TestUserCreation:
    """Tests for user creation"""
    
    def test_create_user_with_defaults (self, user_factory):
        """Should create user with default values"""
        user = user_factory()
        assert user.id is not None
        assert user.name == "Test User"
        assert user.balance == Decimal("1000.00")
    
    def test_create_user_with_custom_values (self, user_factory):
        """Should create user with custom values"""
        user = user_factory (name="Charlie", balance=Decimal("250.00"))
        assert user.name == "Charlie"
        assert user.balance == Decimal("250.00")


class TestPaymentProcessing:
    """Tests for payment processing"""
    
    def test_process_payment_success (self, payment_service, alice, payment_factory):
        """Should process valid payment successfully"""
        payment = payment_factory (user=alice, amount=Decimal("100.00"))
        
        result = payment_service.process_payment (payment)
        
        assert result is True
        assert payment.status == "completed"
        assert alice.balance == Decimal("400.00")  # 500 - 100
    
    def test_process_payment_insufficient_funds (self, payment_service, alice, payment_factory):
        """Should fail when insufficient funds"""
        payment = payment_factory (user=alice, amount=Decimal("600.00"))
        
        result = payment_service.process_payment (payment)
        
        assert result is False
        assert payment.status == "failed"
        assert alice.balance == Decimal("500.00")  # Unchanged
    
    def test_transfer_between_users (self, payment_service, alice, bob):
        """Should transfer money between users"""
        result = payment_service.transfer(
            from_user=bob,
            to_user=alice,
            amount=Decimal("200.00")
        )
        
        assert result is True
        assert bob.balance == Decimal("800.00")  # 1000 - 200
        assert alice.balance == Decimal("700.00")  # 500 + 200


class TestPaymentRefunds:
    """Tests for payment refunds"""
    
    def test_refund_completed_payment (self, payment_service, alice, payment_factory):
        """Should refund completed payment"""
        payment = payment_factory (user=alice, amount=Decimal("100.00"))
        payment_service.process_payment (payment)
        
        result = payment_service.refund_payment (payment)
        
        assert result is True
        assert payment.status == "refunded"
        assert alice.balance == Decimal("500.00")  # Back to original
    
    def test_cannot_refund_pending_payment (self, payment_service, alice, payment_factory):
        """Should not refund pending payment"""
        payment = payment_factory (user=alice, amount=Decimal("100.00"))
        
        with pytest.raises(ValueError, match="only refund completed"):
            payment_service.refund_payment (payment)
\`\`\`

---

## Best Practices

### 1. One Fixture, One Responsibility

✅ **Good**: Each fixture has clear purpose

\`\`\`python
@pytest.fixture
def db_session():
    """Only provides database session"""
    ...

@pytest.fixture
def user (db_session):
    """Only creates user"""
    ...
\`\`\`

❌ **Bad**: Fixture does too much

\`\`\`python
@pytest.fixture
def everything():
    """Creates database, user, payment, client..."""
    ...
\`\`\`

### 2. Name Fixtures Clearly

✅ **Good**: Clear what fixture provides

\`\`\`python
@pytest.fixture
def authenticated_api_client():
    ...

@pytest.fixture
def clean_database_session():
    ...
\`\`\`

❌ **Bad**: Unclear names

\`\`\`python
@pytest.fixture
def setup():  # Setup what?
    ...

@pytest.fixture
def data():  # What data?
    ...
\`\`\`

### 3. Use Appropriate Scope

Choose scope based on fixture cost and test isolation:

- **function** (default): Test isolation, moderate cost
- **class**: Shared across related tests, expensive setup
- **module**: Shared across file, very expensive
- **session**: Shared across all tests, extremely expensive

### 4. Prefer Composition Over Large Fixtures

✅ **Good**: Small, composable fixtures

\`\`\`python
@pytest.fixture
def db_session():
    ...

@pytest.fixture
def user (db_session):
    ...

@pytest.fixture
def payment (user):
    ...
\`\`\`

❌ **Bad**: Monolithic fixture

\`\`\`python
@pytest.fixture
def complete_setup():
    """Does everything"""
    db = create_db()
    user = create_user (db)
    payment = create_payment (user)
    return db, user, payment
\`\`\`

### 5. Document Fixtures

Add docstrings explaining what fixture provides:

\`\`\`python
@pytest.fixture
def db_session():
    """
    Provides clean database session for each test.
    
    - Session-level transaction (rolled back after test)
    - All tables available
    - Automatically cleaned up
    
    Yields:
        Session: SQLAlchemy session
    """
    ...
\`\`\`

---

## Common Patterns

### Pattern 1: Database with Automatic Cleanup

\`\`\`python
@pytest.fixture (scope="session")
def db_engine():
    """Database engine (created once)"""
    engine = create_engine("postgresql://...")
    Base.metadata.create_all (engine)
    yield engine
    Base.metadata.drop_all (engine)
    engine.dispose()

@pytest.fixture
def db_session (db_engine):
    """Clean database session per test"""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session (bind=connection)
    yield session
    session.close()
    transaction.rollback()  # Clean state
    connection.close()
\`\`\`

### Pattern 2: API Client with Authentication

\`\`\`python
@pytest.fixture
def api_client():
    """Unauthenticated API client"""
    return APIClient (base_url="http://localhost:8000")

@pytest.fixture
def authenticated_client (api_client, user):
    """API client with authentication"""
    token = generate_token (user)
    api_client.set_header("Authorization", f"Bearer {token}")
    return api_client
\`\`\`

### Pattern 3: Mocked External Services

\`\`\`python
@pytest.fixture
def mock_payment_gateway (monkeypatch):
    """Mock external payment gateway"""
    mock = Mock()
    mock.charge.return_value = {"status": "success", "id": "txn_123"}
    monkeypatch.setattr("myapp.services.payment_gateway", mock)
    return mock
\`\`\`

---

## Summary

**Fixtures are pytest's most powerful feature**:

- **Eliminate duplication**: Define setup once, use everywhere
- **Dependency injection**: Automatic parameter passing
- **Flexible scope**: Function, class, module, session
- **Composable**: Fixtures depend on other fixtures
- **Automatic cleanup**: Guaranteed with yield

**Key concepts**:
- Use \`@pytest.fixture\` decorator
- Request fixtures by parameter name
- Use \`yield\` for setup/teardown
- Choose appropriate scope for performance
- Compose small fixtures instead of large ones

**Impact**:
- 99% reduction in duplicate code
- Faster test execution with proper scope
- More maintainable test suites
- Easier to understand and debug tests

Master fixtures, and you'll write pytest tests that are **clean, fast, and maintainable**.
`,
};
