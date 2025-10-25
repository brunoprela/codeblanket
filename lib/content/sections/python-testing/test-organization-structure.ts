export const testOrganizationStructure = {
  title: 'Test Organization & Structure',
  id: 'test-organization-structure',
  content: `
# Test Organization & Structure

## Introduction

**Test organization is often overlooked, yet it's critical for maintaining a large test suite.** A well-organized test suite is easy to navigate, maintain, and extend. A poorly organized suite becomes technical debt that slows development.

Consider this: A payment service has 2,500 tests scattered across 80 files with no clear structure. A new developer needs 30 minutes to find where to add a test for a new feature. Tests are duplicated across files. When the database schema changes, 45 test files must be updated.

**Proper test organization saves hours per week and prevents test rot.**

### What You'll Learn

- Directory structures that scale from 10 to 10,000 tests
- Test file naming conventions and discovery
- Organizing tests by type (unit, integration, E2E)
- Using test classes effectively
- Shared test utilities and conftest.py
- Module and package structure
- Real-world patterns from production codebases

---

## Directory Structure Patterns

### Pattern 1: Flat Structure (Simple Projects)

**Best for**: Small projects (<1,000 tests, <10K lines of code)

\`\`\`
my_project/
├── src/
│   ├── calculator.py
│   ├── payment.py
│   └── user.py
└── tests/
    ├── test_calculator.py
    ├── test_payment.py
    └── test_user.py
\`\`\`

**Pros**:
- Simple: Everything in one directory
- Easy to navigate: All tests in one place
- No cognitive overhead: No need to decide where tests go

**Cons**:
- Doesn't scale: 50+ test files become unwieldy
- No separation: Unit and integration tests mixed
- Hard to run selectively: Can't easily run just unit tests

**When to use**: Prototypes, small libraries, learning projects

### Pattern 2: Mirrored Structure (Recommended for Most Projects)

**Best for**: Medium projects (1,000-10,000 tests, 10K-100K lines of code)

\`\`\`
my_project/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── business/
│   │   ├── __init__.py
│   │   ├── payment_processor.py
│   │   └── user_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── payment.py
│   │   └── user.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── tests/
    ├── api/
    │   ├── test_routes.py       # Mirrors src/api/routes.py
    │   └── test_schemas.py      # Mirrors src/api/schemas.py
    ├── business/
    │   ├── test_payment_processor.py
    │   └── test_user_service.py
    ├── models/
    │   ├── test_payment.py
    │   └── test_user.py
    ├── utils/
    │   └── test_helpers.py
    └── conftest.py              # Shared fixtures
\`\`\`

**Pros**:
- Intuitive: Test structure mirrors source structure
- Easy navigation: Know immediately where test file is
- Scalable: Grows naturally with source code
- Maintains context: Business logic tests grouped together

**Cons**:
- Still mixes test types: Unit and integration tests in same directory
- Can't easily run only integration tests

**When to use**: Most production applications, recommended default

### Pattern 3: Test Type Separation (Large Projects)

**Best for**: Large projects (10,000+ tests, 100K+ lines of code)

\`\`\`
my_project/
├── src/
│   └── (same as above)
└── tests/
    ├── unit/                    # Fast, isolated tests
    │   ├── api/
    │   │   ├── test_routes.py
    │   │   └── test_schemas.py
    │   ├── business/
    │   │   ├── test_payment_processor.py
    │   │   └── test_user_service.py
    │   └── models/
    │       ├── test_payment.py
    │       └── test_user.py
    ├── integration/             # Tests with external dependencies
    │   ├── test_database_operations.py
    │   ├── test_redis_cache.py
    │   └── test_api_endpoints.py
    ├── e2e/                     # End-to-end user workflows
    │   ├── test_user_registration_flow.py
    │   ├── test_payment_flow.py
    │   └── test_checkout_flow.py
    ├── performance/             # Performance and load tests
    │   └── test_api_performance.py
    ├── fixtures/                # Shared test data
    │   ├── users.json
    │   └── payments.json
    └── conftest.py              # Root-level fixtures
\`\`\`

**Pros**:
- Clear separation: Easy to run only unit tests (\`pytest tests/unit\`)
- Different speeds: Fast unit tests vs slow E2E tests
- CI/CD friendly: Run unit tests on every commit, E2E nightly
- Professional: Industry standard for large codebases

**Cons**:
- More directories: Slightly more complex navigation
- Duplicate structure: Need to mirror in unit/ and integration/

**When to use**: Large production applications, microservices, enterprise systems

**Run tests selectively**:

\`\`\`bash
# Run only unit tests (2 minutes)
pytest tests/unit

# Run only integration tests (10 minutes)
pytest tests/integration

# Run only E2E tests (30 minutes)
pytest tests/e2e

# Run unit and integration (12 minutes)
pytest tests/unit tests/integration

# CI: Run unit tests on every commit, E2E nightly
\`\`\`

---

## File Naming Conventions

pytest discovers tests based on naming patterns. **Consistency is critical**.

### Test File Names

**Convention**: \`test_<module_name>.py\` or \`<module_name>_test.py\`

✅ **Good** (recommended):
\`\`\`
test_calculator.py      # Tests for calculator.py
test_payment.py         # Tests for payment.py
test_user_service.py    # Tests for user_service.py
\`\`\`

✅ **Also valid** (less common):
\`\`\`
calculator_test.py
payment_test.py
user_service_test.py
\`\`\`

❌ **Bad** (won't be discovered):
\`\`\`
calculator.py           # Missing test_ prefix
tests_calculator.py     # Wrong prefix
calculator_tests.py     # Plural (inconsistent)
\`\`\`

**Recommendation**: Use \`test_\` prefix (more Pythonic, matches pytest convention)

### Test Function Names

**Convention**: \`test_<what_is_tested>\`

✅ **Good** (descriptive):
\`\`\`python
def test_add_positive_numbers():
    ...

def test_add_negative_numbers():
    ...

def test_divide_by_zero_raises_error():
    ...

def test_payment_processing_succeeds_with_valid_card():
    ...
\`\`\`

❌ **Bad** (unclear):
\`\`\`python
def test_add():           # What aspect of add?
    ...

def test_1():             # What is test 1?
    ...

def test_payment():       # Too vague
    ...
\`\`\`

**Guidelines**:
- Be specific: What condition/input is tested?
- Include expected behavior: "succeeds", "raises_error", "returns_none"
- Use underscores: \`test_user_can_login\` not \`testUserCanLogin\`
- Avoid abbreviations: \`test_user_authentication\` not \`test_usr_auth\`

### Test Class Names

**Convention**: \`Test<ClassName>\` (no \`__init__\` method)

✅ **Good**:
\`\`\`python
class TestCalculator:
    ...

class TestPaymentProcessor:
    ...

class TestUserAuthentication:
    ...
\`\`\`

❌ **Bad**:
\`\`\`python
class Calculator:         # Missing Test prefix
    ...

class TestCalculatorClass:  # Redundant "Class" suffix
    ...

class Test_Calculator:   # Underscore unnecessary
    ...
\`\`\`

---

## Organizing Tests with Classes

Classes help group related tests and share setup logic.

### Example: Grouping Related Tests

\`\`\`python
"""Test payment processing module"""
import pytest
from payment import PaymentProcessor, Payment


class TestPaymentCreation:
    """Tests for creating Payment objects"""
    
    def test_create_payment_with_valid_data(self):
        """Should create payment with valid data"""
        payment = Payment(amount=100.0, currency="USD")
        assert payment.amount == 100.0
        assert payment.currency == "USD"
    
    def test_create_payment_with_negative_amount(self):
        """Should raise error for negative amount"""
        with pytest.raises(ValueError):
            Payment(amount=-100.0, currency="USD")
    
    def test_create_payment_with_zero_amount(self):
        """Should raise error for zero amount"""
        with pytest.raises(ValueError):
            Payment(amount=0.0, currency="USD")


class TestPaymentProcessing:
    """Tests for processing payments"""
    
    def test_process_payment_success(self):
        """Should process valid payment successfully"""
        processor = PaymentProcessor(api_key="test")
        payment = Payment(amount=100.0, currency="USD")
        result = processor.process_payment(payment)
        assert result is True
    
    def test_process_payment_without_api_key(self):
        """Should fail without API key"""
        processor = PaymentProcessor(api_key="")
        payment = Payment(amount=100.0, currency="USD")
        with pytest.raises(ValueError):
            processor.process_payment(payment)
    
    def test_process_payment_already_processed(self):
        """Should not process payment twice"""
        processor = PaymentProcessor(api_key="test")
        payment = Payment(amount=100.0, currency="USD")
        payment.status = "completed"
        with pytest.raises(ValueError):
            processor.process_payment(payment)


class TestPaymentRefunds:
    """Tests for refunding payments"""
    
    def test_refund_completed_payment(self):
        """Should refund completed payment"""
        processor = PaymentProcessor(api_key="test")
        payment = Payment(amount=100.0, currency="USD")
        processor.process_payment(payment)
        result = processor.refund_payment(payment)
        assert result is True
    
    def test_refund_pending_payment(self):
        """Should not refund pending payment"""
        processor = PaymentProcessor(api_key="test")
        payment = Payment(amount=100.0, currency="USD")
        with pytest.raises(ValueError):
            processor.refund_payment(payment)
\`\`\`

### Benefits of Class-Based Organization

1. **Logical Grouping**: Related tests together (creation, processing, refunds)
2. **Clear Hierarchy**: Test report shows class structure
3. **Shared Setup**: Use fixtures at class level
4. **Better Navigation**: Jump to "TestPaymentProcessing" class
5. **Documentation**: Class docstrings explain test group purpose

### When to Use Classes vs Functions

**Use classes when**:
- Testing a class/module with multiple related methods
- Need shared setup for multiple tests
- Want to group tests logically for reporting
- Have >5 tests for same functionality

**Use functions when**:
- Testing simple functions
- Tests are independent with no shared setup
- Simplicity is preferred over grouping
- <5 tests for functionality

---

## conftest.py: Shared Fixtures and Utilities

\`conftest.py\` is pytest's mechanism for sharing fixtures and utilities across multiple test files.

### How conftest.py Works

**Discovery**:
- pytest automatically discovers \`conftest.py\` files
- No imports needed—fixtures are automatically available
- Scope: Fixtures in \`conftest.py\` available to all tests in same directory and subdirectories

**Multiple conftest.py files**:
- Can have multiple at different levels
- Root \`conftest.py\`: Available to all tests
- \`tests/unit/conftest.py\`: Available only to unit tests
- Fixtures can override: Lower-level overrides higher-level

### Root conftest.py Example

\`\`\`python
"""Root conftest.py - shared fixtures for all tests"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from redis import Redis
from myapp import create_app
from myapp.database import Base


@pytest.fixture(scope="session")
def db_engine():
    """Database engine for testing (created once per session)"""
    engine = create_engine("postgresql://test:test@localhost/test_db")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup after all tests
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Clean database session for each test"""
    connection = db_engine.connect()
    transaction = connection.begin()
    
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    # Rollback transaction (clean database)
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="module")
def redis_client():
    """Redis client for testing"""
    client = Redis(host="localhost", port=6379, db=15)
    
    yield client
    
    # Cleanup
    client.flushdb()
    client.close()


@pytest.fixture
def api_client():
    """Flask/FastAPI test client"""
    app = create_app(config="testing")
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def authenticated_api_client(api_client, db_session):
    """API client with authentication"""
    # Create test user
    user = User(username="testuser", email="test@example.com")
    user.set_password("password")
    db_session.add(user)
    db_session.commit()
    
    # Login
    response = api_client.post("/auth/login", json={
        "username": "testuser",
        "password": "password"
    })
    token = response.json["token"]
    
    # Set authorization header
    api_client.headers["Authorization"] = f"Bearer {token}"
    
    return api_client
\`\`\`

### Using Fixtures from conftest.py

**No imports needed**—fixtures are automatically available:

\`\`\`python
"""tests/api/test_users.py"""

def test_create_user(db_session, api_client):
    """Test creating user via API"""
    # db_session and api_client from conftest.py (no import needed)
    
    response = api_client.post("/users", json={
        "username": "newuser",
        "email": "new@example.com"
    })
    
    assert response.status_code == 201
    
    # Verify in database
    user = db_session.query(User).filter_by(username="newuser").first()
    assert user is not None
    assert user.email == "new@example.com"


def test_get_user_profile(authenticated_api_client):
    """Test getting user profile (requires authentication)"""
    # authenticated_api_client from conftest.py
    
    response = authenticated_api_client.get("/users/me")
    
    assert response.status_code == 200
    assert response.json["username"] == "testuser"
\`\`\`

### Multiple conftest.py Files

\`\`\`
tests/
├── conftest.py                 # Root (db_session, api_client)
├── unit/
│   ├── conftest.py             # Unit-specific fixtures
│   └── test_calculator.py
└── integration/
    ├── conftest.py             # Integration-specific fixtures
    └── test_api_integration.py
\`\`\`

**Root conftest.py** (shared by all tests):
\`\`\`python
import pytest

@pytest.fixture
def db_session():
    """Database session available to all tests"""
    ...
\`\`\`

**tests/unit/conftest.py** (only for unit tests):
\`\`\`python
import pytest

@pytest.fixture
def mock_db():
    """Mocked database for fast unit tests"""
    return MockDatabase()
\`\`\`

**tests/integration/conftest.py** (only for integration tests):
\`\`\`python
import pytest
import docker

@pytest.fixture(scope="module")
def docker_services():
    """Start Docker containers for integration tests"""
    client = docker.from_env()
    
    # Start PostgreSQL container
    db_container = client.containers.run(
        "postgres:15",
        environment={"POSTGRES_PASSWORD": "test"},
        ports={"5432/tcp": 5432},
        detach=True
    )
    
    yield
    
    # Stop container after tests
    db_container.stop()
    db_container.remove()
\`\`\`

---

## Shared Test Utilities

Beyond fixtures, create shared test utilities for common operations.

### utilities/assertions.py

\`\`\`python
"""Custom assertions for tests"""

def assert_payment_valid(payment):
    """Assert payment has valid structure"""
    assert payment.id is not None
    assert payment.amount > 0
    assert payment.currency in ["USD", "EUR", "GBP"]
    assert payment.status in ["pending", "completed", "failed"]


def assert_response_success(response):
    """Assert API response is successful"""
    assert response.status_code in [200, 201]
    assert "error" not in response.json


def assert_user_authenticated(user):
    """Assert user is properly authenticated"""
    assert user is not None
    assert user.is_authenticated
    assert user.token is not None
\`\`\`

### utilities/factories.py

\`\`\`python
"""Test data factories using FactoryBoy"""
import factory
from myapp.models import User, Payment


class UserFactory(factory.Factory):
    """Create test users"""
    
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    age = factory.Faker("random_int", min=18, max=80)


class PaymentFactory(factory.Factory):
    """Create test payments"""
    
    class Meta:
        model = Payment
    
    amount = factory.Faker("pyfloat", min_value=10, max_value=1000)
    currency = factory.Iterator(["USD", "EUR", "GBP"])
    status = "pending"
\`\`\`

### utilities/helpers.py

\`\`\`python
"""Test helper functions"""
import random
import string

def generate_random_email():
    """Generate random email for testing"""
    username = ''.join(random.choices(string.ascii_lowercase, k=10))
    return f"{username}@example.com"


def create_test_user(db_session, **kwargs):
    """Create test user with defaults"""
    defaults = {
        "username": "testuser",
        "email": "test@example.com",
        "age": 25
    }
    defaults.update(kwargs)
    
    user = User(**defaults)
    db_session.add(user)
    db_session.commit()
    
    return user


def wait_for_async_task(task_id, timeout=10):
    """Wait for async task to complete"""
    import time
    start = time.time()
    
    while time.time() - start < timeout:
        task = get_task_status(task_id)
        if task.status in ["completed", "failed"]:
            return task
        time.sleep(0.1)
    
    raise TimeoutError(f"Task {task_id} did not complete in {timeout}s")
\`\`\`

### Using Shared Utilities

\`\`\`python
"""tests/api/test_users.py"""
from tests.utilities.assertions import assert_user_authenticated
from tests.utilities.factories import UserFactory
from tests.utilities.helpers import generate_random_email


def test_user_creation(db_session):
    """Test creating user"""
    user = UserFactory.create(
        username="alice",
        email=generate_random_email()
    )
    
    db_session.add(user)
    db_session.commit()
    
    assert user.id is not None


def test_user_authentication(db_session, api_client):
    """Test user authentication"""
    user = UserFactory.create()
    db_session.add(user)
    db_session.commit()
    
    response = api_client.post("/auth/login", json={
        "username": user.username,
        "password": "password"
    })
    
    assert_user_authenticated(response.json["user"])
\`\`\`

---

## Complete Example: Large Project Structure

\`\`\`
payment_platform/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── users.py
│   │   │   └── payments.py
│   │   └── schemas/
│   │       ├── user.py
│   │       └── payment.py
│   ├── business/
│   │   ├── payment_processor.py
│   │   └── user_service.py
│   ├── models/
│   │   ├── user.py
│   │   └── payment.py
│   └── utils/
│       └── validators.py
└── tests/
    ├── conftest.py                    # Root fixtures (db, api client)
    ├── unit/
    │   ├── conftest.py                # Mock fixtures for unit tests
    │   ├── api/
    │   │   ├── routes/
    │   │   │   ├── test_users.py
    │   │   │   └── test_payments.py
    │   │   └── schemas/
    │   │       ├── test_user.py
    │   │       └── test_payment.py
    │   ├── business/
    │   │   ├── test_payment_processor.py
    │   │   └── test_user_service.py
    │   └── models/
    │       ├── test_user.py
    │       └── test_payment.py
    ├── integration/
    │   ├── conftest.py                # Docker/database fixtures
    │   ├── test_payment_flow.py
    │   ├── test_user_registration.py
    │   └── test_database_operations.py
    ├── e2e/
    │   ├── conftest.py                # Full system fixtures
    │   ├── test_checkout_flow.py
    │   └── test_user_journey.py
    ├── performance/
    │   └── test_api_load.py
    └── utilities/
        ├── __init__.py
        ├── assertions.py              # Custom assertions
        ├── factories.py               # FactoryBoy factories
        └── helpers.py                 # Helper functions
\`\`\`

---

## Best Practices

### 1. Keep Related Tests Together

✅ **Good**:
\`\`\`python
class TestPaymentProcessing:
    def test_process_success(self):
        ...
    
    def test_process_insufficient_funds(self):
        ...
    
    def test_process_invalid_card(self):
        ...
\`\`\`

❌ **Bad**: Tests scattered across multiple files

### 2. Use Descriptive Names

✅ **Good**: \`test_payment_processing_fails_with_expired_card\`  
❌ **Bad**: \`test_payment_1\`

### 3. Mirror Source Structure

If \`src/payment/processor.py\` exists, create \`tests/unit/payment/test_processor.py\`

### 4. Separate Test Types

- **Unit tests**: \`tests/unit/\` (fast, no external deps)
- **Integration tests**: \`tests/integration/\` (databases, APIs)
- **E2E tests**: \`tests/e2e/\` (full user workflows)

### 5. Use conftest.py for Shared Fixtures

Don't duplicate database/API client setup across 50 files—put in \`conftest.py\` once.

### 6. Create Shared Utilities

Common patterns (assertions, factories, helpers) → \`tests/utilities/\`

### 7. Document Test Structure

Add \`tests/README.md\`:

\`\`\`markdown
# Test Organization

## Structure

- \`unit/\`: Fast unit tests (run on every commit)
- \`integration/\`: Integration tests (run before merge)
- \`e2e/\`: End-to-end tests (run nightly)

## Running Tests

\`\`\`bash
# All tests
pytest

# Only unit tests (fast)
pytest tests/unit

# Only integration tests
pytest tests/integration
\`\`\`

## Adding New Tests

1. Determine test type (unit/integration/e2e)
2. Mirror source structure (\`src/api/routes.py\` → \`tests/unit/api/test_routes.py\`)
3. Use fixtures from \`conftest.py\`
4. Follow naming convention: \`test_<what>_<condition>_<expected>\`
\`\`\`

---

## Summary

**Test organization impacts maintainability**:
- **Flat structure**: Simple, but doesn't scale beyond 1,000 tests
- **Mirrored structure**: Recommended for most projects, intuitive navigation
- **Type separation**: Industry standard for large projects, enables selective test runs

**Naming conventions**:
- Files: \`test_<module>.py\`
- Functions: \`test_<what>_<condition>_<expected>\`
- Classes: \`Test<ClassName>\`

**conftest.py**:
- Share fixtures across multiple test files
- No imports needed (automatic discovery)
- Multiple conftest.py files at different levels

**Shared utilities**:
- \`utilities/assertions.py\`: Custom assertions
- \`utilities/factories.py\`: Test data factories
- \`utilities/helpers.py\`: Helper functions

**Real-world impact**:
- Well-organized: Find test location in 5 seconds
- Poorly organized: Spend 30 minutes searching
- Difference: 5 hours saved per developer per month

Invest time in organization upfront—pays dividends as test suite grows to thousands of tests.
`,
};
