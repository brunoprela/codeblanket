export const testingFastapi = {
  title: 'Testing FastAPI Applications',
  id: 'testing-fastapi',
  content: `
# Testing FastAPI Applications

## Introduction

Testing ensures your API works correctly and prevents regressions. FastAPI's TestClient enables comprehensive testing without running a server, making tests fast, reliable, and easy to write. Production APIs need thorough test coverage: unit tests, integration tests, and end-to-end tests.

**Why testing matters:**
- **Prevent regressions**: Catch bugs before production
- **Refactoring confidence**: Change code without fear
- **Documentation**: Tests show how APIs should be used
- **Quality assurance**: Ensure business logic works correctly
- **CI/CD integration**: Automated testing in pipelines

**Testing challenges in APIs:**
- Database state management
- Authentication and authorization
- External service dependencies
- Asynchronous operations
- WebSocket connections
- Background tasks

In this section, you'll master:
- TestClient for endpoint testing
- Pytest fixtures and setup
- Dependency overrides for mocking
- Database testing strategies
- Authentication testing patterns
- Mocking external services
- Coverage and reporting
- Integration vs unit tests
- Production testing patterns

---

## TestClient Fundamentals

### Basic Testing

\`\`\`python
"""
Testing FastAPI endpoints with TestClient
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

# Test client
client = TestClient(app)

def test_read_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_read_item():
    """Test parameterized endpoint"""
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42}

def test_invalid_item_id():
    """Test validation error"""
    response = client.get("/items/invalid")
    assert response.status_code == 422  # Validation error
\`\`\`

### Testing POST Requests

\`\`\`python
"""
Testing request bodies and validation
"""

from pydantic import BaseModel

class User(BaseModel):
    email: str
    username: str
    age: int

@app.post("/users/", status_code=201)
async def create_user(user: User):
    return {"id": 1, **user.dict()}

def test_create_user():
    """Test user creation"""
    response = client.post(
        "/users/",
        json={
            "email": "test@example.com",
            "username": "testuser",
            "age": 25
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["id"] == 1

def test_create_user_validation_error():
    """Test validation fails for invalid data"""
    response = client.post(
        "/users/",
        json={
            "email": "not-an-email",  # Invalid email
            "username": "te",  # Too short
            "age": -1  # Invalid age
        }
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert len(errors) > 0
\`\`\`

---

## Pytest Fixtures

### Basic Fixtures

\`\`\`python
"""
Reusable test setup with pytest fixtures
"""

import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Fixture providing test client"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sample_user():
    """Fixture providing sample user data"""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "age": 25
    }

def test_with_fixtures(client, sample_user):
    """Test using fixtures"""
    response = client.post("/users/", json=sample_user)
    assert response.status_code == 201
\`\`\`

### Database Fixtures

\`\`\`python
"""
Database setup/teardown with fixtures
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# In-memory SQLite for fast tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

@pytest.fixture
def db():
    """
    Fixture providing database session
    Creates tables before test, drops after
    """
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        # Drop tables
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client_with_db(db):
    """
    Test client with database dependency override
    """
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear overrides
    app.dependency_overrides.clear()

def test_create_user_in_db(client_with_db, sample_user):
    """Test user creation with real database"""
    response = client_with_db.post("/users/", json=sample_user)
    assert response.status_code == 201
    
    # Verify user in database
    user_id = response.json()["id"]
    response = client_with_db.get(f"/users/{user_id}")
    assert response.status_code == 200
\`\`\`

---

## Dependency Overrides

### Mocking Authentication

\`\`\`python
"""
Override authentication dependency for testing
"""

from fastapi import Depends, HTTPException

# Production dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate token and return user"""
    user = decode_token(token)
    if not user:
        raise HTTPException(status_code=401)
    return user

# Test override
def get_current_user_override():
    """Return test user without authentication"""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com"
    }

@pytest.fixture
def authenticated_client():
    """Client with authentication bypassed"""
    app.dependency_overrides[get_current_user] = get_current_user_override
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()

def test_protected_endpoint(authenticated_client):
    """Test protected endpoint without real auth"""
    response = authenticated_client.get("/protected")
    assert response.status_code == 200
    assert "data" in response.json()
\`\`\`

### Mocking External Services

\`\`\`python
"""
Mock external API calls for testing
"""

from unittest.mock import Mock, patch
import httpx

@app.get("/external-data")
async def get_external_data():
    """Fetch data from external API"""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.external.com/data")
        return response.json()

@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_external_data(mock_get):
    """Test with mocked external API"""
    # Mock response
    mock_response = Mock()
    mock_response.json.return_value = {"data": "mocked"}
    mock_get.return_value = mock_response
    
    # Test endpoint
    response = client.get("/external-data")
    assert response.status_code == 200
    assert response.json() == {"data": "mocked"}
    
    # Verify mock called
    mock_get.assert_called_once()
\`\`\`

---

## Database Testing Strategies

### Strategy 1: In-Memory SQLite

\`\`\`python
"""
Fast tests with in-memory database
"""

# Pros:
# - Very fast (no disk I/O)
# - Clean state for each test
# - No external dependencies

# Cons:
# - SQLite != PostgreSQL (different SQL dialects)
# - Missing some PostgreSQL features

@pytest.fixture(scope="function")
def db_session():
    """Fresh database for each test"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()
\`\`\`

### Strategy 2: Test Database

\`\`\`python
"""
Use separate test database (PostgreSQL)
"""

# Pros:
# - Same database engine as production
# - Test PostgreSQL-specific features
# - More realistic tests

# Cons:
# - Slower than in-memory
# - Requires database setup

import pytest
from sqlalchemy import create_engine

TEST_DATABASE_URL = "postgresql://test:test@localhost/test_db"

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    
    yield engine
    
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(test_engine):
    """Transaction-wrapped session"""
    connection = test_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()  # Rollback after each test
    connection.close()
\`\`\`

### Strategy 3: Docker Test Database

\`\`\`python
"""
Spin up PostgreSQL in Docker for tests
"""

import pytest
import docker
import time

@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for tests"""
    client = docker.from_env()
    
    container = client.containers.run(
        "postgres:15",
        environment={
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_DB": "test_db"
        },
        ports={"5432/tcp": 5433},
        detach=True,
        remove=True
    )
    
    # Wait for PostgreSQL to be ready
    time.sleep(3)
    
    yield container
    
    container.stop()
\`\`\`

---

## Testing Authentication

### JWT Testing

\`\`\`python
"""
Test JWT authentication flow
"""

from jose import jwt
from datetime import datetime, timedelta

def create_test_token(user_id: int = 1):
    """Create valid JWT token for testing"""
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def test_login():
    """Test login endpoint"""
    response = client.post(
        "/auth/login",
        json={"username": "testuser", "password": "testpass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_protected_with_valid_token():
    """Test protected endpoint with valid token"""
    token = create_test_token()
    
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

def test_protected_without_token():
    """Test protected endpoint without token"""
    response = client.get("/protected")
    assert response.status_code == 401

def test_protected_with_expired_token():
    """Test with expired token"""
    # Create expired token
    payload = {
        "sub": "1",
        "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
\`\`\`

---

## Integration Testing

### Full Flow Testing

\`\`\`python
"""
Test complete user workflows
"""

def test_user_registration_flow(client_with_db):
    """Test complete registration → login → access flow"""
    
    # 1. Register user
    register_data = {
        "email": "newuser@example.com",
        "username": "newuser",
        "password": "SecurePass123!"
    }
    response = client_with_db.post("/auth/register", json=register_data)
    assert response.status_code == 201
    user_id = response.json()["id"]
    
    # 2. Login
    login_data = {
        "username": "newuser",
        "password": "SecurePass123!"
    }
    response = client_with_db.post("/auth/login", json=login_data)
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # 3. Access protected resource
    headers = {"Authorization": f"Bearer {token}"}
    response = client_with_db.get("/users/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == user_id

def test_order_creation_flow(authenticated_client, db):
    """Test order creation with inventory check"""
    
    # Setup: Create product with inventory
    product = Product(id=1, name="Test Product", stock=10)
    db.add(product)
    db.commit()
    
    # 1. Create order
    order_data = {
        "items": [{"product_id": 1, "quantity": 3}]
    }
    response = authenticated_client.post("/orders/", json=order_data)
    assert response.status_code == 201
    order_id = response.json()["id"]
    
    # 2. Verify inventory reduced
    product = db.query(Product).filter(Product.id == 1).first()
    assert product.stock == 7
    
    # 3. Verify order status
    response = authenticated_client.get(f"/orders/{order_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
\`\`\`

---

## Testing WebSockets

### WebSocket Testing

\`\`\`python
"""
Test WebSocket connections
"""

def test_websocket():
    """Test WebSocket endpoint"""
    with client.websocket_connect("/ws") as websocket:
        # Send message
        websocket.send_json({"message": "Hello"})
        
        # Receive response
        data = websocket.receive_json()
        assert data["message"] == "Hello"

def test_websocket_authentication():
    """Test WebSocket with authentication"""
    token = create_test_token()
    
    with client.websocket_connect(f"/ws?token={token}") as websocket:
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"
\`\`\`

---

## Test Coverage

### Measuring Coverage

\`\`\`bash
# Install coverage
pip install pytest-cov

# Run tests with coverage
pytest tests/ --cov=app --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
\`\`\`

### Coverage Configuration

\`\`\`ini
# .coveragerc
[run]
source = app
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
\`\`\`

---

## Parametrized Tests

### Testing Multiple Scenarios

\`\`\`python
"""
Test same logic with different inputs
"""

import pytest

@pytest.mark.parametrize("email,expected_status", [
    ("valid@example.com", 201),
    ("invalid-email", 422),
    ("", 422),
    ("test@", 422),
])
def test_user_creation_validation(client, email, expected_status):
    """Test email validation with various inputs"""
    response = client.post(
        "/users/",
        json={"email": email, "username": "testuser", "age": 25}
    )
    assert response.status_code == expected_status

@pytest.mark.parametrize("age,valid", [
    (18, True),
    (100, True),
    (17, False),  # Too young
    (151, False),  # Too old
    (-1, False),  # Negative
])
def test_age_validation(client, age, valid):
    """Test age validation"""
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "username": "test", "age": age}
    )
    
    if valid:
        assert response.status_code == 201
    else:
        assert response.status_code == 422
\`\`\`

---

## Async Testing

### Testing Async Endpoints

\`\`\`python
"""
Test async operations
"""

import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_operation():
    """Test async endpoint"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/async-endpoint")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test multiple concurrent requests"""
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        tasks = [
            client.get(f"/items/{i}")
            for i in range(10)
        ]
        responses = await asyncio.gather(*tasks)
        
        assert all(r.status_code == 200 for r in responses)
\`\`\`

---

## Production Testing Patterns

### Test Organization

\`\`\`
tests/
├── conftest.py          # Shared fixtures
├── unit/
│   ├── test_models.py
│   ├── test_schemas.py
│   └── test_utils.py
├── integration/
│   ├── test_auth.py
│   ├── test_users.py
│   └── test_orders.py
└── e2e/
    └── test_workflows.py
\`\`\`

### CI/CD Integration

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
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
\`\`\`

---

## Summary

✅ **TestClient**: Test endpoints without running server  
✅ **Pytest fixtures**: Reusable setup/teardown  
✅ **Dependency overrides**: Mock authentication, database, services  
✅ **Database strategies**: In-memory, test DB, Docker  
✅ **Authentication testing**: JWT validation, protected endpoints  
✅ **Integration tests**: Complete user workflows  
✅ **WebSocket testing**: Test real-time connections  
✅ **Coverage**: Measure and improve test coverage  
✅ **Parametrized tests**: Test multiple scenarios efficiently  
✅ **Async testing**: Test async endpoints and operations  

### Best Practices

**1. Test structure**: Unit → Integration → E2E
**2. Fast tests**: Use in-memory database when possible
**3. Isolation**: Each test should be independent
**4. Coverage**: Aim for 80%+ coverage
**5. CI/CD**: Run tests automatically on every commit
**6. Mocking**: Mock external services, not business logic
**7. Fixtures**: Share setup code, keep tests DRY

### Next Steps

In the next section, we'll explore **Async FastAPI Patterns**: mastering asynchronous programming for high-performance APIs.
`,
};
