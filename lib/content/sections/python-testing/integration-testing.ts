export const integrationTesting = {
  title: 'Integration Testing',
  id: 'integration-testing',
  content: `
# Integration Testing

## Introduction

**Integration testing verifies that multiple components work together correctly**—databases with APIs, services with external dependencies, complete user workflows from request to response. While unit tests verify individual functions in isolation, integration tests catch the bugs that only appear when components interact.

Integration tests are slower and more complex than unit tests, but they're essential for confidence that your system works end-to-end. This section covers professional integration testing strategies, from testing API endpoints with real databases to orchestrating complex microservice interactions.

---

## Integration vs Unit Tests: Understanding the Difference

### The Testing Pyramid

\`\`\`
        /\\
       /  \\
      / E2E \\        Few (10s) - Slow, expensive, high confidence
     /-------\\
    /        \\
   / Integration\\   Medium (100s) - Moderate speed, good coverage
  /-------------\\
 /              \\
/   Unit Tests   \\  Many (1000s) - Fast, cheap, focused
------------------
\`\`\`

### Detailed Comparison

| Aspect | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| **Scope** | Single function/class | Multiple components together |
| **Dependencies** | Mocked/stubbed | Real (database, Redis, APIs) |
| **Speed** | Very fast (<10ms) | Slower (100ms-5s) |
| **Count** | Many (1000s) | Fewer (100s) |
| **Isolation** | Perfect (no side effects) | Shared state (database, files) |
| **Purpose** | Verify logic correctness | Verify component integration |
| **Confidence** | High for unit logic | High for system behavior |
| **Debugging** | Easy (small scope) | Harder (multiple components) |
| **CI Time** | Seconds | Minutes |

### When to Use Each

**Unit tests** for:
- Business logic
- Algorithms
- Data transformations
- Validation functions
- Pure functions

**Integration tests** for:
- API endpoints
- Database queries
- External API calls
- Authentication flows
- Payment processing
- Email sending
- File uploads

---

## Testing API Endpoints with Real Database

### Basic API Integration Test

\`\`\`python
# test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from myapp import create_app
from myapp.models import User

@pytest.fixture
def client (db_session):
    """Test client with real database"""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient (app)

def test_create_user_endpoint (client, db_session):
    """Test complete POST /users flow"""
    # Make API request
    response = client.post("/api/users", json={
        "username": "alice",
        "email": "alice@example.com",
        "age": 30
    })

    # Verify HTTP response
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert data["id"] is not None

    # Verify data persisted in database
    user = db_session.query(User).filter_by (username="alice").first()
    assert user is not None
    assert user.email == "alice@example.com"
    assert user.age == 30
\`\`\`

**What this tests**:
1. HTTP request handling
2. Request validation
3. Database insertion
4. Response serialization
5. Status code correctness

### Testing Error Cases

\`\`\`python
def test_create_user_duplicate_username (client):
    """Test duplicate username returns 409"""
    # Create first user
    client.post("/api/users", json={
        "username": "alice",
        "email": "alice@example.com"
    })

    # Try to create duplicate
    response = client.post("/api/users", json={
        "username": "alice",
        "email": "different@example.com"
    })

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()

def test_create_user_invalid_email (client):
    """Test invalid email returns 422"""
    response = client.post("/api/users", json={
        "username": "alice",
        "email": "not-an-email"
    })

    assert response.status_code == 422
    error = response.json()
    assert "email" in str (error).lower()

def test_create_user_missing_fields (client):
    """Test missing required fields returns 422"""
    response = client.post("/api/users", json={
        "username": "alice"
        # Missing email
    })

    assert response.status_code == 422
    error = response.json()
    assert any (e["loc"] == ["body", "email"] for e in error["detail"])
\`\`\`

### Testing Complete CRUD Operations

\`\`\`python
def test_user_crud_workflow (client, db_session):
    """Test complete Create, Read, Update, Delete workflow"""

    # CREATE
    create_response = client.post("/api/users", json={
        "username": "alice",
        "email": "alice@example.com",
        "age": 30
    })
    assert create_response.status_code == 201
    user_id = create_response.json()["id"]

    # READ (get specific)
    get_response = client.get (f"/api/users/{user_id}")
    assert get_response.status_code == 200
    assert get_response.json()["username"] == "alice"

    # READ (list all)
    list_response = client.get("/api/users")
    assert list_response.status_code == 200
    users = list_response.json()
    assert len (users) == 1
    assert users[0]["username"] == "alice"

    # UPDATE
    update_response = client.put (f"/api/users/{user_id}", json={
        "username": "alice_updated",
        "email": "alice_new@example.com",
        "age": 31
    })
    assert update_response.status_code == 200
    assert update_response.json()["username"] == "alice_updated"

    # Verify update in database
    user = db_session.query(User).get (user_id)
    assert user.username == "alice_updated"
    assert user.age == 31

    # DELETE
    delete_response = client.delete (f"/api/users/{user_id}")
    assert delete_response.status_code == 204

    # Verify deletion
    get_after_delete = client.get (f"/api/users/{user_id}")
    assert get_after_delete.status_code == 404

    # Verify in database
    user = db_session.query(User).get (user_id)
    assert user is None
\`\`\`

---

## Testing with Real Dependencies

### Docker Compose for Test Dependencies

\`\`\`yaml
# docker-compose.test.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: test_db
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: test_user
      RABBITMQ_DEFAULT_PASS: test_password
    ports:
      - "5673:5672"
      - "15673:15672"
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
\`\`\`

### Managing Docker Services in Tests

\`\`\`python
# conftest.py
import pytest
import subprocess
import time
import psycopg2
from psycopg2 import OperationalError

@pytest.fixture (scope="session", autouse=True)
def docker_services():
    """Start all Docker services for tests"""
    # Start containers
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"],
        check=True
    )

    # Wait for services to be healthy
    wait_for_postgres()
    wait_for_redis()
    wait_for_rabbitmq()

    yield

    # Cleanup
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.test.yml", "down", "-v"],
        check=True
    )

def wait_for_postgres (max_attempts=30):
    """Wait for PostgreSQL to be ready"""
    for attempt in range (max_attempts):
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                user="test_user",
                password="test_password",
                dbname="test_db"
            )
            conn.close()
            return
        except OperationalError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1)

def wait_for_redis (max_attempts=30):
    """Wait for Redis to be ready"""
    import redis
    for attempt in range (max_attempts):
        try:
            r = redis.Redis (host="localhost", port=6380)
            r.ping()
            return
        except redis.ConnectionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1)

def wait_for_rabbitmq (max_attempts=30):
    """Wait for RabbitMQ to be ready"""
    import pika
    for attempt in range (max_attempts):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host="localhost",
                    port=5673,
                    credentials=pika.PlainCredentials("test_user", "test_password")
                )
            )
            connection.close()
            return
        except pika.exceptions.AMQPConnectionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1)
\`\`\`

### Testing with Redis

\`\`\`python
import pytest
import redis

@pytest.fixture
def redis_client():
    """Redis client for testing"""
    client = redis.Redis (host="localhost", port=6380, decode_responses=True)
    yield client
    # Cleanup: flush test data
    client.flushdb()

def test_caching_integration (client, redis_client):
    """Test API caching with Redis"""
    # First request: Cache miss, hits database
    response1 = client.get("/api/users/1")
    assert response1.status_code == 200

    # Verify cached in Redis
    cached_data = redis_client.get("user:1")
    assert cached_data is not None

    # Second request: Cache hit, no database query
    response2 = client.get("/api/users/1")
    assert response2.status_code == 200
    assert response2.json() == response1.json()

    # Update user: Cache should be invalidated
    client.put("/api/users/1", json={"username": "updated"})

    # Verify cache cleared
    cached_after_update = redis_client.get("user:1")
    assert cached_after_update is None
\`\`\`

---

## Testing External APIs

### Using VCR.py for HTTP Recording

**Problem**: External APIs are slow, rate-limited, and unreliable in tests.

**Solution**: VCR.py records real API responses, replays them in tests.

\`\`\`bash
pip install vcrpy
\`\`\`

\`\`\`python
import pytest
import vcr
import requests

my_vcr = vcr.VCR(
    cassette_library_dir="tests/cassettes",
    record_mode="once",  # Record once, replay thereafter
)

@pytest.mark.integration
@my_vcr.use_cassette("github_user.yaml")
def test_fetch_github_user():
    """Test GitHub API integration with VCR"""
    response = requests.get("https://api.github.com/users/octocat")

    assert response.status_code == 200
    data = response.json()
    assert data["login"] == "octocat"
    assert data["public_repos"] > 0

    # First run: Makes real API call, records to cassette
    # Subsequent runs: Replays from cassette (instant, no API call)
\`\`\`

**Cassette file** (tests/cassettes/github_user.yaml):
\`\`\`yaml
interactions:
- request:
    uri: https://api.github.com/users/octocat
    method: GET
  response:
    status: {code: 200, message: OK}
    body: {string: '{"login":"octocat","public_repos":8,...}'}
\`\`\`

### Testing with Test Doubles (Test-Specific External Service)

\`\`\`python
@pytest.fixture
def mock_payment_api (mocker):
    """Mock external payment API"""
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "transaction_id": "txn_12345",
        "status": "success",
        "amount": 99.99
    }

    mocker.patch("requests.post", return_value=mock_response)
    return mock_response

def test_payment_processing (client, mock_payment_api):
    """Test payment processing with mocked external API"""
    response = client.post("/api/checkout", json={
        "amount": 99.99,
        "card_token": "tok_visa"
    })

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["transaction_id"] == "txn_12345"
\`\`\`

---

## End-to-End Workflow Testing

### Complete User Registration Flow

\`\`\`python
def test_user_registration_workflow (client, db_session, redis_client, mocker):
    """Test complete user registration and authentication flow"""

    # Mock email service
    mock_email = mocker.patch("myapp.email.send_email")

    # Step 1: Register new user
    registration_response = client.post("/api/register", json={
        "username": "alice",
        "email": "alice@example.com",
        "password": "SecurePass123!"
    })

    assert registration_response.status_code == 201
    user_data = registration_response.json()
    assert user_data["username"] == "alice"
    assert "password" not in user_data  # Password should be hidden

    # Verify user in database
    user = db_session.query(User).filter_by (username="alice").first()
    assert user is not None
    assert not user.email_confirmed  # Not confirmed yet

    # Verify confirmation email sent
    mock_email.assert_called_once()
    email_call = mock_email.call_args
    assert "alice@example.com" in str (email_call)
    assert "confirm" in str (email_call).lower()

    # Step 2: Confirm email
    # Extract confirmation token from Redis
    confirmation_token = redis_client.get (f"confirm:{user.id}")
    assert confirmation_token is not None

    confirm_response = client.get (f"/api/confirm-email?token={confirmation_token}")
    assert confirm_response.status_code == 200

    # Verify email confirmed in database
    db_session.refresh (user)
    assert user.email_confirmed

    # Verify token removed from Redis
    assert redis_client.get (f"confirm:{user.id}") is None

    # Step 3: Login
    login_response = client.post("/api/login", json={
        "username": "alice",
        "password": "SecurePass123!"
    })

    assert login_response.status_code == 200
    login_data = login_response.json()
    assert "access_token" in login_data
    assert "refresh_token" in login_data

    # Step 4: Access protected endpoint
    access_token = login_data["access_token"]
    protected_response = client.get(
        "/api/profile",
        headers={"Authorization": f"Bearer {access_token}"}
    )

    assert protected_response.status_code == 200
    profile_data = protected_response.json()
    assert profile_data["username"] == "alice"
    assert profile_data["email"] == "alice@example.com"
\`\`\`

### E-commerce Checkout Flow

\`\`\`python
def test_ecommerce_checkout_workflow (client, db_session, redis_client, factories):
    """Test complete e-commerce checkout flow"""

    # Setup: Create products
    product1 = factories["Product"].create (name="Widget", price=29.99, stock=10)
    product2 = factories["Product"].create (name="Gadget", price=49.99, stock=5)

    # Setup: Create and login user
    user = factories["User"].create (username="alice")
    login_response = client.post("/api/login", json={
        "username": "alice",
        "password": "password"
    })
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Add items to cart
    cart_response = client.post("/api/cart/add", headers=headers, json={
        "product_id": product1.id,
        "quantity": 2
    })
    assert cart_response.status_code == 200

    client.post("/api/cart/add", headers=headers, json={
        "product_id": product2.id,
        "quantity": 1
    })

    # Verify cart
    cart_view = client.get("/api/cart", headers=headers)
    assert len (cart_view.json()["items"]) == 2
    assert cart_view.json()["total"] == (29.99 * 2) + 49.99

    # Step 2: Apply discount code
    discount_response = client.post("/api/cart/discount", headers=headers, json={
        "code": "SAVE10"
    })
    assert discount_response.status_code == 200

    cart_with_discount = client.get("/api/cart", headers=headers)
    expected_total = ((29.99 * 2) + 49.99) * 0.9  # 10% off
    assert abs (cart_with_discount.json()["total"] - expected_total) < 0.01

    # Step 3: Checkout
    checkout_response = client.post("/api/checkout", headers=headers, json={
        "payment_method": "card",
        "card_token": "tok_visa",
        "shipping_address": {
            "street": "123 Main St",
            "city": "Springfield",
            "zip": "12345"
        }
    })

    assert checkout_response.status_code == 200
    order_data = checkout_response.json()
    assert order_data["status"] == "confirmed"
    assert order_data["total"] == expected_total

    # Verify order in database
    order = db_session.query(Order).get (order_data["order_id"])
    assert order is not None
    assert order.user_id == user.id
    assert len (order.items) == 2

    # Verify stock reduced
    db_session.refresh (product1)
    db_session.refresh (product2)
    assert product1.stock == 8  # 10 - 2
    assert product2.stock == 4  # 5 - 1

    # Verify cart cleared
    cart_after_checkout = client.get("/api/cart", headers=headers)
    assert len (cart_after_checkout.json()["items"]) == 0
\`\`\`

---

## Testing Microservices Communication

\`\`\`python
def test_microservices_communication (docker_services):
    """Test Service A → Service B communication"""
    from testcontainers.compose import DockerCompose

    with DockerCompose(".", compose_file_name="docker-compose.services.yml") as compose:
        # Wait for services
        compose.wait_for("http://localhost:8001/health")
        compose.wait_for("http://localhost:8002/health")

        # Test: Service A calls Service B
        response = requests.post("http://localhost:8001/api/process", json={
            "data": "test payload"
        })

        assert response.status_code == 200
        result = response.json()

        # Verify Service B processed request
        assert result["processed_by"] == "service_b"
        assert result["status"] == "success"

        # Verify in Service B's database
        service_b_db = connect_to_service_b_db()
        record = service_b_db.query(ProcessedRequest).first()
        assert record.data == "test payload"
\`\`\`

---

## Marking Integration Tests

### Using pytest Markers

\`\`\`python
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests (slow, uses real services)
    unit: marks tests as unit tests (fast, isolated)
    slow: marks tests as slow running tests
\`\`\`

\`\`\`python
import pytest

@pytest.mark.integration
def test_database_integration():
    """Integration test with real database"""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_external_api():
    """Slow integration test with external API"""
    pass
\`\`\`

### Running Tests Selectively

\`\`\`bash
# Run only unit tests (fast, for development)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run all except integration (for quick feedback)
pytest -m "not integration"

# Run slow tests
pytest -m slow

# Run integration but not slow
pytest -m "integration and not slow"
\`\`\`

---

## Best Practices

1. **Use real dependencies** (database, Redis) not mocks
2. **Docker for isolation** and reproducibility
3. **Test critical paths** (registration, checkout, payment)
4. **Mark as @pytest.mark.integration** for selective running
5. **Clean state** between tests (transaction rollback)
6. **VCR.py for external APIs** (fast, reliable)
7. **Test error scenarios** (network failures, timeouts)
8. **Separate unit and integration** (run unit frequently, integration before merge)

---

## Integration Test Patterns

| Pattern | When to Use | Example |
|---------|-------------|---------|
| **API + Database** | Testing endpoints | POST /users → verify in DB |
| **Service + Cache** | Testing caching logic | Redis cache invalidation |
| **Service + Queue** | Testing async tasks | RabbitMQ message processing |
| **Microservices** | Testing service communication | Service A → Service B |
| **External API** | Testing third-party integrations | Payment gateway, email service |
| **E2E Workflow** | Testing user journeys | Registration → login → checkout |

---

## Summary

**Integration testing essentials**:
- **Real dependencies**: Database, Redis, external services
- **Docker Compose**: Isolated, reproducible environments
- **E2E workflows**: Complete user journeys
- **Selective execution**: Run separately from unit tests
- **VCR.py**: Record/replay external API calls
- **Markers**: Organize and selectively run tests

Integration tests ensure **components work together in production-like environments**.
`,
};
