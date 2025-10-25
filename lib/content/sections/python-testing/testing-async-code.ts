export const testingAsyncCode = {
  title: 'Testing Async Code',
  id: 'testing-async-code',
  content: `
# Testing Async Code

## Introduction

**Async Python (async/await) has transformed how we build high-performance applications**—from FastAPI web services to asyncpg database clients to aiohttp API calls. But testing async code requires special approaches. Testing async code incorrectly leads to false positives (tests pass but code is broken), deadlocks that hang CI/CD, and flaky tests that pass locally but fail in production.

This section covers professional async testing with pytest-asyncio, from basic async functions to complex scenarios like concurrent operations, timeouts, and async context managers.

---

## Understanding Async Testing Challenges

### Why Async Code Needs Special Testing

**Problem 1**: Regular pytest doesn't await async functions
\`\`\`python
# This doesn't work!
def test_async_function():
    result = fetch_data_async()
    assert result == "data"  # result is coroutine object, not "data"
\`\`\`

**Problem 2**: Event loop management
\`\`\`python
# Manually managing event loop is error-prone
def test_with_manual_loop():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(fetch_data_async())
    # Forgot to close loop—affects other tests
\`\`\`

**Problem 3**: Async mocking requires special mocks
\`\`\`python
# Regular Mock doesn't work for async
mock = Mock(return_value="data")
result = await async_function(mock)  # TypeError: object Mock can't be used in await
\`\`\`

**Solution**: pytest-asyncio handles event loops, awaiting, and provides async-aware fixtures.

---

## Setup: pytest-asyncio

### Installation

\`\`\`bash
pip install pytest-asyncio
\`\`\`

### Configuration

\`\`\`ini
# pytest.ini
[pytest]
asyncio_mode = auto  # Auto-detect async tests
\`\`\`

**Modes**:
- \`auto\`: Automatically detect async tests (recommended)
- \`strict\`: Require explicit \`@pytest.mark.asyncio\` (more control)
- \`legacy\`: Backward compatibility

---

## Testing Async Functions

### Basic Async Test

\`\`\`python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test basic async function"""
    result = await fetch_data_async()
    assert result == "data"
\`\`\`

**What happens**:
1. pytest-asyncio detects \`@pytest.mark.asyncio\`
2. Creates event loop
3. Runs test coroutine in event loop
4. Closes loop after test
5. Clean state for next test

### Multiple Awaits

\`\`\`python
@pytest.mark.asyncio
async def test_multiple_awaits():
    """Test function with multiple async calls"""
    # Sequential awaits
    data1 = await fetch_data(1)
    data2 = await fetch_data(2)
    data3 = await fetch_data(3)
    
    assert data1 != data2 != data3
    assert all(data is not None for data in [data1, data2, data3])
\`\`\`

### Testing Error Handling

\`\`\`python
@pytest.mark.asyncio
async def test_async_exception():
    """Test async function raises exception"""
    with pytest.raises(ValueError, match="Invalid input"):
        await process_data_async(invalid_input)

@pytest.mark.asyncio
async def test_async_timeout_handling():
    """Test function handles timeout gracefully"""
    import asyncio
    
    async def slow_operation():
        await asyncio.sleep(10)
        return "done"
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=1.0)
\`\`\`

---

## Async Fixtures

### Basic Async Fixture

\`\`\`python
import pytest
import aiohttp

@pytest.fixture
async def async_client():
    """Async fixture providing aiohttp client"""
    async with aiohttp.ClientSession() as session:
        yield session
    # Cleanup happens automatically after yield

@pytest.mark.asyncio
async def test_with_async_fixture(async_client):
    """Use async fixture in test"""
    async with async_client.get("https://api.example.com/data") as resp:
        data = await resp.json()
        assert data["status"] == "ok"
\`\`\`

### Session-Scoped Async Fixture

\`\`\`python
@pytest.fixture(scope="session")
async def async_db_engine():
    """Session-scoped async database engine"""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    engine = create_async_engine("postgresql+asyncpg://test:test@localhost/testdb")
    
    # Setup: Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Teardown: Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def async_db_session(async_db_engine):
    """Function-scoped async session with rollback"""
    from sqlalchemy.ext.asyncio import AsyncSession
    
    async with async_db_engine.connect() as connection:
        async with connection.begin() as transaction:
            session = AsyncSession(bind=connection)
            
            yield session
            
            await session.close()
            await transaction.rollback()
\`\`\`

### Composing Async Fixtures

\`\`\`python
@pytest.fixture
async def async_redis():
    """Async Redis connection"""
    import aioredis
    
    redis = await aioredis.create_redis_pool("redis://localhost")
    yield redis
    redis.close()
    await redis.wait_closed()

@pytest.fixture
async def async_cache(async_redis):
    """Cache service using async Redis"""
    from myapp.cache import Cache
    
    cache = Cache(async_redis)
    await cache.clear()  # Start with clean cache
    
    yield cache
    
    await cache.clear()  # Cleanup

@pytest.mark.asyncio
async def test_with_composed_fixtures(async_cache):
    """Test using composed async fixtures"""
    await async_cache.set("key", "value")
    result = await async_cache.get("key")
    assert result == "value"
\`\`\`

---

## Mocking Async Code

### AsyncMock Basics

\`\`\`python
from unittest.mock import AsyncMock
import pytest

@pytest.mark.asyncio
async def test_with_async_mock():
    """Mock async function"""
    # Create AsyncMock
    mock_fetch = AsyncMock()
    mock_fetch.return_value = {"id": 1, "name": "Alice"}
    
    # Use in test
    result = await mock_fetch()
    
    assert result["name"] == "Alice"
    mock_fetch.assert_called_once()
\`\`\`

### Mocking with pytest-mock

\`\`\`python
@pytest.mark.asyncio
async def test_with_mocker(mocker):
    """Use pytest-mock's mocker fixture"""
    # Patch async function
    mock_fetch = mocker.patch("myapp.services.fetch_data", new_callable=AsyncMock)
    mock_fetch.return_value = {"status": "success", "data": [1, 2, 3]}
    
    # Call function that uses fetch_data
    result = await process_user_data()
    
    assert result["status"] == "success"
    assert len(result["data"]) == 3
    mock_fetch.assert_called_once()

@pytest.mark.asyncio
async def test_mock_side_effect(mocker):
    """Mock with side effect (different return per call)"""
    mock_api = mocker.patch("myapp.api.call_external_api", new_callable=AsyncMock)
    mock_api.side_effect = [
        {"page": 1, "items": [1, 2, 3]},
        {"page": 2, "items": [4, 5, 6]},
        {"page": 3, "items": []},  # Last page
    ]
    
    # Function paginating through API
    all_items = await fetch_all_pages()
    
    assert len(all_items) == 6
    assert mock_api.call_count == 3
\`\`\`

### Mocking Async Context Managers

\`\`\`python
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_mock_async_context_manager(mocker):
    """Mock async context manager"""
    # Create mock that supports async with
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = AsyncMock()
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={"data": "test"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    mock_session.get.return_value = mock_response
    
    # Patch ClientSession
    mocker.patch("aiohttp.ClientSession", return_value=mock_session)
    
    # Test code using aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as resp:
            data = await resp.json()
            assert data["data"] == "test"
\`\`\`

---

## Testing Concurrent Operations

### Testing Parallel Execution with asyncio.gather

\`\`\`python
@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test multiple concurrent async operations"""
    import asyncio
    
    # Execute 10 requests concurrently
    results = await asyncio.gather(
        fetch_data(1),
        fetch_data(2),
        fetch_data(3),
        fetch_data(4),
        fetch_data(5),
        fetch_data(6),
        fetch_data(7),
        fetch_data(8),
        fetch_data(9),
        fetch_data(10),
    )
    
    assert len(results) == 10
    assert all(r is not None for r in results)
    
    # Verify all IDs are unique
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 10  # All unique

@pytest.mark.asyncio
async def test_concurrent_with_comprehension():
    """Test concurrent operations with comprehension"""
    import asyncio
    
    # More elegant syntax
    results = await asyncio.gather(*[
        fetch_data(i) for i in range(1, 101)
    ])
    
    assert len(results) == 100
\`\`\`

### Testing Race Conditions

\`\`\`python
@pytest.mark.asyncio
async def test_race_condition_handling():
    """Test code handles race conditions"""
    import asyncio
    
    # Shared resource
    counter = {"value": 0}
    lock = asyncio.Lock()
    
    async def increment_with_lock():
        async with lock:
            current = counter["value"]
            await asyncio.sleep(0.001)  # Simulate slow operation
            counter["value"] = current + 1
    
    # Run 100 concurrent increments
    await asyncio.gather(*[increment_with_lock() for _ in range(100)])
    
    # With lock: counter = 100 (correct)
    assert counter["value"] == 100

@pytest.mark.asyncio
async def test_race_condition_without_lock():
    """Demonstrate race condition without lock"""
    import asyncio
    
    counter = {"value": 0}
    
    async def increment_without_lock():
        current = counter["value"]
        await asyncio.sleep(0.001)
        counter["value"] = current + 1
    
    await asyncio.gather(*[increment_without_lock() for _ in range(100)])
    
    # Without lock: counter < 100 (race condition)
    assert counter["value"] < 100  # Demonstrates the problem
\`\`\`

### Testing Error Handling in Concurrent Operations

\`\`\`python
@pytest.mark.asyncio
async def test_gather_with_exceptions():
    """Test asyncio.gather handles exceptions"""
    import asyncio
    
    async def success():
        return "ok"
    
    async def failure():
        raise ValueError("Error")
    
    # By default, gather raises first exception
    with pytest.raises(ValueError):
        await asyncio.gather(
            success(),
            failure(),
            success(),
        )
    
    # return_exceptions=True collects exceptions
    results = await asyncio.gather(
        success(),
        failure(),
        success(),
        return_exceptions=True
    )
    
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert results[2] == "ok"
\`\`\`

---

## Testing Async Context Managers

\`\`\`python
@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager"""
    async with DatabaseConnection() as conn:
        result = await conn.query("SELECT 1")
        assert result is not None
    # Connection automatically closed

@pytest.mark.asyncio
async def test_async_context_manager_exception():
    """Test context manager handles exceptions"""
    connection_closed = False
    
    class TestConnection:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            nonlocal connection_closed
            connection_closed = True
            return False  # Don't suppress exception
        
        async def query(self):
            raise ValueError("Query failed")
    
    with pytest.raises(ValueError):
        async with TestConnection() as conn:
            await conn.query()
    
    # Verify __aexit__ was called
    assert connection_closed
\`\`\`

---

## Testing Async Generators

\`\`\`python
@pytest.mark.asyncio
async def test_async_generator():
    """Test async generator"""
    async def async_range(n):
        for i in range(n):
            await asyncio.sleep(0.01)  # Simulate async operation
            yield i
    
    items = []
    async for item in async_range(5):
        items.append(item)
    
    assert items == [0, 1, 2, 3, 4]

@pytest.mark.asyncio
async def test_async_generator_with_data():
    """Test async generator with database streaming"""
    async def stream_users():
        # Simulate streaming large dataset
        for i in range(100):
            await asyncio.sleep(0.001)
            yield {"id": i, "name": f"user{i}"}
    
    # Process in batches
    batch = []
    count = 0
    async for user in stream_users():
        batch.append(user)
        count += 1
        
        if len(batch) == 10:
            # Process batch
            assert all("name" in u for u in batch)
            batch.clear()
    
    assert count == 100
\`\`\`

---

## Testing Timeouts

\`\`\`python
@pytest.mark.asyncio
async def test_timeout_with_wait_for():
    """Test operation times out appropriately"""
    import asyncio
    
    async def slow_operation():
        await asyncio.sleep(10)
        return "done"
    
    # Operation should timeout
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=1.0)

@pytest.mark.asyncio
async def test_timeout_success():
    """Test fast operation doesn't timeout"""
    import asyncio
    
    async def fast_operation():
        await asyncio.sleep(0.1)
        return "done"
    
    # Should complete successfully
    result = await asyncio.wait_for(fast_operation(), timeout=5.0)
    assert result == "done"

@pytest.mark.asyncio
async def test_custom_timeout():
    """Test custom timeout implementation"""
    import asyncio
    
    async def operation_with_timeout(timeout):
        try:
            async with asyncio.timeout(timeout):  # Python 3.11+
                await asyncio.sleep(2)
                return "completed"
        except asyncio.TimeoutError:
            return "timed out"
    
    result_timeout = await operation_with_timeout(0.5)
    assert result_timeout == "timed out"
    
    result_success = await operation_with_timeout(5.0)
    assert result_success == "completed"
\`\`\`

---

## Testing FastAPI Applications

\`\`\`python
import pytest
from httpx import AsyncClient
from myapp import app

@pytest.mark.asyncio
async def test_fastapi_endpoint():
    """Test FastAPI endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/users")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_fastapi_post(async_db_session):
    """Test FastAPI POST with database"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/users", json={
            "username": "alice",
            "email": "alice@example.com"
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "alice"
        assert data["id"] is not None

@pytest.mark.asyncio
async def test_fastapi_websocket():
    """Test FastAPI WebSocket"""
    from fastapi.testclient import TestClient
    
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("Hello")
            data = websocket.receive_text()
            assert data == "Hello"
\`\`\`

---

## Common Patterns & Best Practices

### Pattern 1: Async Test Fixture for Database

\`\`\`python
@pytest.fixture
async def async_test_db():
    """Complete async database fixture"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    
    # Create engine
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()
    
    # Cleanup
    await engine.dispose()
\`\`\`

### Pattern 2: Async Test Data Factory

\`\`\`python
class AsyncUserFactory:
    """Factory for creating async test users"""
    _counter = 0
    
    @classmethod
    async def create(cls, session, **kwargs):
        cls._counter += 1
        user = User(
            username=kwargs.get("username", f"user{cls._counter}"),
            email=kwargs.get("email", f"user{cls._counter}@example.com"),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    @classmethod
    async def create_batch(cls, session, count):
        users = []
        for _ in range(count):
            user = await cls.create(session)
            users.append(user)
        return users
\`\`\`

### Pattern 3: Testing Retry Logic

\`\`\`python
@pytest.mark.asyncio
async def test_retry_logic(mocker):
    """Test async retry mechanism"""
    mock_api = mocker.patch("myapp.api.external_call", new_callable=AsyncMock)
    
    # First 2 calls fail, 3rd succeeds
    mock_api.side_effect = [
        Exception("Network error"),
        Exception("Timeout"),
        {"status": "ok"}
    ]
    
    # Function with retry logic
    result = await call_with_retry(max_attempts=3)
    
    assert result["status"] == "ok"
    assert mock_api.call_count == 3
\`\`\`

---

## Best Practices

1. **Use @pytest.mark.asyncio** for all async tests
2. **AsyncMock for mocking** async functions
3. **Async fixtures** for async setup (database, API clients)
4. **Test concurrent behavior** with asyncio.gather
5. **Test timeout handling** with asyncio.wait_for
6. **Test error propagation** in async chains
7. **Use AsyncClient** for FastAPI testing
8. **Clean up resources** in async fixtures (await dispose)

---

## Common Pitfalls

❌ **Forgetting await**
\`\`\`python
result = async_function()  # Returns coroutine, not result
\`\`\`

❌ **Using regular Mock for async**
\`\`\`python
mock = Mock()  # TypeError: can't be used in await
\`\`\`

❌ **Not handling exceptions in gather**
\`\`\`python
await asyncio.gather(task1(), task2())  # First exception aborts all
\`\`\`

✅ **Correct patterns**
\`\`\`python
result = await async_function()
mock = AsyncMock()
await asyncio.gather(task1(), task2(), return_exceptions=True)
\`\`\`

---

## Summary

**Async testing essentials**:
- **pytest-asyncio**: @pytest.mark.asyncio decorator
- **Async fixtures**: async def with yield
- **AsyncMock**: Mock async functions
- **Concurrent testing**: asyncio.gather for parallelism
- **Timeout testing**: asyncio.wait_for
- **FastAPI**: AsyncClient for endpoint testing

Master async testing for **reliable FastAPI, aiohttp, and asyncpg applications**.
`,
};
