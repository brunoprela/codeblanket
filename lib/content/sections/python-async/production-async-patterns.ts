export const productionAsyncPatterns = {
  title: 'Production Async Patterns & Best Practices',
  id: 'production-async-patterns',
  content: `
# Production Async Patterns & Best Practices

## Introduction

Building production async systems requires more than just knowing async/await. You need **robust error handling, monitoring, graceful shutdown, resource management, and performance optimization.** This section covers battle-tested patterns for production deployments.

### Production Requirements

\`\`\`python
"""
Production Checklist
"""

# ✅ Graceful Shutdown: Handle SIGTERM, cleanup resources
# ✅ Health Checks: Monitor application health
# ✅ Metrics & Monitoring: Track performance, errors
# ✅ Connection Pooling: Reuse connections efficiently
# ✅ Timeouts: Every external call has timeout
# ✅ Error Handling: No silent failures
# ✅ Resource Limits: Prevent resource exhaustion
# ✅ Backpressure: Handle slow consumers
# ✅ Testing: Comprehensive async test coverage
\`\`\`

By the end of this section, you'll master:
- Production-ready application structure
- Graceful shutdown patterns
- Health checks and monitoring
- Connection pooling
- Performance optimization
- Deployment best practices

---

## Production Application Structure

### Complete Async Application

\`\`\`python
"""
Production Async Application Structure
"""

import asyncio
import signal
import logging
from contextlib import asynccontextmanager
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncApplication:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.background_tasks: List[asyncio.Task] = []
        self.db_pool = None
        self.http_session = None
    
    async def startup(self):
        """Initialize application resources"""
        logger.info("Starting application...")
        
        # Initialize database pool
        self.db_pool = await self._create_db_pool()
        
        # Initialize HTTP session
        self.http_session = await self._create_http_session()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_task()),
            asyncio.create_task(self._metrics_task()),
            asyncio.create_task(self._cleanup_task()),
        ]
        
        logger.info("Application started")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close connections
        if self.http_session:
            await self.http_session.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Shutdown complete")
    
    async def run(self):
        """Main application loop"""
        try:
            await self.startup()
            await self.shutdown_event.wait()
        finally:
            await self.shutdown()
    
    async def _create_db_pool(self):
        """Create database connection pool"""
        import asyncpg
        return await asyncpg.create_pool(
            "postgresql://localhost/db",
            min_size=10,
            max_size=20,
            command_timeout=60.0
        )
    
    async def _create_http_session(self):
        """Create HTTP session"""
        import aiohttp
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def _health_check_task(self):
        """Background health check"""
        while not self.shutdown_event.is_set():
            try:
                # Check database
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                logger.info("Health check: OK")
            except Exception as e:
                logger.error(f"Health check failed: {e}")
            
            await asyncio.sleep(30)
    
    async def _metrics_task(self):
        """Background metrics collection"""
        while not self.shutdown_event.is_set():
            try:
                tasks = len(asyncio.all_tasks())
                logger.info(f"Metrics: {tasks} active tasks")
            except Exception as e:
                logger.error(f"Metrics failed: {e}")
            
            await asyncio.sleep(60)
    
    async def _cleanup_task(self):
        """Background cleanup"""
        while not self.shutdown_event.is_set():
            try:
                logger.info("Running cleanup...")
                # Cleanup logic here
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
            
            await asyncio.sleep(300)

async def main():
    app = AsyncApplication()
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(app.shutdown())
        )
    
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

---

## Graceful Shutdown

### Signal Handling

\`\`\`python
"""
Graceful Shutdown with Signal Handling
"""

import asyncio
import signal
import logging

logger = logging.getLogger(__name__)

class GracefulShutdown:
    def __init__(self, grace_period: float = 30.0):
        self.grace_period = grace_period
        self.shutdown_event = asyncio.Event()
        self.workers: List[asyncio.Task] = []
    
    async def worker(self, name: str):
        """Worker that respects shutdown"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Simulate work
                    await asyncio.wait_for(
                        self._do_work(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            logger.info(f"{name} cancelled, cleaning up...")
            # Cleanup logic
            raise
    
    async def _do_work(self):
        """Actual work"""
        await asyncio.sleep(0.5)
    
    async def start(self, n_workers: int = 5):
        """Start workers"""
        self.workers = [
            asyncio.create_task(self.worker(f"Worker-{i}"))
            for i in range(n_workers)
        ]
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutdown initiated...")
        
        # Signal workers to stop
        self.shutdown_event.set()
        
        # Wait for workers with grace period
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=self.grace_period
            )
            logger.info("All workers stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning("Grace period expired, cancelling workers")
            for worker in self.workers:
                worker.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("Shutdown complete")

async def main():
    app = GracefulShutdown(grace_period=30.0)
    await app.start(n_workers=10)
    
    # Setup signal handler
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGTERM,
        lambda: asyncio.create_task(app.shutdown())
    )
    
    # Wait indefinitely
    await asyncio.Event().wait()

asyncio.run(main())
\`\`\`

---

## Health Checks & Monitoring

### Health Check System

\`\`\`python
"""
Production Health Check System
"""

import asyncio
from datetime import datetime
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.last_results = {}
    
    def register(self, name: str, check_func):
        """Register health check"""
        self.checks[name] = check_func
    
    async def check(self, name: str) -> dict:
        """Run single health check"""
        start = datetime.now()
        try:
            await asyncio.wait_for(
                self.checks[name](),
                timeout=5.0
            )
            status = HealthStatus.HEALTHY
            message = "OK"
        except asyncio.TimeoutError:
            status = HealthStatus.DEGRADED
            message = "Timeout"
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = str(e)
        
        duration = (datetime.now() - start).total_seconds()
        
        result = {
            'status': status.value,
            'message': message,
            'duration_ms': duration * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        self.last_results[name] = result
        return result
    
    async def check_all(self) -> dict:
        """Run all health checks"""
        results = await asyncio.gather(*[
            self.check(name) for name in self.checks.keys()
        ], return_exceptions=True)
        
        checks_dict = dict(zip(self.checks.keys(), results))
        
        # Determine overall status
        statuses = [r['status'] for r in checks_dict.values()]
        if any(s == HealthStatus.UNHEALTHY.value for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED.value for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            'status': overall.value,
            'checks': checks_dict,
            'timestamp': datetime.now().isoformat()
        }

# Usage
checker = HealthChecker()

async def check_database():
    """Check database connectivity"""
    # Actual database check
    await asyncio.sleep(0.1)

async def check_redis():
    """Check Redis connectivity"""
    await asyncio.sleep(0.1)

checker.register('database', check_database)
checker.register('redis', check_redis)

# In FastAPI:
# @app.get("/health")
# async def health():
#     return await checker.check_all()
\`\`\`

---

## Connection Pooling

### Database Connection Pool

\`\`\`python
"""
Production Database Pool Configuration
"""

import asyncpg

async def create_production_pool():
    """Create production-ready database pool"""
    return await asyncpg.create_pool(
        host='localhost',
        database='mydb',
        user='user',
        password='password',
        
        # Pool size
        min_size=10,     # Keep 10 connections warm
        max_size=20,     # Allow burst to 20
        
        # Timeouts
        command_timeout=60.0,  # Query timeout
        timeout=30.0,          # Connection timeout
        
        # Connection lifetime
        max_inactive_connection_lifetime=300.0,  # Close idle after 5min
        max_queries=50000,  # Recycle after 50K queries
        
        # Health checks
        setup=setup_connection,
    )

async def setup_connection(conn):
    """Setup new connection"""
    # Set timezone
    await conn.execute("SET timezone TO 'UTC'")
    
    # Prepare commonly used queries
    await conn.execute("PREPARE user_by_id AS SELECT * FROM users WHERE id = $1")

# Usage
pool = await create_production_pool()

# Acquire connection
async with pool.acquire() as conn:
    user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", 123)

# Close pool on shutdown
await pool.close()
\`\`\`

---

## Performance Optimization

### uvloop for Speed

\`\`\`python
"""
Use uvloop for 2-4× Performance Boost
"""

import asyncio

# Install: pip install uvloop

# Option 1: Set as default event loop policy
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def main():
    # Your async code here
    pass

# Run normally
asyncio.run(main())

# Option 2: Use directly
import uvloop
uvloop.run(main())

# Performance gain: 2-4× faster than default asyncio
# Use in production for better throughput
\`\`\`

### Caching Pattern

\`\`\`python
"""
Async Cache with TTL
"""

import asyncio
import time
from functools import wraps

class AsyncCache:
    def __init__(self, ttl: float = 300.0):
        self.cache = {}
        self.ttl = ttl
        self.lock = asyncio.Lock()
    
    def cached(self, func):
        """Cache decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            async with self.lock:
                if key in self.cache:
                    value, timestamp = self.cache[key]
                    if time.time() - timestamp < self.ttl:
                        return value
            
            # Cache miss or expired
            value = await func(*args, **kwargs)
            
            async with self.lock:
                self.cache[key] = (value, time.time())
            
            return value
        
        return wrapper

# Usage
cache = AsyncCache(ttl=60.0)

@cache.cached
async def expensive_operation(param):
    """Expensive operation"""
    await asyncio.sleep(2)
    return f"Result for {param}"

# First call: takes 2s
result = await expensive_operation("test")

# Second call within 60s: instant
result = await expensive_operation("test")
\`\`\`

---

## Testing Async Code

### Async Test Patterns

\`\`\`python
"""
Testing Async Code with pytest
"""

import pytest
import asyncio

# pytest-asyncio required: pip install pytest-asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test async function"""
    result = await some_async_function()
    assert result == expected_value

@pytest.mark.asyncio
async def test_with_timeout():
    """Test with timeout"""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            slow_function(),
            timeout=1.0
        )

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations"""
    results = await asyncio.gather(
        operation1(),
        operation2(),
        operation3()
    )
    assert len(results) == 3

# Mocking async functions
@pytest.mark.asyncio
async def test_with_mock(mocker):
    """Test with mocked async function"""
    mock_db = mocker.AsyncMock()
    mock_db.fetch.return_value = [{'id': 1, 'name': 'Test'}]
    
    result = await process_users(mock_db)
    assert len(result) == 1
    mock_db.fetch.assert_called_once()
\`\`\`

---

## Deployment Best Practices

### Production Configuration

\`\`\`python
"""
Production Configuration
"""

import os

class Config:
    # Database
    DB_POOL_MIN_SIZE = int(os.getenv('DB_POOL_MIN_SIZE', '10'))
    DB_POOL_MAX_SIZE = int(os.getenv('DB_POOL_MAX_SIZE', '20'))
    DB_COMMAND_TIMEOUT = float(os.getenv('DB_COMMAND_TIMEOUT', '60.0'))
    
    # HTTP
    HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '30.0'))
    HTTP_MAX_CONNECTIONS = int(os.getenv('HTTP_MAX_CONNECTIONS', '100'))
    
    # Application
    WORKER_COUNT = int(os.getenv('WORKER_COUNT', '10'))
    SHUTDOWN_GRACE_PERIOD = float(os.getenv('SHUTDOWN_GRACE_PERIOD', '30.0'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Monitoring
    HEALTH_CHECK_INTERVAL = float(os.getenv('HEALTH_CHECK_INTERVAL', '30.0'))
    METRICS_INTERVAL = float(os.getenv('METRICS_INTERVAL', '60.0'))

# Use in application
config = Config()
pool = await create_pool(
    min_size=config.DB_POOL_MIN_SIZE,
    max_size=config.DB_POOL_MAX_SIZE
)
\`\`\`

---

## Summary

### Production Checklist

- ✅ **Graceful Shutdown**: Handle SIGTERM, cleanup resources
- ✅ **Health Checks**: /health endpoint with dependency checks
- ✅ **Connection Pooling**: Reuse DB/HTTP connections
- ✅ **Timeouts**: Every external call has timeout
- ✅ **Error Handling**: No silent failures, log all errors
- ✅ **Monitoring**: Metrics, tracing, logging
- ✅ **Testing**: Comprehensive async test coverage
- ✅ **Configuration**: Environment-based config
- ✅ **Performance**: uvloop, caching, optimization

### Key Patterns

1. **Application Structure**: Startup, shutdown, background tasks
2. **Signal Handling**: SIGTERM for graceful shutdown
3. **Health Checks**: Monitor dependencies
4. **Connection Pools**: Database, HTTP sessions
5. **Caching**: Reduce expensive operations
6. **Testing**: pytest-asyncio for async tests

### Performance Tips

- Use uvloop (2-4× faster)
- Connection pooling (reuse connections)
- Batch operations (reduce round-trips)
- Caching (avoid repeated work)
- Monitor and profile (find bottlenecks)

### Deployment

- Container-friendly (handle SIGTERM)
- Configuration via environment variables
- Health checks for orchestration
- Structured logging for analysis
- Metrics for monitoring

**Congratulations!** You've mastered asynchronous Python from fundamentals to production deployment!
\`,
};
