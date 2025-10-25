export const productionPatterns = {
  title: 'SQLAlchemy Production Patterns',
  id: 'production-patterns',
  content: `
# SQLAlchemy Production Patterns

## Introduction

Production SQLAlchemy applications require careful attention to connection pooling, caching, monitoring, error handling, and architectural patterns. This section covers production-ready patterns for reliable, scalable database applications.

In this section, you'll master:
- Connection pool configuration
- Caching strategies (query cache, Redis)
- Monitoring & observability
- Error handling & retries
- Repository pattern
- Unit of Work pattern
- Production deployment
- Performance optimization

### Why Production Patterns Matter

**Production reality**: Poor connection pooling → connection exhaustion. No caching → slow queries. No monitoring → blind to issues. Bad error handling → data corruption. Production patterns prevent outages and ensure reliability.

---

## Connection Pooling

### Pool Configuration

\`\`\`python
"""
Production Connection Pool Configuration
"""

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Production engine with proper pooling
engine = create_engine(
    "postgresql://user:pass@localhost/mydb",
    
    # Pool settings
    poolclass=QueuePool,  # Default, supports multiple threads
    pool_size=20,          # Permanent connections (adjust based on load)
    max_overflow=40,       # Additional connections when pool exhausted
    pool_timeout=30,       # Wait 30s for connection before error
    pool_recycle=3600,     # Recycle connections after 1 hour (prevent stale)
    pool_pre_ping=True,    # Test connection before checkout (catch disconnects)
    
    # Engine settings
    echo=False,            # Don't log SQL (use logging module instead)
    future=True,           # Use SQLAlchemy 2.0 style
)

# Pool sizing formula:
# pool_size = number_of_application_threads
# max_overflow = 2 * pool_size (for burst capacity)
# Total connections = pool_size + max_overflow

# Example: 
# 10 gunicorn workers * 2 threads = 20 threads
# pool_size = 20, max_overflow = 40
# Max connections = 60
\`\`\`

### Pool Events & Monitoring

\`\`\`python
"""
Monitor Connection Pool
"""

from sqlalchemy import event
import logging

logger = logging.getLogger(__name__)

@event.listens_for (engine, "connect")
def receive_connect (dbapi_conn, connection_record):
    """Log when connection created"""
    logger.info("New database connection created")

@event.listens_for (engine, "checkout")
def receive_checkout (dbapi_conn, connection_record, connection_proxy):
    """Log when connection checked out from pool"""
    logger.debug("Connection checked out from pool")

@event.listens_for (engine, "checkin")
def receive_checkin (dbapi_conn, connection_record):
    """Log when connection returned to pool"""
    logger.debug("Connection returned to pool")

# Pool status
def get_pool_status():
    """Get current pool statistics"""
    pool = engine.pool
    return {
        "size": pool.size(),           # Current pool size
        "checked_in": pool.checkedin(), # Available connections
        "checked_out": pool.checkedout(),  # In-use connections
        "overflow": pool.overflow(),   # Overflow connections
    }

# Usage
stats = get_pool_status()
logger.info (f"Pool stats: {stats}")
\`\`\`

---

## Caching Strategies

### Query Result Caching

\`\`\`python
"""
Cache Query Results with Redis
"""

import redis
import json
from functools import wraps

redis_client = redis.Redis (host='localhost', port=6379, db=0)

def cache_query (key_prefix: str, ttl: int = 300):
    """Cache query results decorator"""
    def decorator (func):
        @wraps (func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{args}:{kwargs}"
            
            # Check cache
            cached = redis_client.get (cache_key)
            if cached:
                logger.info (f"Cache hit: {cache_key}")
                return json.loads (cached)
            
            # Execute query
            result = func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps (result, default=str)  # default=str for datetime
            )
            logger.info (f"Cache miss: {cache_key}")
            
            return result
        return wrapper
    return decorator

# Usage
@cache_query("user_by_id", ttl=600)
def get_user (user_id: int):
    with Session() as session:
        user = session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one()
        return {
            "id": user.id,
            "email": user.email,
            "created_at": user.created_at
        }

# First call: database query + cache store
user = get_user(123)

# Second call: cache hit (no database query)
user = get_user(123)
\`\`\`

### Cache Invalidation

\`\`\`python
"""
Cache Invalidation Strategies
"""

def invalidate_user_cache (user_id: int):
    """Invalidate cached user data"""
    redis_client.delete (f"user_by_id:{user_id}")

def invalidate_pattern (pattern: str):
    """Invalidate all keys matching pattern"""
    for key in redis_client.scan_iter (pattern):
        redis_client.delete (key)

# Invalidate on update
def update_user (user_id: int, **kwargs):
    with Session() as session:
        session.execute(
            update(User)
            .where(User.id == user_id)
            .values(**kwargs)
        )
        session.commit()
        
        # Invalidate cache
        invalidate_user_cache (user_id)
        invalidate_pattern (f"users_list:*")  # Invalidate list queries

# Two strategies:
# 1. TTL (time-to-live): Cache expires after X seconds
# 2. Explicit invalidation: Invalidate on write
# Production: Use both - TTL as fallback, invalidate on write for accuracy
\`\`\`

---

## Monitoring & Observability

### Query Logging

\`\`\`python
"""
Production Query Logging
"""

import logging
import time
from sqlalchemy import event

logger = logging.getLogger("sqlalchemy.engine")
logger.setLevel (logging.INFO)

@event.listens_for (engine, "before_cursor_execute")
def before_cursor_execute (conn, cursor, statement, parameters, context, executemany):
    """Log query start and store start time"""
    context._query_start_time = time.time()
    logger.debug (f"Query started: {statement[:100]}")

@event.listens_for (engine, "after_cursor_execute")
def after_cursor_execute (conn, cursor, statement, parameters, context, executemany):
    """Log query completion and duration"""
    duration = time.time() - context._query_start_time
    
    # Log slow queries (> 100ms)
    if duration > 0.1:
        logger.warning(
            f"SLOW QUERY ({duration:.2f}s): {statement[:200]}"
        )
    else:
        logger.debug (f"Query completed in {duration:.3f}s")
\`\`\`

### Prometheus Metrics

\`\`\`python
"""
Expose Database Metrics to Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge

# Metrics
db_query_count = Counter(
    'db_query_total',
    'Total database queries',
    ['operation']  # SELECT, INSERT, UPDATE, DELETE
)

db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation']
)

db_pool_size = Gauge(
    'db_pool_size',
    'Connection pool size'
)

db_pool_checked_out = Gauge(
    'db_pool_checked_out',
    'Checked out connections'
)

# Track metrics
@event.listens_for (engine, "before_cursor_execute")
def track_query_start (conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()
    
    # Determine operation
    operation = statement.strip().split()[0].upper()
    db_query_count.labels (operation=operation).inc()

@event.listens_for (engine, "after_cursor_execute")
def track_query_duration (conn, cursor, statement, parameters, context, executemany):
    duration = time.time() - context._query_start_time
    operation = statement.strip().split()[0].upper()
    db_query_duration.labels (operation=operation).observe (duration)

# Update pool metrics periodically
def update_pool_metrics():
    """Update pool metrics (call every 10s)"""
    pool = engine.pool
    db_pool_size.set (pool.size())
    db_pool_checked_out.set (pool.checkedout())

# Expose metrics endpoint
from flask import Flask
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware (app.wsgi_app, {
    '/metrics': make_wsgi_app()
})
\`\`\`

---

## Error Handling & Retries

### Retry Logic

\`\`\`python
"""
Retry Database Operations
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy.exc import OperationalError, DBAPIError

@retry(
    stop=stop_after_attempt(3),           # Max 3 attempts
    wait=wait_exponential (multiplier=1, min=1, max=10),  # Exponential backoff
    retry=retry_if_exception_type(OperationalError)  # Only retry operational errors
)
def query_with_retry (session, stmt):
    """Execute query with automatic retry"""
    return session.execute (stmt).scalars().all()

# Usage
try:
    users = query_with_retry (session, select(User))
except OperationalError as e:
    logger.error (f"Database query failed after retries: {e}")
    raise
\`\`\`

### Graceful Error Handling

\`\`\`python
"""
Production Error Handling
"""

from sqlalchemy.exc import IntegrityError, DataError

def safe_create_user (email: str):
    """Create user with proper error handling"""
    try:
        with Session() as session:
            user = User (email=email)
            session.add (user)
            session.commit()
            return {"success": True, "user_id": user.id}
    
    except IntegrityError as e:
        # Constraint violation (duplicate email, etc.)
        logger.warning (f"Integrity error creating user: {e}")
        return {"success": False, "error": "User already exists"}
    
    except DataError as e:
        # Invalid data (email too long, invalid type, etc.)
        logger.warning (f"Data error creating user: {e}")
        return {"success": False, "error": "Invalid data"}
    
    except OperationalError as e:
        # Database connection issues
        logger.error (f"Database operational error: {e}")
        return {"success": False, "error": "Database temporarily unavailable"}
    
    except Exception as e:
        # Unexpected error
        logger.exception (f"Unexpected error creating user: {e}")
        return {"success": False, "error": "Internal server error"}
\`\`\`

---

## Repository Pattern

### Implementation

\`\`\`python
"""
Repository Pattern for Data Access
"""

from abc import ABC, abstractmethod
from typing import Optional, List

class UserRepository(ABC):
    """Abstract repository interface"""
    
    @abstractmethod
    def find_by_id (self, user_id: int) -> Optional[User]:
        pass
    
    @abstractmethod
    def find_by_email (self, email: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def find_all (self, skip: int = 0, limit: int = 100) -> List[User]:
        pass
    
    @abstractmethod
    def create (self, user: User) -> User:
        pass
    
    @abstractmethod
    def update (self, user: User) -> User:
        pass
    
    @abstractmethod
    def delete (self, user_id: int) -> bool:
        pass

class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy implementation"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def find_by_id (self, user_id: int) -> Optional[User]:
        return self.session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()
    
    def find_by_email (self, email: str) -> Optional[User]:
        return self.session.execute(
            select(User).where(User.email == email)
        ).scalar_one_or_none()
    
    def find_all (self, skip: int = 0, limit: int = 100) -> List[User]:
        return self.session.execute(
            select(User).offset (skip).limit (limit)
        ).scalars().all()
    
    def create (self, user: User) -> User:
        self.session.add (user)
        self.session.flush()  # Get user.id without commit
        return user
    
    def update (self, user: User) -> User:
        self.session.merge (user)
        return user
    
    def delete (self, user_id: int) -> bool:
        result = self.session.execute(
            delete(User).where(User.id == user_id)
        )
        return result.rowcount > 0

# Usage
def get_user_service (user_id: int):
    with Session() as session:
        repo = SQLAlchemyUserRepository (session)
        user = repo.find_by_id (user_id)
        return user
\`\`\`

---

## Unit of Work Pattern

### Implementation

\`\`\`python
"""
Unit of Work Pattern
"""

class UnitOfWork:
    """Manage transaction and repositories"""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.session: Optional[Session] = None
    
    def __enter__(self):
        self.session = self.session_factory()
        self.users = SQLAlchemyUserRepository (self.session)
        self.posts = SQLAlchemyPostRepository (self.session)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        self.session.close()
    
    def commit (self):
        """Commit transaction"""
        self.session.commit()
    
    def rollback (self):
        """Rollback transaction"""
        self.session.rollback()

# Usage
def create_user_with_post (email: str, post_title: str):
    """Create user and post in single transaction"""
    with UnitOfWork(Session) as uow:
        # Create user
        user = User (email=email)
        uow.users.create (user)
        
        # Create post
        post = Post (title=post_title, user=user)
        uow.posts.create (post)
        
        # Commit both (or rollback both on error)
        uow.commit()
\`\`\`

---

## Production Deployment

### Configuration Management

\`\`\`python
"""
Environment-Based Configuration
"""

import os
from pydantic import BaseSettings

class DatabaseConfig(BaseSettings):
    """Database configuration from environment"""
    
    database_url: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo_sql: bool = False
    
    class Config:
        env_file = ".env"

# Load config
config = DatabaseConfig()

# Create engine with config
engine = create_engine(
    config.database_url,
    pool_size=config.pool_size,
    max_overflow=config.max_overflow,
    pool_timeout=config.pool_timeout,
    pool_recycle=config.pool_recycle,
    echo=config.echo_sql,
    pool_pre_ping=True
)

# Different configs per environment
# .env.development:
#   DATABASE_URL=postgresql://localhost/dev_db
#   ECHO_SQL=true
#
# .env.production:
#   DATABASE_URL=postgresql://prod-db.example.com/prod_db
#   POOL_SIZE=50
#   ECHO_SQL=false
\`\`\`

### Health Checks

\`\`\`python
"""
Database Health Check Endpoint
"""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for load balancer"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute (text("SELECT 1"))
        
        # Check pool status
        pool = engine.pool
        checked_out = pool.checkedout()
        size = pool.size()
        
        # Alert if pool nearly exhausted
        if checked_out > size * 0.8:
            logger.warning (f"Pool nearly exhausted: {checked_out}/{size}")
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "pool": {
                "size": size,
                "checked_out": checked_out,
                "available": size - checked_out
            }
        }), 200
    
    except Exception as e:
        logger.error (f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str (e)
        }), 503
\`\`\`

---

## Summary

### Key Takeaways

✅ **Connection pooling**: pool_size=20, max_overflow=40, pool_pre_ping=True  
✅ **Caching**: Redis query cache + TTL + invalidation on write  
✅ **Monitoring**: Prometheus metrics, slow query logging, pool stats  
✅ **Error handling**: Retry with exponential backoff, graceful degradation  
✅ **Repository pattern**: Abstract data access, testable, swappable  
✅ **Unit of Work**: Manage transactions across repositories  
✅ **Configuration**: Environment-based, secrets management  
✅ **Health checks**: Database connectivity, pool status

### Production Checklist

✅ Connection pool properly sized  
✅ Query result caching implemented  
✅ Slow query monitoring enabled  
✅ Prometheus metrics exposed  
✅ Retry logic for transient failures  
✅ Repository pattern for data access  
✅ Health check endpoint configured  
✅ Environment-based configuration  
✅ Secrets not in code (use env vars)  
✅ Database connection SSL enabled

### Performance Optimization

✅ **Connection pooling**: Reuse connections  
✅ **Caching**: Avoid repeated queries  
✅ **Indexes**: Fast lookups  
✅ **Eager loading**: Avoid N+1  
✅ **Batch operations**: bulk_insert_mappings()  
✅ **Read replicas**: Scale reads  
✅ **Query optimization**: EXPLAIN ANALYZE

### Next Steps

Congratulations! You've completed **Module 5: SQLAlchemy & Database Mastery**. You now have comprehensive knowledge of:
- Database fundamentals for Python
- SQLAlchemy Core & ORM concepts
- Defining models & relationships
- Query API & advanced filtering
- Relationship loading strategies
- Session management patterns
- Alembic migrations & advanced techniques
- Performance optimization
- Advanced patterns (repository, unit of work)
- Testing with databases
- Multi-database & sharding
- Production patterns & deployment

Continue to **Module 6** to expand your Python mastery!
`,
};
