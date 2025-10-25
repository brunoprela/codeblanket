export const sessionManagement = {
  title: 'Session Management & Patterns',
  id: 'session-management',
  content: `
# Session Management & Patterns

## Introduction

The Session is SQLAlchemy's "workspace" for database operations. Understanding session lifecycle, scoping, and patterns is critical for building robust, thread-safe, production applications. Poor session management is a leading cause of bugs, memory leaks, and concurrency issues.

In this section, you'll master:
- Session lifecycle and states
- Session factory patterns
- Scoped sessions for web applications
- Thread-local session management
- Context managers and dependency injection
- Session states (transient, pending, persistent, detached)
- Expiring and refreshing objects
- Session best practices for production

### Why Session Management Matters

**Production impact**: Session mismanagement causes: memory leaks (unclosed sessions), stale data (cached objects), race conditions (shared sessions), DetachedInstanceError, and connection pool exhaustion. Proper session management is non-negotiable.

---

## Session Lifecycle

### Session States

\`\`\`python
"""
Object States in Session Lifecycle
"""

# State 1: TRANSIENT (not in session, no database identity)
user = User(email="test@example.com")
# Object exists in Python, not tracked by session

# State 2: PENDING (in session, not yet in database)
session.add(user)
# Session tracks object, will INSERT on flush

# State 3: PERSISTENT (in session, in database)
session.commit()  # or session.flush()
# Object has database identity (primary key)

# State 4: DETACHED (was in session, now removed)
session.close()
# Or: session.expunge(user)
# Object still exists in Python, but session doesn't track it

# Check object state
from sqlalchemy.inspect import inspect
state = inspect(user)
print(state.transient)   # True if transient
print(state.pending)     # True if pending
print(state.persistent)  # True if persistent
print(state.detached)    # True if detached
\`\`\`

### Session Operations

\`\`\`python
"""
Core Session Operations
"""

# Create session
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# ADD: Track new object
user = User(email="new@example.com")
session.add(user)  # Now pending

# ADD_ALL: Track multiple objects
users = [User(email=f"user{i}@example.com") for i in range(100)]
session.add_all(users)

# FLUSH: Send pending changes to database (within transaction)
session.flush()
# Executes INSERT/UPDATE/DELETE, objects get primary keys
# Transaction still open, can rollback

# COMMIT: Flush + commit transaction
session.commit()
# Makes changes permanent

# ROLLBACK: Discard changes
user.email = "changed@example.com"
session.rollback()  # Reverts to database state
print(user.email)  # Original value restored

# REFRESH: Reload object from database
session.refresh(user)
# Queries database, updates object with latest values

# EXPUNGE: Remove object from session (becomes detached)
session.expunge(user)
# Object no longer tracked

# EXPUNGE_ALL: Remove all objects
session.expunge_all()

# CLOSE: Close session, return connection to pool
session.close()
# Objects become detached
\`\`\`

---

## Session Factory Pattern

### Basic Session Factory

\`\`\`python
"""
Session Factory with sessionmaker
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create engine (once, global)
engine = create_engine("postgresql://localhost/mydb")

# Create session factory (once, global)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,    # Manual commits (recommended)
    autoflush=True,      # Auto-flush before queries
    expire_on_commit=True  # Expire objects after commit
)

# Create sessions as needed
def get_users():
    session = SessionLocal()
    try:
        users = session.execute(select(User)).scalars().all()
        return users
    finally:
        session.close()
\`\`\`

### Session Configuration Options

\`\`\`python
"""
sessionmaker Configuration
"""

SessionLocal = sessionmaker(
    bind=engine,
    
    # Commit behavior
    autocommit=False,  # False = manual commits (RECOMMENDED)
    # True = auto-commit each statement (not recommended)
    
    # Flush behavior  
    autoflush=True,   # True = auto-flush before queries (RECOMMENDED)
    # False = manual flush required
    
    # Expiry behavior
    expire_on_commit=True,  # True = expire objects after commit
    # False = keep cached values (dangerous: stale data)
    
    # Class to use
    class_=Session,  # Custom session class
    
    # Query class
    query_cls=Query,  # Custom query class
    
    # Execution options
    execution_options={
        "isolation_level": "READ COMMITTED"
    }
)

# Why these defaults?
# autocommit=False: Explicit commits, predictable transactions
# autoflush=True: Ensures queries see pending changes
# expire_on_commit=True: Prevents stale data after commit
\`\`\`

---

## Context Manager Pattern

### Basic Context Manager

\`\`\`python
"""
Session Context Manager
"""

from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Provide transactional scope for database operations"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with get_db_session() as session:
    user = User(email="test@example.com")
    session.add(user)
    # Auto-commits on success, rolls back on exception
# Session automatically closed
\`\`\`

### Advanced Context Manager

\`\`\`python
"""
Production Context Manager with Logging
"""

import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Database session context manager
    
    Features:
    - Automatic commit/rollback
    - Connection error handling
    - Query logging
    - Performance monitoring
    """
    session = SessionLocal()
    start_time = time.time()
    query_count = 0
    
    # Track queries
    @event.listens_for(session, "after_cursor_execute")
    def count_queries(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        query_count += 1
    
    try:
        yield session
        session.commit()
        
        duration = time.time() - start_time
        if duration > 1.0:  # Slow transaction
            logger.warning(
                f"Slow transaction: {duration:.2f}s, {query_count} queries"
            )
            
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
        
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error: {e}")
        raise
        
    finally:
        session.close()
        event.remove(session, "after_cursor_execute", count_queries)
\`\`\`

---

## Scoped Sessions

### Thread-Local Sessions

\`\`\`python
"""
Scoped Session for Multi-Threaded Applications
"""

from sqlalchemy.orm import scoped_session, sessionmaker

# Create scoped session (thread-local)
SessionLocal = scoped_session(
    sessionmaker(bind=engine)
)

# Each thread gets its own session
def thread_function():
    # This thread's session
    session = SessionLocal()
    
    user = User(email="thread@example.com")
    session.add(user)
    session.commit()
    
    # Remove session for this thread
    SessionLocal.remove()

import threading
threads = [threading.Thread(target=thread_function) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Global operations
SessionLocal.remove()  # Remove all thread-local sessions
\`\`\`

### Scoped Session Patterns

\`\`\`python
"""
Scoped Session Usage Patterns
"""

# Pattern 1: Global session object
Session = scoped_session(sessionmaker(bind=engine))

def create_user(email):
    user = User(email=email)
    Session.add(user)
    Session.commit()
    return user

# Pattern 2: Explicit removal
def process_batch():
    for item in items:
        process_item(item)
    Session.remove()  # Clean up

# Pattern 3: Registry pattern
from sqlalchemy.orm import scoped_session

registry = scoped_session(sessionmaker(bind=engine))

class UserRepository:
    def __init__(self, session=None):
        self.session = session or registry()
    
    def create(self, email):
        user = User(email=email)
        self.session.add(user)
        self.session.commit()
        return user

# WARNING: Scoped sessions are tricky!
# - Don't use with async code
# - Careful with web frameworks (request-scoped better)
# - Modern pattern: dependency injection instead
\`\`\`

---

## Web Framework Integration

### FastAPI Integration

\`\`\`python
"""
FastAPI Dependency Injection Pattern (RECOMMENDED)
"""

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint with dependency
@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users")
def create_user(email: str, db: Session = Depends(get_db)):
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)  # Get ID after insert
    return user

# FastAPI automatically:
# - Injects session
# - Closes session after request
# - Handles exceptions
\`\`\`

### Flask Integration

\`\`\`python
"""
Flask Session Management
"""

from flask import Flask, g
from sqlalchemy.orm import scoped_session, sessionmaker

app = Flask(__name__)

# Scoped session (thread-local)
Session = scoped_session(sessionmaker(bind=engine))

@app.before_request
def before_request():
    """Create session before each request"""
    g.db = Session()

@app.teardown_request
def teardown_request(exception=None):
    """Close session after each request"""
    db = g.pop('db', None)
    if db is not None:
        if exception:
            db.rollback()
        Session.remove()

# Route
@app.route('/users/<int:user_id>')
def get_user(user_id):
    user = g.db.get(User, user_id)
    return {"email": user.email}

# Alternative: Flask-SQLAlchemy handles this automatically
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)
\`\`\`

### Django Integration

\`\`\`python
"""
Using SQLAlchemy with Django
"""

# Django has its own ORM, but you can use SQLAlchemy alongside

from django.conf import settings

# Engine from Django settings
engine = create_engine(settings.DATABASES['default']['ENGINE'])
SessionLocal = sessionmaker(bind=engine)

# Middleware for session management
class SQLAlchemyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        request.db = SessionLocal()
        try:
            response = self.get_response(request)
            request.db.commit()
            return response
        except Exception:
            request.db.rollback()
            raise
        finally:
            request.db.close()

# View
def user_detail(request, user_id):
    user = request.db.get(User, user_id)
    return JsonResponse({"email": user.email})
\`\`\`

---

## Session Best Practices

### Pattern 1: One Session Per Request

\`\`\`python
"""
Web Application: One Session Per HTTP Request
"""

# GOOD: New session per request
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.execute(select(User)).scalars().all()

# BAD: Reusing session across requests
global_session = SessionLocal()  # DON'T DO THIS

@app.get("/users")
def get_users():
    return global_session.execute(select(User)).scalars().all()
# Problems: stale data, thread safety, connection leaks
\`\`\`

### Pattern 2: Short-Lived Sessions

\`\`\`python
"""
Keep Sessions Short-Lived
"""

# GOOD: Create, use, close
def process_user(user_id):
    with get_db_session() as session:
        user = session.get(User, user_id)
        user.last_seen = datetime.utcnow()
        # Session closed automatically

# BAD: Long-lived session
session = SessionLocal()
for user_id in user_ids:  # Thousands of users
    user = session.get(User, user_id)
    process(user)
# Session holds objects in memory, grows indefinitely
session.close()

# GOOD: Batch processing with session reset
session = SessionLocal()
for i, user_id in enumerate(user_ids):
    user = session.get(User, user_id)
    process(user)
    
    if i % 1000 == 0:
        session.commit()
        session.expire_all()  # Clear memory
\`\`\`

### Pattern 3: Transaction Boundaries

\`\`\`python
"""
Explicit Transaction Boundaries
"""

# GOOD: Transaction per business operation
def transfer_money(from_id, to_id, amount):
    with get_db_session() as session:
        from_account = session.get(Account, from_id)
        to_account = session.get(Account, to_id)
        
        from_account.balance -= amount
        to_account.balance += amount
        
        # Both updates in same transaction
        # Auto-commits on exit

# BAD: Implicit transactions
from_account.balance -= amount  # Transaction 1
session.commit()
to_account.balance += amount    # Transaction 2
session.commit()
# Money could be lost if crash between commits!
\`\`\`

---

## Object States and Lifecycle

### Expiring and Refreshing

\`\`\`python
"""
Managing Object Freshness
"""

# Expire: Mark object stale (reload on next access)
session.expire(user)
print(user.email)  # Triggers SELECT to reload

# Expire all: Clear session cache
session.expire_all()

# Refresh: Reload immediately
session.refresh(user)

# expire_on_commit (default=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=True)

session = SessionLocal()
user = session.get(User, 1)
print(user.email)  # "old@example.com"

session.commit()
# Objects expired due to expire_on_commit=True

print(user.email)  # Triggers SELECT to reload
# Ensures we don't have stale data after commit
\`\`\`

### Merge and Detached Objects

\`\`\`python
"""
Working with Detached Objects
"""

# Scenario: Object from another session or thread
def get_user_detached(user_id):
    session = SessionLocal()
    user = session.get(User, user_id)
    session.close()  # User now detached
    return user

# Use merge to reattach
user = get_user_detached(1)
# user is detached

session = SessionLocal()
user = session.merge(user)  # Now attached to this session
user.email = "new@example.com"
session.commit()

# merge behavior:
# - If object exists in session: update it
# - If not: SELECT from database, update, add to session
# - Returns persistent object (original still detached)
\`\`\`

---

## Memory Management

### Session Memory Growth

\`\`\`python
"""
Preventing Session Memory Issues
"""

# Problem: Session holds references to all loaded objects
session = SessionLocal()
for i in range(100000):
    user = session.get(User, i)
    process(user)
# Session holds 100,000 User objects in memory!

# Solution 1: expire_all periodically
session = SessionLocal()
for i in range(100000):
    user = session.get(User, i)
    process(user)
    
    if i % 1000 == 0:
        session.expire_all()  # Clear identity map
        session.commit()      # Commit changes

# Solution 2: expunge processed objects
session = SessionLocal()
for i in range(100000):
    user = session.get(User, i)
    process(user)
    session.expunge(user)  # Remove from session

# Solution 3: Batch with new sessions
for batch_start in range(0, 100000, 1000):
    with get_db_session() as session:
        users = session.execute(
            select(User).offset(batch_start).limit(1000)
        ).scalars().all()
        for user in users:
            process(user)
    # Session closed, memory freed
\`\`\`

---

## Summary

### Key Takeaways

✅ **Session per request**: Create new session for each web request  
✅ **Short-lived sessions**: Create, use, close quickly  
✅ **Context managers**: Use \`with\` for automatic cleanup  
✅ **Dependency injection**: Pass sessions as function parameters  
✅ **Transaction boundaries**: One transaction per business operation  
✅ **expire_on_commit=True**: Prevent stale data (default)  
✅ **Avoid global sessions**: Thread safety and memory issues

### Session Lifecycle

\`\`\`
1. Create: SessionLocal()
2. Add objects: session.add()
3. Flush: session.flush() (optional, sends to DB)
4. Commit: session.commit() (flush + commit transaction)
5. Close: session.close() (return connection to pool)
\`\`\`

### Object States

- **Transient**: Not in session, no DB identity
- **Pending**: In session, will INSERT on flush
- **Persistent**: In session, has DB identity
- **Detached**: Was in session, now removed

### Production Patterns

✅ **FastAPI**: Dependency injection with \`Depends(get_db)\`  
✅ **Flask**: Scoped session with before/teardown request  
✅ **Batch jobs**: New session per batch, expire_all periodically  
✅ **Long-running**: expire_all to free memory  
✅ **Async**: Use AsyncSession (separate session class)

### Next Steps

In the next section, we'll explore **Alembic: Database Migrations**: version control for schemas, auto-generating migrations, and zero-downtime deployment strategies.
`,
};
