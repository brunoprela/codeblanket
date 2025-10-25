export const sqlalchemyCore = {
  title: 'SQLAlchemy Core Concepts',
  id: 'sqlalchemy-core',
  content: `
# SQLAlchemy Core Concepts

## Introduction

SQLAlchemy is **two libraries in one**:
1. **SQLAlchemy Core**: SQL expression language and engine (SQL builder)
2. **SQLAlchemy ORM**: Object-relational mapping layer built on Core

Understanding Core concepts is essential for mastering the ORM. Even when using the ORM exclusively, you're leveraging Core underneath.

In this section, you'll learn:
- SQLAlchemy architecture (Core vs ORM layers)
- Engine creation and connection management
- Declarative base and table mappings
- Session lifecycle and patterns
- Transaction handling
- MetaData and reflection
- Connection pooling configuration

### SQLAlchemy 2.0 vs 1.x

**This curriculum focuses on SQLAlchemy 2.0+**, which introduced:
- ✅ Unified async/sync API
- ✅ Type annotations throughout
- ✅ Improved query syntax
- ✅ Better performance
- ✅ Cleaner separation of concerns

\`\`\`python
# Legacy 1.x style (still works in 2.0)
users = session.query(User).filter(User.email == "test@example.com").all()

# Modern 2.0 style (recommended)
from sqlalchemy import select
stmt = select(User).where(User.email == "test@example.com")
users = session.execute(stmt).scalars().all()
\`\`\`

---

## SQLAlchemy Architecture

### Layered Design

\`\`\`
┌─────────────────────────────────────┐
│    SQLAlchemy ORM (High Level)     │
│  - Session                          │
│  - Declarative Models               │
│  - Relationships                    │
│  - Query API                        │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   SQLAlchemy Core (Mid Level)      │
│  - SQL Expression Language          │
│  - Schema / Types                   │
│  - Engine                           │
│  - Connection Pooling               │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│      DB-API (Low Level)            │
│  - psycopg2, pymysql, etc          │
│  - Direct database communication   │
└─────────────────────────────────────┘
\`\`\`

### Core vs ORM

\`\`\`python
"""
Core Example (SQL Expression Language)
"""

from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

# Define table using Core
metadata = MetaData()
users = Table(
    'users',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String(255)),
)

# Create engine
engine = create_engine("postgresql://localhost/mydb")

# Execute SQL using Core
from sqlalchemy import select
with engine.connect() as conn:
    stmt = select(users).where(users.c.email == "test@example.com")
    result = conn.execute(stmt)
    for row in result:
        print(row.id, row.email)

"""
ORM Example (Object-Relational Mapping)
"""

from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255))

# Create engine and session
engine = create_engine("postgresql://localhost/mydb")
Session = sessionmaker(bind=engine)
session = Session()

# Query using ORM
users = session.execute(
    select(User).where(User.email == "test@example.com")
).scalars().all()

for user in users:
    print(user.id, user.email)
\`\`\`

**When to use Core vs ORM**:
- **ORM**: Application development, domain models, relationships (95% of cases)
- **Core**: Data migrations, ETL, direct SQL control, no ORM overhead

---

## Engine: The Database Connection

### Creating an Engine

\`\`\`python
"""
Engine Creation
"""

from sqlalchemy import create_engine

# PostgreSQL
engine = create_engine("postgresql://user:password@localhost:5432/mydb")
engine = create_engine("postgresql+psycopg2://user:password@localhost/mydb")
engine = create_engine("postgresql+asyncpg://user:password@localhost/mydb")  # Async

# MySQL
engine = create_engine("mysql://user:password@localhost/mydb")
engine = create_engine("mysql+pymysql://user:password@localhost/mydb")

# SQLite
engine = create_engine("sqlite:///./database.db")  # File
engine = create_engine("sqlite:///:memory:")       # In-memory

# With parameters
engine = create_engine(
    "postgresql://localhost/mydb",
    echo=True,              # Log all SQL statements
    pool_size=5,           # Connection pool size
    max_overflow=10,       # Extra connections when pool exhausted
    pool_timeout=30,       # Seconds to wait for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True,    # Test connections before using
)
\`\`\`

### Engine Configuration

\`\`\`python
"""
Production Engine Configuration
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

def create_db_engine():
    """Create production-ready database engine"""
    
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://localhost/mydb"
    )
    
    # Production settings
    engine = create_engine(
        database_url,
        
        # Connection Pool Settings
        poolclass=QueuePool,
        pool_size=10,              # Normal connections
        max_overflow=20,           # Burst capacity
        pool_timeout=30,           # Wait time for connection
        pool_recycle=3600,         # Recycle every hour (prevents stale connections)
        pool_pre_ping=True,        # Verify connection health before use
        
        # Query Settings
        echo=False,                # Don't log SQL in production
        echo_pool=False,           # Don't log pool events
        
        # Execution Settings
        isolation_level="READ COMMITTED",  # Transaction isolation
        
        # Performance
        execution_options={
            "compiled_cache_size": 500  # Cache compiled SQL statements
        }
    )
    
    return engine

# Create global engine
engine = create_db_engine()
\`\`\`

### Connection Management

\`\`\`python
"""
Using Connections
"""

from sqlalchemy import text

# Context manager (recommended)
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM users"))
    for row in result:
        print(row)
# Connection automatically returned to pool

# Explicit transaction
with engine.begin() as conn:
    conn.execute(text("INSERT INTO users (email) VALUES (:email)"), {"email": "test@example.com"})
    # Automatically commits on success, rolls back on exception

# Manual connection management (not recommended)
conn = engine.connect()
try:
    result = conn.execute(text("SELECT * FROM users"))
finally:
    conn.close()  # Must manually close
\`\`\`

---

## Declarative Base and Mappings

### Declarative Base

\`\`\`python
"""
Setting Up Declarative Base
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime

# Create base class for all models
Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    # Columns
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"

# Create all tables
Base.metadata.create_all(engine)
\`\`\`

### Modern Declarative Mapping (2.0)

\`\`\`python
"""
SQLAlchemy 2.0 Declarative Mapping with Type Annotations
"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String
from datetime import datetime
from typing import Optional

class Base(DeclarativeBase):
    """Base class for all models"""
    pass

class User(Base):
    """User model with type annotations"""
    __tablename__ = 'users'
    
    # Type-annotated columns
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    username: Mapped[str] = mapped_column(String(50))
    bio: Mapped[Optional[str]] = mapped_column(String(500))  # Nullable
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(email='{self.email}')>"
\`\`\`

### Table Configuration

\`\`\`python
"""
Advanced Table Configuration
"""

from sqlalchemy import Index, CheckConstraint, UniqueConstraint

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    age: Mapped[Optional[int]]
    status: Mapped[str] = mapped_column(String(20), default='active')
    
    # Table-level constraints
    __table_args__ = (
        # Indexes
        Index('idx_email_status', 'email', 'status'),
        
        # Unique constraints
        UniqueConstraint('email', name='uq_user_email'),
        
        # Check constraints
        CheckConstraint('age >= 0 AND age <= 150', name='check_age_range'),
        CheckConstraint("status IN ('active', 'inactive', 'banned')", name='check_status'),
        
        # Table options (PostgreSQL)
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )
\`\`\`

---

## Session Management

### Creating Sessions

\`\`\`python
"""
Session Factory Pattern
"""

from sqlalchemy.orm import sessionmaker, Session as SessionType

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,      # Manual commits (recommended)
    autoflush=True,        # Flush before queries
    expire_on_commit=True  # Expire instances after commit
)

# Create session
session = SessionLocal()

try:
    # Use session
    user = User(email="test@example.com")
    session.add(user)
    session.commit()
finally:
    session.close()
\`\`\`

### Session Context Manager

\`\`\`python
"""
Session as Context Manager
"""

from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Provide a transactional scope for database operations"""
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
    # Automatic commit on success, rollback on exception
\`\`\`

### Scoped Sessions (Thread-Local)

\`\`\`python
"""
Scoped Session for Web Applications
"""

from sqlalchemy.orm import scoped_session, sessionmaker

# Create thread-local session
SessionLocal = scoped_session(
    sessionmaker(bind=engine)
)

# Each thread gets its own session
def create_user(email: str):
    session = SessionLocal()  # Thread-local session
    user = User(email=email)
    session.add(user)
    session.commit()
    return user

# Remove session when done (important!)
SessionLocal.remove()
\`\`\`

---

## Transaction Handling

### Manual Transaction Control

\`\`\`python
"""
Explicit Transaction Management
"""

from sqlalchemy import select

session = SessionLocal()

# Start transaction (implicit)
user = User(email="test@example.com")
session.add(user)

# Query in same transaction
existing = session.execute(
    select(User).where(User.email == "test@example.com")
).scalar_one_or_none()

# Commit transaction
session.commit()

# Start new transaction
user.email = "newemail@example.com"
session.commit()

session.close()
\`\`\`

### Nested Transactions (Savepoints)

\`\`\`python
"""
Savepoints for Partial Rollback
"""

session = SessionLocal()

try:
    # Outer transaction
    user1 = User(email="user1@example.com")
    session.add(user1)
    
    # Create savepoint
    nested = session.begin_nested()
    
    try:
        # Inner transaction
        user2 = User(email="user2@example.com")
        session.add(user2)
        
        # Commit savepoint
        nested.commit()
    except Exception:
        # Rollback to savepoint (user1 preserved)
        nested.rollback()
    
    # Commit outer transaction
    session.commit()
except Exception:
    # Rollback entire transaction
    session.rollback()
finally:
    session.close()
\`\`\`

### Transaction Isolation Levels

\`\`\`python
"""
Setting Isolation Levels
"""

# Engine-level (all connections)
engine = create_engine(
    "postgresql://localhost/mydb",
    isolation_level="REPEATABLE READ"
)

# Connection-level
with engine.connect().execution_options(
    isolation_level="SERIALIZABLE"
) as conn:
    result = conn.execute(text("SELECT * FROM users"))

# Session-level
session = SessionLocal()
session.connection(execution_options={"isolation_level": "READ UNCOMMITTED"})
\`\`\`

**Isolation levels** (from least to most strict):
- \`READ UNCOMMITTED\`: Can read uncommitted changes (dirty reads)
- \`READ COMMITTED\`: Only read committed changes (default for PostgreSQL)
- \`REPEATABLE READ\`: Same data for repeated reads in transaction
- \`SERIALIZABLE\`: Full isolation, as if transactions run serially

---

## MetaData and Reflection

### MetaData: Schema Catalog

\`\`\`python
"""
MetaData Management
"""

from sqlalchemy import MetaData

# MetaData holds all table definitions
metadata = MetaData()

class User(Base):
    __tablename__ = 'users'
    # ...

# Access MetaData
print(Base.metadata.tables.keys())  # ['users', ...]

# Create all tables
Base.metadata.create_all(engine)

# Drop all tables (dangerous!)
Base.metadata.drop_all(engine)

# Create specific table
Base.metadata.tables['users'].create(engine)
\`\`\`

### Database Reflection

\`\`\`python
"""
Reflect Existing Database Schema
"""

from sqlalchemy import MetaData, Table

# Reflect all tables
metadata = MetaData()
metadata.reflect(bind=engine)

# Access reflected tables
users_table = metadata.tables['users']
print(users_table.columns.keys())  # ['id', 'email', ...]

# Reflect specific table
users = Table('users', metadata, autoload_with=engine)
print(users.c.email)  # Column('email', String(255), ...)

# Use reflected table in queries
from sqlalchemy import select
stmt = select(users).where(users.c.email == "test@example.com")
with engine.connect() as conn:
    result = conn.execute(stmt)
\`\`\`

### Automap: ORM from Existing Database

\`\`\`python
"""
Automatically Generate ORM Models from Database
"""

from sqlalchemy.ext.automap import automap_base

# Reflect database into ORM models
Base = automap_base()
Base.prepare(engine, reflect=True)

# Access generated classes
User = Base.classes.users
Post = Base.classes.posts

# Use like normal ORM models
session = SessionLocal()
users = session.query(User).all()
\`\`\`

---

## Connection Pooling Deep Dive

### Pool Types

\`\`\`python
"""
Different Pool Implementations
"""

from sqlalchemy.pool import (
    QueuePool,      # Default, thread-safe queue
    NullPool,       # No pooling, new connection each time
    StaticPool,     # Single connection for all threads
    SingletonThreadPool  # One connection per thread
)

# QueuePool (default, recommended for most cases)
engine = create_engine(
    "postgresql://localhost/mydb",
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
)

# NullPool (for serverless, AWS Lambda)
engine = create_engine(
    "postgresql://localhost/mydb",
    poolclass=NullPool
)

# StaticPool (for SQLite, in-memory databases)
engine = create_engine(
    "sqlite:///:memory:",
    poolclass=StaticPool
)
\`\`\`

### Pool Monitoring

\`\`\`python
"""
Monitor Connection Pool Health
"""

def get_pool_stats(engine):
    """Get connection pool statistics"""
    pool = engine.pool
    return {
        'size': pool.size(),              # Current connections
        'checked_in': pool.checkedin(),   # Available connections
        'checked_out': pool.checkedout(), # In-use connections
        'overflow': pool.overflow(),      # Overflow connections
        'total': pool.size() + pool.overflow()
    }

# Check pool status
stats = get_pool_stats(engine)
print(f"Pool: {stats['checked_out']}/{stats['total']} connections in use")

# Pool events for monitoring
from sqlalchemy import event

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    print(f"New connection: {id(dbapi_conn)}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    print(f"Connection checked out: {id(dbapi_conn)}")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    print(f"Connection checked in: {id(dbapi_conn)}")
\`\`\`

---

## Best Practices

### 1. Engine as Singleton

\`\`\`python
"""
Create engine once, reuse everywhere
"""

# ✅ Correct: Single engine instance
engine = create_engine("postgresql://localhost/mydb", pool_size=10)

def get_users():
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM users")).fetchall()

# ❌ Wrong: Creating engine per function
def get_users_wrong():
    engine = create_engine("postgresql://localhost/mydb")  # Don't do this!
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM users")).fetchall()
\`\`\`

### 2. Always Close Sessions

\`\`\`python
"""
Proper session cleanup
"""

# ✅ Context manager (preferred)
with get_db_session() as session:
    users = session.query(User).all()

# ✅ Try/finally
session = SessionLocal()
try:
    users = session.query(User).all()
finally:
    session.close()

# ❌ No cleanup (leaks connections)
session = SessionLocal()
users = session.query(User).all()  # Session never closed!
\`\`\`

### 3. Dependency Injection Pattern

\`\`\`python
"""
Inject session as dependency (FastAPI example)
"""

from typing import Generator

def get_db() -> Generator[SessionType, None, None]:
    """Database session dependency"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Use in endpoint
from fastapi import Depends

@app.get("/users")
def get_users(session: SessionType = Depends(get_db)):
    return session.query(User).all()
\`\`\`

---

## Summary

### Key Takeaways

✅ **SQLAlchemy has two layers**: Core (SQL builder) and ORM (object mapping)  
✅ **Engine**: Central point for database connections, create once and reuse  
✅ **Declarative Base**: Foundation for ORM models, use type annotations (2.0)  
✅ **Session**: UnitOfWork pattern, tracks changes, manages transactions  
✅ **Connection pooling**: Essential for production, configure pool_size and max_overflow  
✅ **Transactions**: Use context managers, commit/rollback explicitly  
✅ **MetaData**: Schema catalog, supports reflection from existing databases

### Production Checklist

✅ Create engine as application singleton  
✅ Configure connection pooling appropriately  
✅ Use context managers for sessions  
✅ Set proper isolation levels  
✅ Enable pool pre-ping for stale connections  
✅ Monitor pool usage  
✅ Use dependency injection for testability

### Next Steps

In the next section, we'll explore **Defining Models & Relationships**: table definitions, column types, foreign keys, and all relationship types (one-to-one, one-to-many, many-to-many, self-referential, polymorphic).
`,
};
