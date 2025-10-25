export const databaseFundamentals = {
  title: 'Database Fundamentals for Python',
  id: 'database-fundamentals',
  content: `
# Database Fundamentals for Python

## Introduction

Databases are the foundation of production applications. While NoSQL databases have their place, **relational databases (RDBMS)** remain the gold standard for transactional systems, data integrity, and complex queries. Understanding how Python interacts with databases is essential for building production-grade applications.

In this section, you'll learn:
- Core RDBMS concepts and why they matter
- Python database drivers and the DB-API specification
- When and why to use an ORM
- SQLAlchemy vs alternatives
- When to use raw SQL instead

### Why Relational Databases?

Modern production systems require:
- **ACID properties**: Atomicity, Consistency, Isolation, Durability
- **Data integrity**: Foreign keys, constraints, transactions
- **Complex queries**: JOINs, aggregations, subqueries
- **Scalability**: Indexes, query optimization, replication
- **Reliability**: Battle-tested over decades

**Real-world truth**: Every major tech company uses relational databases for core transactional data. Even "NoSQL companies" use PostgreSQL for critical data.

---

## RDBMS Core Concepts

### Tables, Rows, and Columns

\`\`\`sql
-- Table definition
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Rows are individual records
INSERT INTO users (email) VALUES ('user@example.com');

-- Columns are attributes with specific types
-- id: integer, email: string, created_at: timestamp, is_active: boolean
\`\`\`

### Primary Keys

**Primary Key**: Unique identifier for each row

\`\`\`sql
-- Single column primary key
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- Auto-incrementing integer
    email VARCHAR(255)
);

-- Composite primary key (multiple columns)
CREATE TABLE user_roles (
    user_id INTEGER,
    role_id INTEGER,
    PRIMARY KEY (user_id, role_id)
);

-- UUID as primary key (recommended for distributed systems)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER,
    token VARCHAR(255)
);
\`\`\`

### Foreign Keys and Relationships

**Foreign Key**: Column that references a primary key in another table

\`\`\`sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title VARCHAR(255),
    content TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
\`\`\`

**ON DELETE behaviors**:
- \`CASCADE\`: Delete related rows
- \`SET NULL\`: Set foreign key to NULL
- \`RESTRICT\`: Prevent deletion if references exist
- \`NO ACTION\`: Similar to RESTRICT

### Indexes

**Indexes** speed up queries by creating data structures for fast lookups:

\`\`\`sql
-- Single column index
CREATE INDEX idx_users_email ON users (email);

-- Composite index (order matters!)
CREATE INDEX idx_posts_user_created ON posts (user_id, created_at);

-- Unique index
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- Partial index (PostgreSQL)
CREATE INDEX idx_active_users ON users (email) WHERE is_active = TRUE;
\`\`\`

**Index trade-offs**:
- ✅ Faster SELECT queries
- ❌ Slower INSERT/UPDATE/DELETE
- ❌ More disk space
- **Rule**: Index foreign keys and frequently queried columns

### Transactions and ACID

**Transaction**: Group of operations that succeed or fail together

\`\`\`sql
BEGIN;

-- Transfer money between accounts
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- If anything fails, ROLLBACK undoes everything
-- If successful, COMMIT makes changes permanent
COMMIT;
\`\`\`

**ACID Properties**:
- **Atomicity**: All operations succeed or all fail
- **Consistency**: Database constraints always maintained
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed data survives crashes

---

## Python Database Drivers

### DB-API 2.0 Specification

Python\'s standard interface for database access (PEP 249):

\`\`\`python
"""
DB-API 2.0 Example (Raw SQL)
"""

import psycopg2  # PostgreSQL driver

# Connect to database
conn = psycopg2.connect(
    dbname="mydb",
    user="postgres",
    password="secret",
    host="localhost",
    port=5432
)

# Create cursor
cursor = conn.cursor()

try:
    # Execute query
    cursor.execute("SELECT id, email FROM users WHERE is_active = %s", (True,))
    
    # Fetch results
    users = cursor.fetchall()
    for user_id, email in users:
        print(f"User {user_id}: {email}")
    
    # Insert with transaction
    cursor.execute(
        "INSERT INTO users (email) VALUES (%s) RETURNING id",
        ("newuser@example.com",)
    )
    new_id = cursor.fetchone()[0]
    
    # Commit transaction
    conn.commit()
    print(f"Created user with ID: {new_id}")
    
except Exception as e:
    # Rollback on error
    conn.rollback()
    print(f"Error: {e}")
    
finally:
    # Always close resources
    cursor.close()
    conn.close()
\`\`\`

### Popular Python Database Drivers

\`\`\`python
"""
PostgreSQL Drivers
"""

# psycopg2: Most popular, C-based, fast
import psycopg2
conn = psycopg2.connect("postgresql://user:pass@localhost/dbname")

# psycopg3 (psycopg): Modern rewrite, async support
import psycopg
conn = psycopg.connect("postgresql://user:pass@localhost/dbname")

# asyncpg: High-performance async driver
import asyncpg
conn = await asyncpg.connect("postgresql://user:pass@localhost/dbname")

"""
MySQL Drivers
"""

# pymysql: Pure Python, easy to install
import pymysql
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="secret",
    database="mydb"
)

# mysqlclient: C-based, faster
import MySQLdb
conn = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="secret",
    db="mydb"
)

# aiomysql: Async MySQL
import aiomysql
conn = await aiomysql.connect(
    host="localhost",
    user="root",
    password="secret",
    db="mydb"
)

"""
SQLite (Built-in)
"""

import sqlite3
conn = sqlite3.connect("database.db")  # File-based
conn = sqlite3.connect(":memory:")      # In-memory
\`\`\`

### Connection Management Best Practices

\`\`\`python
"""
Proper Connection Handling
"""

from contextlib import contextmanager
import psycopg2

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = psycopg2.connect(
        dbname="mydb",
        user="postgres",
        password="secret",
        host="localhost"
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
# Connection automatically closed, committed or rolled back
\`\`\`

---

## Why Use an ORM?

### Raw SQL vs ORM

**Raw SQL Approach**:
\`\`\`python
# Verbose, error-prone, no type safety
cursor.execute(
    "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
    (email, password_hash)
)
conn.commit()

# Manual result mapping
cursor.execute("SELECT id, email, created_at FROM users WHERE id = %s", (user_id,))
row = cursor.fetchone()
user = {
    'id': row[0],
    'email': row[1],
    'created_at': row[2]
}
\`\`\`

**ORM Approach** (SQLAlchemy):
\`\`\`python
# Clean, Pythonic, type-safe
user = User (email=email, password_hash=password_hash)
session.add (user)
session.commit()

# Automatic mapping to objects
user = session.query(User).filter_by (id=user_id).first()
print(user.email, user.created_at)
\`\`\`

### ORM Benefits

✅ **Productivity**: Write less code, focus on business logic  
✅ **Type Safety**: IDE autocomplete, catch errors early  
✅ **Database Agnostic**: Switch databases with minimal code changes  
✅ **Relationship Management**: Automatic JOIN handling  
✅ **Query Composition**: Build complex queries programmatically  
✅ **Migration Management**: Schema version control  
✅ **Security**: Automatic SQL injection prevention  
✅ **Testing**: Easier to mock and test

### ORM Limitations

❌ **Performance Overhead**: Slight overhead vs raw SQL  
❌ **Learning Curve**: Must understand both SQL and ORM  
❌ **Complex Queries**: Some advanced SQL hard to express  
❌ **Abstraction Leaks**: Must understand underlying SQL  
❌ **Debugging**: Harder to see generated SQL

**Production truth**: Use ORM for 95% of queries, raw SQL for complex analytics or performance-critical paths.

---

## SQLAlchemy vs Alternatives

### SQLAlchemy

**The industry standard ORM for Python**

\`\`\`python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)

engine = create_engine("postgresql://localhost/mydb")
Session = sessionmaker (bind=engine)
session = Session()

# Query users
users = session.query(User).filter(User.email.like("%@example.com")).all()
\`\`\`

**Strengths**:
- ✅ Most mature and feature-complete
- ✅ Excellent documentation
- ✅ Large ecosystem and community
- ✅ Supports all major databases
- ✅ Both ORM and Core (SQL builder)
- ✅ Production-proven at scale

**Use when**: Building production applications

### Django ORM

**Integrated with Django framework**

\`\`\`python
from django.db import models

class User (models.Model):
    email = models.EmailField (unique=True)
    created_at = models.DateTimeField (auto_now_add=True)

# Query users
users = User.objects.filter (email__endswith="@example.com")
\`\`\`

**Strengths**:
- ✅ Simple and intuitive API
- ✅ Integrated with Django admin
- ✅ Built-in migrations
- ✅ Great for rapid development

**Limitations**:
- ❌ Tied to Django framework
- ❌ Less flexible than SQLAlchemy
- ❌ Harder to use complex queries

**Use when**: Building Django applications

### Peewee

**Lightweight ORM, simpler than SQLAlchemy**

\`\`\`python
from peewee import *

db = PostgresqlDatabase('mydb', user='postgres')

class User(Model):
    email = CharField (unique=True)
    
    class Meta:
        database = db

# Query users
users = User.select().where(User.email.endswith("@example.com"))
\`\`\`

**Strengths**:
- ✅ Simple API, easy to learn
- ✅ Lightweight, minimal dependencies
- ✅ Good for small to medium projects

**Limitations**:
- ❌ Less feature-complete
- ❌ Smaller community
- ❌ Limited advanced features

**Use when**: Prototyping or small applications

### Tortoise ORM

**Async ORM inspired by Django ORM**

\`\`\`python
from tortoise import fields
from tortoise.models import Model

class User(Model):
    id = fields.IntField (pk=True)
    email = fields.CharField (max_length=255, unique=True)

# Async queries
users = await User.filter (email__endswith="@example.com")
\`\`\`

**Strengths**:
- ✅ Async/await support
- ✅ Django-like API
- ✅ Good FastAPI integration

**Limitations**:
- ❌ Newer, less mature
- ❌ Smaller ecosystem
- ❌ Limited advanced features

**Use when**: Building async Python applications

### Comparison Matrix

| Feature | SQLAlchemy | Django ORM | Peewee | Tortoise |
|---------|-----------|-----------|--------|----------|
| Maturity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Learning Curve | Medium-High | Easy | Easy | Easy |
| Async Support | ✅ (2.0+) | ❌ | ❌ | ✅ |
| Flexibility | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Community | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Database Support | All | All | Most | Most |
| Migrations | Alembic | Built-in | Manual | Aerich |
| Production Ready | ✅ | ✅ | ✅ | ⚠️ |

---

## When to Use Raw SQL

ORMs are powerful, but sometimes raw SQL is better:

### 1. Complex Analytics Queries

\`\`\`python
"""
Complex window functions, CTEs
"""

from sqlalchemy import text

query = text("""
    WITH monthly_revenue AS (
        SELECT 
            DATE_TRUNC('month', created_at) as month,
            SUM(amount) as revenue,
            LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', created_at)) as prev_revenue
        FROM orders
        GROUP BY month
    )
    SELECT 
        month,
        revenue,
        revenue - prev_revenue as growth,
        (revenue - prev_revenue) / prev_revenue * 100 as growth_pct
    FROM monthly_revenue
    WHERE prev_revenue IS NOT NULL
    ORDER BY month DESC
""")

result = session.execute (query).fetchall()
\`\`\`

### 2. Performance-Critical Bulk Operations

\`\`\`python
"""
Bulk inserts/updates with COPY
"""

# SQLAlchemy way (slower)
users = [User (email=f"user{i}@example.com") for i in range(10000)]
session.bulk_save_objects (users)

# Raw SQL with COPY (10-100x faster)
import io
data = io.StringIO()
for i in range(10000):
    data.write (f"user{i}@example.com\\n")
data.seek(0)

cursor.copy_from (data, 'users', columns=('email',))
\`\`\`

### 3. Database-Specific Features

\`\`\`python
"""
PostgreSQL full-text search
"""

query = text("""
    SELECT * FROM articles
    WHERE to_tsvector('english', title || ' ' || content) 
          @@ to_tsquery('english', :search_term)
    ORDER BY ts_rank(
        to_tsvector('english', title || ' ' || content),
        to_tsquery('english', :search_term)
    ) DESC
""")

results = session.execute (query, {"search_term": "python & database"}).fetchall()
\`\`\`

### 4. Schema Migrations

\`\`\`python
"""
Complex schema changes best done in raw SQL
"""

# In Alembic migration
def upgrade():
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_users_email_gin 
        ON users USING gin (email gin_trgm_ops);
    """)
\`\`\`

### Best Practice: Hybrid Approach

\`\`\`python
"""
Use ORM for CRUD, raw SQL for complex queries
"""

class UserRepository:
    def create_user (self, email: str) -> User:
        """Use ORM for simple operations"""
        user = User (email=email)
        session.add (user)
        session.commit()
        return user
    
    def get_user_analytics (self) -> List[dict]:
        """Use raw SQL for complex analytics"""
        query = text("""
            SELECT 
                DATE_TRUNC('day', created_at) as date,
                COUNT(*) as signups,
                COUNT(*) FILTER (WHERE is_active) as active
            FROM users
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """)
        return session.execute (query).mappings().all()
\`\`\`

---

## Connection Pooling

**Connection pooling** reuses database connections for better performance:

\`\`\`python
"""
SQLAlchemy Connection Pooling
"""

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://localhost/mydb",
    poolclass=QueuePool,
    pool_size=5,          # Keep 5 connections open
    max_overflow=10,      # Allow 10 additional connections when busy
    pool_timeout=30,      # Wait 30s for connection before timing out
    pool_recycle=3600,    # Recycle connections after 1 hour
    pool_pre_ping=True    # Verify connections before use
)

# Each session gets a connection from the pool
Session = sessionmaker (bind=engine)
session = Session()

# Use session...

# Connection returned to pool when session closed
session.close()
\`\`\`

**Why connection pooling matters**:
- Creating database connections is expensive (100-1000ms)
- Connection pools reuse connections (< 1ms)
- Essential for production performance
- Prevents connection exhaustion

---

## Summary

### Key Takeaways

✅ **RDBMS fundamentals**: Tables, keys, indexes, transactions, ACID  
✅ **Python DB-API**: Standard interface for database access  
✅ **Database drivers**: psycopg2 (PostgreSQL), pymysql (MySQL), sqlite3  
✅ **ORMs**: SQLAlchemy for production, Django ORM for Django apps  
✅ **Hybrid approach**: ORM for CRUD, raw SQL for complex queries  
✅ **Connection pooling**: Essential for production performance

### When to Use What

**SQLAlchemy ORM**: 95% of application code  
**Raw SQL**: Complex analytics, bulk operations, database-specific features  
**Django ORM**: When using Django framework  
**Peewee**: Small projects, prototyping  
**Tortoise**: Async Python applications

### Next Steps

In the next section, we'll dive deep into **SQLAlchemy Core Concepts**: engines, sessions, declarative base, and connection management. You'll learn how to set up SQLAlchemy properly for production applications.

**Production mindset**: Understand both the ORM and the underlying SQL. ORMs are tools that enhance productivity, not magic that replaces SQL knowledge.
`,
};
