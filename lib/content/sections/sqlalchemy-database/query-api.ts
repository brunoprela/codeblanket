export const queryApi = {
  title: 'Query API Deep Dive',
  id: 'query-api',
  content: `
# Query API Deep Dive

## Introduction

Querying is the heart of database interaction. SQLAlchemy\'s query API is powerful, expressive, and type-safe. Mastering queries is essential for building efficient applications.

In this section, you'll learn:
- SELECT queries with modern 2.0 syntax
- Filtering and conditional logic
- Ordering and pagination
- Joins (inner, outer, explicit)
- Subqueries and CTEs
- Aggregations and GROUP BY
- Window functions
- Union and set operations
- Query optimization techniques

---

## Basic SELECT Queries

### Simple SELECT

\`\`\`python
"""
Basic SELECT Queries
"""

from sqlalchemy import select

# Select all columns
stmt = select(User)
users = session.execute (stmt).scalars().all()

# Select specific columns
stmt = select(User.email, User.username)
results = session.execute (stmt).all()  # Returns tuples

# Select single object
stmt = select(User).where(User.id == 1)
user = session.execute (stmt).scalar_one()  # Raises if not found
user = session.execute (stmt).scalar_one_or_none()  # Returns None if not found

# Select first result
stmt = select(User).where(User.is_active == True)
user = session.execute (stmt).scalars().first()  # Returns first or None
\`\`\`

### Result Methods

\`\`\`python
"""
Different Ways to Fetch Results
"""

stmt = select(User)

# scalars() - extract first column (returns entities)
users = session.execute (stmt).scalars().all()  # List[User]
user = session.execute (stmt).scalars().first()  # User | None
user = session.execute (stmt).scalar_one()  # User (raises if not exactly 1)

# all() - returns Row objects (tuples)
rows = session.execute (stmt).all()  # List[Row]
for row in rows:
    print(row[0], row.User, row._mapping)  # Multiple ways to access

# fetchmany (size) - batch fetch
result = session.execute (stmt)
while batch := result.scalars().fetchmany(100):
    process_batch (batch)

# fetchone() - one at a time
result = session.execute (stmt)
while row := result.scalars().fetchone():
    process_row (row)
\`\`\`

---

## Filtering with WHERE

### Basic Filtering

\`\`\`python
"""
WHERE Clauses
"""

# Equality
stmt = select(User).where(User.email == "test@example.com")

# Inequality
stmt = select(User).where(User.age >= 18)
stmt = select(User).where(User.status != "banned")

# Multiple conditions (AND)
stmt = select(User).where(
    User.is_active == True,
    User.age >= 18
)

# Alternative AND syntax
from sqlalchemy import and_
stmt = select(User).where (and_(User.is_active == True, User.age >= 18))

# OR conditions
from sqlalchemy import or_
stmt = select(User).where (or_(User.role == "admin", User.role == "moderator"))

# NOT
from sqlalchemy import not_
stmt = select(User).where (not_(User.is_banned))

# IN
stmt = select(User).where(User.role.in_(["admin", "moderator"]))

# NOT IN
stmt = select(User).where(User.role.not_in(["banned", "suspended"]))

# LIKE
stmt = select(User).where(User.email.like("%@example.com"))
stmt = select(User).where(User.email.ilike("%@EXAMPLE.COM"))  # Case-insensitive

# BETWEEN
stmt = select(User).where(User.age.between(18, 65))

# IS NULL / IS NOT NULL
stmt = select(User).where(User.deleted_at.is_(None))
stmt = select(User).where(User.deleted_at.is_not(None))
\`\`\`

### Dynamic Filtering

\`\`\`python
"""
Build Queries Dynamically
"""

def get_users (email: str | None = None, min_age: int | None = None, is_active: bool | None = None):
    """Build query dynamically based on provided filters"""
    stmt = select(User)
    
    if email:
        stmt = stmt.where(User.email == email)
    
    if min_age:
        stmt = stmt.where(User.age >= min_age)
    
    if is_active is not None:
        stmt = stmt.where(User.is_active == is_active)
    
    return session.execute (stmt).scalars().all()

# Usage
users = get_users (min_age=18, is_active=True)
\`\`\`

---

## Ordering and Limiting

\`\`\`python
"""
ORDER BY and LIMIT
"""

# Order by single column
stmt = select(User).order_by(User.created_at)  # ASC
stmt = select(User).order_by(User.created_at.desc())  # DESC

# Order by multiple columns
stmt = select(User).order_by(User.last_name, User.first_name)

# NULL handling
from sqlalchemy import nullsfirst, nullslast
stmt = select(User).order_by (nullslast(User.deleted_at))

# LIMIT
stmt = select(User).limit(10)

# OFFSET (pagination)
stmt = select(User).offset(20).limit(10)  # Skip 20, take 10

# Combined
stmt = (
    select(User)
    .where(User.is_active == True)
    .order_by(User.created_at.desc())
    .limit(100)
)
\`\`\`

### Pagination Helper

\`\`\`python
"""
Production Pagination Pattern
"""

from typing import Generic, TypeVar, List
from pydantic import BaseModel

T = TypeVar('T')

class Page(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int

def paginate (stmt, page: int = 1, page_size: int = 20) -> Page:
    """Paginate query results"""
    
    # Count total (without limit/offset)
    count_stmt = select (func.count()).select_from (stmt.subquery())
    total = session.execute (count_stmt).scalar()
    
    # Get page of results
    stmt = stmt.offset((page - 1) * page_size).limit (page_size)
    items = session.execute (stmt).scalars().all()
    
    return Page(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size
    )

# Usage
page = paginate (select(User).where(User.is_active == True), page=2, page_size=50)
\`\`\`

---

## Joins

### Inner Join

\`\`\`python
"""
INNER JOIN
"""

# Implicit join (using relationship)
stmt = select(User).join(User.posts)

# Explicit join
stmt = select(User).join(Post, User.id == Post.user_id)

# Select from both tables
stmt = select(User, Post).join(Post, User.id == Post.user_id)

for user, post in session.execute (stmt):
    print(f"{user.email}: {post.title}")

# Join with filter
stmt = (
    select(User)
    .join(User.posts)
    .where(Post.published == True)
)
\`\`\`

### Outer Join

\`\`\`python
"""
LEFT OUTER JOIN
"""

# Get users even if they have no posts
stmt = select(User).outerjoin(User.posts)

# Count posts per user (including users with 0 posts)
from sqlalchemy import func
stmt = (
    select(User, func.count(Post.id).label('post_count'))
    .outerjoin(User.posts)
    .group_by(User.id)
)

for user, post_count in session.execute (stmt):
    print(f"{user.email}: {post_count} posts")
\`\`\`

### Multiple Joins

\`\`\`python
"""
Joining Multiple Tables
"""

# User -> Post -> Comment
stmt = (
    select(User)
    .join(User.posts)
    .join(Post.comments)
    .where(Comment.text.like("%great%"))
)

# Complex multi-table query
stmt = (
    select(User.username, Post.title, Comment.text)
    .join(Post, User.id == Post.user_id)
    .join(Comment, Post.id == Comment.post_id)
    .where(User.is_active == True)
    .order_by(Post.created_at.desc())
)
\`\`\`

---

## Aggregations

\`\`\`python
"""
Aggregate Functions
"""

from sqlalchemy import func

# COUNT
stmt = select (func.count(User.id))
count = session.execute (stmt).scalar()

# COUNT with condition
stmt = select (func.count(User.id)).where(User.is_active == True)

# SUM
stmt = select (func.sum(Order.total))
total_revenue = session.execute (stmt).scalar()

# AVG
stmt = select (func.avg(Product.price))
avg_price = session.execute (stmt).scalar()

# MIN / MAX
stmt = select (func.min(User.created_at), func.max(User.created_at))
first_user, last_user = session.execute (stmt).one()

# Multiple aggregates
stmt = select(
    func.count(Order.id).label('order_count'),
    func.sum(Order.total).label('total_revenue'),
    func.avg(Order.total).label('avg_order_value')
)
result = session.execute (stmt).one()
\`\`\`

### GROUP BY

\`\`\`python
"""
GROUP BY Queries
"""

# Group by single column
stmt = (
    select(User.city, func.count(User.id).label('user_count'))
    .group_by(User.city)
    .order_by (func.count(User.id).desc())
)

for city, count in session.execute (stmt):
    print(f"{city}: {count} users")

# Group by multiple columns
stmt = (
    select(
        User.country,
        User.city,
        func.count(User.id).label('count')
    )
    .group_by(User.country, User.city)
)

# HAVING (filter on aggregates)
stmt = (
    select(User.city, func.count(User.id).label('count'))
    .group_by(User.city)
    .having (func.count(User.id) > 100)
)
\`\`\`

### Real-World Analytics Query

\`\`\`python
"""
Complex Analytics Query
"""

from sqlalchemy import extract, func, case

# Daily signups for last 30 days
stmt = (
    select(
        func.date_trunc('day', User.created_at).label('date'),
        func.count(User.id).label('signups'),
        func.count(User.id).filter(User.email.like('%@gmail.com')).label('gmail_users'),
        func.avg(User.age).label('avg_age')
    )
    .where(User.created_at >= func.now() - func.interval('30 days'))
    .group_by (func.date_trunc('day', User.created_at))
    .order_by('date')
)

for row in session.execute (stmt):
    print(f"{row.date}: {row.signups} signups, {row.gmail_users} Gmail, avg age {row.avg_age:.1f}")
\`\`\`

---

## Subqueries

\`\`\`python
"""
Subqueries
"""

# Scalar subquery (single value)
subq = select (func.avg(User.age)).scalar_subquery()
stmt = select(User).where(User.age > subq)  # Users older than average

# Correlated subquery
subq = (
    select (func.count(Post.id))
    .where(Post.user_id == User.id)
    .correlate(User)
    .scalar_subquery()
)
stmt = select(User.username, subq.label('post_count'))

# Subquery as table
subq = (
    select(Post.user_id, func.count(Post.id).label('post_count'))
    .group_by(Post.user_id)
    .subquery()
)
stmt = (
    select(User.username, subq.c.post_count)
    .join (subq, User.id == subq.c.user_id)
)
\`\`\`

### Common Table Expressions (CTEs)

\`\`\`python
"""
CTEs (WITH clauses)
"""

# Basic CTE
cte = (
    select(User.city, func.count(User.id).label('count'))
    .group_by(User.city)
    .cte('city_counts')
)
stmt = select (cte).where (cte.c.count > 1000)

# Recursive CTE (hierarchical data)
# Get all children of a category
cte = (
    select(Category.id, Category.parent_id, Category.name)
    .where(Category.id == 1)
    .cte('category_tree', recursive=True)
)

cte = cte.union_all(
    select(Category.id, Category.parent_id, Category.name)
    .join (cte, Category.parent_id == cte.c.id)
)

stmt = select (cte)
\`\`\`

---

## Window Functions

\`\`\`python
"""
Window Functions
"""

from sqlalchemy import over, func

# ROW_NUMBER
stmt = select(
    User.username,
    User.created_at,
    func.row_number().over (order_by=User.created_at).label('row_num')
)

# RANK
stmt = select(
    User.username,
    User.score,
    func.rank().over (order_by=User.score.desc()).label('rank')
)

# Partition by
stmt = select(
    User.city,
    User.username,
    User.age,
    func.avg(User.age).over (partition_by=User.city).label('city_avg_age')
)

# LAG / LEAD
stmt = select(
    Order.created_at,
    Order.total,
    func.lag(Order.total).over (order_by=Order.created_at).label('previous_order'),
    func.lead(Order.total).over (order_by=Order.created_at).label('next_order')
)

# NTILE (distribute into N buckets)
stmt = select(
    User.email,
    User.score,
    func.ntile(4).over (order_by=User.score).label('quartile')
)
\`\`\`

---

## Union and Set Operations

\`\`\`python
"""
UNION, INTERSECT, EXCEPT
"""

# UNION (removes duplicates)
stmt1 = select(User.email).where(User.city == "New York")
stmt2 = select(User.email).where(User.age > 30)
stmt = stmt1.union (stmt2)

# UNION ALL (keeps duplicates, faster)
stmt = stmt1.union_all (stmt2)

# INTERSECT (common to both)
stmt = stmt1.intersect (stmt2)

# EXCEPT (in first, not in second)
stmt = stmt1.except_(stmt2)
\`\`\`

---

## Query Optimization

### EXPLAIN Analysis

\`\`\`python
"""
Analyze Query Performance
"""

stmt = select(User).where(User.email == "test@example.com")

# Get compiled SQL
print(stmt.compile (compile_kwargs={"literal_binds": True}))

# EXPLAIN (PostgreSQL)
from sqlalchemy import text
explain_stmt = text (f"EXPLAIN ANALYZE {stmt}")
result = session.execute (explain_stmt)
for row in result:
    print(row)
\`\`\`

### Index Usage

\`\`\`python
"""
Ensure Queries Use Indexes
"""

# ✅ Uses index on email
stmt = select(User).where(User.email == "test@example.com")

# ❌ Function on column prevents index use
stmt = select(User).where (func.lower(User.email) == "test@example.com")

# ✅ Use functional index or computed column
# In migration: Index('idx_email_lower', func.lower(User.email))

# ✅ Composite index (query must use prefix)
# Index: (user_id, created_at)
stmt = select(Post).where(Post.user_id == 1)  # Uses index
stmt = select(Post).where(Post.user_id == 1, Post.created_at > date)  # Uses index
stmt = select(Post).where(Post.created_at > date)  # Doesn't use index!
\`\`\`

---

## Summary

### Key Takeaways

✅ **Modern syntax**: Use \`select()\` and \`session.execute()\` (2.0 style)  
✅ **scalars()**: Extracts entities from Row objects  
✅ **WHERE**: Chain conditions, use \`and_()\`, \`or_()\`, \`in_()\`  
✅ **Joins**: \`join()\` for inner, \`outerjoin()\` for left outer  
✅ **Aggregations**: \`func.count()\`, \`func.sum()\`, with \`group_by()\`  
✅ **Subqueries**: \`subquery()\` for complex queries  
✅ **CTEs**: Cleaner than subqueries, supports recursion  
✅ **Window functions**: Ranking, running totals, partitions  
✅ **Optimization**: Analyze with EXPLAIN, ensure index usage

### Best Practices

✅ Use \`select()\` instead of \`session.query()\`  
✅ Call \`scalars()\` for single entity queries  
✅ Avoid functions on indexed columns in WHERE  
✅ Use CTEs for complex queries (more readable)  
✅ Paginate with LIMIT/OFFSET, include total count  
✅ Use window functions for analytics  
✅ Always EXPLAIN complex queries  
✅ Composite indexes: query must use prefix

### Next Steps

In the next section, we'll explore **Advanced Filtering & Expressions**: JSON querying, array operations, full-text search, custom operators, and database-specific features.
`,
};
