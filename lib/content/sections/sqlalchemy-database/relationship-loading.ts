export const relationshipLoading = {
  title: 'Relationship Loading Strategies',
  id: 'relationship-loading',
  content: `
# Relationship Loading Strategies

## Introduction

The N+1 query problem is the most common performance issue in ORM applications. Understanding relationship loading strategies is critical for building fast, scalable applications. SQLAlchemy provides multiple loading strategies, each optimized for different scenarios.

In this section, you'll master:
- The N+1 query problem and how to detect it
- Lazy loading (default behavior)
- Eager loading strategies (joinedload, selectinload, subqueryload)
- Relationship loading options and when to use each
- Avoiding common performance pitfalls
- Advanced loading techniques
- Production optimization patterns

### Why Loading Strategies Matter

**Real-world impact**: A single N+1 problem can make your API 100x slower. Loading 100 users with their posts can execute 101 queries (N+1) instead of 2. Understanding loading strategies is non-negotiable for production applications.

---

## The N+1 Query Problem

### Understanding N+1

\`\`\`python
"""
The N+1 Problem Demonstration
"""

# BAD: N+1 queries
users = session.execute (select(User).limit(10)).scalars().all()
# Query 1: SELECT * FROM users LIMIT 10

for user in users:
    print(f"{user.email}: {len (user.posts)} posts")
    # Query 2-11: SELECT * FROM posts WHERE user_id = ?
    # Executes 10 more queries (one per user)
    
# Total: 1 + 10 = 11 queries for 10 users
# With 1000 users: 1 + 1000 = 1001 queries! 🔥

# GOOD: 2 queries with eager loading
stmt = select(User).options (selectinload(User.posts)).limit(10)
users = session.execute (stmt).scalars().all()
# Query 1: SELECT * FROM users LIMIT 10
# Query 2: SELECT * FROM posts WHERE user_id IN (1,2,3,4,5,6,7,8,9,10)

for user in users:
    print(f"{user.email}: {len (user.posts)} posts")
    # No additional queries!
    
# Total: 2 queries regardless of user count ✅
\`\`\`

### Detecting N+1 Problems

\`\`\`python
"""
Tools to Detect N+1 Issues
"""

# Method 1: Enable SQL logging
engine = create_engine("postgresql://localhost/mydb", echo=True)
# Prints every SQL query - look for repeated patterns

# Method 2: Query counter
from sqlalchemy import event

query_count = 0

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute (conn, cursor, statement, parameters, context, executemany):
    global query_count
    query_count += 1
    print(f"Query #{query_count}: {statement[:100]}...")

# Method 3: Context manager counter
class QueryCounter:
    def __init__(self):
        self.count = 0
    
    def __enter__(self):
        self.count = 0
        
        @event.listens_for(Engine, "before_cursor_execute")
        def increment (conn, cursor, statement, parameters, context, executemany):
            self.count += 1
        
        self.listener = increment
        return self
    
    def __exit__(self, *args):
        event.remove(Engine, "before_cursor_execute", self.listener)

# Usage
with QueryCounter() as counter:
    users = session.execute (select(User).limit(10)).scalars().all()
    for user in users:
        _ = user.posts
    print(f"Executed {counter.count} queries")  # Should be 2, not 11

# Method 4: Production APM tools
# - Datadog APM
# - New Relic
# - Scout APM
# These show query counts per endpoint
\`\`\`

---

## Lazy Loading (Default)

### How Lazy Loading Works

\`\`\`python
"""
Lazy Loading Behavior
"""

class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column (primary_key=True)
    posts: Mapped[list["Post"]] = relationship (back_populates="user")
    # Default: lazy="select"

class Post(Base):
    __tablename__ = 'posts'
    id: Mapped[int] = mapped_column (primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    user: Mapped["User"] = relationship (back_populates="posts")

# Lazy loading behavior
user = session.get(User, 1)
# Query: SELECT * FROM users WHERE id = 1

print(user.posts)
# NOW it queries: SELECT * FROM posts WHERE user_id = 1
# Posts loaded on first access

# Accessing again: no query (cached in session)
print(user.posts)  # No query

# New session: queries again
session.close()
session = SessionLocal()
user = session.get(User, 1)
print(user.posts)  # Queries again
\`\`\`

### Lazy Loading Variants

\`\`\`python
"""
Different Lazy Loading Options
"""

# lazy='select' (default)
posts: Mapped[list["Post"]] = relationship (lazy='select')
# Loads related objects with separate SELECT when accessed

# lazy='immediate'
posts: Mapped[list["Post"]] = relationship (lazy='immediate')
# Loads related objects immediately after parent loaded (separate query)

# lazy='joined'
posts: Mapped[list["Post"]] = relationship (lazy='joined')
# Loads with JOIN in same query as parent

# lazy='subquery'
posts: Mapped[list["Post"]] = relationship (lazy='subquery')
# Loads with subquery after parent query

# lazy='selectin'
posts: Mapped[list["Post"]] = relationship (lazy='selectin')
# Loads with SELECT IN after parent query (recommended)

# lazy='noload'
posts: Mapped[list["Post"]] = relationship (lazy='noload')
# Never loads, always returns empty list

# lazy='raise'
posts: Mapped[list["Post"]] = relationship (lazy='raise')
# Raises error if accessed (prevents lazy loading bugs)

# lazy='raise_on_sql'
posts: Mapped[list["Post"]] = relationship (lazy='raise_on_sql')
# Raises error if would execute SQL (allows already-loaded data)
\`\`\`

---

## Eager Loading Strategies

### joinedload (Single Query with JOIN)

\`\`\`python
"""
joinedload: Single Query with LEFT OUTER JOIN
"""

from sqlalchemy.orm import joinedload

# Load users with posts in single query
stmt = select(User).options (joinedload(User.posts))
users = session.execute (stmt).scalars().unique().all()

# SQL Generated:
# SELECT users.*, posts.*
# FROM users
# LEFT OUTER JOIN posts ON users.id = posts.user_id

# Key characteristics:
# ✅ Single query (minimal network roundtrips)
# ✅ Good for one-to-one relationships
# ❌ Cartesian product (user repeated for each post)
# ❌ Poor for large collections (100 posts = 100 duplicate user rows)
# ❌ Doesn't work well with LIMIT (applies after JOIN)

# Accessing posts: no additional query
for user in users:
    print(user.posts)  # Already loaded

# Multiple relationships
stmt = select(User).options(
    joinedload(User.posts),
    joinedload(User.profile)
)

# Nested joinedload
stmt = select(User).options(
    joinedload(User.posts).joinedload(Post.comments)
)
# Loads users -> posts -> comments in single query
\`\`\`

### selectinload (Two Queries with IN)

\`\`\`python
"""
selectinload: Separate Query with IN Clause (RECOMMENDED)
"""

from sqlalchemy.orm import selectinload

# Load users, then posts in single query
stmt = select(User).options (selectinload(User.posts))
users = session.execute (stmt).scalars().all()

# SQL Generated:
# Query 1: SELECT * FROM users
# Query 2: SELECT * FROM posts WHERE user_id IN (1, 2, 3, 4, 5)

# Key characteristics:
# ✅ No cartesian product (efficient memory usage)
# ✅ Scales well with LIMIT on parent query
# ✅ Works well with large collections
# ✅ Better than joinedload for one-to-many
# ✅ RECOMMENDED default for eager loading
# ❌ Two queries instead of one (minimal overhead)

# Multiple relationships
stmt = select(User).options(
    selectinload(User.posts),
    selectinload(User.comments),
    selectinload(User.profile)
)
# 4 queries total (1 for users, 1 for each relationship)

# Nested selectinload
stmt = select(User).options(
    selectinload(User.posts).selectinload(Post.comments)
)
# Query 1: SELECT * FROM users
# Query 2: SELECT * FROM posts WHERE user_id IN (...)
# Query 3: SELECT * FROM comments WHERE post_id IN (...)
\`\`\`

### subqueryload (Two Queries with Subquery)

\`\`\`python
"""
subqueryload: Separate Query with Subquery JOIN
"""

from sqlalchemy.orm import subqueryload

# Load users, then posts with subquery
stmt = select(User).options (subqueryload(User.posts))
users = session.execute (stmt).scalars().all()

# SQL Generated:
# Query 1: SELECT * FROM users
# Query 2: SELECT * FROM posts WHERE user_id IN (
#     SELECT users.id FROM users
# )

# Key characteristics:
# ✅ No cartesian product
# ❌ Less efficient than selectinload (subquery overhead)
# ❌ Legacy approach, use selectinload instead
# ⚠️ Can have issues with LIMIT on parent query

# When to use:
# - Legacy codebases already using it
# - Specific database optimizers that prefer subqueries
# - Generally: use selectinload instead
\`\`\`

### Comparison Matrix

\`\`\`python
"""
Loading Strategy Comparison
"""

# Scenario: 100 users, each with 10 posts

# lazy='select' (N+1)
# Queries: 1 + 100 = 101
# Memory: Minimal
# Use when: Not accessing relationships

# joinedload
# Queries: 1
# Memory: High (100 users × 10 posts = 1000 rows with duplicate user data)
# Use when: One-to-one, small one-to-many

# selectinload (RECOMMENDED)
# Queries: 2
# Memory: Optimal (100 user rows + 1000 post rows, no duplication)
# Use when: One-to-many (default choice)

# subqueryload
# Queries: 2 (with subquery overhead)
# Memory: Optimal
# Use when: Legacy codebases

# Performance test
import time

# N+1
start = time.time()
users = session.execute (select(User).limit(100)).scalars().all()
for user in users:
    _ = user.posts
print(f"N+1: {time.time() - start:.2f}s")  # ~2.0s

# joinedload
start = time.time()
users = session.execute(
    select(User).options (joinedload(User.posts)).limit(100)
).scalars().unique().all()
print(f"joinedload: {time.time() - start:.2f}s")  # ~0.3s

# selectinload
start = time.time()
users = session.execute(
    select(User).options (selectinload(User.posts)).limit(100)
).scalars().all()
print(f"selectinload: {time.time() - start:.2f}s")  # ~0.2s
\`\`\`

---

## Advanced Loading Techniques

### Contains Eager

\`\`\`python
"""
contains_eager: When You've Already Joined
"""

from sqlalchemy.orm import contains_eager

# When you explicitly join for filtering
stmt = (
    select(User)
    .join(User.posts)
    .where(Post.published == True)
    .options (contains_eager(User.posts))
)

# SQL: Single query with your explicit JOIN
# SELECT users.*, posts.* FROM users
# JOIN posts ON users.id = posts.user_id
# WHERE posts.published = true

# WITHOUT contains_eager: posts loaded twice
# (once in JOIN for filter, again with lazy loading)

# WITH contains_eager: uses JOIN results (efficient)
\`\`\`

### Raiseload (Prevent Lazy Loading)

\`\`\`python
"""
raiseload: Catch Lazy Loading Bugs
"""

from sqlalchemy.orm import raiseload

# Prevent accidental lazy loading
stmt = select(User).options(
    raiseload(User.posts),  # Raise error if posts accessed
    selectinload(User.comments)  # Load comments eagerly
)

users = session.execute (stmt).scalars().all()

for user in users:
    print(user.comments)  # OK: eagerly loaded
    print(user.posts)  # ERROR: would lazy load

# Use raiseload in development to catch N+1 issues
# Remove or replace with selectinload for production
\`\`\`

### Load Only Specific Columns

\`\`\`python
"""
Load Only Required Columns
"""

from sqlalchemy.orm import load_only

# Load only specific columns
stmt = select(User).options(
    load_only(User.id, User.email),  # Only these columns
    selectinload(User.posts).load_only(Post.id, Post.title)
)

# Reduces data transfer (important for large text columns)
\`\`\`

### Defer Column Loading

\`\`\`python
"""
Defer Expensive Columns
"""

from sqlalchemy.orm import defer

# Defer large columns (load only when accessed)
stmt = select(Post).options(
    defer(Post.content),  # Don't load content initially
    defer(Post.rendered_html)
)

posts = session.execute (stmt).scalars().all()

for post in posts:
    print(post.title)  # OK: loaded
    print(post.content)  # Queries now (lazy load deferred column)

# Use for: Large TEXT/BLOB columns not always needed
\`\`\`

---

## Relationship Loading Patterns

### Per-Query Options

\`\`\`python
"""
Override Relationship Lazy Setting Per Query
"""

# Model with lazy='select' (default)
class User(Base):
    __tablename__ = 'users'
    posts: Mapped[list["Post"]] = relationship()  # lazy='select'

# Query 1: Eager load posts
stmt = select(User).options (selectinload(User.posts))

# Query 2: Don't load posts at all
stmt = select(User).options (noload(User.posts))

# Query 3: Use default (lazy='select')
stmt = select(User)  # No options
\`\`\`

### Default Loading Strategy

\`\`\`python
"""
Set Default Loading Strategy on Relationship
"""

class User(Base):
    __tablename__ = 'users'
    
    # Always eager load (not recommended)
    profile: Mapped["UserProfile"] = relationship (lazy='joined')
    
    # Recommended: Keep lazy='select', use options() per query
    posts: Mapped[list["Post"]] = relationship (lazy='select')
    
    # Prevent lazy loading (development)
    comments: Mapped[list["Comment"]] = relationship (lazy='raise')

# Per-query override still works
stmt = select(User).options (lazyload(User.profile))  # Override joined
\`\`\`

### Loading Strategies by Use Case

\`\`\`python
"""
Choose Strategy Based on Use Case
"""

# API endpoint: List users
# Load users without relationships (faster response)
stmt = select(User).options(
    noload(User.posts),
    noload(User.comments)
)

# API endpoint: User detail
# Load user with all relationships
stmt = select(User).where(User.id == user_id).options(
    selectinload(User.posts).selectinload(Post.comments),
    selectinload(User.profile)
)

# Admin dashboard: User list with counts
# Load users, count relationships without loading all
from sqlalchemy import func
stmt = select(
    User,
    func.count(Post.id).label('post_count')
).outerjoin(User.posts).group_by(User.id)

# Background job: Process all users
# Batch load with selectinload, process in chunks
for offset in range(0, total_users, 1000):
    stmt = (
        select(User)
        .options (selectinload(User.posts))
        .offset (offset)
        .limit(1000)
    )
    users = session.execute (stmt).scalars().all()
    process_users (users)
    session.expire_all()  # Free memory
\`\`\`

---

## Common Pitfalls and Solutions

### Pitfall 1: LIMIT with joinedload

\`\`\`python
"""
joinedload + LIMIT Issues
"""

# PROBLEM: LIMIT applies AFTER JOIN
stmt = select(User).options (joinedload(User.posts)).limit(10)

# If users have many posts:
# - Joins all posts first
# - Then limits to 10 rows (not 10 users!)
# - Might get 1 user with 10 posts instead of 10 users

# SOLUTION: Use selectinload
stmt = select(User).options (selectinload(User.posts)).limit(10)
# Limits users first, then loads posts for those 10 users
\`\`\`

### Pitfall 2: Multiple joinedloads

\`\`\`python
"""
Multiple joinedload Causes Cartesian Product
"""

# PROBLEM: Exponential row growth
stmt = select(User).options(
    joinedload(User.posts),      # 10 posts per user
    joinedload(User.comments)    # 20 comments per user
)
# Result: 10 × 20 = 200 rows per user!

# SOLUTION: Use selectinload
stmt = select(User).options(
    selectinload(User.posts),    # 2 queries
    selectinload(User.comments)  # 3 queries total
)
\`\`\`

### Pitfall 3: Forgetting unique()

\`\`\`python
"""
joinedload Requires unique()
"""

# With joinedload, duplicate User objects without unique()
stmt = select(User).options (joinedload(User.posts))
users = session.execute (stmt).scalars().all()  # Duplicates!

# CORRECT: Use unique()
users = session.execute (stmt).scalars().unique().all()

# selectinload doesn't need unique()
stmt = select(User).options (selectinload(User.posts))
users = session.execute (stmt).scalars().all()  # No duplicates
\`\`\`

### Pitfall 4: Detached Instance Access

\`\`\`python
"""
Accessing Relationships After Session Close
"""

# PROBLEM
session = SessionLocal()
user = session.get(User, 1)
session.close()

print(user.posts)  # ERROR: DetachedInstanceError

# SOLUTION 1: Eager load before closing
session = SessionLocal()
stmt = select(User).where(User.id == 1).options (selectinload(User.posts))
user = session.execute (stmt).scalar_one()
session.close()
print(user.posts)  # OK: already loaded

# SOLUTION 2: Keep session open
session = SessionLocal()
user = session.get(User, 1)
print(user.posts)  # OK: session still open
session.close()

# SOLUTION 3: expunge and merge (advanced)
session = SessionLocal()
user = session.get(User, 1)
session.expunge (user)  # Detach from session
# user is now detached, can pass across threads
\`\`\`

---

## Production Optimization

### Loading Strategy Decision Tree

\`\`\`
Need to load relationship?
│
├─ No → noload() or raiseload()
│
└─ Yes
   │
   ├─ One-to-one or small one-to-many (< 10 items)?
   │  └─ Use joinedload (single query)
   │
   └─ One-to-many or many-to-many?
      │
      ├─ Need to filter/order parent by child?
      │  └─ Manual JOIN + contains_eager
      │
      └─ Regular eager loading?
         └─ Use selectinload (recommended)
\`\`\`

### Monitoring Query Performance

\`\`\`python
"""
Production Monitoring
"""

import time
from contextlib import contextmanager

@contextmanager
def query_timer (operation: str):
    """Time database operations"""
    start = time.time()
    query_count_start = session.info.get('query_count', 0)
    
    yield
    
    duration = time.time() - start
    query_count = session.info.get('query_count', 0) - query_count_start
    
    if duration > 0.1:  # Slow query threshold
        logger.warning(
            f"Slow operation: {operation} took {duration:.2f}s "
            f"with {query_count} queries"
        )

# Usage
with query_timer("Load users with posts"):
    stmt = select(User).options (selectinload(User.posts)).limit(100)
    users = session.execute (stmt).scalars().all()
\`\`\`

---

## Summary

### Key Takeaways

✅ **N+1 problem**: Most common ORM performance issue (1 + N queries)  
✅ **selectinload**: Recommended default for eager loading (2 queries, no cartesian)  
✅ **joinedload**: Use for one-to-one, small one-to-many (1 query, cartesian product)  
✅ **Lazy loading**: Default behavior, causes N+1 if not careful  
✅ **raiseload**: Use in development to catch lazy loading issues  
✅ **contains_eager**: When you've already joined for filtering  
✅ **unique()**: Required with joinedload, not with selectinload

### Performance Rules

✅ Default: lazy='select' on relationships  
✅ Per-query: Use .options (selectinload(...)) for eager loading  
✅ One-to-many: selectinload (not joinedload)  
✅ LIMIT queries: Always use selectinload (not joinedload)  
✅ Multiple relationships: selectinload (not multiple joinedloads)  
✅ Development: Use raiseload to catch N+1 issues early

### Decision Matrix

| Scenario | Strategy | Queries | Notes |
|----------|----------|---------|-------|
| One-to-one | joinedload | 1 | Single query efficient |
| Small one-to-many (<10) | joinedload | 1 | Cartesian acceptable |
| Large one-to-many | selectinload | 2 | No cartesian |
| Many-to-many | selectinload | 2 | Recommended |
| With LIMIT | selectinload | 2 | joinedload breaks LIMIT |
| Multiple relationships | selectinload | N+1 | Avoid multiple joinedload |
| Already joined | contains_eager | 1 | Reuse existing JOIN |
| Never load | noload/raiseload | 0 | Prevent lazy load |

### Next Steps

In the next section, we'll explore **Session Management & Patterns**: session lifecycle, scoping, web framework integration, and production session patterns.
`,
};
