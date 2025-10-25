export const definingModels = {
  title: 'Defining Models & Relationships',
  id: 'defining-models',
  content: `
# Defining Models & Relationships

## Introduction

Database relationships are the foundation of relational databases. SQLAlchemy\'s ORM provides powerful tools for modeling one-to-one, one-to-many, many-to-many, self-referential, and polymorphic relationships.

In this section, you'll learn:
- Table definitions with declarative syntax
- Column types and constraints
- Primary and foreign keys
- All relationship types and when to use each
- Relationship configuration and loading strategies
- Real-world schema patterns

Understanding relationships is crucial for building normalized, maintainable database schemas.

---

## Table Definitions

### Basic Table Definition

\`\`\`python
"""
Basic Model Definition
"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, Boolean
from datetime import datetime
from typing import Optional

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    
    # Primary key
    id: Mapped[int] = mapped_column (primary_key=True)
    
    # Required columns
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(50))
    
    # Optional columns
    bio: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column (default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column (onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<User (id={self.id}, email='{self.email}')>"
\`\`\`

### Column Types

\`\`\`python
"""
Comprehensive Column Types
"""

from sqlalchemy import (
    Integer, BigInteger, SmallInteger,
    String, Text, VARCHAR, CHAR,
    Boolean,
    Date, DateTime, Time, Interval,
    Float, Numeric, DECIMAL,
    JSON, ARRAY,
    LargeBinary, BLOB,
    Enum as SQLEnum,
    UUID
)
from enum import Enum as PyEnum
import uuid

class UserStatus(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BANNED = "banned"

class Product(Base):
    __tablename__ = 'products'
    
    # Integers
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[int] = mapped_column(SmallInteger)
    
    # Strings
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    sku: Mapped[str] = mapped_column(CHAR(10))
    
    # Numeric
    price: Mapped[float] = mapped_column(Numeric(10, 2))  # DECIMAL(10,2)
    weight: Mapped[float] = mapped_column(Float)
    
    # Boolean
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Dates and Times
    created_date: Mapped[datetime] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    delivery_time: Mapped[datetime] = mapped_column(Time)
    
    # PostgreSQL-specific
    metadata_json: Mapped[dict] = mapped_column(JSON)  # JSON/JSONB
    tags: Mapped[list] = mapped_column(ARRAY(String))  # Array
    
    # UUID
    uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4)
    
    # Enum
    status: Mapped[UserStatus] = mapped_column(SQLEnum(UserStatus))
    
    # Binary data
    image: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
\`\`\`

### Constraints

\`\`\`python
"""
Column and Table Constraints
"""

from sqlalchemy import CheckConstraint, UniqueConstraint, Index

class Account(Base):
    __tablename__ = 'accounts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String(50), index=True)
    age: Mapped[Optional[int]]
    balance: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
    status: Mapped[str] = mapped_column(String(20))
    
    # Table-level constraints
    __table_args__ = (
        # Check constraints
        CheckConstraint('age >= 18', name='check_adult'),
        CheckConstraint('balance >= 0', name='check_positive_balance'),
        CheckConstraint("status IN ('active', 'suspended', 'closed')", name='check_status'),
        
        # Unique constraints
        UniqueConstraint('email', name='uq_account_email'),
        
        # Composite unique constraint
        UniqueConstraint('username', 'email', name='uq_username_email'),
        
        # Indexes
        Index('idx_account_status', 'status'),
        Index('idx_account_created', 'created_at'),
        Index('idx_account_email_status', 'email', 'status'),  # Composite
        
        # Partial index (PostgreSQL)
        Index('idx_active_accounts', 'email', postgresql_where='status = \\'active\\''),
    )
\`\`\`

---

## Primary Keys

### Auto-Incrementing Integer

\`\`\`python
"""
Standard auto-incrementing primary key
"""

class User(Base):
    __tablename__ = 'users'
    
    # PostgreSQL: SERIAL
    # MySQL: AUTO_INCREMENT
    # SQLite: AUTOINCREMENT
    id: Mapped[int] = mapped_column (primary_key=True)
\`\`\`

### UUID Primary Key

\`\`\`python
"""
UUID primary keys for distributed systems
"""

import uuid
from sqlalchemy import UUID as SQLUUID

class Session(Base):
    __tablename__ = 'sessions'
    
    # UUID as primary key
    id: Mapped[uuid.UUID] = mapped_column(
        SQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[int]
    token: Mapped[str] = mapped_column(String(255))

# Benefits of UUID:
# - Globally unique (no coordination needed)
# - Can generate client-side
# - Good for distributed systems, microservices
# - Harder to enumerate/guess

# Drawbacks:
# - Larger (16 bytes vs 4 bytes for int)
# - Slower index lookups
# - Random UUIDs cause index fragmentation
\`\`\`

### Composite Primary Key

\`\`\`python
"""
Multiple columns as primary key
"""

class UserRole(Base):
    __tablename__ = 'user_roles'
    
    user_id: Mapped[int] = mapped_column (primary_key=True)
    role_id: Mapped[int] = mapped_column (primary_key=True)
    granted_at: Mapped[datetime] = mapped_column (default=datetime.utcnow)
    
    # No single 'id' column
    # Primary key is (user_id, role_id) combination
\`\`\`

---

## Foreign Keys

### Basic Foreign Key

\`\`\`python
"""
Foreign Key Column
"""

from sqlalchemy import ForeignKey

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
\`\`\`

### Foreign Key with Cascade Options

\`\`\`python
"""
Foreign Key Cascade Behaviors
"""

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    
    # CASCADE: Delete posts when user deleted
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE', onupdate='CASCADE')
    )
    
    # SET NULL: Keep post but set author to NULL
    author_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL')
    )
    
    # RESTRICT: Prevent deletion if posts exist (default)
    category_id: Mapped[int] = mapped_column(
        ForeignKey('categories.id', ondelete='RESTRICT')
    )

# Cascade options:
# - CASCADE: Delete children when parent deleted
# - SET NULL: Set foreign key to NULL
# - RESTRICT: Prevent deletion (raise error)
# - NO ACTION: Similar to RESTRICT
# - SET DEFAULT: Set to default value
\`\`\`

---

## One-to-Many Relationships

### Basic One-to-Many

\`\`\`python
"""
One User has Many Posts
"""

from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    
    # Relationship: one user has many posts
    posts: Mapped[list["Post"]] = relationship (back_populates="user")

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    title: Mapped[str] = mapped_column(String(200))
    
    # Relationship: post belongs to one user
    user: Mapped["User"] = relationship (back_populates="posts")

# Usage
user = session.get(User, 1)
print(user.posts)  # List of Post objects
post = user.posts[0]
print(post.user)   # User object
\`\`\`

### One-to-Many with Cascade

\`\`\`python
"""
Cascade Delete and Orphan Removal
"""

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    
    posts: Mapped[list["Post"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",  # Delete posts when user deleted
        passive_deletes=True  # Let database handle CASCADE
    )

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE')
    )
    user: Mapped["User"] = relationship (back_populates="posts")

# Cascade options:
# - save-update: Cascade add() and merge()
# - delete: Cascade delete()
# - delete-orphan: Delete when removed from parent
# - all: All of the above
# - passive_deletes: Let database handle ON DELETE CASCADE
\`\`\`

---

## One-to-One Relationships

\`\`\`python
"""
One-to-One Relationship
"""

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    
    # One-to-one: uselist=False
    profile: Mapped["UserProfile"] = relationship(
        back_populates="user",
        uselist=False,  # Returns single object, not list
        cascade="all, delete-orphan"
    )

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), unique=True)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    user: Mapped["User"] = relationship (back_populates="profile")

# Usage
user = session.get(User, 1)
print(user.profile.bio)  # Single UserProfile object (not list)
\`\`\`

---

## Many-to-Many Relationships

### Association Table

\`\`\`python
"""
Many-to-Many with Association Table
"""

from sqlalchemy import Table

# Association table (no class needed)
user_groups = Table(
    'user_groups',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('group_id', ForeignKey('groups.id'), primary_key=True),
    Column('joined_at', DateTime, default=datetime.utcnow)  # Extra column
)

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    
    # Many-to-many relationship
    groups: Mapped[list["Group"]] = relationship(
        secondary=user_groups,
        back_populates="users"
    )

class Group(Base):
    __tablename__ = 'groups'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    
    users: Mapped[list["User"]] = relationship(
        secondary=user_groups,
        back_populates="groups"
    )

# Usage
user = session.get(User, 1)
user.groups.append (group)  # Add user to group
session.commit()
\`\`\`

### Association Object Pattern

\`\`\`python
"""
Many-to-Many with Extra Data
"""

class UserGroup(Base):
    """Association object with extra columns"""
    __tablename__ = 'user_groups'
    
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey('groups.id'), primary_key=True)
    role: Mapped[str] = mapped_column(String(50), default='member')
    joined_at: Mapped[datetime] = mapped_column (default=datetime.utcnow)
    
    # Relationships to parent objects
    user: Mapped["User"] = relationship (back_populates="group_associations")
    group: Mapped["Group"] = relationship (back_populates="user_associations")

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    
    group_associations: Mapped[list["UserGroup"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Convenience property
    @property
    def groups (self):
        return [assoc.group for assoc in self.group_associations]

class Group(Base):
    __tablename__ = 'groups'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    
    user_associations: Mapped[list["UserGroup"]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan"
    )

# Usage
user = session.get(User, 1)
group = session.get(Group, 1)

# Create association with extra data
assoc = UserGroup (user=user, group=group, role='admin')
session.add (assoc)
session.commit()

# Access extra data
for assoc in user.group_associations:
    print(f"{assoc.group.name}: {assoc.role} (joined {assoc.joined_at})")
\`\`\`

---

## Self-Referential Relationships

\`\`\`python
"""
Self-Referential: Tree Structure
"""

class Category(Base):
    __tablename__ = 'categories'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    parent_id: Mapped[Optional[int]] = mapped_column(ForeignKey('categories.id'))
    
    # Parent relationship (many-to-one)
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        remote_side=[id],  # Indicates which side is "remote"
        back_populates="children"
    )
    
    # Children relationship (one-to-many)
    children: Mapped[list["Category"]] = relationship(
        "Category",
        back_populates="parent"
    )

# Usage
root = Category (name="Electronics")
child1 = Category (name="Computers", parent=root)
child2 = Category (name="Phones", parent=root)
grandchild = Category (name="Laptops", parent=child1)

session.add (root)
session.commit()

# Traverse tree
for child in root.children:
    print(f"- {child.name}")
    for grandchild in child.children:
        print(f"  - {grandchild.name}")
\`\`\`

### Adjacency List Pattern

\`\`\`python
"""
Self-Referential: Social Network (Friends)
"""

# Association table for bidirectional friendship
friendships = Table(
    'friendships',
    Base.metadata,
    Column('user_id', ForeignKey('users.id'), primary_key=True),
    Column('friend_id', ForeignKey('users.id'), primary_key=True),
    Column('created_at', DateTime, default=datetime.utcnow)
)

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    username: Mapped[str] = mapped_column(String(50))
    
    # Self-referential many-to-many
    friends: Mapped[list["User"]] = relationship(
        "User",
        secondary=friendships,
        primaryjoin=id == friendships.c.user_id,
        secondaryjoin=id == friendships.c.friend_id,
        back_populates="friends"
    )

# Usage
user1 = User (username="alice")
user2 = User (username="bob")
user1.friends.append (user2)  # Alice friends Bob
session.commit()
\`\`\`

---

## Polymorphic Associations

### Single Table Inheritance

\`\`\`python
"""
Single Table Inheritance
"""

class Employee(Base):
    __tablename__ = 'employees'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    type: Mapped[str] = mapped_column(String(50))  # Discriminator
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255))
    
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'employee'
    }

class Engineer(Employee):
    """Engineer-specific attributes"""
    programming_language: Mapped[Optional[str]] = mapped_column(String(50))
    
    __mapper_args__ = {
        'polymorphic_identity': 'engineer'
    }

class Manager(Employee):
    """Manager-specific attributes"""
    department: Mapped[Optional[str]] = mapped_column(String(100))
    
    __mapper_args__ = {
        'polymorphic_identity': 'manager'
    }

# Usage
engineer = Engineer (name="Alice", programming_language="Python")
manager = Manager (name="Bob", department="Engineering")

# Query returns correct subclass
employees = session.query(Employee).all()
for emp in employees:
    if isinstance (emp, Engineer):
        print(f"Engineer: {emp.programming_language}")
    elif isinstance (emp, Manager):
        print(f"Manager: {emp.department}")
\`\`\`

### Joined Table Inheritance

\`\`\`python
"""
Joined Table Inheritance (normalized)
"""

class Employee(Base):
    __tablename__ = 'employees'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    type: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(100))
    
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'employee'
    }

class Engineer(Employee):
    __tablename__ = 'engineers'
    
    id: Mapped[int] = mapped_column(ForeignKey('employees.id'), primary_key=True)
    programming_language: Mapped[str] = mapped_column(String(50))
    
    __mapper_args__ = {
        'polymorphic_identity': 'engineer'
    }

class Manager(Employee):
    __tablename__ = 'managers'
    
    id: Mapped[int] = mapped_column(ForeignKey('employees.id'), primary_key=True)
    department: Mapped[str] = mapped_column(String(100))
    
    __mapper_args__ = {
        'polymorphic_identity': 'manager'
    }

# Separate tables, joined on query
# More normalized, but requires JOIN
\`\`\`

---

## Summary

### Key Takeaways

✅ **Column types**: Choose appropriate types for data (String vs Text, Integer vs BigInteger)  
✅ **Primary keys**: Auto-increment for single DB, UUID for distributed systems  
✅ **Foreign keys**: Always add indexes, choose CASCADE behavior carefully  
✅ **One-to-many**: Most common relationship, use \`back_populates\`  
✅ **Many-to-many**: Use association table, or association object for extra data  
✅ **Self-referential**: Use \`remote_side\` for parent-child hierarchies  
✅ **Polymorphic**: Single table for simple, joined table for normalized

### Best Practices

✅ Always use \`back_populates\` for bidirectional relationships  
✅ Add indexes to foreign key columns  
✅ Use \`cascade="all, delete-orphan"\` for strong ownership  
✅ Use \`passive_deletes=True\` with ON DELETE CASCADE for performance  
✅ Type annotate with \`Mapped[]\` for type safety  
✅ Use \`Optional[]\` for nullable columns  
✅ Add \`__repr__\` methods for debugging

### Next Steps

In the next section, we'll explore **Query API Deep Dive**: SELECT queries, filtering, ordering, joins, subqueries, aggregations, and query optimization.
`,
};
