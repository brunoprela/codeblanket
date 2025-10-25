export const databaseIntegrationSqlalchemy = {
  title: 'Database Integration (SQLAlchemy + FastAPI)',
  id: 'database-integration-sqlalchemy',
  content: `
# Database Integration (SQLAlchemy + FastAPI)

## Introduction

Production APIs need databases. FastAPI + SQLAlchemy is the **gold standard** combination for Python database access—FastAPI's dependency injection elegantly manages SQLAlchemy sessions, while SQLAlchemy's ORM provides type-safe, efficient database operations.

**Why SQLAlchemy + FastAPI is powerful:**
- **Dependency injection**: Automatic session management and cleanup
- **Type safety**: Pydantic models + SQLAlchemy models work together
- **Async support**: SQLAlchemy 2.0+ has first-class async support
- **Transaction management**: Automatic commit/rollback
- **Testing**: Easy to mock database for tests

In production, this combination solves:
- Connection pooling and resource management
- Transaction handling and rollbacks
- Relationship loading (N+1 query problem)
- Database migrations with Alembic
- Multi-database and read replicas

In this section, you'll master:
- Setting up SQLAlchemy with FastAPI
- Dependency injection for sessions
- CRUD operations with repository pattern
- Async database operations
- Relationship management
- Transactions and error handling
- Testing database code
- Production patterns

### The Integration Stack

\`\`\`
FastAPI Endpoint
    ↓
Pydantic Model (request validation)
    ↓
Dependency: get_db() → SQLAlchemy Session
    ↓
Repository/Service (business logic)
    ↓
SQLAlchemy ORM Model
    ↓
Database (PostgreSQL, MySQL, SQLite)
    ↓
Pydantic Model (response serialization)
\`\`\`

---

## Setup and Configuration

### Database Connection

\`\`\`python
"""
SQLAlchemy Setup with FastAPI
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import FastAPI, Depends

# Database URL (use environment variables in production)
DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # Number of permanent connections
    max_overflow=20,           # Additional connections when pool exhausted
    pool_timeout=30,           # Seconds to wait for connection
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True,        # Verify connection health before use
    echo=False,                # Set True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,          # Manual transaction control
    autoflush=False,           # Manual flush control
    bind=engine
)

# Base class for models
Base = declarative_base()

app = FastAPI()

# Dependency for database session
def get_db() -> Session:
    """
    Dependency that provides database session
    
    Yields session, ensures cleanup
    Automatic rollback on exception
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Use in endpoints
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Database session automatically injected
    Connection returned to pool after request
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
\`\`\`

### SQLAlchemy Models

\`\`\`python
"""
Define Database Models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author")

class Post(Base):
    """Blog post model"""
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    published = Column(Boolean, default=False, nullable=False)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    author = relationship("User", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

class Comment(Base):
    """Comment model"""
    __tablename__ = "comments"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    post_id = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    author_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")

# Create tables (use Alembic in production)
# Base.metadata.create_all(bind=engine)
\`\`\`

### Pydantic Schemas

\`\`\`python
"""
Pydantic Schemas for Request/Response
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True  # Allow from SQLAlchemy models

# Post schemas
class PostBase(BaseModel):
    title: str = Field(..., min_length=10, max_length=200)
    content: str = Field(..., min_length=50)
    published: bool = False

class PostCreate(PostBase):
    pass

class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=10, max_length=200)
    content: Optional[str] = Field(None, min_length=50)
    published: Optional[bool] = None

class PostResponse(PostBase):
    id: int
    author_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

# Post with author
class PostWithAuthor(PostResponse):
    author: UserResponse

# Comment schemas
class CommentBase(BaseModel):
    content: str = Field(..., min_length=1, max_length=1000)

class CommentCreate(CommentBase):
    post_id: int

class CommentResponse(CommentBase):
    id: int
    post_id: int
    author_id: int
    created_at: datetime
    author: UserResponse
    
    class Config:
        orm_mode = True
\`\`\`

---

## CRUD Operations

### Basic CRUD

\`\`\`python
"""
Basic CRUD Operations
"""

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session

app = FastAPI()

# CREATE
@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Create new user
    """
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )
    
    # Hash password (use bcrypt in production)
    hashed_password = hash_password(user.password)
    
    # Create user
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

# READ (single)
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# READ (list)
@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List users with pagination"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

# UPDATE
@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user: UserUpdate,
    db: Session = Depends(get_db)
):
    """Update user"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    update_data = user.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = hash_password(update_data.pop("password"))
    
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    
    return db_user

# DELETE
@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Delete user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    
    return None
\`\`\`

### Repository Pattern

\`\`\`python
"""
Repository Pattern for Clean Architecture
"""

from typing import List, Optional, Generic, TypeVar, Type
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base repository with generic CRUD operations
    """
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        """Get by ID"""
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get multiple with pagination"""
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, obj_in: CreateSchemaType) -> ModelType:
        """Create new object"""
        obj_data = obj_in.dict()
        db_obj = self.model(**obj_data)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def update(
        self,
        id: int,
        obj_in: UpdateSchemaType
    ) -> Optional[ModelType]:
        """Update existing object"""
        db_obj = self.get(id)
        if not db_obj:
            return None
        
        update_data = obj_in.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_obj, key, value)
        
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, id: int) -> bool:
        """Delete object"""
        db_obj = self.get(id)
        if not db_obj:
            return False
        
        self.db.delete(db_obj)
        self.db.commit()
        return True

class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """User-specific repository"""
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get only active users"""
        return (
            self.db.query(User)
            .filter(User.is_active == True)
            .offset(skip)
            .limit(limit)
            .all()
        )

class PostRepository(BaseRepository[Post, PostCreate, PostUpdate]):
    """Post-specific repository"""
    
    def get_by_author(
        self,
        author_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Post]:
        """Get posts by author"""
        return (
            self.db.query(Post)
            .filter(Post.author_id == author_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_published(self, skip: int = 0, limit: int = 100) -> List[Post]:
        """Get only published posts"""
        return (
            self.db.query(Post)
            .filter(Post.published == True)
            .order_by(Post.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

# Dependency for repositories
def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(User, db)

def get_post_repository(db: Session = Depends(get_db)) -> PostRepository:
    return PostRepository(Post, db)

# Use in endpoints
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    repo: UserRepository = Depends(get_user_repository)
):
    """Clean endpoint using repository"""
    user = repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    user: UserCreate,
    repo: UserRepository = Depends(get_user_repository)
):
    """Create user with repository"""
    # Check duplicates
    if repo.get_by_username(user.username):
        raise HTTPException(status_code=409, detail="Username exists")
    
    # Hash password
    user.password = hash_password(user.password)
    
    return repo.create(user)
\`\`\`

---

## Relationship Management

### Loading Strategies

\`\`\`python
"""
Relationship Loading Strategies
"""

from sqlalchemy.orm import joinedload, selectinload, subqueryload

# Lazy loading (default) - N+1 problem
@app.get("/posts-lazy")
async def get_posts_lazy(db: Session = Depends(get_db)):
    """
    BAD: N+1 query problem
    
    1 query for posts
    + N queries for each post's author
    = 1 + N queries!
    """
    posts = db.query(Post).all()
    
    # This triggers N additional queries
    return [
        {
            "title": post.title,
            "author": post.author.username  # Separate query for each!
        }
        for post in posts
    ]

# Eager loading with joinedload (JOIN)
@app.get("/posts-joined")
async def get_posts_joined(db: Session = Depends(get_db)):
    """
    GOOD: Single query with JOIN
    
    1 query total
    """
    posts = (
        db.query(Post)
        .options(joinedload(Post.author))  # JOIN authors table
        .all()
    )
    
    # No additional queries!
    return [
        {
            "title": post.title,
            "author": post.author.username
        }
        for post in posts
    ]

# Selectin loading (separate SELECT IN)
@app.get("/posts-selectin")
async def get_posts_selectin(db: Session = Depends(get_db)):
    """
    GOOD: 2 queries total
    
    1 query for posts
    + 1 query for all authors (SELECT IN)
    = 2 queries regardless of N
    """
    posts = (
        db.query(Post)
        .options(selectinload(Post.author))
        .all()
    )
    
    return [
        {
            "title": post.title,
            "author": post.author.username
        }
        for post in posts
    ]

# Multiple levels of relationships
@app.get("/posts-with-comments")
async def get_posts_with_comments(db: Session = Depends(get_db)):
    """
    Load posts, authors, comments, and comment authors
    """
    posts = (
        db.query(Post)
        .options(
            joinedload(Post.author),
            selectinload(Post.comments).joinedload(Comment.author)
        )
        .all()
    )
    
    return posts
\`\`\`

### Working with Relationships

\`\`\`python
"""
Create and Manage Relationships
"""

@app.post("/posts", response_model=PostWithAuthor)
async def create_post(
    post: PostCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create post with author relationship
    """
    db_post = Post(
        **post.dict(),
        author_id=current_user.id
    )
    
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    
    # Load author relationship
    db.refresh(db_post, ["author"])
    
    return db_post

@app.post("/posts/{post_id}/comments", response_model=CommentResponse)
async def create_comment(
    post_id: int,
    comment: CommentBase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create comment with post and author relationships
    """
    # Verify post exists
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    db_comment = Comment(
        content=comment.content,
        post_id=post_id,
        author_id=current_user.id
    )
    
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    
    # Load relationships for response
    db.refresh(db_comment, ["author", "post"])
    
    return db_comment

@app.get("/users/{user_id}/posts", response_model=List[PostResponse])
async def get_user_posts(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all posts by user (relationship traversal)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Access relationship
    return user.posts
\`\`\`

---

## Transactions and Error Handling

### Transaction Management

\`\`\`python
"""
Transaction Management Patterns
"""

@app.post("/transfer")
async def transfer_funds(
    from_account_id: int,
    to_account_id: int,
    amount: float,
    db: Session = Depends(get_db)
):
    """
    Transaction example: All or nothing
    """
    try:
        # Fetch accounts
        from_account = db.query(Account).filter(Account.id == from_account_id).first()
        to_account = db.query(Account).filter(Account.id == to_account_id).first()
        
        if not from_account or not to_account:
            raise HTTPException(status_code=404, detail="Account not found")
        
        if from_account.balance < amount:
            raise HTTPException(status_code=400, detail="Insufficient funds")
        
        # Perform transfer
        from_account.balance -= amount
        to_account.balance += amount
        
        # Create transaction record
        transaction = Transaction(
            from_account_id=from_account_id,
            to_account_id=to_account_id,
            amount=amount
        )
        db.add(transaction)
        
        # Commit all changes atomically
        db.commit()
        
        return {"status": "success", "transaction_id": transaction.id}
        
    except HTTPException:
        # Rollback on business logic error
        db.rollback()
        raise
    except Exception as e:
        # Rollback on unexpected error
        db.rollback()
        raise HTTPException(status_code=500, detail="Transaction failed")

# Manual transaction control
@app.post("/complex-operation")
async def complex_operation(db: Session = Depends(get_db)):
    """
    Manual transaction with savepoints
    """
    try:
        # Start transaction
        db.begin_nested()  # Savepoint
        
        # Operation 1
        user = User(username="test")
        db.add(user)
        db.flush()  # Write to DB but don't commit
        
        # Operation 2 (might fail)
        try:
            post = Post(title="Test", content="Content", author_id=user.id)
            db.add(post)
            db.flush()
        except Exception:
            # Rollback to savepoint
            db.rollback()
            raise
        
        # Commit everything
        db.commit()
        
        return {"user_id": user.id, "post_id": post.id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
\`\`\`

---

## Async Database Operations

### Async SQLAlchemy

\`\`\`python
"""
Async Database Operations (SQLAlchemy 2.0+)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

# Async engine
async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=False,
    pool_size=10,
    max_overflow=20
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Async dependency
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

# Async endpoints
@app.get("/users/{user_id}")
async def get_user_async(
    user_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Async database query
    Non-blocking I/O
    """
    result = await db.execute(
        select(User).filter(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.get("/users")
async def list_users_async(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Async query with pagination
    """
    result = await db.execute(
        select(User).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    
    return users

@app.post("/users")
async def create_user_async(
    user: UserCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Async insert
    """
    db_user = User(**user.dict())
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    return db_user
\`\`\`

---

## Testing

### Database Testing Patterns

\`\`\`python
"""
Testing with Database
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Test database
TEST_DATABASE_URL = "sqlite:///./test.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(bind=test_engine)

@pytest.fixture
def db():
    """Test database fixture"""
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
    
    # Drop tables after test
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def client(db):
    """Test client with database"""
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as c:
        yield c
    
    app.dependency_overrides.clear()

def test_create_user(client):
    """Test user creation"""
    response = client.post(
        "/users",
        json={"username": "testuser", "email": "test@example.com", "password": "password123"}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert "id" in data

def test_get_user(client, db):
    """Test get user"""
    # Create test user
    user = User(username="testuser", email="test@example.com", hashed_password="hashed")
    db.add(user)
    db.commit()
    
    response = client.get(f"/users/{user.id}")
    
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
\`\`\`

---

## Summary

### Key Takeaways

✅ **Dependency injection**: get_db() provides session, automatic cleanup  
✅ **Repository pattern**: Encapsulate data access, clean architecture  
✅ **Relationship loading**: Avoid N+1 with joinedload, selectinload  
✅ **Transactions**: Automatic commit/rollback with try/except  
✅ **Async support**: SQLAlchemy 2.0+ async for high performance  
✅ **Pydantic + SQLAlchemy**: orm_mode bridges the two worlds  
✅ **Testing**: Override dependencies, use test database

### Best Practices

**1. Session management**:
- Always use dependency injection
- Never create sessions manually in endpoints
- Use context managers (yield) for cleanup

**2. Avoid N+1 queries**:
- Use joinedload or selectinload
- Profile queries with SQL echo
- Measure with query count logging

**3. Transactions**:
- Wrap multi-operation flows in try/except
- Call commit() explicitly for multi-step operations
- Use rollback() on any error

**4. Testing**:
- Use SQLite for fast tests
- Override get_db dependency
- Create/drop tables per test or use transaction rollback

### Next Steps

In the next section, we'll explore **Authentication (JWT, OAuth2)**: implementing secure authentication with JSON Web Tokens, OAuth2 flows, and integrating with our database layer for user management.

**Production mindset**: Database integration is the foundation of your API. Master session management, transactions, and the repository pattern for clean, maintainable code.
`,
};
