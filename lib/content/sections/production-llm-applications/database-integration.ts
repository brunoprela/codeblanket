export const databaseIntegrationContent = `
# Database Integration

## Introduction

Production LLM applications need databases for storing conversation history, user data, caching, and application state. This section covers PostgreSQL patterns, connection pooling, ORMs, and database optimization for LLM workloads.

## PostgreSQL for LLM Applications

\`\`\`python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    system_prompt = Column(Text)
    metadata = Column(JSON)

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), index=True)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer)
    cost = Column(Numeric(10, 6))
    created_at = Column(DateTime, default=datetime.utcnow)
    
# Connection with pooling
engine = create_engine(
    'postgresql://user:pass@localhost/llm_db',
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker (bind=engine)

def get_conversation_history (conversation_id: int):
    """Get conversation with messages."""
    session = SessionLocal()
    try:
        conversation = session.query(Conversation).filter_by (id=conversation_id).first()
        messages = session.query(Message).filter_by (conversation_id=conversation_id).all()
        return conversation, messages
    finally:
        session.close()
\`\`\`

## Connection Pooling

\`\`\`python
from sqlalchemy.pool import QueuePool
import psycopg2
from psycopg2 import pool

# SQLAlchemy pooling
engine = create_engine(
    'postgresql://localhost/db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600
)

# Direct psycopg2 pooling
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host="localhost",
    database="llm_db",
    user="user",
    password="pass"
)
\`\`\`

## pgvector for Embeddings

\`\`\`python
from pgvector.sqlalchemy import Vector

class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(String(255), index=True)
    content = Column(Text)
    embedding = Column(Vector(1536))  # OpenAI embedding size
    
# Query by similarity
def find_similar_documents (query_embedding, limit=10):
    session = SessionLocal()
    results = session.query(DocumentEmbedding).order_by(
        DocumentEmbedding.embedding.cosine_distance (query_embedding)
    ).limit (limit).all()
    return results
\`\`\`

## Caching Queries

\`\`\`python
from functools import lru_cache
import redis

redis_client = redis.Redis()

def cached_query (query_key: str):
    """Cache database query results in Redis."""
    # Check cache
    cached = redis_client.get (f"query:{query_key}")
    if cached:
        return json.loads (cached)
    
    # Query database
    result = expensive_database_query()
    
    # Cache result
    redis_client.setex (f"query:{query_key}", 3600, json.dumps (result))
    
    return result
\`\`\`

## Transaction Management

\`\`\`python
from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
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
    conversation = Conversation (user_id="user_123")
    session.add (conversation)
    session.flush()
    
    message = Message(
        conversation_id=conversation.id,
        role="user",
        content="Hello"
    )
    session.add (message)
\`\`\`

## Best Practices

1. **Use connection pooling** to avoid overhead
2. **Index frequently queried columns** (user_id, conversation_id)
3. **Use pgvector** for semantic search
4. **Cache expensive queries** in Redis
5. **Use transactions** for consistency
6. **Monitor query performance** and optimize slow queries
7. **Use read replicas** for read-heavy workloads
8. **Archive old conversations** to keep tables small
9. **Use JSONB** for flexible metadata storage
10. **Set up database backups** for disaster recovery

Proper database integration ensures your LLM application scales while maintaining data integrity and performance.
`;
