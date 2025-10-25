export const multiDatabaseSharding = {
  title: 'Multi-Database & Sharding',
  id: 'multi-database-sharding',
  content: `
# Multi-Database & Sharding

## Introduction

As applications scale, single databases become bottlenecks. This section covers horizontal scaling strategies: read replicas, multi-database configurations, sharding patterns, and distributed database management with SQLAlchemy.

In this section, you'll master:
- Read/write splitting with replicas
- Multi-database configurations
- Horizontal sharding strategies
- Routing queries to correct databases
- Distributed transactions
- Data consistency challenges
- Migration across shards
- Production patterns

### Why Multi-Database Matters

**Scalability reality**: Single database limits: CPU, memory, I/O, connection limits. Solutions: Vertical scaling (limited, expensive), horizontal scaling (distributed, cost-effective). Multi-database architectures enable massive scale (millions of users, billions of records).

---

## Read Replicas & Write Master

### Basic Read/Write Splitting

\`\`\`python
"""
Read Replicas for Query Scaling
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Master database (writes)
master_engine = create_engine(
    "postgresql://master.db.example.com/myapp",
    pool_size=20,
    max_overflow=40
)

# Read replicas (reads)
replica_engines = [
    create_engine(
        "postgresql://replica1.db.example.com/myapp",
        pool_size=50,
        max_overflow=100
    ),
    create_engine(
        "postgresql://replica2.db.example.com/myapp",
        pool_size=50,
        max_overflow=100
    )
]

# Session factories
MasterSession = sessionmaker(bind=master_engine)
ReplicaSession = sessionmaker(bind=replica_engines[0])

# Usage
@contextmanager
def get_write_session():
    """Session for writes (master)"""
    session = MasterSession()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

@contextmanager
def get_read_session():
    """Session for reads (replica)"""
    session = ReplicaSession()
    try:
        yield session
    finally:
        session.close()

# Application code
def create_user(email: str):
    with get_write_session() as session:
        user = User(email=email)
        session.add(user)
        # Commits to master

def get_users():
    with get_read_session() as session:
        return session.execute(select(User)).scalars().all()
        # Reads from replica
\`\`\`

### Automatic Routing

\`\`\`python
"""
Automatic Read/Write Routing
"""

from sqlalchemy.orm import Session

class RoutingSession(Session):
    """Session that routes reads to replicas, writes to master"""
    
    def __init__(self, master_engine, replica_engines, **kwargs):
        super().__init__(bind=master_engine, **kwargs)
        self.master_engine = master_engine
        self.replica_engines = replica_engines
        self._replica_index = 0
    
    def get_bind(self, mapper=None, clause=None):
        """Route to appropriate database"""
        # Writes: INSERT, UPDATE, DELETE
        if self._flushing:
            return self.master_engine
        
        # Reads: SELECT
        if clause is not None:
            if clause.is_select:
                # Round-robin across replicas
                engine = self.replica_engines[self._replica_index]
                self._replica_index = (self._replica_index + 1) % len(self.replica_engines)
                return engine
        
        # Default: master
        return self.master_engine

# Usage
session = RoutingSession(master_engine, replica_engines)

# SELECT queries go to replicas
users = session.execute(select(User)).scalars().all()

# INSERTs go to master
user = User(email="test@example.com")
session.add(user)
session.commit()
\`\`\`

---

## Multi-Database Configuration

### Multiple Application Databases

\`\`\`python
"""
Multiple Databases (e.g., Users DB + Analytics DB)
"""

# Database engines
users_engine = create_engine("postgresql://localhost/users_db")
analytics_engine = create_engine("postgresql://localhost/analytics_db")

# Bind models to databases
class Base(DeclarativeBase):
    pass

class AnalyticsBase(DeclarativeBase):
    pass

# User models (users_db)
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str]

# Analytics models (analytics_db)
class PageView(AnalyticsBase):
    __tablename__ = "page_views"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int]
    page: Mapped[str]
    timestamp: Mapped[datetime]

# Session with multiple binds
Session = sessionmaker()
Session.configure(binds={
    User: users_engine,
    PageView: analytics_engine
})

session = Session()

# Queries automatically route to correct DB
user = session.execute(select(User)).scalar_one()      # -> users_db
views = session.execute(select(PageView)).scalars().all()  # -> analytics_db
\`\`\`

### Database Context Manager

\`\`\`python
"""
Explicit Database Selection
"""

class MultiDatabaseSession:
    """Manage multiple database connections"""
    
    def __init__(self, databases: dict[str, Engine]):
        self.databases = databases
        self.sessions: dict[str, Session] = {}
    
    def get_session(self, db_name: str) -> Session:
        """Get session for specific database"""
        if db_name not in self.sessions:
            engine = self.databases[db_name]
            SessionLocal = sessionmaker(bind=engine)
            self.sessions[db_name] = SessionLocal()
        return self.sessions[db_name]
    
    def commit_all(self):
        """Commit all sessions"""
        for session in self.sessions.values():
            session.commit()
    
    def rollback_all(self):
        """Rollback all sessions"""
        for session in self.sessions.values():
            session.rollback()
    
    def close_all(self):
        """Close all sessions"""
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()

# Usage
databases = {
    "users": create_engine("postgresql://localhost/users_db"),
    "analytics": create_engine("postgresql://localhost/analytics_db")
}

multi_db = MultiDatabaseSession(databases)

# Use specific databases
users_session = multi_db.get_session("users")
analytics_session = multi_db.get_session("analytics")

user = users_session.execute(select(User)).scalar_one()
analytics_session.add(PageView(user_id=user.id, page="/home"))

multi_db.commit_all()
multi_db.close_all()
\`\`\`

---

## Sharding Strategies

### Hash-Based Sharding

\`\`\`python
"""
Hash-Based Sharding by User ID
"""

import hashlib

class ShardedDatabase:
    """Shard users across multiple databases"""
    
    def __init__(self, shard_engines: list[Engine]):
        self.shard_engines = shard_engines
        self.num_shards = len(shard_engines)
    
    def get_shard(self, user_id: int) -> Engine:
        """Get shard for user ID"""
        shard_index = user_id % self.num_shards
        return self.shard_engines[shard_index]
    
    def get_shard_by_hash(self, key: str) -> Engine:
        """Get shard by hash of key"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.num_shards
        return self.shard_engines[shard_index]

# Setup shards
shard_engines = [
    create_engine("postgresql://localhost/shard_0"),
    create_engine("postgresql://localhost/shard_1"),
    create_engine("postgresql://localhost/shard_2"),
    create_engine("postgresql://localhost/shard_3")
]

sharded_db = ShardedDatabase(shard_engines)

# Get user from correct shard
def get_user(user_id: int) -> User:
    engine = sharded_db.get_shard(user_id)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        return session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one()

# Create user in correct shard
def create_user(user_id: int, email: str):
    engine = sharded_db.get_shard(user_id)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        user = User(id=user_id, email=email)
        session.add(user)
        session.commit()
\`\`\`

### Range-Based Sharding

\`\`\`python
"""
Range-Based Sharding (e.g., by User ID Ranges)
"""

class RangeShardedDatabase:
    """Shard by ranges"""
    
    def __init__(self):
        self.shard_ranges = [
            (0, 1_000_000, create_engine("postgresql://localhost/shard_0")),
            (1_000_000, 2_000_000, create_engine("postgresql://localhost/shard_1")),
            (2_000_000, 3_000_000, create_engine("postgresql://localhost/shard_2")),
        ]
    
    def get_shard(self, user_id: int) -> Engine:
        """Get shard for user ID range"""
        for start, end, engine in self.shard_ranges:
            if start <= user_id < end:
                return engine
        raise ValueError(f"No shard for user_id {user_id}")

# Usage
range_db = RangeShardedDatabase()
engine = range_db.get_shard(500_000)  # Returns shard_0
\`\`\`

### Geographic Sharding

\`\`\`python
"""
Geographic Sharding (e.g., by Region)
"""

class GeographicShardedDatabase:
    """Shard by geographic region"""
    
    def __init__(self):
        self.shard_map = {
            "us-east": create_engine("postgresql://us-east.db.example.com/myapp"),
            "us-west": create_engine("postgresql://us-west.db.example.com/myapp"),
            "eu": create_engine("postgresql://eu.db.example.com/myapp"),
            "asia": create_engine("postgresql://asia.db.example.com/myapp"),
        }
    
    def get_shard(self, region: str) -> Engine:
        """Get shard for region"""
        return self.shard_map.get(region)

# Usage
geo_db = GeographicShardedDatabase()

def get_user_by_region(user_id: int, region: str):
    engine = geo_db.get_shard(region)
    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        return session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one()
\`\`\`

---

## Query Routing

### Shard-Aware Session

\`\`\`python
"""
Session that Routes to Correct Shard
"""

class ShardedSession(Session):
    """Session with shard routing"""
    
    def __init__(self, sharded_db: ShardedDatabase, shard_key: int, **kwargs):
        engine = sharded_db.get_shard(shard_key)
        super().__init__(bind=engine, **kwargs)
        self.sharded_db = sharded_db
        self.shard_key = shard_key

# Usage
def get_user_posts(user_id: int):
    """Get user and their posts (same shard)"""
    session = ShardedSession(sharded_db, shard_key=user_id)
    
    user = session.execute(
        select(User).where(User.id == user_id)
    ).scalar_one()
    
    posts = session.execute(
        select(Post).where(Post.user_id == user_id)
    ).scalars().all()
    
    session.close()
    return user, posts
\`\`\`

### Cross-Shard Queries

\`\`\`python
"""
Query Across Multiple Shards
"""

def get_all_active_users(sharded_db: ShardedDatabase) -> list[User]:
    """Query all shards and merge results"""
    all_users = []
    
    for engine in sharded_db.shard_engines:
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as session:
            users = session.execute(
                select(User).where(User.is_active == True)
            ).scalars().all()
            all_users.extend(users)
    
    return all_users

def get_user_count_by_shard(sharded_db: ShardedDatabase) -> dict[int, int]:
    """Count users in each shard"""
    counts = {}
    
    for index, engine in enumerate(sharded_db.shard_engines):
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as session:
            count = session.execute(
                select(func.count()).select_from(User)
            ).scalar()
            counts[index] = count
    
    return counts
\`\`\`

---

## Distributed Transactions

### Two-Phase Commit (2PC)

\`\`\`python
"""
Two-Phase Commit Across Shards
"""

from sqlalchemy.engine import Engine

def distributed_transaction(
    engines: list[Engine],
    operations: list[callable]
):
    """Execute operations across multiple databases with 2PC"""
    connections = []
    transactions = []
    
    try:
        # Phase 1: Prepare all transactions
        for engine, operation in zip(engines, operations):
            conn = engine.connect()
            transaction = conn.begin_twophase()
            connections.append(conn)
            transactions.append(transaction)
            
            # Execute operation
            operation(conn)
            
            # Prepare transaction
            transaction.prepare()
        
        # Phase 2: Commit all transactions
        for transaction in transactions:
            transaction.commit()
    
    except Exception as e:
        # Rollback all transactions on error
        for transaction in transactions:
            try:
                transaction.rollback()
            except:
                pass
        raise e
    
    finally:
        for conn in connections:
            conn.close()

# Usage
def transfer_credits(from_user_id: int, to_user_id: int, amount: int):
    """Transfer credits between users (possibly different shards)"""
    from_shard = sharded_db.get_shard(from_user_id)
    to_shard = sharded_db.get_shard(to_user_id)
    
    def deduct_credits(conn):
        conn.execute(
            update(User)
            .where(User.id == from_user_id)
            .values(credits=User.credits - amount)
        )
    
    def add_credits(conn):
        conn.execute(
            update(User)
            .where(User.id == to_user_id)
            .values(credits=User.credits + amount)
        )
    
    distributed_transaction(
        [from_shard, to_shard],
        [deduct_credits, add_credits]
    )
\`\`\`

---

## Data Consistency

### Replication Lag

\`\`\`python
"""
Handle Replication Lag
"""

def get_user_with_fallback(user_id: int):
    """Read from replica, fallback to master if not found"""
    # Try replica first
    with get_read_session() as session:
        user = session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()
        
        if user is not None:
            return user
    
    # Fallback to master (for just-created records)
    with get_write_session() as session:
        return session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one()

def read_your_writes(user_id: int):
    """Force read from master after write"""
    # Write to master
    with get_write_session() as session:
        user = User(id=user_id, email="test@example.com")
        session.add(user)
        session.commit()
    
    # Read from master (not replica) to guarantee consistency
    with get_write_session() as session:
        return session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one()
\`\`\`

---

## Migration Across Shards

### Shard Migration Script

\`\`\`python
"""
Run Alembic Migration Across All Shards
"""

from alembic.config import Config
from alembic import command

def migrate_all_shards(shard_urls: list[str]):
    """Run migrations on all shards"""
    for index, url in enumerate(shard_urls):
        print(f"Migrating shard {index}: {url}")
        
        # Configure Alembic for this shard
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", url)
        
        # Run migration
        command.upgrade(alembic_cfg, "head")
        
        print(f"Shard {index} migrated successfully")

# Usage
shard_urls = [
    "postgresql://localhost/shard_0",
    "postgresql://localhost/shard_1",
    "postgresql://localhost/shard_2",
    "postgresql://localhost/shard_3"
]

migrate_all_shards(shard_urls)
\`\`\`

---

## Production Patterns

### Shard Management Service

\`\`\`python
"""
Production Shard Management
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ShardInfo:
    index: int
    url: str
    engine: Engine
    is_active: bool
    capacity: int  # Max users
    current_count: int

class ShardManager:
    """Manage shards in production"""
    
    def __init__(self):
        self.shards: list[ShardInfo] = []
        self.load_shard_config()
    
    def load_shard_config(self):
        """Load shard configuration from database"""
        # In practice: load from config DB
        pass
    
    def get_shard_for_new_user(self) -> ShardInfo:
        """Find shard with capacity for new user"""
        for shard in self.shards:
            if shard.is_active and shard.current_count < shard.capacity:
                return shard
        raise Exception("No available shards")
    
    def rebalance_shards(self):
        """Move users between shards to balance load"""
        # Implementation: gradual migration
        pass
    
    def add_new_shard(self, url: str):
        """Add new shard to cluster"""
        engine = create_engine(url)
        shard = ShardInfo(
            index=len(self.shards),
            url=url,
            engine=engine,
            is_active=True,
            capacity=1_000_000,
            current_count=0
        )
        self.shards.append(shard)
        
        # Run migrations on new shard
        self.migrate_shard(shard)
\`\`\`

---

## Summary

### Key Takeaways

✅ **Read replicas**: Scale reads, master for writes  
✅ **Automatic routing**: RoutingSession, get_bind()  
✅ **Multi-database**: Separate DBs for different domains  
✅ **Sharding strategies**: Hash, range, geographic  
✅ **Cross-shard queries**: Iterate shards, merge results  
✅ **Distributed transactions**: Two-phase commit (2PC)  
✅ **Consistency**: Handle replication lag, read-your-writes  
✅ **Migrations**: Run across all shards

### Sharding Decision Tree

✅ **< 1M users**: Single database  
✅ **1M-10M users**: Read replicas + caching  
✅ **10M+ users**: Horizontal sharding  
✅ **Geographic**: Region-based sharding  
✅ **Multi-tenant**: Tenant-based sharding

### Production Checklist

✅ Read/write splitting configured  
✅ Replication lag monitoring  
✅ Shard routing tested  
✅ Cross-shard query strategy  
✅ Migration scripts for all shards  
✅ Shard rebalancing plan  
✅ Monitoring per shard

### Next Steps

In the next section, we'll explore **SQLAlchemy Production Patterns**: connection pooling, caching, monitoring, and production-ready architectures.
`,
};
