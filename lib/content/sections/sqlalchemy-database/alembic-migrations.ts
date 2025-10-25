export const alembicMigrations = {
  title: 'Alembic: Database Migrations',
  id: 'alembic-migrations',
  content: `
# Alembic: Database Migrations

## Introduction

Database migrations are version control for your schema. Alembic is SQLAlchemy's migration tool, enabling you to evolve database schemas safely in production. Without migrations, schema changes require manual SQL scripts, risking errors and downtime.

In this section, you'll master:
- Why database migrations are essential
- Alembic installation and configuration
- Creating and running migrations
- Auto-generating migrations from models
- Migration file structure and operations
- Downgrade strategies and rollbacks
- Team collaboration with migrations
- Production migration workflows

### Why Alembic Matters

**Production reality**: Schema changes are frequent. Adding columns, indexes, or tables must be reproducible across dev, staging, and production. Manual SQL scripts cause: version drift, human error, deployment failures, and rollback nightmares. Alembic solves this with version-controlled, testable migrations.

---

## Installation and Setup

### Installing Alembic

\`\`\`bash
# Install Alembic
pip install alembic

# Initialize Alembic in your project
alembic init alembic

# Creates:
# alembic/
#   ├── versions/          # Migration files
#   ├── env.py            # Migration environment
#   ├── script.py.mako    # Migration template
#   └── README
# alembic.ini             # Configuration
\`\`\`

### Project Structure

\`\`\`
myproject/
├── alembic/
│   ├── versions/
│   │   └── (migration files)
│   ├── env.py
│   ├── script.py.mako
│   └── README
├── alembic.ini
├── models.py           # SQLAlchemy models
└── database.py         # Engine/session setup
\`\`\`

---

## Configuration

### alembic.ini

\`\`\`ini
# alembic.ini

[alembic]
# Path to migration scripts
script_location = alembic

# Database URL
sqlalchemy.url = postgresql://user:password@localhost/mydb

# Logging
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# Output encoding
output_encoding = utf-8
\`\`\`

### env.py Configuration

\`\`\`python
"""
alembic/env.py - Configure migration environment
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os

# Import your models' Base
from myapp.models import Base

# Alembic Config object
config = context.config

# Configure logging
fileConfig(config.config_file_name)

# Add your model's MetaData for autogenerate
target_metadata = Base.metadata

# Read DB URL from environment
def get_url():
    return os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))

def run_migrations_offline():
    """Run migrations in 'offline' mode"""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode"""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
\`\`\`

---

## Creating Migrations

### Manual Migration Creation

\`\`\`bash
# Create empty migration
alembic revision -m "create users table"

# Creates: alembic/versions/20240101_1200_abc123_create_users_table.py
\`\`\`

\`\`\`python
"""
Manual Migration File
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = 'abc123'
down_revision = None  # First migration
branch_labels = None
depends_on = None

def upgrade():
    """Apply migration"""
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Create index
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    """Revert migration"""
    op.drop_index('ix_users_email')
    op.drop_table('users')
\`\`\`

### Auto-Generate Migrations

\`\`\`bash
# Auto-generate from model changes
alembic revision --autogenerate -m "add posts table"

# Alembic compares:
# - Current database schema
# - Models in Base.metadata
# - Generates migration for differences
\`\`\`

\`\`\`python
"""
Auto-Generated Migration
"""

from alembic import op
import sqlalchemy as sa

revision = 'def456'
down_revision = 'abc123'  # Previous migration

def upgrade():
    # Alembic auto-generated this
    op.create_table(
        'posts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_posts_user_id'), 'posts', ['user_id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_posts_user_id'), table_name='posts')
    op.drop_table('posts')
\`\`\`

---

## Running Migrations

### Basic Commands

\`\`\`bash
# Show current revision
alembic current

# Show migration history
alembic history --verbose

# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade abc123

# Upgrade one step
alembic upgrade +1

# Downgrade one step
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade abc123

# Downgrade all (dangerous!)
alembic downgrade base

# Show SQL without executing
alembic upgrade head --sql
\`\`\`

### Migration Workflow

\`\`\`bash
# 1. Make model changes
# Edit models.py: Add new column

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String(255))
    bio = Column(Text)  # NEW COLUMN

# 2. Auto-generate migration
alembic revision --autogenerate -m "add bio column to users"

# 3. Review generated migration
# Check alembic/versions/xxx_add_bio_column_to_users.py

# 4. Test migration
alembic upgrade head  # Apply migration

# 5. Test downgrade
alembic downgrade -1  # Revert migration

# 6. Verify
alembic upgrade head  # Re-apply

# 7. Commit migration file
git add alembic/versions/xxx_add_bio_column_to_users.py
git commit -m "Add bio column to users"
\`\`\`

---

## Migration Operations

### Table Operations

\`\`\`python
"""
Table Creation and Deletion
"""

def upgrade():
    # Create table
    op.create_table(
        'products',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('price', sa.Numeric(10, 2), nullable=False),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )
    
    # Drop table
    op.drop_table('old_table')

def downgrade():
    op.drop_table('products')
\`\`\`

### Column Operations

\`\`\`python
"""
Column Modifications
"""

def upgrade():
    # Add column
    op.add_column('users', sa.Column('phone', sa.String(20)))
    
    # Add column with default
    op.add_column('users', sa.Column('is_active', sa.Boolean, server_default='true'))
    
    # Drop column
    op.drop_column('users', 'old_column')
    
    # Rename column
    op.alter_column('users', 'username', new_column_name='user_name')
    
    # Change column type
    op.alter_column('users', 'age',
                    existing_type=sa.String(),
                    type_=sa.Integer(),
                    postgresql_using='age::integer')
    
    # Change nullable
    op.alter_column('users', 'email', nullable=False)
    
    # Change default
    op.alter_column('users', 'status',
                    server_default='active')
\`\`\`

### Index and Constraint Operations

\`\`\`python
"""
Indexes and Constraints
"""

def upgrade():
    # Create index
    op.create_index('ix_users_email', 'users', ['email'])
    
    # Create unique index
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    
    # Create composite index
    op.create_index('ix_posts_user_created', 'posts', ['user_id', 'created_at'])
    
    # Drop index
    op.drop_index('ix_old_index')
    
    # Add foreign key
    op.create_foreign_key(
        'fk_posts_user_id',
        'posts', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    
    # Drop foreign key
    op.drop_constraint('fk_old_constraint', 'table_name', type_='foreignkey')
    
    # Add unique constraint
    op.create_unique_constraint('uq_users_email', 'users', ['email'])
    
    # Add check constraint
    op.create_check_constraint(
        'ck_users_age',
        'users',
        'age >= 0 AND age <= 150'
    )
\`\`\`

---

## Data Migrations

### Modifying Data

\`\`\`python
"""
Data Migration with SQL
"""

def upgrade():
    # Add column
    op.add_column('users', sa.Column('full_name', sa.String(200)))
    
    # Populate data
    op.execute("""
        UPDATE users 
        SET full_name = first_name || ' ' || last_name
        WHERE full_name IS NULL
    """)
    
    # Make NOT NULL after populating
    op.alter_column('users', 'full_name', nullable=False)

def downgrade():
    op.drop_column('users', 'full_name')
\`\`\`

### Using ORM in Migrations

\`\`\`python
"""
ORM-Based Data Migration
"""

from sqlalchemy.orm import Session
from sqlalchemy import Table, Column, Integer, String, MetaData

def upgrade():
    # Create column
    op.add_column('users', sa.Column('slug', sa.String(100)))
    
    # Use ORM to update data
    bind = op.get_bind()
    session = Session(bind=bind)
    
    # Define table (can't import models - circular dependency)
    metadata = MetaData()
    users = Table('users', metadata,
        Column('id', Integer, primary_key=True),
        Column('username', String),
        Column('slug', String)
    )
    
    # Generate slugs
    for user in session.execute(sa.select(users)).mappings():
        slug = user['username'].lower().replace(' ', '-')
        session.execute(
            users.update().where(users.c.id == user['id']).values(slug=slug)
        )
    
    session.commit()

def downgrade():
    op.drop_column('users', 'slug')
\`\`\`

---

## Migration Best Practices

### Review Auto-Generated Migrations

\`\`\`python
"""
Always Review Autogenerate Output!
"""

# Alembic may miss:
# - Rename operations (sees drop + create)
# - Table renames
# - Enum changes
# - Custom types
# - Server defaults
# - Complex constraints

# After autogenerate, manually verify and fix:
def upgrade():
    # Alembic generated drop + create (wrong!)
    # op.drop_column('users', 'username')
    # op.add_column('users', sa.Column('user_name', sa.String(50)))
    
    # Correct: Rename operation
    op.alter_column('users', 'username', new_column_name='user_name')
\`\`\`

### Zero-Downtime Migrations

\`\`\`python
"""
Safe Production Migration Pattern
"""

# Phase 1: Add nullable column
def upgrade():
    op.add_column('users', sa.Column('new_email', sa.String(255)))

# Deploy code that writes to both old_email and new_email

# Phase 2: Backfill data
def upgrade():
    op.execute("UPDATE users SET new_email = email WHERE new_email IS NULL")

# Phase 3: Make NOT NULL
def upgrade():
    op.alter_column('users', 'new_email', nullable=False)

# Deploy code that only uses new_email

# Phase 4: Drop old column
def upgrade():
    op.drop_column('users', 'email')
    op.alter_column('users', 'new_email', new_column_name='email')
\`\`\`

### Testing Migrations

\`\`\`python
"""
Test Migrations
"""

import pytest
from alembic.command import upgrade, downgrade
from alembic.config import Config

def test_migration_up_down():
    """Test migration can apply and revert"""
    alembic_cfg = Config("alembic.ini")
    
    # Upgrade
    upgrade(alembic_cfg, "head")
    
    # Downgrade
    downgrade(alembic_cfg, "-1")
    
    # Re-upgrade
    upgrade(alembic_cfg, "head")

def test_migration_idempotent():
    """Test migration can run multiple times"""
    alembic_cfg = Config("alembic.ini")
    
    # Run twice
    upgrade(alembic_cfg, "head")
    upgrade(alembic_cfg, "head")  # Should not error
\`\`\`

---

## Team Collaboration

### Handling Migration Conflicts

\`\`\`bash
# Two developers create migrations simultaneously

# Developer A:
alembic revision -m "add column A"
# Creates: abc123_add_column_a.py (down_revision='xyz789')

# Developer B (parallel):
alembic revision -m "add column B"  
# Creates: def456_add_column_b.py (down_revision='xyz789')

# Both have same down_revision - CONFLICT!

# Solution: Merge migrations
alembic merge -m "merge heads" abc123 def456
# Creates: merge_abc123_def456.py
# Has: down_revision = ('abc123', 'def456')

# Or: Manually set down_revision
# Edit def456: down_revision = 'abc123'
\`\`\`

### Migration Naming Convention

\`\`\`python
"""
Clear Migration Names
"""

# GOOD names:
# - 20240115_add_user_email_column.py
# - 20240115_create_posts_table.py
# - 20240115_add_users_email_index.py
# - 20240115_populate_user_slugs.py

# BAD names:
# - migration.py
# - update.py
# - fix.py
\`\`\`

---

## Production Deployment

### Deployment Workflow

\`\`\`bash
#!/bin/bash
# deploy.sh

set -e  # Exit on error

echo "Running database migrations..."

# 1. Show pending migrations
alembic history --verbose | grep -A5 "current"

# 2. Show SQL that will be executed
alembic upgrade head --sql > migration.sql

# 3. Review (in production, send to Slack/PagerDuty)
cat migration.sql

# 4. Backup database (if needed)
pg_dump mydb > backup_$(date +%Y%m%d_%H%M%S).sql

# 5. Run migration
alembic upgrade head

# 6. Verify
alembic current

echo "Migrations complete!"
\`\`\`

### Rollback Strategy

\`\`\`bash
# If deployment fails, rollback migration

# Downgrade one step
alembic downgrade -1

# Or downgrade to specific revision
alembic downgrade abc123

# Then rollback application code
git revert HEAD
\`\`\`

---

## Summary

### Key Takeaways

✅ **Alembic = version control for database schema**  
✅ **Auto-generate**: Use \`--autogenerate\`, but always review  
✅ **Migrations are code**: Version control, test, review  
✅ **upgrade() + downgrade()**: Always provide rollback path  
✅ **Zero-downtime**: Add nullable, backfill, make NOT NULL, drop old  
✅ **Test migrations**: Up, down, and idempotency  
✅ **Team collaboration**: Merge conflicting migrations

### Workflow Checklist

✅ Make model changes  
✅ Generate migration: \`alembic revision --autogenerate\`  
✅ Review migration file  
✅ Test: \`alembic upgrade head\`  
✅ Test rollback: \`alembic downgrade -1\`  
✅ Commit migration file  
✅ Deploy to staging first  
✅ Run in production  
✅ Verify: \`alembic current\`

### Common Operations

- **Create table**: \`op.create_table()\`
- **Add column**: \`op.add_column()\`
- **Create index**: \`op.create_index()\`
- **Data migration**: \`op.execute()\` or ORM with Session
- **Rename column**: \`op.alter_column(new_column_name=...)\`
- **Foreign key**: \`op.create_foreign_key()\`

### Next Steps

In the next section, we'll explore **Advanced Alembic Techniques**: branching, multiple databases, custom operations, and complex migration scenarios.
`,
};
