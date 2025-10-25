export const advancedAlembic = {
  title: 'Advanced Alembic Techniques',
  id: 'advanced-alembic',
  content: `
# Advanced Alembic Techniques

## Introduction

Beyond basic migrations, Alembic supports advanced scenarios: custom operations, multiple database migrations, branching strategies, and complex data transformations. These techniques are essential for large-scale production systems with sophisticated requirements.

**Topics covered:**
- Custom migration operations
- Batch operations for SQLite
- Multiple database migrations
- Branching and merging strategies
- Testing migrations thoroughly
- Production migration patterns
- Zero-downtime advanced techniques

---

## Custom Migration Operations

### Creating Custom Operations

\`\`\`python
"""
Custom Alembic Operations
"""

from alembic.operations import Operations, MigrateOperation

@Operations.register_operation("create_view")
class CreateViewOp(MigrateOperation):
    """Custom operation to create database view"""
    
    def __init__(self, view_name, select_query):
        self.view_name = view_name
        self.select_query = select_query
    
    @classmethod
    def create_view (cls, operations, view_name, select_query):
        op = CreateViewOp (view_name, select_query)
        return operations.invoke (op)

@Operations.implementation_for(CreateViewOp)
def create_view (operations, operation):
    operations.execute(
        f"CREATE VIEW {operation.view_name} AS {operation.select_query}"
    )

# Usage in migration
def upgrade():
    op.create_view(
        'active_users_view',
        'SELECT * FROM users WHERE is_active = true'
    )

def downgrade():
    op.execute('DROP VIEW active_users_view')
\`\`\`

---

## Batch Operations (SQLite)

\`\`\`python
"""
Batch Mode for SQLite
"""

# SQLite doesn't support many ALTER TABLE operations
# Batch mode: Creates temp table, copies data, renames

def upgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column (sa.Column('age', sa.Integer()))
        batch_op.create_index('ix_users_age', ['age'])
        batch_op.drop_column('old_column')
    
    # Alembic:
    # 1. CREATE TABLE users_temp WITH new schema
    # 2. INSERT INTO users_temp SELECT FROM users
    # 3. DROP TABLE users
    # 4. ALTER TABLE users_temp RENAME TO users
\`\`\`

---

## Multiple Database Migrations

\`\`\`python
"""
Multi-Database Setup
"""

# alembic.ini
[app_db]
sqlalchemy.url = postgresql://localhost/app_db

[analytics_db]
sqlalchemy.url = postgresql://localhost/analytics_db

# env.py
def run_migrations_online():
    # Get database name from config
    db_name = context.config.get_section_option(
        context.config.config_ini_section,
        'database'
    ) or 'app_db'
    
    configuration = config.get_section (db_name)
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure (connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

# Run for specific database
# alembic -n app_db upgrade head
# alembic -n analytics_db upgrade head
\`\`\`

---

## Summary

✅ Custom operations for database-specific features  
✅ Batch mode for SQLite ALTER TABLE limitations  
✅ Multiple database support with sections  
✅ Advanced testing and validation  
✅ Production-ready migration patterns
`,
};
