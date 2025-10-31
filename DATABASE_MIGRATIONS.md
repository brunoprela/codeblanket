# Database Migrations Guide

## TL;DR

❌ **Neon does NOT auto-update tables** when you change your code
✅ **You must manually run SQL migrations** to update the schema

## Current Approach: Manual SQL Migrations

### How It Works Now

1. You change your schema (e.g., add a new column)
2. You create a migration SQL file
3. You run it in Neon SQL Editor (or via CLI)
4. Changes are applied

**Pros:**

- ✅ Simple, no extra tools
- ✅ Full control over SQL
- ✅ Works for any database

**Cons:**

- ❌ Easy to forget migrations
- ❌ No rollback tracking
- ❌ Hard to coordinate across team/environments

## Recommended Approach: Drizzle ORM

Drizzle is a lightweight TypeScript ORM that handles migrations automatically.

### Install Drizzle (Optional, for future)

```bash
npm install drizzle-orm drizzle-kit
npm install -D @types/pg
```

### Why Drizzle?

- ✅ **Auto-generates migrations** from TypeScript schema
- ✅ **Version controlled** migrations (in Git)
- ✅ **Type-safe** database queries
- ✅ **Rollback support** (undo migrations)
- ✅ **Works perfectly with Neon** (official integration)

## Migration Workflow Options

### Option 1: Manual SQL Files (Current - Simple)

**Directory structure:**

```
lib/db/
├── schema.sql           # Initial schema
├── migrations/
│   ├── 001_initial.sql
│   ├── 002_add_user_preferences.sql
│   ├── 003_add_video_thumbnails.sql
│   └── README.md
```

**When you need to update schema:**

1. Create new migration file: `lib/db/migrations/00X_description.sql`
2. Write the SQL:

   ```sql
   -- Migration: Add user preferences table
   -- Date: 2024-10-31

   CREATE TABLE IF NOT EXISTS user_preferences (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     user_id TEXT NOT NULL,
     theme TEXT DEFAULT 'dark',
     created_at TIMESTAMPTZ DEFAULT NOW()
   );
   ```

3. Run in Neon SQL Editor
4. Commit to Git

**Track what's been run:**

```sql
-- Optional: Track migrations
CREATE TABLE IF NOT EXISTS migrations (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  executed_at TIMESTAMPTZ DEFAULT NOW()
);

-- After running a migration, record it
INSERT INTO migrations (name) VALUES ('002_add_user_preferences');
```

### Option 2: Drizzle Kit (Recommended - Automated)

**1. Define schema in TypeScript:**

```typescript
// lib/db/schema.ts
import { pgTable, text, uuid, timestamp, jsonb } from 'drizzle-orm/pg-core';

export const userProgress = pgTable('user_progress', {
  id: uuid('id').defaultRandom().primaryKey(),
  userId: text('user_id').notNull(),
  key: text('key').notNull(),
  value: jsonb('value').notNull(),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow(),
});
```

**2. Generate migration:**

```bash
npx drizzle-kit generate:pg
# Creates: drizzle/0001_migration.sql automatically
```

**3. Run migration:**

```bash
npx drizzle-kit push:pg
# Applies to Neon database
```

**Benefits:**

- Migrations auto-generated from TypeScript
- Type-safe queries
- Version controlled
- Easy rollbacks

### Option 3: Prisma (Alternative ORM)

```bash
npm install prisma @prisma/client
npx prisma init
```

Define schema in `prisma/schema.prisma`, then:

```bash
npx prisma migrate dev --name add_user_preferences
```

## How to Update Tables (Step-by-Step)

### Scenario: You Want to Add a New Column

**Example:** Add `last_login` to track when users last accessed the app

#### Step 1: Create Migration File

Create: `lib/db/migrations/002_add_last_login.sql`

```sql
-- Migration 002: Add last_login tracking
-- Date: 2024-10-31
-- Description: Track user's last login time

-- Add column to user_progress (or create new table)
ALTER TABLE user_progress
ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_user_progress_last_login
ON user_progress(user_id, last_login);
```

#### Step 2: Run in Neon Console

1. Go to Neon Console → SQL Editor
2. Copy the SQL from your migration file
3. Click **Run**
4. Verify: `\d user_progress` (shows table structure)

#### Step 3: Update TypeScript Types

```typescript
// lib/db/neon.ts
export interface ProgressRecord {
  id: string;
  user_id: string;
  key: string;
  value: unknown;
  created_at: string;
  updated_at: string;
  last_login?: string; // Add new field
}
```

#### Step 4: Commit to Git

```bash
git add lib/db/migrations/002_add_last_login.sql
git commit -m "Add last_login tracking to user_progress"
```

## Best Practices

### 1. Always Use Migrations (Never Direct ALTER TABLE)

**❌ Bad (Direct changes):**

```sql
-- Running random ALTER TABLE commands
ALTER TABLE user_progress ADD COLUMN random_field TEXT;
```

**✅ Good (Versioned migrations):**

```
migrations/
├── 001_initial.sql
├── 002_add_feature_x.sql
└── 003_add_feature_y.sql
```

### 2. Make Migrations Idempotent

Use `IF NOT EXISTS`, `IF EXISTS`, etc:

```sql
-- Safe to run multiple times
ALTER TABLE user_progress
ADD COLUMN IF NOT EXISTS new_field TEXT;

-- Not safe (errors if run twice)
ALTER TABLE user_progress
ADD COLUMN new_field TEXT;
```

### 3. Include Rollback Instructions

```sql
-- Migration 002: Add user preferences
-- UP:
ALTER TABLE user_progress ADD COLUMN preferences JSONB;

-- DOWN (rollback):
-- ALTER TABLE user_progress DROP COLUMN preferences;
```

### 4. Test Migrations on Dev First

1. Run on development database
2. Verify it works
3. Test your application
4. Then run on production

## Common Schema Changes

### Add Column

```sql
ALTER TABLE user_progress
ADD COLUMN IF NOT EXISTS new_field TEXT DEFAULT 'value';
```

### Add Index

```sql
CREATE INDEX IF NOT EXISTS idx_name
ON table_name(column_name);
```

### Add Table

```sql
CREATE TABLE IF NOT EXISTS new_table (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL
);
```

### Modify Column (Careful!)

```sql
-- Change column type (may require data conversion)
ALTER TABLE user_progress
ALTER COLUMN some_field TYPE BIGINT USING some_field::BIGINT;
```

### Remove Column (Dangerous!)

```sql
-- Make sure no code depends on this first!
ALTER TABLE user_progress
DROP COLUMN IF EXISTS old_field;
```

## Production Deployment Workflow

### Development → Production

**Development:**

1. Create migration file
2. Run on dev database
3. Test thoroughly
4. Commit to Git

**Production:**

1. Deploy code (Vercel auto-deploys from Git)
2. Run same migration on production database
3. Verify it worked

### Automated Deployment (Advanced)

**Using GitHub Actions + Neon API:**

```yaml
# .github/workflows/migrate.yml
name: Database Migration
on:
  push:
    branches: [main]
    paths:
      - 'lib/db/migrations/**'

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run migrations
        run: |
          psql ${{ secrets.DATABASE_URL }} < lib/db/migrations/latest.sql
```

## Migration Tracking Table (Optional)

Keep track of which migrations have been run:

```sql
-- Create migrations table
CREATE TABLE IF NOT EXISTS schema_migrations (
  id SERIAL PRIMARY KEY,
  migration_name TEXT NOT NULL UNIQUE,
  executed_at TIMESTAMPTZ DEFAULT NOW(),
  checksum TEXT -- Optional: verify file hasn't changed
);

-- After running each migration
INSERT INTO schema_migrations (migration_name)
VALUES ('002_add_user_preferences')
ON CONFLICT (migration_name) DO NOTHING;

-- Check what's been run
SELECT * FROM schema_migrations ORDER BY id;
```

## Your Current Setup

Right now, you're using **manual migrations**:

1. Schema defined in: `lib/db/schema.sql`
2. Migration script available: `lib/db/schema-blob-migration.sql`
3. Run manually in Neon SQL Editor

**This is fine for now!** As your app grows, consider Drizzle ORM for automation.

## When to Run Migrations

### You Need a Migration When:

- ✅ Adding a new table
- ✅ Adding/removing columns
- ✅ Adding/removing indexes
- ✅ Changing column types
- ✅ Adding constraints (UNIQUE, FOREIGN KEY, etc.)

### You DON'T Need a Migration When:

- ❌ Changing application code
- ❌ Updating environment variables
- ❌ Modifying frontend components
- ❌ Changing API logic (unless schema changes)

## Quick Reference Commands

### Check Current Schema

```sql
-- List all tables
\dt

-- Describe table structure
\d user_progress

-- Show indexes
\di

-- Show all columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'user_progress';
```

### Backup Before Migration

```sql
-- Backup table
CREATE TABLE user_progress_backup AS
SELECT * FROM user_progress;

-- Run migration
ALTER TABLE user_progress ADD COLUMN new_field TEXT;

-- If something goes wrong, restore
DROP TABLE user_progress;
ALTER TABLE user_progress_backup RENAME TO user_progress;
```

## Summary

**For now (manual approach):**

1. Create `.sql` file in `lib/db/migrations/`
2. Run in Neon SQL Editor
3. Commit to Git
4. Document in README

**Future upgrade (Drizzle ORM):**

1. Define schema in TypeScript
2. Run `drizzle-kit generate`
3. Run `drizzle-kit push`
4. Automatic migration history

Your current manual approach is totally fine for development! Consider upgrading to Drizzle when you have more complex migrations or a team. 🚀
