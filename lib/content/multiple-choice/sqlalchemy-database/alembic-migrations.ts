import { MultipleChoiceQuestion } from '@/lib/types';

export const alembicMigrationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-alembic-mc-1',
    question:
      'What is the purpose of the downgrade() function in an Alembic migration?',
    options: [
      'To improve database performance',
      'To revert the migration if deployment fails',
      'To compress the database',
      'To backup data before migration',
    ],
    correctAnswer: 1,
    explanation:
      'The downgrade() function provides a rollback path if migration or deployment fails. It should reverse all operations in upgrade(). Example: if upgrade() adds a column, downgrade() drops it. Critical for production: if deployment fails, you can "alembic downgrade -1" to revert schema changes. Always implement downgrade() - even if you think you won\'t need it, production emergencies happen. Some operations are irreversible (data deletion), document this in comments.',
  },
  {
    id: 'sql-alembic-mc-2',
    question: 'When should you manually review auto-generated migrations?',
    options: [
      'Never, auto-generate is always correct',
      'Always - Alembic may miss renames, enum changes, and custom operations',
      'Only for production environments',
      'Only if errors occur',
    ],
    correctAnswer: 1,
    explanation:
      "Always review auto-generated migrations! Alembic autogenerate has limitations: (1) Detects drop+create instead of rename (loses data!), (2) May miss enum/type changes, (3) Doesn't detect table renames, (4) Server defaults may not be detected, (5) Custom SQL or complex constraints missed. Example: renaming column generates op.drop_column() + op.add_column() (DATA LOSS) instead of op.alter_column (new_column_name=...). Always verify the generated migration makes sense before committing.",
  },
  {
    id: 'sql-alembic-mc-3',
    question: 'What does "alembic upgrade head" do?',
    options: [
      'Creates a new migration',
      'Applies all pending migrations to reach the latest version',
      'Rolls back the last migration',
      'Shows migration history',
    ],
    correctAnswer: 1,
    explanation:
      '"alembic upgrade head" applies all pending migrations from current revision to the latest (head). Example: current at revision abc123, latest is xyz789 with 3 migrations in between - runs all 3 to reach xyz789. "head" is a special identifier meaning "latest migration". Alternative commands: "alembic upgrade +1" (one step forward), "alembic upgrade abc123" (to specific revision). Production: Always run "alembic upgrade head --sql" first to review SQL before executing.',
  },
  {
    id: 'sql-alembic-mc-4',
    question:
      'What is the correct approach for zero-downtime column addition with a NOT NULL constraint?',
    options: [
      'Add column with NOT NULL and default in single migration',
      'Add nullable column, backfill data, then make NOT NULL in separate phases',
      'Add column, use ALTER TABLE SET NOT NULL immediately',
      'Drop table and recreate',
    ],
    correctAnswer: 1,
    explanation:
      'Zero-downtime requires phases: Phase 1: Add nullable column (instant, no lock). Phase 2: Backfill data in batches. Phase 3: Make NOT NULL after all data populated. Why not single migration with NOT NULL + default? Large tables: Adding NOT NULL column locks table for minutes (rewrites all rows). Batched approach: No exclusive locks, application stays online. Example: 10M row table, adding NOT NULL column takes 10 minutes with exclusive lock vs 0 downtime with phased approach.',
  },
  {
    id: 'sql-alembic-mc-5',
    question: 'What is the down_revision field in a migration file?',
    options: [
      'The database version being downgraded',
      'The previous migration this one builds upon',
      'The lowest supported database version',
      'The backup revision ID',
    ],
    correctAnswer: 1,
    explanation:
      'down_revision specifies the previous migration in the chain. It creates a linked list of migrations: None (first migration) → abc123 → def456 → xyz789 (head). Alembic uses this to determine order: to reach xyz789 from None, applies abc123, then def456, then xyz789. When two migrations have same down_revision, causes "multiple heads" error (both claim to be next). Migration conflicts occur when parallel development creates this scenario. Check with: alembic history --verbose to see the chain.',
  },
];
