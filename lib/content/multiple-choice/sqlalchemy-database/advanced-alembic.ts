import { MultipleChoiceQuestion } from '@/lib/types';

export const advancedAlembicMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'sql-adv-alembic-mc-1',
    question: 'What is the purpose of batch_alter_table() in Alembic?',
    options: [
      'Speed up migrations',
      'Work around SQLite ALTER TABLE limitations',
      'Batch multiple tables',
      'Improve performance',
    ],
    correctAnswer: 1,
    explanation:
      "batch_alter_table() works around SQLite\'s limited ALTER TABLE support. SQLite can't drop columns, add foreign keys, or modify columns directly. Batch mode creates temp table with new schema, copies data, drops old table, renames temp. Required for SQLite migrations with these operations.",
  },
  {
    id: 'sql-adv-alembic-mc-2',
    question: "Why doesn't Alembic autogenerate detect database views?",
    options: [
      'Bug in Alembic',
      'Views are not schema objects',
      'Views are not in MetaData - must create custom operations',
      'Performance reasons',
    ],
    correctAnswer: 2,
    explanation:
      "Views aren't part of SQLAlchemy MetaData (only tables/columns are). Autogenerate compares MetaData to database schema - views aren't in MetaData so can't be detected. Must create custom operations: @Operations.register_operation(\"create_view\") to handle views in migrations.",
  },
  {
    id: 'sql-adv-alembic-mc-3',
    question:
      'What happens if you have circular dependencies between migrations?',
    options: [
      'Auto-resolved',
      'Migration fails - must fix dependency chain',
      'Ignored',
      'Performance penalty',
    ],
    correctAnswer: 1,
    explanation:
      'Circular dependencies (A depends on B, B depends on A) break migration chain. Alembic can\'t determine order to apply migrations. Error: "Can\'t locate revision". Solution: Restructure migrations to remove circular dependencies, or use branching with merge migrations.',
  },
  {
    id: 'sql-adv-alembic-mc-4',
    question: 'How do you run migrations for multiple databases?',
    options: [
      'Not supported',
      'Multiple alembic.ini sections with -n flag',
      'Separate projects',
      'Manual SQL',
    ],
    correctAnswer: 1,
    explanation:
      'Multiple databases: Create sections in alembic.ini for each database. Run with -n flag: "alembic -n app_db upgrade head", "alembic -n analytics_db upgrade head". Each section has its own sqlalchemy.url. Useful for microservices with separate databases or sharding.',
  },
  {
    id: 'sql-adv-alembic-mc-5',
    question: 'What is a merge migration?',
    options: [
      'Combines multiple tables',
      'Merges two parallel migration branches into one',
      'Database merge',
      'Data merge',
    ],
    correctAnswer: 1,
    explanation:
      "Merge migration resolves multiple heads (parallel development creating divergent migration branches). Created with: alembic merge heads. Has down_revision = ('rev1', 'rev2') - multiple parents. After merge, single head again. Essential for team collaboration when developers create migrations in parallel branches.",
  },
];
