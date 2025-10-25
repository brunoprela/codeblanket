/**
 * Module: SQLAlchemy & Database Mastery
 * Module 5 of Python Curriculum
 */

import { Module } from '../../types';

// Section imports
import { databaseFundamentals } from '../sections/sqlalchemy-database/database-fundamentals';
import { sqlalchemyCore } from '../sections/sqlalchemy-database/sqlalchemy-core';
import { definingModels } from '../sections/sqlalchemy-database/defining-models';
import { queryApi } from '../sections/sqlalchemy-database/query-api';
import { advancedFiltering } from '../sections/sqlalchemy-database/advanced-filtering';
import { relationshipLoading } from '../sections/sqlalchemy-database/relationship-loading';
import { sessionManagement } from '../sections/sqlalchemy-database/session-management';
import { alembicMigrations } from '../sections/sqlalchemy-database/alembic-migrations';
import { advancedAlembic } from '../sections/sqlalchemy-database/advanced-alembic';
import { performanceOptimization } from '../sections/sqlalchemy-database/performance-optimization';
import { advancedPatterns } from '../sections/sqlalchemy-database/advanced-patterns';
import { testingWithDatabases } from '../sections/sqlalchemy-database/testing-databases';
import { multiDatabaseSharding } from '../sections/sqlalchemy-database/multi-database-sharding';
import { productionPatterns } from '../sections/sqlalchemy-database/production-patterns';

// Quiz imports
import { databaseFundamentalsQuiz } from '../quizzes/sqlalchemy-database/database-fundamentals';
import { sqlalchemyCoreQuiz } from '../quizzes/sqlalchemy-database/sqlalchemy-core';
import { definingModelsQuiz } from '../quizzes/sqlalchemy-database/defining-models';
import { queryApiQuiz } from '../quizzes/sqlalchemy-database/query-api';
import { advancedFilteringQuiz } from '../quizzes/sqlalchemy-database/advanced-filtering';
import { relationshipLoadingQuiz } from '../quizzes/sqlalchemy-database/relationship-loading';
import { sessionManagementQuiz } from '../quizzes/sqlalchemy-database/session-management';
import { alembicMigrationsQuiz } from '../quizzes/sqlalchemy-database/alembic-migrations';
import { advancedAlembicQuiz } from '../quizzes/sqlalchemy-database/advanced-alembic';
import { performanceOptimizationQuiz } from '../quizzes/sqlalchemy-database/performance-optimization';
import { advancedPatternsQuiz } from '../quizzes/sqlalchemy-database/advanced-patterns';
import { testingDatabasesQuiz } from '../quizzes/sqlalchemy-database/testing-databases';
import { multiDatabaseShardingQuiz } from '../quizzes/sqlalchemy-database/multi-database-sharding';
import { productionPatternsQuiz } from '../quizzes/sqlalchemy-database/production-patterns';

// Multiple choice imports
import { databaseFundamentalsMultipleChoice } from '../multiple-choice/sqlalchemy-database/database-fundamentals';
import { sqlalchemyCoreMultipleChoice } from '../multiple-choice/sqlalchemy-database/sqlalchemy-core';
import { definingModelsMultipleChoice } from '../multiple-choice/sqlalchemy-database/defining-models';
import { queryApiMultipleChoice } from '../multiple-choice/sqlalchemy-database/query-api';
import { advancedFilteringMultipleChoice } from '../multiple-choice/sqlalchemy-database/advanced-filtering';
import { relationshipLoadingMultipleChoice } from '../multiple-choice/sqlalchemy-database/relationship-loading';
import { sessionManagementMultipleChoice } from '../multiple-choice/sqlalchemy-database/session-management';
import { alembicMigrationsMultipleChoice } from '../multiple-choice/sqlalchemy-database/alembic-migrations';
import { advancedAlembicMultipleChoice } from '../multiple-choice/sqlalchemy-database/advanced-alembic';
import { performanceOptimizationMultipleChoice } from '../multiple-choice/sqlalchemy-database/performance-optimization';
import { advancedPatternsMultipleChoice } from '../multiple-choice/sqlalchemy-database/advanced-patterns';
import { testingDatabasesMultipleChoice } from '../multiple-choice/sqlalchemy-database/testing-databases';
import { multiDatabaseShardingMultipleChoice } from '../multiple-choice/sqlalchemy-database/multi-database-sharding';
import { productionPatternsMultipleChoice } from '../multiple-choice/sqlalchemy-database/production-patterns';

export const sqlalchemyDatabaseModule: Module = {
  id: 'sqlalchemy-database',
  title: 'SQLAlchemy & Database Mastery',
  description:
    'Master SQLAlchemy ORM and database management from fundamentals to production. Learn database concepts (RDBMS, DB-API), SQLAlchemy Core & ORM architecture, model definitions with complex relationships (one-to-many, many-to-many, polymorphic), advanced query API (joins, subqueries, CTEs, window functions), filtering & expressions, relationship loading strategies (lazy, eager, selectin), session management patterns, Alembic migrations (auto-generation, branching, zero-downtime), performance optimization (indexes, bulk operations, EXPLAIN), advanced patterns (Repository, Unit of Work), testing strategies (fixtures, factories, mocks), multi-database & sharding (read replicas, hash/range sharding, distributed transactions), and production patterns (connection pooling, caching, monitoring, health checks). Build production-ready database applications with proper architecture, testing, and deployment.',
  icon: '🗄️',
  keyTakeaways: [
    'Understand RDBMS concepts: ACID, normalization, indexes, constraints',
    'Master SQLAlchemy architecture: Core vs ORM, Engine, Session, MetaData',
    'Define models with relationships: one-to-one, one-to-many, many-to-many, polymorphic',
    'Query API mastery: select(), filter(), joins, subqueries, CTEs, aggregations',
    'Advanced filtering: comparison operators, IN/LIKE, NULL handling, JSON querying',
    'Optimize queries: relationship loading strategies (lazy vs eager), avoid N+1 problem',
    'Session patterns: lifecycle, scoped sessions, context managers, web integration',
    'Alembic migrations: auto-generate, upgrade/downgrade, branching, data migrations',
    'Performance optimization: indexes (B-tree, composite, partial), bulk operations, query profiling',
    'Advanced patterns: Repository pattern, Unit of Work, query objects, specification pattern',
    'Testing: pytest fixtures, Factory Boy, mocks, transaction rollback, CI/CD integration',
    'Scale horizontally: read replicas, sharding (hash, range, geographic), cross-shard queries',
    'Production patterns: connection pooling (pool_size, max_overflow), caching (Redis), monitoring (Prometheus)',
    'Deploy confidently: health checks, error handling, retries, configuration management',
  ],
  sections: [
    {
      ...databaseFundamentals,
      quiz: databaseFundamentalsQuiz,
      multipleChoice: databaseFundamentalsMultipleChoice,
    },
    {
      ...sqlalchemyCore,
      quiz: sqlalchemyCoreQuiz,
      multipleChoice: sqlalchemyCoreMultipleChoice,
    },
    {
      ...definingModels,
      quiz: definingModelsQuiz,
      multipleChoice: definingModelsMultipleChoice,
    },
    {
      ...queryApi,
      quiz: queryApiQuiz,
      multipleChoice: queryApiMultipleChoice,
    },
    {
      ...advancedFiltering,
      quiz: advancedFilteringQuiz,
      multipleChoice: advancedFilteringMultipleChoice,
    },
    {
      ...relationshipLoading,
      quiz: relationshipLoadingQuiz,
      multipleChoice: relationshipLoadingMultipleChoice,
    },
    {
      ...sessionManagement,
      quiz: sessionManagementQuiz,
      multipleChoice: sessionManagementMultipleChoice,
    },
    {
      ...alembicMigrations,
      quiz: alembicMigrationsQuiz,
      multipleChoice: alembicMigrationsMultipleChoice,
    },
    {
      ...advancedAlembic,
      quiz: advancedAlembicQuiz,
      multipleChoice: advancedAlembicMultipleChoice,
    },
    {
      ...performanceOptimization,
      quiz: performanceOptimizationQuiz,
      multipleChoice: performanceOptimizationMultipleChoice,
    },
    {
      ...advancedPatterns,
      quiz: advancedPatternsQuiz,
      multipleChoice: advancedPatternsMultipleChoice,
    },
    {
      ...testingWithDatabases,
      quiz: testingDatabasesQuiz,
      multipleChoice: testingDatabasesMultipleChoice,
    },
    {
      ...multiDatabaseSharding,
      quiz: multiDatabaseShardingQuiz,
      multipleChoice: multiDatabaseShardingMultipleChoice,
    },
    {
      ...productionPatterns,
      quiz: productionPatternsQuiz,
      multipleChoice: productionPatternsMultipleChoice,
    },
  ],
};
