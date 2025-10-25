export const testingDatabasesSqlalchemyQuiz = [
  {
    id: 'tds-q-1',
    question:
      'Design a database testing strategy for a multi-tenant SaaS application. Address: (1) test isolation between tenants, (2) schema migrations testing, (3) performance with 10K test cases, (4) data cleanup strategies, (5) CI/CD database setup.',
    sampleAnswer:
      'Multi-tenant database testing: (1) Tenant isolation: Separate schemas per tenant or row-level with tenant_id. Test: Create tenant in fixture, queries filtered by tenant_id. @pytest.fixture def tenant (db_session): return Tenant.create(); tests verify tenant_id in all queries. (2) Migration testing: Test migrations up/down: alembic upgrade head; run tests; alembic downgrade -1; verify rollback. Use separate test DB, apply migrations before tests. (3) Performance: In-memory SQLite (fast, 10K tests in 2 min), session-scoped engine, function-scoped session with rollback. Parallel: pytest -n auto. (4) Cleanup: Transaction rollback (fastest, no deletion needed), truncate tables (fallback), or drop/recreate DB (slow). (5) CI: Docker PostgreSQL container, apply migrations in setup, run tests, teardown container.',
    keyPoints: [
      'Tenant isolation: Row-level tenant_id filtering, test tenant creation in fixtures',
      'Migrations: Test upgrade/downgrade, separate DB, alembic commands in CI',
      'Performance: In-memory SQLite (2 min for 10K tests), session engine + function session',
      'Cleanup: Transaction rollback preferred (fast), truncate fallback, avoid drop/recreate',
      'CI: Docker PostgreSQL, apply migrations, parallel execution with pytest -n auto',
    ],
  },
];
