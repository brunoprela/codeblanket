import { Quiz } from '@/lib/types';

export const databasesFinancialDataMultipleChoice: Quiz = {
  title: 'Databases for Financial Data Quiz',
  description:
    'Test your knowledge of database technologies and practices for financial data storage.',
  questions: [
    {
      id: 'db-1',
      question:
        'What is the primary advantage of using TimescaleDB hypertables over regular PostgreSQL tables for storing financial time-series data?',
      options: [
        'Hypertables automatically encrypt data for security',
        'Hypertables automatically partition data by time and optimize time-based queries, providing 10-100x performance improvement',
        'Hypertables use less disk space by storing only changed values',
        'Hypertables automatically generate technical indicators',
      ],
      correctAnswer: 1,
      explanation:
        'TimescaleDB hypertables automatically partition data into time-based chunks (e.g., weekly or monthly), dramatically improving query performance for time-range filters. When you query data from last month, TimescaleDB only scans the relevant chunk instead of the entire table. This provides 10-100x speedup on typical financial queries. Additionally, hypertables enable automatic compression (90%+ space savings), continuous aggregates (pre-computed OHLCV bars), and retention policies (auto-delete old data). All while maintaining full PostgreSQL compatibility and SQL support.',
    },
    {
      id: 'db-2',
      question:
        'Why is it important to use connection pooling when querying a database from a Python trading application?',
      options: [
        'Connection pooling encrypts queries for security',
        'Connection pooling automatically optimizes SQL queries',
        'Connection pooling reuses database connections instead of creating new ones for each query, reducing overhead and improving performance',
        'Connection pooling is required by PostgreSQL for all applications',
      ],
      correctAnswer: 2,
      explanation:
        "Creating a new database connection is expensive (network handshake, authentication, resource allocation) - typically 50-200ms. For a trading application making hundreds of queries per second, this overhead is devastating. Connection pooling maintains a pool of active connections that are reused across queries. SQLAlchemy's QueuePool maintains 10-30 connections ready to use, reducing query latency from 150ms to 5ms. Critical for real-time trading where milliseconds matter. Configuration: `pool_size=10` (core connections), `max_overflow=20` (additional when busy), `pool_recycle=3600` (refresh hourly), `pool_pre_ping=True` (verify health).",
    },
    {
      id: 'db-3',
      question:
        'What SQL constraint should you add to a `prices` table to prevent data quality issues like having a HIGH price that is lower than the LOW price?',
      options: [
        'PRIMARY KEY (date, ticker)',
        'UNIQUE (date, ticker)',
        'CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close)',
        'FOREIGN KEY (ticker) REFERENCES tickers(ticker_id)',
      ],
      correctAnswer: 2,
      explanation:
        'CHECK constraints enforce data quality rules at the database level, preventing invalid data from entering. For OHLCV data: `CHECK (high >= low)` ensures HIGH is highest, `CHECK (high >= open AND high >= close)` ensures HIGH bounds OPEN/CLOSE, `CHECK (low <= open AND low <= close)` ensures LOW bounds OPEN/CLOSE, `CHECK (close > 0)` prevents negative prices. When these constraints are violated, INSERT/UPDATE fails immediately with clear error. This catches data errors early (during ETL) rather than later (during analysis). Much better than validating in application code - database constraints are enforced regardless of which application writes data.',
    },
    {
      id: 'db-4',
      question:
        'In a database schema for financial data, why would you create separate tables for different data frequencies (tick, minute, daily) rather than storing all in one table?',
      options: [
        'Separate tables are required by SQL standards',
        'Different frequencies have vastly different data volumes and query patterns; separate tables enable appropriate indexing, partitioning, and retention policies for each',
        'Separate tables use less total disk space',
        'Separate tables automatically synchronize data between frequencies',
      ],
      correctAnswer: 1,
      explanation:
        "Data volume and access patterns differ dramatically by frequency: Tick data (billions of rows, short retention, rarely queried after 30 days), Minute data (hundreds of millions, medium retention, frequent queries), Daily data (millions, long retention, very frequent queries). Separate tables enable: (1) Appropriate retention - delete tick after 30 days but keep daily for 10 years, (2) Different compression - aggressive compression on tick, lighter on daily, (3) Tailored indexes - tick needs (ticker, time, exchange), daily needs (ticker, date), (4) Query optimization - daily queries don't scan tick table, (5) Maintenance - vacuum/analyze tick frequently, daily rarely. Combined table would be huge, slow, and wasteful.",
    },
    {
      id: 'db-5',
      question:
        'What is the purpose of using `EXPLAIN ANALYZE` before a SQL query in a trading database?',
      options: [
        'To automatically fix slow queries',
        'To see the query execution plan, identify performance bottlenecks, and verify that indexes are being used properly',
        'To encrypt the query results',
        'To create backups of queried data',
      ],
      correctAnswer: 1,
      explanation:
        '`EXPLAIN ANALYZE` shows exactly how PostgreSQL executes your query: (1) Query plan - which indexes used, (2) Actual execution time per step, (3) Rows scanned vs returned (efficiency), (4) Join methods used. Critical for optimization: "Seq Scan" (sequential scan) = BAD, means no index used, scanning entire table; "Index Scan" = GOOD, using index efficiently. Example: Query taking 30 seconds shows Seq Scan on 100M rows - add index, re-run EXPLAIN ANALYZE, now shows Index Scan in 0.5 seconds (60x faster). Essential tool for database performance tuning in production trading systems where query speed directly impacts strategy profitability.',
    },
  ],
};
