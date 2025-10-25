import { Discussion } from '@/lib/types';

export const databasesFinancialDataQuiz: Discussion = {
  title: 'Databases for Financial Data Discussion Questions',
  description:
    'Deep dive into database design, optimization, and operations for quantitative trading systems.',
  questions: [
    {
      id: 'db-disc-1',
      question:
        'Design a comprehensive database schema for a multi-strategy quantitative trading system that tracks: (1) OHLCV data at multiple frequencies, (2) fundamental data, (3) alternative data sources, (4) trade execution history, and (5) strategy performance metrics. Include table definitions, indexes, constraints, and explain your design decisions.',
      sampleAnswer: `[Comprehensive schema design covering: ticker reference table with sectors/industries, prices tables partitioned by frequency (tick/minute/daily), fundamentals table with quarterly data, alternative data tables (news, social sentiment, economic indicators), trades table with execution details, positions table for current holdings, strategy_performance table with daily metrics, orders table for order management. Include proper foreign keys, indexes for common query patterns, CHECK constraints for data quality, partitioning strategies for large tables, and explain why each design choice was made for financial data characteristics.]`,
    },
    {
      id: 'db-disc-2',
      question:
        'Explain the complete process of optimizing a slow database query in a trading system. Cover performance measurement, bottleneck identification, index strategy, query rewriting, and validation of improvements. Provide specific examples with EXPLAIN ANALYZE output.',
      sampleAnswer: `[Detailed optimization workflow: (1) Measure baseline with EXPLAIN ANALYZE, (2) Identify bottlenecks (sequential scans, nested loops on large tables, missing indexes), (3) Create appropriate indexes (composite, covering, partial), (4) Consider query rewrites (CTE vs subquery, EXISTS vs IN), (5) Update statistics with ANALYZE, (6) Re-measure and validate improvement, (7) Monitor in production. Include example of slow query taking 30s with Seq Scan, adding indexes to improve to 0.5s with Index Scan, showing before/after EXPLAIN ANALYZE output, and discussing tradeoffs of indexes.]`,
    },
    {
      id: 'db-disc-3',
      question:
        'Describe a production-ready data pipeline architecture for ingesting real-time market data into a database, including error handling, data validation, monitoring, and recovery procedures. How would you ensure data integrity and handle failures?',
      sampleAnswer: `[Complete pipeline architecture: Data sources (market data API, websocket feeds) → Ingestion layer (validation, deduplication, normalization) → Queue (RabbitMQ/Kafka for buffering) → Database writer (batch inserts with connection pooling) → Monitoring (data quality checks, latency alerts). Error handling: retry with exponential backoff, dead letter queue for persistent failures, circuit breaker for failing sources. Data validation: price bounds, OHLC relationships, volume sanity checks, duplicate detection. Recovery: automatic reconnection, gap detection and backfill, checkpoint/resume capability. Monitoring: Grafana dashboards for latency/throughput, PagerDuty alerts for failures, daily data quality reports. Include code examples for each component.]`,
    },
  ],
};
