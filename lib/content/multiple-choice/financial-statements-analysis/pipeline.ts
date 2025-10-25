export const pipelineMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'Why use PostgreSQL over SQLite for production financial data pipeline?',
    options: [
      'SQLite is faster',
      'PostgreSQL supports concurrent writes, ACID compliance, advanced indexing, and scales to terabytes; SQLite is single-writer and limited to ~1TB',
      'SQLite has better Python support',
      'PostgreSQL is free while SQLite costs money',
      "They're identical in capabilities",
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. PostgreSQL is production-grade: (1) Multi-user concurrent access (multiple analysts querying simultaneously), (2) ACID transactions (data integrity), (3) Advanced features (JSON, time-series, partitioning), (4) Scales to terabytes, (5) Replication/backup. SQLite is file-based, single-writer, good for small apps but not production data systems with hundreds of companies and years of history. For financial data requiring reliability and concurrent access, PostgreSQL (or TimescaleDB for time-series) is industry standard.`,
  },

  {
    id: 2,
    question:
      'Pipeline runs nightly at 6 PM ET. SEC filings can be released anytime. How do you ensure timely data?',
    options: [
      'Run pipeline every hour',
      'Check SEC RSS feed every 15 minutes, trigger pipeline immediately when relevant filings detected; use event-driven architecture instead of fixed schedule',
      'Manual monitoring of SEC website',
      'Daily batch is sufficient',
      'Increase to twice daily',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Event-driven architecture is superior: (1) Poll SEC RSS feed every 15 min for new 10-Q/10-K/8-K filings, (2) When detected, immediately trigger pipeline for that specific company, (3) Gets data within minutes vs waiting for nightly batch, (4) Critical for time-sensitive analysis (earnings, covenant breaches). Implementation: AWS Lambda checking RSS feed → triggers Step Function pipeline → stores to RDS. This combines best of both: scheduled full refresh (nightly) + event-driven updates (real-time). Trading firms use this pattern for competitive advantage.`,
  },

  {
    id: 3,
    question:
      "What's the best way to handle historical data revisions when companies restate earnings?",
    options: [
      'Delete old data and insert new',
      'Overwrite existing records',
      'Use temporal tables / versioning; keep both original and revised data with timestamps, allowing point-in-time queries',
      'Ignore restatements',
      'Manual spreadsheet tracking',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Temporal/versioned data model: (1) Never delete - keep original AND revised figures, (2) Add version_id and valid_from/valid_to timestamps, (3) Allows queries like "show as-reported in 2020" vs "show restated values", (4) Audit trail for research/compliance. Schema: Add columns: version INT, valid_from DATE, valid_to DATE, is_restated BOOLEAN. When restatement occurs: Set valid_to on old record, insert new record with version+1. This is critical for research (analyzing management accuracy) and avoiding "look-ahead bias" in backtests. Many hedge funds maintain both reported and restated data for this reason.`,
  },

  {
    id: 4,
    question:
      'Pipeline takes 2 hours to process 500 companies. How to speed up?',
    options: [
      'Buy faster server',
      'Parallel processing: Use multiprocessing/threading to process multiple companies simultaneously; add caching for repeated API calls; optimize database queries',
      'Reduce number of companies analyzed',
      'Run less frequently',
      'Cannot be optimized',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Optimization strategies: (1) **Parallel processing**: Process 10-20 companies simultaneously using Python multiprocessing (limited by CPU cores) or distributed with Celery/RabbitMQ, (2) **Caching**: Cache API responses (SEC data rarely changes), (3) **Database optimization**: Batch inserts instead of row-by-row, use COPY for bulk loads, (4) **Profiling**: Identify bottlenecks (network I/O? parsing? DB writes?). Example: 500 companies sequentially at 14.4 sec each = 2 hours. With 20 parallel workers: 2 hours / 20 = 6 minutes. Real implementations achieve sub-hour processing for 1000+ companies.`,
  },

  {
    id: 5,
    question: 'How should you monitor pipeline health in production?',
    options: [
      'Check manually each morning',
      'Implement comprehensive monitoring: Track success/failure rates, processing time per company, data completeness, alert on anomalies; use tools like Datadog or custom dashboards',
      'Only monitor if errors occur',
      'Log files are sufficient',
      'No monitoring needed',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Production monitoring requirements: (1) **Success metrics**: % of companies processed successfully (target: >98%), (2) **Performance**: Processing time per company (alert if >30 sec), total pipeline duration, (3) **Data quality**: Check for nulls, outliers, schema violations, (4) **Alerting**: PagerDuty/Slack alerts for pipeline failures or >2% failure rate, (5) **Dashboards**: Grafana showing real-time metrics. Example alert: "Pipeline failed for 15/500 companies (3%) - investigate" or "Pipeline took 4 hours (2x normal) - performance degradation". Without monitoring, silent failures can go undetected for days, corrupting downstream analysis.`,
  },
];
