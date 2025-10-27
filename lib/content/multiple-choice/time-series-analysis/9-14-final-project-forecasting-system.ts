export const finalProjectForecastingSystemMultipleChoice = [
  {
    id: 1,
    question:
      'In a production forecasting system, which deployment strategy minimizes downtime while ensuring quality?',
    options: [
      'Direct deployment to production (fastest)',
      'Blue-green deployment with automated rollback',
      'Canary deployment with gradual traffic increase',
      'Shadow deployment with parallel systems',
      'Manual deployment with testing in production',
    ],
    correctAnswer: 2,
    explanation:
      "Canary deployment with gradual traffic. Strategy: Deploy new model to small % of traffic (5%), monitor metrics (forecast accuracy, latency, errors), if metrics acceptable → increase to 25%, 50%, 100% over days, if problems → instant rollback to old model. Why best for forecasting: (1) Real production data validation, (2) Minimal blast radius if model fails, (3) Easy rollback, (4) Gradual confidence building. Blue-green: Good but all-or-nothing switch (risky for untested model changes), Shadow: Safe but doesn't serve real users (can't validate business impact), Direct: Too risky for production ML. Implementation: Load balancer routes traffic by model version, monitoring dashboard tracks canary vs baseline metrics, automated rollback if error rate > threshold. Key metrics: forecast RMSE, API latency, error rate, business KPIs (trader satisfaction, P&L).",
    difficulty: 'advanced',
  },
  {
    id: 2,
    question:
      'Your forecasting API experiences 500ms latency, violating the 100ms SLA. Profiling shows: data loading 200ms, model inference 250ms, result formatting 50ms. Which optimization has highest impact?',
    options: [
      'Cache model predictions for repeated requests',
      'Use faster hardware (GPU for inference)',
      'Preload and cache recent data in Redis',
      'Optimize result formatting code',
      'Implement request batching',
    ],
    correctAnswer: 2,
    explanation:
      "Preload and cache data in Redis. Analysis: Data loading (200ms) is largest bottleneck (40%). Solution: Redis caching → reduces to ~5-10ms → total latency ~100ms ✓ SLA. Why other options less impactful: GPU: model inference already fast (250ms), ARIMA/GARCH not GPU-bottlenecked, marginal improvement. Cache predictions: Only helps repeated requests (forecasts change frequently), doesn't solve root cause. Result formatting: Only 10% of latency (50ms), optimization maybe saves 20ms. Batching: Helps throughput but not single-request latency. Implementation: Redis key schema: ticker:date:feature → value, TTL = 1 hour (balance freshness vs performance), warm cache on startup, async background updates. Expected result: 200ms → 10ms data loading, total latency: 310ms → 110ms (close to SLA), further optimization: model caching, async preprocessing. General principle: Optimize the bottleneck first (Amdahl's law), measure before and after.",
    difficulty: 'advanced',
  },
  {
    id: 3,
    question:
      'In forecast evaluation, you find: Train RMSE = 1.0%, Val RMSE = 1.5%, Test RMSE = 3.0%. This pattern suggests:',
    options: [
      'Model is well-calibrated and reliable',
      'Test set is too small for reliable estimate',
      'Overfitting to training data',
      'Regime shift between validation and test periods',
      'Random noise in test period',
    ],
    correctAnswer: 3,
    explanation:
      'Regime shift between validation and test. Analysis: Train → Val degradation (1.0% → 1.5%) = normal (50% increase acceptable), Val → Test degradation (1.5% → 3.0%) = 100% increase = ALARM! Likely causes: (1) Structural break in test period (e.g., COVID-19, Fed policy change), (2) Distribution shift (volatility regime change), (3) Model assumes stationarity but relationship changed. Why not overfitting? Would see large Train-Val gap, not Val-Test. Solution: (A) Investigate test period: plot residuals over time, identify outlier periods, check if corresponds to known events (crisis, earnings), (B) Regime-conditional evaluation: separate metrics by volatility/trend regime, (C) Rolling retraining: more frequent updates to adapt, (D) Robustness: test on multiple time periods, stress test on historical crises. Production implication: Build regime detection into system, wider prediction intervals during regime uncertainty, human-in-loop for major regime changes.',
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'For a forecasting system serving 100 concurrent users with 1000 requests/minute, which database is most appropriate for storing time series data?',
    options: [
      'PostgreSQL with TimescaleDB extension',
      'MongoDB (document database)',
      'MySQL (traditional RDBMS)',
      'Redis (in-memory cache)',
      'Flat files on S3',
    ],
    correctAnswer: 0,
    explanation:
      "PostgreSQL with TimescaleDB. Requirements: Time series data (indexed by timestamp), high write throughput (1000 req/min), complex queries (aggregations, joins), reliability and consistency. TimescaleDB advantages: (1) Time-series optimized: automatic partitioning by time (hypertables), efficient time-based queries, compression for old data; (2) SQL interface: familiar, powerful analytics, joins with other tables; (3) Scalability: handles millions of data points, continuous aggregates (materialized views), (4) Reliability: ACID guarantees, replication, backup; Performance: Insert 1000 rows/min easily handled, query recent data <10ms, aggregations fast with indexing. Alternatives: MongoDB: Good for flexible schema but slower for time series queries, MySQL: Works but no time-series optimizations, Redis: Great for caching but not primary storage (volatile), S3: Good for archival but high latency for queries. Architecture: TimescaleDB for operational data (last 1 year), S3 for archival (>1 year), Redis for caching hot data (today's forecasts). Implementation: Hypertable on (ticker, timestamp), continuous aggregate for daily stats, retention policy for old data.",
    difficulty: 'advanced',
  },
  {
    id: 5,
    question:
      'To ensure forecasting system reliability, which monitoring approach is most comprehensive?',
    options: [
      'Track API uptime and response time only',
      'Monitor model accuracy metrics daily',
      'Four Golden Signals: latency, traffic, errors, saturation',
      'Alert on any forecast error > threshold',
      'Manual daily review of forecasts',
    ],
    correctAnswer: 2,
    explanation:
      "Four Golden Signals (Google SRE). Comprehensive monitoring requires: (1) Latency: p50, p95, p99 response times → detect performance degradation, SLA compliance; (2) Traffic: requests/second, unique users → understand load patterns, capacity planning; (3) Errors: error rate %, error types → detect system failures, model crashes; (4) Saturation: CPU/memory/disk usage → predict capacity issues before failures. Why comprehensive: Covers infrastructure AND application health. For forecasting, ADD: (5) Model performance: rolling RMSE, MAE, direction accuracy → detect model degradation, (6) Data quality: missing data %, outliers, staleness → catch upstream issues, (7) Business metrics: forecast usage, trader satisfaction, P&L attribution → measure actual value. Implementation: Prometheus for metrics collection, Grafana for dashboards, PagerDuty for alerts. Alert priorities: P0 (wake up): API down, error rate >10%, P1 (next day): latency > SLA, model RMSE > 2× baseline, P2 (weekly review): gradual performance drift. Avoid: Alert fatigue (too many low-priority alerts), vanity metrics (doesn't predict problems), missing business context.",
    difficulty: 'advanced',
  },
];
