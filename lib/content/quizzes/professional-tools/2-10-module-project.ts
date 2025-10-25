import { Discussion } from '@/lib/types';

export const moduleProjectQuiz: Discussion = {
  title: 'Module Project Discussion Questions',
  description:
    'Deep dive into design decisions, architecture patterns, and production considerations for the Financial Data Aggregator.',
  questions: [
    {
      id: 'project-disc-1',
      question:
        'Design a complete error handling and recovery strategy for the Financial Data Aggregator. Cover: API failures, database connection issues, data validation errors, rate limiting, and partial failures. How would you ensure data completeness despite failures?',
      sampleAnswer: `[Comprehensive error handling strategy covering: (1) API failures - retry with exponential backoff, fallback to alternative sources, log failed tickers for manual review, (2) Database issues - connection pooling with health checks, automatic reconnection, transaction rollback on failure, (3) Validation errors - log problematic data, continue with other tickers, daily data quality report, (4) Rate limiting - track API call times, auto-throttle before hitting limits, queue requests during heavy load, (5) Partial failures - checkpoint progress, resume from last successful ticker, idempotent operations. Include code examples for each scenario and monitoring dashboards to track failure rates.]`,
      keyPoints: [],
    },
    {
      id: 'project-disc-2',
      question:
        'Explain how you would extend the Financial Data Aggregator to handle real-time streaming data (WebSocket feeds) in addition to batch historical data. What architecture changes would be needed? Consider: data ingestion, storage, API design, and user access patterns.',
      sampleAnswer: `[Architecture evolution from batch to real-time: (1) Ingestion layer - add WebSocket client connecting to data provider, message queue (RabbitMQ/Kafka) to buffer high-frequency updates, worker processes consuming queue and writing to database, (2) Storage - separate tables for real-time (tick) vs historical (daily), use TimescaleDB continuous aggregates to roll up ticks to minute/hour bars, implement retention policies (ticks 30 days, minute 1 year, daily 10 years), (3) API changes - add WebSocket endpoint for live streaming to clients, Server-Sent Events alternative for simpler use case, caching layer (Redis) for latest prices, (4) Access patterns - historical queries unchanged, real-time subscriptions via WebSocket, hybrid queries combining historical + latest. Include code examples and discuss tradeoffs.]`,
      keyPoints: [],
    },
    {
      id: 'project-disc-3',
      question:
        'Design a comprehensive monitoring and alerting system for the production Financial Data Aggregator. What metrics would you track? What alerts would you configure? How would you visualize system health? Consider both infrastructure and data quality monitoring.',
      sampleAnswer: `[Complete monitoring strategy: (1) Infrastructure metrics - API response times (alert >2s), database query performance (alert >500ms), disk space (alert >80% full), memory usage, CPU utilization, network I/O, (2) Data quality metrics - daily collection success rate (alert <95%), data gaps detection, OHLC validation failure rate, source availability (Yahoo vs Alpha Vantage uptime), (3) Business metrics - number of tickers collected, total rows inserted daily, API quota usage (alert >80%), data freshness (alert if >1 hour stale), (4) Visualization - Grafana dashboards showing all metrics, historical trends, comparison to baseline, (5) Alerting - PagerDuty for critical failures, Slack for warnings, email for daily summaries, escalation policies. Include example Grafana dashboard JSON and alerting rules configuration.]`,
      keyPoints: [],
    },
  ],
};
