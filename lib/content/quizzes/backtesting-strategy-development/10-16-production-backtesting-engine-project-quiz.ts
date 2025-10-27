import { MultipleChoiceQuestion } from '@/lib/types';

const productionBacktestingEngineProjectQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'prod-1',
    question:
      'In a production backtesting system handling 100+ concurrent backtests, where should you cache frequently accessed historical data?',
    options: [
      'In-memory Python dictionaries (fastest access)',
      'PostgreSQL with proper indexing',
      'Redis with TTL expiration (distributed cache)',
      'Local file system (CSV files)',
    ],
    correctAnswer: 2,
    explanation:
      "Redis provides distributed, in-memory caching perfect for production systems. It's fast (sub-millisecond access), shared across multiple workers, supports TTL for automatic expiration, and handles concurrent access safely. Option A (Python dicts) doesn't share across processes. Option B (PostgreSQL) is too slow for caching (though good for persistent storage). Option D (files) doesn't scale for concurrent access. In production, typical architecture: PostgreSQL for persistent storage, Redis for caching hot data, with 1-hour TTL. Two Sigma and Citadel use similar tiered storage architectures with distributed caching layers.",
    difficulty: 'intermediate',
  },
  {
    id: 'prod-2',
    question:
      'Your production backtest system runs in Kubernetes. A pod crashes mid-backtest. What should happen?',
    options: [
      'The backtest is lost and must be manually resubmitted',
      'Kubernetes automatically restarts the pod and resumes the backtest from a checkpoint',
      'The system marks the backtest as failed and notifies the user',
      'The backtest is automatically retried from the beginning with exponential backoff',
    ],
    correctAnswer: 3,
    explanation:
      'Production systems should automatically retry failed jobs with exponential backoff. The backtest is idempotent (same inputs always produce same outputs), so rerunning from start is safe. Option B (checkpointing) is ideal but complex to implement. Option A makes the system brittle. Option C alone is insufficient—should retry first, then notify if failures persist. Typical implementation: max 3 retries with 1min, 5min, 15min backoff using a job queue (Celery, RabbitMQ, or Kubernetes Jobs). After 3 failures, mark as failed and alert on-call engineer. This pattern is standard in distributed systems at scale.',
    difficulty: 'advanced',
  },
  {
    id: 'prod-3',
    question:
      "What is the primary benefit of using FastAPI for a production backtesting engine's REST API?",
    options: [
      'FastAPI is the only framework that supports async Python',
      'Automatic API documentation, type validation, and async support with high performance',
      'FastAPI requires less code than other frameworks',
      'FastAPI automatically scales to handle millions of requests',
    ],
    correctAnswer: 1,
    explanation:
      "FastAPI provides automatic OpenAPI documentation, Pydantic-based request/response validation, native async/await support, and performance comparable to Node.js/Go. This combination is ideal for production APIs. Option A is wrong (asyncio works with any framework). Option C is subjective. Option D is misleading—scaling requires infrastructure (Kubernetes, load balancers), not just the framework. FastAPI's automatic validation prevents malformed requests from reaching your business logic, reducing bugs. Its auto-generated docs (Swagger UI) help frontend developers integrate quickly. Jane Street and other quant firms use FastAPI for internal trading tools.",
    difficulty: 'intermediate',
  },
  {
    id: 'prod-4',
    question:
      'For a production system storing backtest results, which database design is most appropriate?',
    options: [
      'Single PostgreSQL table with JSON blob for all metrics',
      'TimescaleDB (PostgreSQL extension) with separate tables for metadata, equity curves, and trades',
      'MongoDB for flexible schema',
      'Redis for fast access to all historical results',
    ],
    correctAnswer: 1,
    explanation:
      "TimescaleDB (PostgreSQL with time-series optimizations) is ideal: structured schema for queries, efficient time-series storage for equity curves, relational integrity for joins, and PostgreSQL's reliability. Option A (JSON blobs) sacrifices query ability. Option C (MongoDB) lacks transactions and is overkill for structured data. Option D (Redis) is for caching, not persistent storage (though useful for recent results). Proper schema: `backtests` table (metadata), `equity_curves` hypertable (time-series), `trades` table (individual fills). This enables queries like 'show all backtests with Sharpe >1.5 in 2023' and 'plot equity curve for backtest X.'",
    difficulty: 'advanced',
  },
  {
    id: 'prod-5',
    question:
      "In production, how should your backtesting engine handle a strategy that's taking >10x longer than expected?",
    options: [
      'Let it run to completion—all backtests must finish',
      'Kill it immediately to free resources',
      'Implement timeouts with configurable limits and graceful shutdown',
      'Reduce data size dynamically to speed up execution',
    ],
    correctAnswer: 2,
    explanation:
      "Production systems need timeout mechanisms with graceful shutdown. Set reasonable timeout (e.g., 30 minutes for daily strategy), send SIGTERM for cleanup, then SIGKILL if unresponsive. Option A risks resource exhaustion. Option B risks corruption (open connections, partial writes). Option D masks problems instead of fixing them. Implementation: use asyncio.wait_for() with timeout, catch TimeoutError, log partial results, mark as failed with 'timeout' reason. Alert if timeouts become frequent (indicates code bug or infrastructure issue). This pattern is standard in distributed systems—every operation should have a timeout.",
    difficulty: 'beginner',
  },
];

export default productionBacktestingEngineProjectQuiz;
