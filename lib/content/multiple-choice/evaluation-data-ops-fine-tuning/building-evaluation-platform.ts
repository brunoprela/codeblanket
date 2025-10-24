/**
 * Multiple choice questions for Building an Evaluation Platform section
 */

export const buildingEvaluationPlatformMultipleChoice = [
  {
    id: 'eval-platform-mc-1',
    question:
      "You're designing an evaluation platform. Should test sets be stored in: (A) PostgreSQL (with examples as JSONB), (B) S3 (as JSONL files), (C) MongoDB, or (D) PostgreSQL for metadata + S3 for examples?",
    options: [
      'PostgreSQL with JSONB (everything in one place)',
      'S3 only (cheap and scalable)',
      'MongoDB (flexible schema)',
      'PostgreSQL (metadata) + S3 (examples)',
    ],
    correctAnswer: 3,
    explanation:
      "Option D (PostgreSQL + S3 hybrid) is best. Here's why: PostgreSQL (metadata): Store test_set_id, name, model_type, num_examples, tags, created_at. Fast queries for searching/filtering. Relational integrity (foreign keys). S3 (examples): Store actual test examples as JSONL files. Cheap storage ($0.023/GB vs $0.115/GB for PostgreSQL). Scales to millions of examples. Can use S3 Select for filtering. Trade-offs: Option A (JSONB only): PostgreSQL expensive at scale, 10K test sets × 1000 examples × 1KB = 10GB → $115/month + slow queries. Option B (S3 only): Can't efficiently query/search metadata, need to download+parse files. Option C (MongoDB): Works but most companies have PostgreSQL already, not worth adding new DB. Best of both worlds: Fast metadata queries (PostgreSQL) + cheap storage (S3).",
  },
  {
    id: 'eval-platform-mc-2',
    question:
      'Your evaluation platform runs 1,000 evaluations/day. Each takes 2-5 minutes. Should you use: (A) Synchronous API (wait for result), (B) Async with Celery/RQ, (C) Serverless (Lambda), or (D) Kubernetes Jobs?',
    options: [
      'Synchronous API (simple)',
      'Async with Celery/RQ (queue-based)',
      'Serverless Lambda (pay-per-use)',
      'Kubernetes Jobs (container-based)',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (Async with Celery) is best for evaluation platform. Why: 1,000 evals/day × 3 min avg = 3,000 minutes/day = 50 hours compute/day. Async is essential—can't keep API connection open for 3 minutes. Celery benefits: Queue jobs (handle bursts), distributed workers (scale horizontally), retry logic (handle failures), priority queues (urgent evals first), result backend (track status), monitoring (Flower dashboard). Option A (sync): Times out after 30-60s, ties up web server. Option C (Lambda): 15min timeout (might hit it), cold starts add latency, harder to debug, expensive at scale ($0.20/1M requests). Option D (K8s Jobs): Works but overhead of spinning up pods per eval, Celery is simpler. Implementation: FastAPI endpoint → Queue Celery task → Return eval_run_id → Client polls status.",
  },
  {
    id: 'eval-platform-mc-3',
    question:
      'You have 50K evaluation runs in PostgreSQL. Dashboard query "get average accuracy over last 30 days" takes 8 seconds. Best fix?',
    options: [
      'Add index on (model_id, started_at)',
      'Use Redis to cache query results',
      'Create materialized view with daily aggregates',
      'All of the above',
    ],
    correctAnswer: 3,
    explanation:
      "Option D (all of the above) is best—they complement each other. Index (Option A): CREATE INDEX idx_model_time ON evaluation_runs(model_id, started_at DESC);. Makes WHERE + GROUP BY faster: 8s → 2s. Still slow because aggregating 1000s of rows. Cache (Option B): Cache result in Redis for 1 hour, 70%+ cache hit rate. Cached queries: <10ms. But doesn't help first query or cache misses. Materialized view (Option C): Pre-compute daily aggregates, refresh nightly. Query pre-computed view instead of raw data: 2s → 0.05s (40x faster). Combined effect: Materialized view: 8s → 0.05s (for aggregate query), Index: Speeds up other queries (filtering, sorting), Cache: Makes repeated queries instant (<10ms). This is a standard three-tier optimization: Index for query efficiency, materialized view for pre-computation, cache for hot data. Use all three for production dashboards.",
  },
  {
    id: 'eval-platform-mc-4',
    question:
      'Engineers complain your evaluation platform is "too much work". What\'s the BEST way to increase adoption?',
    options: [
      'Build better documentation',
      'Build a CLI tool (one command to evaluate)',
      'Mandate use via policy',
      'Offer training sessions',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (CLI tool) is most effective. Why: Engineers want convenience, not documentation/training/mandates. Friction reduction: Instead of: Login to web UI → Upload test set → Configure → Wait → Download results (5 minutes). Now: eval-platform run --model my-model --test-set prod-samples (one command, 30 seconds). Adoption pattern: Week 0: Web UI only → 10% adoption. Week 2: Add CLI → 40% adoption. Week 4: Add Python SDK → 60% adoption. Week 8: Add CI/CD integration (automatic evals) → 90% adoption. Option A (docs): Helpful but doesn't reduce friction. Option C (mandate): Creates resentment, workarounds. Option D (training): Doesn't scale, people forget. Best practice: Make platform easier to use than NOT using it. CLI + SDK + CI/CD integration = invisible evaluation (happens automatically).",
  },
  {
    id: 'eval-platform-mc-5',
    question:
      "Your platform stores 10M predictions (1GB). Database queries are slow. What's the BEST data archival strategy?",
    options: [
      'Delete predictions older than 90 days',
      'Move predictions older than 90 days to S3 Glacier',
      'Keep all predictions but compress them',
      'Use time-series database (InfluxDB)',
    ],
    correctAnswer: 1,
    explanation:
      "Option B (archive to S3 Glacier) is best. Archival strategy: Hot data (0-90 days): PostgreSQL for fast queries (most queries are recent). Cold data (90+ days): S3 Glacier for cheap storage. Retrieve on-demand (rare). Cost comparison: PostgreSQL: 1GB × $0.115/GB/month = $115/month. S3 Standard: 1GB × $0.023/GB/month = $23/month (5x cheaper). S3 Glacier: 1GB × $0.001/GB/month = $1/month (115x cheaper!). If 70% of data is >90 days old: Current cost (all in PostgreSQL): $115/month. With archival: Hot 0.3GB in PG ($34.50) + Cold 0.7GB in Glacier ($0.70) = $35.20/month → 70% savings. Option A (delete): Lose valuable data, can't debug past issues. Option C (compress): Still in expensive PostgreSQL. Option D (InfluxDB): Over-engineering, PostgreSQL fine with archival. Best practice: Keep recent data hot (fast queries), archive old data (cheap storage), retrieve on-demand (rare).",
  },
];
