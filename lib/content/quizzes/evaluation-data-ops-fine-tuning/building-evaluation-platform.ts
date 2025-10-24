/**
 * Discussion questions for Building an Evaluation Platform section
 */

export const buildingEvaluationPlatformQuiz = [
  {
    id: 'eval-platform-q-1',
    question:
      "You're building an internal evaluation platform for your company's 10 AI models (chatbots, classifiers, generators). Design the architecture and key features of this platform, including: (1) Data management (test sets, labels), (2) Evaluation execution (running evals, metrics), (3) Results tracking (dashboards, trends), (4) Human evaluation workflows. What tech stack would you use and why?",
    hint: 'Consider scalability (10 models → 100 models), reusability of test sets across models, integration with ML pipelines, and human-in-the-loop evaluation.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Architecture: Test set management, evaluation execution, results dashboard, human evaluation, alerting',
      'Tech stack: FastAPI (API), PostgreSQL (metadata), S3 (test sets), Celery (distributed execution), React (dashboard)',
      'Test sets: Reusable across models, stored in S3, searchable by model type and tags',
      'Evaluation: Async via Celery, scalable workers, compute metrics per model type (classifier, generator, etc.)',
      'Dashboard: Performance trends over time, model comparisons on same test set, exportable reports',
      'Human evaluation: Smart sampling (low confidence, wrong predictions), annotation queue, inter-annotator agreement',
      'API-first: Integrate with CI/CD, trigger evals on model deployment, block releases if metrics degrade',
    ],
  },
  {
    id: 'eval-platform-q-2',
    question:
      "You've built an evaluation platform used by 50 ML engineers across 20 teams. Usage is low—most teams still run ad-hoc evaluation scripts. How do you drive adoption and make the platform indispensable?",
    hint: 'Consider integration points (CI/CD, notebooks), reducing friction (CLI, SDKs), demonstrating value (dashboards, reports), and building a community.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Phase 1 - Reduce friction: CLI tool (one command), Python SDK (3 lines), CI/CD integration (automatic)',
      'Phase 2 - Demonstrate value: Auto weekly reports, comparison dashboards, historical trends, root cause analysis',
      'Phase 3 - Build community: Shared test sets, leaderboards, weekly office hours, best practices sharing',
      'Phase 4 - Make required: Deployment gates (block if accuracy < threshold), quarterly model reviews',
      'Adoption: 10% → 60% (Phase 1-2) → 95% (Phase 3-4) in 3-4 months',
      'ROI: $210K cost, $550K savings (time + prevented incidents) = 2.6x return in first year',
      "Key: Meet engineers where they are (notebooks, CI/CD), provide value they can't get elsewhere (trends, comparisons)",
    ],
  },
  {
    id: 'eval-platform-q-3',
    question:
      'Your evaluation platform has been running for 1 year and stores 50K evaluation runs, 500 test sets, 10M predictions. Database queries are slow (10s+), costs are $5K/month. Design an optimization strategy for: (1) Query performance, (2) Storage costs, (3) Long-term scalability.',
    hint: 'Consider data archival, indexing strategies, caching, query optimization, and data lifecycle management.',
    sampleAnswer:
      'This is a comprehensive answer that would be displayed in the UI.',
    keyPoints: [
      'Indexing: Add indexes on model_id + started_at, GIN on JSONB, partial for recent data (8s → 0.3s, 26x faster)',
      'Denormalize: Extract hot metrics (accuracy, precision) from JSONB to columns (8s → 0.1s, 80x faster)',
      'Caching: Redis for dashboard queries, 70% hit rate, <10ms for cached (reduces DB load 75%)',
      'Archival: Move predictions >90 days to S3 Glacier, 10M → 3M rows (-70%), saves $2K/month',
      'Query optimization: Fix N+1 queries, use joins (2s → 0.2s, 10x faster)',
      'Partitioning: Range partition by date (quarterly), enables fast queries + easy cleanup',
      'Materialized views: Pre-compute aggregations, refresh daily (15s → 0.05s, 300x faster)',
    ],
  },
];
