import { MultipleChoiceQuestion } from '../../../types';

export const devopsDeploymentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-dd-mc-1',
    question:
      'For an early-stage AI product (1-100 users), what is the most cost-effective deployment strategy?',
    options: [
      'Full Kubernetes cluster',
      'Docker Compose on single VPS ($50/mo) + Modal for GPU ($20/mo usage)',
      'Serverless Lambda for everything',
      'Multiple cloud providers',
    ],
    correctAnswer: 1,
    explanation:
      'Early stage optimization: (1) Docker Compose: API + workers + databases on single $50 VPS, simple, local dev matches production, (2) Modal for GPU: Pay-per-use ($0.002/sec), no idle cost, fast cold start, (3) Total: $70/month vs $500+ for Kubernetes or Lambda. Scale later when needed. Avoid premature optimization - complexity kills early-stage products. Simple deployment enables fast iteration.',
  },
  {
    id: 'bcap-dd-mc-2',
    question: 'What is the most effective cost optimization for LLM API usage?',
    options: [
      'Use the cheapest model for everything',
      'Semantic caching (30-40% hit rate) + model routing + prompt optimization = 60-70% cost reduction',
      'Never cache responses',
      'Only optimize infrastructure, not LLM costs',
    ],
    correctAnswer: 1,
    explanation:
      'Multi-layer LLM optimization: (1) Semantic caching: 30-40% hit rate, $31.5k savings on $90k spend, (2) Model routing: 50% queries use cheaper Haiku vs Sonnet (3x cheaper), $30k savings, (3) Prompt optimization: Reduce input tokens 20%, $18k savings, (4) Combined: $90k → $31.5k (65% reduction). LLM is 60% of costs - optimize here first before infrastructure. Caching alone provides biggest ROI with minimal code changes.',
  },
  {
    id: 'bcap-dd-mc-3',
    question:
      'How should GPU workers be managed for image generation at scale?',
    options: [
      'All on-demand instances',
      'Auto-scaling GPU node pool: 80% spot (70% discount) + 20% on-demand, scale on queue depth',
      'Fixed number of GPUs',
      'No GPUs - use only CPU',
    ],
    correctAnswer: 1,
    explanation:
      'GPU auto-scaling strategy: (1) 80% spot instances (70% cheaper but can terminate), 20% on-demand (reliable), (2) Auto-scale based on queue depth (if queue >50, add GPUs), (3) Separate GPU pool from API servers (independent scaling), (4) Use Tesla T4 for images ($0.35/hr), A100 for video ($2.50/hr), (5) Batch processing: 4-8 images per GPU simultaneously. This balances cost (spot savings) with reliability (on-demand backup) and performance.',
  },
  {
    id: 'bcap-dd-mc-4',
    question:
      'What metrics should trigger alerts in an AI application monitoring system?',
    options: [
      'Only track uptime',
      'Critical alerts: p95 latency >10s, error rate >10%, GPU queue >100, daily cost >2x expected',
      'Never set up alerts',
      'Alert on every error',
    ],
    correctAnswer: 1,
    explanation:
      'Tiered alerting: (1) Critical (page on-call): p95 >10s (users frustrated), error rate >10% (systemic issue), GPU queue >100 (backlog), cost >2x (runaway spend), (2) Warning (Slack): p95 >5s, error rate >5%, cost >1.5x, (3) Info (daily summary): Trends, new errors, cost increases. Avoid alert fatigue - only page for user-impacting or financial issues. Too many alerts = ignored alerts = missed real problems.',
  },
  {
    id: 'bcap-dd-mc-5',
    question:
      'Why is ClickHouse preferred over PostgreSQL for AI application analytics?',
    options: [
      'ClickHouse is always better than PostgreSQL',
      'Columnar storage makes aggregations 100x faster for time-series data (1M events/day)',
      'PostgreSQL cannot store time-series data',
      'ClickHouse is free',
    ],
    correctAnswer: 1,
    explanation:
      'ClickHouse advantages for analytics: (1) Columnar storage: Only read needed columns (vs row-based reading entire rows), (2) 100x faster aggregations (SUM, AVG) on large datasets, (3) Handles billions of rows efficiently, (4) Optimized for time-series (common in analytics), (5) Cost-effective at scale. PostgreSQL: Great for transactional data (<100k events/day), but slow for analytical queries at scale (1M+ events/day). Use both: PostgreSQL for app data, ClickHouse for analytics.',
  },
];
