import { MultipleChoiceQuestion } from '../../../types';

export const productAnalyticsMetricsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-pam-mc-1',
    question: 'What is the most important metric for an AI chat application?',
    options: [
      'Total number of messages',
      'Thumbs up rate (user satisfaction with responses)',
      'Number of signups',
      'Server uptime',
    ],
    correctAnswer: 1,
    explanation:
      'Thumbs up rate (quality metric): (1) Direct measure of user satisfaction with AI responses, (2) Target: >30% thumbs up rate, (3) Tracks: Is AI providing value?, (4) Actionable: Low rate → improve prompts, switch models, add context. Other metrics important but secondary: Messages (engagement), signups (acquisition), uptime (reliability). Quality (thumbs up) is the north star - without quality, users churn regardless of other metrics.',
  },
  {
    id: 'bcap-pam-mc-2',
    question: 'How should AI application costs be tracked per user?',
    options: [
      "Don't track per user",
      'Log: user_id, tokens (input/output), cost_usd per request, aggregate daily/monthly per user',
      'Only track total costs',
      'Track only for paid users',
    ],
    correctAnswer: 1,
    explanation:
      'Per-user cost tracking: (1) Log each LLM request: user_id, model, input_tokens, output_tokens, cost_usd, (2) Aggregate: Daily cost per user, monthly trends, (3) Segment: Power users (>$10/mo), light users (<$1/mo), (4) Action: If user costs >$30/mo on $20 plan, raise prices or limit usage, (5) Profitability: Track LTV vs cost per cohort. This enables: pricing optimization, identifying high-cost users, calculating true unit economics.',
  },
  {
    id: 'bcap-pam-mc-3',
    question:
      'For real-time analytics (dashboard updates every 5s), what architecture is best?',
    options: [
      'Query PostgreSQL every 5 seconds',
      'Kafka → Flink (real-time agg) → Redis, dashboard polls Redis',
      'No real-time analytics possible',
      'Manual spreadsheet updates',
    ],
    correctAnswer: 1,
    explanation:
      'Streaming analytics: (1) Events → Kafka (buffer), (2) Flink/Kafka Streams: Aggregate last 5s (requests, tokens, errors), (3) Redis: Store aggregated metrics (dashboard_metrics:last_5s), (4) Dashboard polls Redis every 5s, (5) Historical: Separate pipeline → ClickHouse for long-term storage. PostgreSQL too slow (100ms+ queries) for real-time. Redis in-memory provides <1ms lookback. This enables live monitoring without database load.',
  },
  {
    id: 'bcap-pam-mc-4',
    question:
      'What sample size is needed for A/B testing conversion rate improvement from 3% to 4%?',
    options: [
      '100 users per variant',
      '~2,600 users total (1,300 per variant) for 95% confidence',
      '10,000 users per variant',
      '10 users total',
    ],
    correctAnswer: 1,
    explanation:
      'Sample size calculation: n = 2 × (1.96 + 0.84)^2 × p × (1-p) / (MDE)^2, where p=0.03 (baseline), MDE=0.01 (3% → 4% = 1% absolute increase). Result: ~1,300 per variant, 2,600 total. Key factors: (1) Smaller baseline = larger sample needed, (2) Smaller effect = larger sample, (3) 95% confidence standard. Run test 7+ days to capture weekly patterns. Insufficient sample = false conclusions (80% of failed A/B tests).',
  },
  {
    id: 'bcap-pam-mc-5',
    question: 'What indicates product-market fit for an AI SaaS product?',
    options: [
      'Just having users',
      '>40% retention at month 6, NPS >50, >20% organic growth, <5% churn on price increase',
      'Any positive revenue',
      '100 total users',
    ],
    correctAnswer: 1,
    explanation:
      'PMF indicators: (1) Retention: >40% users still active after 6 months (product is sticky), (2) NPS >50: Users enthusiastically recommend (word-of-mouth), (3) Organic growth: >20% new users from referrals (viral growth), (4) Usage: >3 sessions/week (high engagement), (5) Willingness to pay: <5% churn when price increases (value justifies cost). All together indicate: Users love product, it solves real problem, they tell others, willing to pay. Missing any = weak PMF.',
  },
];
