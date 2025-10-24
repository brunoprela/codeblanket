export const productAnalyticsMetricsQuiz = [
  {
    id: 'bcap-pam-q-1',
    question:
      'Design the event tracking system for an AI chat application. What events should you track? For each event, what properties do you capture? How do you handle: (1) High volume (1M events/day), (2) Real-time analytics (dashboard updates every 5s), (3) Historical analysis (monthly trends), (4) User privacy (GDPR compliance)? Compare: storing in PostgreSQL vs streaming to analytics pipeline (Kafka → ClickHouse).',
    sampleAnswer:
      'Event schema: (1) chat_message_sent: {user_id, session_id, conversation_id, message_length, timestamp}. (2) ai_response_generated: {user_id, conversation_id, model, input_tokens, output_tokens, latency_ms, cost_usd, provider}. (3) response_feedback: {user_id, message_id, rating (thumbs up/down), feedback_text}. (4) feature_used: {user_id, feature (code_execution, image_gen), success}. (5) user_session_started/ended: {user_id, session_id, duration_seconds}. High volume (1M/day = 12 events/sec): Use streaming pipeline. Architecture: App → Kafka (buffer) → Consumers: (a) Real-time aggregator (Flink/Kafka Streams) → Redis (dashboard data), (b) Warehouse writer → ClickHouse (historical data), (c) Backup → S3 (raw events). Real-time: Pre-aggregate metrics every 5s (e.g., "requests last 5s"), store in Redis, dashboard polls Redis. Lookback window: 1 hour (720 data points). Historical: Batch job (daily) aggregates to: daily_metrics (date, metric, value). Query from ClickHouse (columnar, fast). Privacy: (1) Anonymize user_id (hash), never store PII in events. (2) Retention: Events 90 days, aggregates 2 years. (3) Deletion: User requests deletion → Delete from events (7 days), anonymize in aggregates. (4) Opt-out: If user opts out, stop sending events. PostgreSQL comparison: Works for <100k events/day, but: slow for aggregations, scales vertically (expensive), not optimized for time-series. ClickHouse: Columnar storage, 100x faster for aggregations, handles billions of rows, cost-effective. Recommendation: Kafka + ClickHouse for scale, PostgreSQL for small apps (<10k users).',
    keyPoints: [
      'Track: message sent, response generated (tokens/cost), feedback, feature usage',
      'Streaming: Kafka → Flink (real-time agg) → Redis (dashboard) + ClickHouse (historical)',
      'Real-time: pre-aggregate every 5s to Redis, dashboard polls',
      'Privacy: anonymize user_id, 90-day retention, support deletion/opt-out',
      'Scale: ClickHouse 100x faster than PostgreSQL for time-series aggregations',
    ],
  },
  {
    id: 'bcap-pam-q-2',
    question:
      'You need to calculate ROI for your AI product. Track: (1) Cost per user (LLM + infrastructure), (2) Revenue per user (subscription + overage), (3) User retention (cohorts), (4) Feature adoption. For a product at $20/mo with 1000 users, current cost is $15/user/mo. How do you: analyze profitability, identify high-value users, optimize for retention? What metrics indicate product-market fit?',
    sampleAnswer:
      'Profitability analysis: Current: Revenue = 1000 × $20 = $20k/mo. Cost = 1000 × $15 = $15k/mo. Profit = $5k/mo (25% margin). Break-even at 750 users. Metrics to track: (1) Unit economics: LTV (Lifetime Value) = $20 × avg_months_retained. CAC (Customer Acquisition Cost) = marketing_spend / new_users. LTV:CAC ratio target: >3:1. (2) Cohort retention: Track monthly cohorts, % still active after 1/3/6/12 months. Good retention: >40% at month 6. Calculate LTV: $20 × (1 + 0.85 + 0.7 + 0.6 + ...) for each cohort. (3) Cost per user: Segment by usage tier. Power users cost $30/mo, light users $5/mo. (4) Revenue per user: Base $20 + overage (tokens, features). Analyze: Do power users pay more (overage) or churn? High-value users: (1) High LTV: Active >6 months, low churn risk, high engagement. (2) Feature adoption: Use multiple features (chat + image gen), indicate stickiness. (3) Payment history: Upgraded to higher tier, purchased add-ons. (4) Referrals: Users who invited others (viral coefficient). Retention optimization: (1) Identify churn triggers: Last activity >14 days → 80% churn. (2) Intervention: Email "We miss you" with discount. (3) Feature gaps: Survey churned users, build requested features. (4) Pricing: If cost >$12/user, margin too thin, raise price or reduce costs. Product-market fit indicators: (1) Retention: >40% at month 6 (product is sticky). (2) NPS: >50 (users love it). (3) Organic growth: >20% users from referrals. (4) Usage: >3 sessions/week (high engagement). (5) Willingness to pay: <5% churn on price increase.',
    keyPoints: [
      'Unit economics: LTV:CAC >3:1, LTV = price × avg_months_retained',
      'Cohort retention: track monthly cohorts, >40% at month 6 = good',
      'Segment costs: power users ($30) vs light users ($5), optimize separately',
      'High-value users: long retention, feature adoption, referrals',
      'PMF indicators: >40% retention month 6, NPS >50, >20% organic growth',
    ],
  },
  {
    id: 'bcap-pam-q-3',
    question:
      'Design an A/B testing framework for AI product features. Example tests: (1) GPT-4o vs Claude Sonnet (quality vs cost), (2) Response length (short vs detailed), (3) Pricing ($15 vs $20 vs $25). How do you: assign users to variants, track metrics, determine statistical significance, handle contamination (user switches variants)? What sample size is needed for 95% confidence?',
    sampleAnswer:
      'A/B testing framework: (1) Assignment: Hash user_id → deterministic variant (user always sees same variant). Stratified randomization: Balance by: user tier, signup date, usage level. Store in DB: experiments (id, name, variants, rollout_percent), user_experiments (user_id, experiment_id, variant). (2) Rollout: Start 5% (test), 95% (control). Monitor for 1 week. If metrics good, ramp to 50/50. If neutral/negative, rollback. (3) Metrics: Primary (e.g., thumbs up rate), Secondary (cost, latency), Guardrail (error rate, churn). Track per variant. (4) Significance: Use t-test or chi-square. Require: p-value <0.05 (95% confidence), run for ≥7 days (capture weekly patterns), ≥100 conversions per variant (avoid noise). (5) Sample size: For thumbs up rate: Baseline 30%, MDE (Minimum Detectable Effect) 5% (want to detect 30%→35%). Calculator: n = 2 × (1.96 + 0.84)^2 × p × (1-p) / (MDE)^2 = 2 × 7.84 × 0.3 × 0.7 / 0.05^2 = 1,310 users per variant. Total: 2,620 users. If 10k active users, test reaches significance in ~2-3 days. Contamination: (1) Prevent: No switching variants mid-test. (2) Detect: Log if user sees multiple variants (should never happen). (3) Handle: Exclude contaminated users from analysis (<1% typically). Example tests: (1) GPT-4o vs Claude: Metrics: quality (thumbs up), cost ($0.03 vs $0.02), latency (500ms vs 800ms). Decision: If Claude quality ≥95% of GPT-4o and cost 33% lower → switch to Claude. (2) Response length: Metrics: completion rate, engagement (follow-up questions), satisfaction. Hypothesis: Detailed better for tutorials, short better for quick questions. Segment by query type. (3) Pricing: Track: conversion rate (free→paid), churn rate, revenue. Avoid: testing large price increases without qualitative research (risk alienating users).',
    keyPoints: [
      'Assignment: hash user_id (deterministic), stratified randomization, store in DB',
      'Rollout: start 5/95, monitor, ramp to 50/50 if positive',
      'Metrics: primary (thumbs up), secondary (cost), guardrail (churn)',
      'Sample size: ~1,300 per variant for 30%→35% detection (95% confidence)',
      'Contamination: prevent switching, log if occurs, exclude from analysis',
    ],
  },
];
