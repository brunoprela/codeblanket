export const costManagementQuiz = [
  {
    id: 'pllm-q-13-1',
    question:
      'Design a comprehensive cost tracking and optimization system for an LLM application. How would you track costs per user, per feature, and per model? Include specific metrics, alerts, and optimization strategies.',
    sampleAnswer:
      'Cost tracking: Record every API call in database: {user_id, model, input_tokens, output_tokens, cost, timestamp, feature, request_id}. Calculate cost immediately after API call using pricing table. Aggregate: per user daily/monthly, per model, per feature, per time period. Real-time dashboard: Current hourly burn rate, daily/monthly spend, cost by model breakdown, top 10 spending users, projected monthly cost. Per-user tracking: Store in CostRecords table, index on (user_id, date), cache today totals in Redis. Per-feature: Tag requests with feature_name (chat, summarization, code_gen), analyze which features cost most. Per-model: Compare GPT-4 vs GPT-3.5 costs, identify opportunities to downgrade. Alerts: Hourly cost >$20 (Slack), daily cost >$400 (Email + Slack), user approaching budget (Email at 80%), anomalous spike (>2x normal, PagerDuty). Optimization strategies: 1) Aggressive caching (80% hit rate = 80% savings), 2) Model routing (use GPT-3.5 when quality difference minimal), 3) Prompt optimization (reduce tokens by 30% through compression), 4) Max tokens limits (cap output to prevent runaway), 5) Rate limiting (prevent abuse), 6) Batch processing (off-peak hours). ROI calculation: If optimization saves $100/day, costs $5K dev time, ROI in 50 days.',
    keyPoints: [
      'Granular tracking per user/model/feature with real-time aggregation',
      'Multi-level alerts for different cost thresholds',
      'Multiple optimization strategies with measurable ROI',
    ],
  },
  {
    id: 'pllm-q-13-2',
    question:
      'Explain cost-based rate limiting where users have daily budgets rather than request limits. How do you handle estimation errors, prevent budget overruns, and provide good user experience?',
    sampleAnswer:
      'Budget system: Each user has daily_budget (Free: $1, Pro: $50, Enterprise: $1000), track current_spend in Redis, check before each request. Request flow: 1) Estimate cost (prompt_tokens/1000 * input_price + estimated_output_tokens/1000 * output_price), 2) Check if current_spend + estimated_cost > budget, 3) If yes: reject with budget error, else: reserve estimated cost, 4) Make API call, 5) Calculate actual cost, 6) Adjust: refund if over-estimated, charge extra if under. Handle estimation errors: Conservative estimates (assume max_tokens for output), track accuracy (actual_cost / estimated_cost), adjust multiplier based on historical data (if avg ratio is 0.6, multiply estimates by 0.6). Prevent overruns: Hard stop at 100% budget, soft warnings at 80% and 90%, use Redis WATCH for atomic check-and-increment (prevent race conditions), reserve maximum possible cost upfront for streaming. Good UX: Real-time budget dashboard showing spent/remaining, cost estimates before expensive operations, email notifications at milestones, allow budget rollover (10% to next day), suggest optimizations (Your summaries cost $0.50 each, try shorter documents), provide cost calculator. Emergency handling: Admin can temporarily increase budget, implement credit system for overages, offer one-time budget increases for good customers.',
    keyPoints: [
      'Conservative estimation with adjust-after-actual pattern',
      'Atomic Redis operations to prevent race conditions and overruns',
      'Transparent UX with real-time tracking and proactive notifications',
    ],
  },
  {
    id: 'pllm-q-13-3',
    question:
      'How would you optimize LLM costs by 50% without degrading user experience? Provide specific strategies with expected savings and implementation complexity.',
    sampleAnswer:
      'Strategy 1: Aggressive caching (expected: 60-80% savings on cache hits). Implement semantic caching with 0.95 similarity threshold, 70% hit rate achievable. Savings: 70% * 60% = 42% total cost. Complexity: Medium (2 weeks). Strategy 2: Model routing (expected: 20-30% savings). Use GPT-3.5 ($0.002) instead of GPT-4 ($0.06) when quality difference <10% (simple questions, well-defined tasks). Route 50% of requests to cheaper model. Savings: 50% * 97% = 48.5% on routed requests = 24% total. Complexity: Medium (1 week). Strategy 3: Prompt optimization (expected: 15-25% savings). Compress prompts using techniques: remove redundancy, use abbreviations, optimize system prompts. Reduce average tokens by 20%. Savings: 20% of input costs = 10% total (input is ~50% of cost). Complexity: Low (3 days). Strategy 4: Output limits (expected: 10-20% savings). Set max_tokens based on use case (summaries: 200, chat: 500, code: 1500). Prevents runaway generation. Savings: 15% total. Complexity: Low (1 day). Strategy 5: Batching (expected: 5-10% savings). Batch similar requests, process during off-peak hours, negotiate volume discounts. Savings: 7% total. Complexity: High (3 weeks). Combined strategy: Implement 1-4 for total 91% savings without batching complexity. Gradual rollout: Start with caching (quick win), add model routing (A/B test quality), optimize prompts, set output limits. Monitor quality metrics throughout.',
    keyPoints: [
      'Multi-faceted approach targeting different cost drivers',
      'Quantified savings with realistic hit rates and implementation effort',
      'Gradual rollout with quality monitoring at each step',
    ],
  },
];
