/**
 * Quiz questions for Cost Tracking & Optimization section
 */

export const costtrackingoptimizationQuiz = [
    {
        id: 'q1',
        question:
            'Your application currently uses GPT-4 for all tasks and costs $3,000/month. Management wants to reduce costs by 80% while maintaining acceptable quality. Describe a comprehensive strategy to achieve this goal.',
        sampleAnswer:
            'To reduce from $3,000/month to $600/month while maintaining quality, I would implement a multi-tier strategy: (1) Task classification and routing (estimated 60% savings): Analyze current usage by task type, classify into simple (extraction, classification, short Q&A), medium (summarization, simple writing), and complex (code generation, deep analysis). Route simple tasks to GPT-3.5 (20x cheaper) - likely 40-50% of tasks; medium tasks to Claude Sonnet (3x cheaper than GPT-4) - likely 30-40% of tasks; complex tasks stay on GPT-4 - only 10-20% of tasks. Impact: If 40% simple, 40% medium, 20% complex, cost becomes: 0.4×$150 (GPT-3.5) + 0.4×$1000 (Claude Sonnet) + 0.2×$3000 (GPT-4) = $1,060 (65% reduction). (2) Prompt optimization (15% savings): Remove unnecessary context and repetitive instructions from prompts, use templates to avoid regenerating same instructions, truncate prompts to necessary information only. Average prompt reduction of 30% across all tasks saves 15% on input tokens. (3) Caching (10% additional savings): Implement Redis cache for exact matches, semantic cache for similar queries. Even 20% cache hit rate saves 10% total costs. (4) Response length control (5% savings): Set appropriate max_tokens per task type - do not allow 2000 tokens for tasks needing 200. Output tokens cost 2-3x input. (5) A/B testing quality: Before full rollout, A/B test cheaper models on representative tasks, measure quality metrics (accuracy, user satisfaction), iterate on prompt engineering to improve cheaper model performance. Implementation timeline: Week 1 - implement task classification, Week 2 - deploy routing for simple tasks only, Week 3 - expand to medium complexity if quality acceptable, Week 4 - implement caching. Result: Achieve $600-900/month target while maintaining 90%+ quality score on most tasks.',
        keyPoints: [
            'Route tasks by complexity to appropriate models',
            'GPT-3.5 for simple tasks is 20x cheaper',
            'Optimize prompts to reduce token usage',
            'Implement caching for repeated queries',
            'A/B test quality before full rollout'
        ]
    },
    {
        id: 'q2',
        question:
            'Explain the difference between tracking costs at the request level versus the user level, and why both are important for a production LLM application.',
        sampleAnswer:
            'Request-level cost tracking measures cost per individual API call - captures model, prompt/completion tokens, and calculated cost for each request. Important because: (1) Identifies expensive operations - find which specific requests cost the most, (2) Debugging cost spikes - when costs jump, trace to specific request patterns, (3) Optimization targets - know exactly which operations to optimize first, and (4) Granular analysis - understand cost by task type, prompt length, response length. User-level cost tracking aggregates costs per user - totals per user per day/month, lifetime value, and cost relative to revenue. Important because: (1) User profitability - some users cost more than they generate in revenue, (2) Abuse detection - users consuming excessive resources (intentionally or bugs), (3) Tiered pricing decisions - set free tier limits based on actual costs, (4) Cost allocation - in multi-tenant apps, bill customers accurately. Real example: Request-level analysis shows "code generation" costs $0.05/request. User-level shows: User A: 10 requests/day × 30 days = $15/month. User B: 1000 requests/day × 30 days = $1,500/month. If both are on $50/month plan, User B is unprofitable. Action: (1) Implement rate limits per user, (2) Add usage tiers - higher limits for paid users, (3) Optimize expensive operations identified from request-level data, (4) Block/warn users exceeding reasonable thresholds. Implementation: Store request logs with user_id, aggregate daily by user_id for alerting, dashboard showing both views - top expensive requests AND top expensive users, alerts when user exceeds budget thresholds. Both views together enable: technical optimization (request-level) and business sustainability (user-level).',
        keyPoints: [
            'Request-level identifies expensive operations',
            'User-level reveals profitability and abuse',
            'Both needed for technical and business optimization',
            'User-level enables rate limiting and tier decisions',
            'Track and alert on both dimensions'
        ]
    },
    {
        id: 'q3',
        question:
            'You notice that 30% of your LLM costs come from a single feature used by only 5% of users. How would you approach optimizing or handling this situation?',
        sampleAnswer:
            'This scenario requires balancing cost optimization with user value. Analysis approach: (1) Understand the feature - is it high-value (critical for those 5%) or low-value (nice-to-have)? Survey those users. (2) Check user type - are the 5% paying customers, power users, or free tier? High-value customers justify higher costs. (3) Measure engagement - do these users have higher retention, satisfaction, or revenue? (4) Compare alternatives - could the feature work with cheaper models? Optimization strategies by scenario: If high-value to important users: (1) Optimize the expensive feature - can it use GPT-3.5 instead of GPT-4? Batch requests? Cache results? (2) Make it a premium feature - charge extra for expensive functionality. Users who value it will pay. (3) Implement usage limits - free tier gets N uses/month, paid gets unlimited. If low-value or abused: (1) Deprecate the feature - 95% of users do not use it anyway. (2) Add aggressive rate limiting - reduce from expensive problem to manageable cost. (3) Move to opt-in/premium only - reduce visibility, require explicit enabling. If medium-value: (1) Hybrid approach - optimize heavily (switch models, cache, batch), implement soft limits with upgrade paths, monitor closely for abuse patterns. Implementation example: Feature is "Advanced Code Review" costing $900/month for 100 users. After optimization - switch to GPT-3.5, add caching, set limit of 10 reviews/day: cost drops to $200/month. Add "Pro Code Review" tier for $20/month: 20 users upgrade = $400/month revenue, net profit $200/month. Feature transforms from cost center to profit center. Key principle: Not all features should be equally accessible. Expensive features should create value that justifies costs, either through revenue or critical user retention.',
        keyPoints: [
            'Analyze feature value vs cost trade-off',
            'Identify if users are high-value or low-value',
            'Optimize expensive features or make them premium',
            'Consider deprecation if low-value',
            'Transform cost centers into revenue opportunities'
        ]
    }
];

