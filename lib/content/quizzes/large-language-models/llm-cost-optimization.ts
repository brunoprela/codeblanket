export const llmCostOptimizationQuiz = {
  title: 'LLM Cost Optimization Discussion',
  id: 'llm-cost-optimization-quiz',
  sectionId: 'llm-cost-optimization',
  questions: [
    {
      id: 1,
      question:
        'LLM API costs can quickly escalate in production. Analyze the cost structure: input tokens, output tokens, and their different pricing. Why are output tokens more expensive? Discuss optimization strategies across the stack: prompt engineering, caching, model selection, and batching. How can you achieve 10x cost reduction while maintaining quality?',
      expectedAnswer:
        'Should cover: per-token pricing models, output tokens costing 2-5x input tokens due to autoregressive generation, prompt length impact on costs, caching responses for repeated queries (70-90% cost reduction), semantic caching for similar queries, model cascading (cheap model first, escalate if needed), shorter prompts via compression, batching requests to amortize overhead, quantization for self-hosted models, monitoring and alerting on costs, setting budgets per user/feature, and achieving major savings through combination of techniques.',
    },
    {
      id: 2,
      question:
        'Compare the economics of API-based models (OpenAI, Anthropic) versus self-hosting (vLLM, TGI). At what scale does self-hosting become cost-effective? Include infrastructure costs, engineering time, model quality, and maintenance. How do you make this build-vs-buy decision?',
      expectedAnswer:
        'Should analyze: API costs linear with usage, self-hosting having fixed infrastructure costs, break-even point typically 5-10M tokens/day, GPU costs (A100 ~$2-3/hr, H100 ~$5-8/hr), engineering effort for deployment and maintenance, model quality gap between open-source and frontier models, latency and throughput considerations, scaling flexibility, updating to better models, when to self-host (high volume, cost-sensitive, data privacy), when to use APIs (rapid iteration, latest models, variable load), and hybrid approaches.',
    },
    {
      id: 3,
      question:
        "Prompt caching (like Claude's prompt caching feature) can dramatically reduce costs by reusing common prompt prefixes.Explain how prompt caching works and what patterns benefit most.Discuss the implementation considerations: cache invalidation, stale data, and cache hit rates.How do you design prompts to maximize caching benefits?",
      expectedAnswer:
        'Should cover: caching long static prefixes (system prompts, few-shot examples, documentation), charged once then free/reduced on cache hits, 5-10 minute TTL typical, hit rates of 70-90% in some applications, savings calculations (90% of tokens cached → 90% cost reduction), prompt structure for caching (static prefix, variable suffix), cache invalidation on prompt changes, version control for prompts, monitoring cache hit rates, warming caches proactively, tradeoffs with dynamic content, and quantifying ROI of caching implementation.',
    },
  ],
};
