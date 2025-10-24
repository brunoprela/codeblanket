export const llmCostOptimizationMC = {
  title: 'LLM Cost Optimization Quiz',
  id: 'llm-cost-optimization-mc',
  sectionId: 'llm-cost-optimization',
  questions: [
    {
      id: 1,
      question:
        'Why are output tokens typically 2-5x more expensive than input tokens?',
      options: [
        'Output tokens are higher quality',
        'Autoregressive generation requires sequential computation for each output token',
        'Output tokens use more memory',
        'Providers charge more arbitrarily',
      ],
      correctAnswer: 1,
      explanation:
        'Output generation is sequential (autoregressive)—each token requires a full forward pass through the model. Input tokens can be processed in parallel in a single pass. This computational difference is reflected in pricing: Claude Sonnet charges $3/M input, $15/M output (5x).',
    },
    {
      id: 2,
      question:
        'What is the typical cost savings from implementing aggressive response caching?',
      options: [
        '10-20% reduction',
        '30-40% reduction',
        '70-90% reduction',
        'No significant savings',
      ],
      correctAnswer: 2,
      explanation:
        'With good cache hit rates (70-90% for common queries), you can achieve 70-90% cost reduction. If 80% of queries hit cache and you only pay for the first query, costs drop dramatically. Semantic caching extends this to similar queries.',
    },
    {
      id: 3,
      question:
        'At approximately what usage level does self-hosting LLMs become more cost-effective than APIs?',
      options: [
        '100K tokens/day',
        '1M tokens/day',
        '5-10M tokens/day',
        '100M tokens/day',
      ],
      correctAnswer: 2,
      explanation:
        'Break-even is typically 5-10M tokens/day. Below this, API costs are lower than infrastructure costs (GPUs, engineers, maintenance). Above this, dedicated infrastructure becomes cheaper. The exact point depends on model size, quality requirements, and engineering resources.',
    },
    {
      id: 4,
      question: 'What is "model cascading" for cost optimization?',
      options: [
        'Using multiple models in parallel',
        'Trying a cheap model first, escalating to expensive model only if needed',
        'Training smaller models from larger ones',
        'Falling back to older model versions',
      ],
      correctAnswer: 1,
      explanation:
        'Model cascading tries a cheap/fast model (GPT-3.5) first. If confidence is low or the task seems complex, escalate to expensive model (GPT-4). Many queries (70%+) can be handled by cheaper models, significantly reducing average cost.',
    },
    {
      id: 5,
      question: "What is Claude's prompt caching feature designed to optimize?",
      options: [
        'Model loading time',
        'Reusing common static prompt prefixes across requests',
        'Caching final responses',
        'Pre-computing embeddings',
      ],
      correctAnswer: 1,
      explanation:
        'Prompt caching caches long static prefixes (system prompts, documentation, few-shot examples) so subsequent requests only pay for cache lookup (~10% cost) plus new tokens. With a 50k token prefix, this can reduce costs 90% if the prefix is reused frequently.',
    },
  ],
};
