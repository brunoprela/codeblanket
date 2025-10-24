export const llmDeploymentProductionMC = {
  title: 'LLM Deployment & Production Quiz',
  id: 'llm-deployment-production-mc',
  sectionId: 'llm-deployment-production',
  questions: [
    {
      id: 1,
      question:
        'What is the primary benefit of token streaming in production LLM systems?',
      options: [
        'Lower costs',
        'Better accuracy',
        'Improved perceived latency and user experience',
        'Easier caching',
      ],
      correctAnswer: 2,
      explanation:
        "Streaming shows incremental output as it's generated, making responses feel faster even though total latency is similar.Users see progress immediately rather than waiting for complete response.Critical for long outputs and real- time feel.",
    },
    {
      id: 2,
      question: 'What is vLLM designed to optimize?',
      options: [
        'Model training speed',
        'High-throughput inference for self-hosted LLMs',
        'Model compression',
        'Fine-tuning efficiency',
      ],
      correctAnswer: 1,
      explanation:
        'vLLM is an inference server optimized for throughput via PagedAttention (efficient KV cache management) and continuous batching. It achieves 10-20x higher throughput than naive serving with HuggingFace transformers, crucial for cost-effective self-hosting.',
    },
    {
      id: 3,
      question:
        'Why is a "circuit breaker" pattern important in production LLM systems?',
      options: [
        'To save electricity',
        'To prevent cascading failures by stopping requests to failing services',
        'To break long contexts into chunks',
        'To interrupt generation mid-stream',
      ],
      correctAnswer: 1,
      explanation:
        'Circuit breakers detect when a downstream service (API provider, database) is failing and stop sending requests temporarily, preventing cascading failures. After a timeout, they try again. This protects system stability when dependencies fail.',
    },
    {
      id: 4,
      question:
        'What is the recommended approach for handling multiple LLM provider failures?',
      options: [
        'Return error immediately',
        'Retry the same provider indefinitely',
        'Fallback to alternative providers (OpenAI → Anthropic → self-hosted)',
        'Cache old responses forever',
      ],
      correctAnswer: 2,
      explanation:
        'Production systems should have fallback providers. If OpenAI fails, try Anthropic; if that fails, try self-hosted. This maximizes availability. Different providers have different outage patterns, and fallbacks can maintain 99.9%+ uptime.',
    },
    {
      id: 5,
      question:
        'What key metric indicates LLM output quality degradation that standard error monitoring might miss?',
      options: [
        'HTTP status codes',
        'Response latency',
        'User feedback scores or LLM-as-judge evaluations',
        'Memory usage',
      ],
      correctAnswer: 2,
      explanation:
        "Standard monitoring (errors, latency) won't catch quality issues—the API returns 200 OK with bad answers. You need qualitative metrics: user feedback, LLM-as-judge scores, A/B test results, or sample review. Quality monitoring is distinct from availability monitoring.",
    },
  ],
};
