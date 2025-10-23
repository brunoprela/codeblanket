/**
 * Quiz questions for LLM APIs & Providers section
 */

export const llmapisprovidersQuiz = [
  {
    id: 'q1',
    question:
      'You are building an application that processes 10,000 requests per day with an average of 2,000 input tokens and 1,000 output tokens per request. Compare the monthly costs and trade-offs between using GPT-4 Turbo, GPT-3.5 Turbo, and Claude 3 Sonnet. Which would you choose and why?',
    sampleAnswer:
      'Let me calculate monthly costs: GPT-4 Turbo would cost ~$1,200/month ($10 input + $30 output per 1M tokens × 30M input + 30M output tokens), GPT-3.5 Turbo ~$60/month ($0.50 input + $1.50 output), and Claude 3 Sonnet ~$450/month ($3 input + $15 output). For most production applications, I would choose GPT-3.5 Turbo because: (1) It offers the best cost-to-performance ratio at just 5% of GPT-4 cost, (2) Quality is sufficient for most tasks (90% of GPT-4 quality), (3) It has a 16K context window which handles most use cases, and (4) The $660/month savings can be reinvested into other features. However, I would use GPT-4 Turbo for complex tasks like code generation or detailed analysis where quality justifies the 20x cost increase, and Claude 3 Sonnet for tasks requiring large context windows (200K tokens) at medium quality.',
    keyPoints: [
      'Calculate actual costs per request and monthly totals',
      'GPT-3.5 Turbo is 20x cheaper than GPT-4 Turbo',
      'Claude offers 200K context window advantage',
      'Choose based on quality requirements vs budget',
      'Consider hybrid approach using different models for different tasks',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the advantages and disadvantages of building a unified LLM client that supports multiple providers (OpenAI, Anthropic, Google) versus committing to a single provider. What architectural decisions would you make?',
    sampleAnswer:
      'A unified client offers significant advantages: (1) Vendor flexibility - switch providers without rewriting code, critical if a provider changes pricing or has outages, (2) Cost optimization - route requests to cheapest provider for each task type, (3) A/B testing - easily compare quality across providers, (4) Fallback strategy - use secondary provider if primary fails, and (5) Future-proofing - new providers can be added with minimal changes. However, disadvantages include: (1) Complexity - maintaining adapters for different API formats, (2) Lowest common denominator - features specific to one provider may not be available, (3) Testing overhead - must test across all providers, and (4) Performance - abstraction layer adds latency. My architecture would include: a Provider interface with standard methods (chat, complete, embed), provider-specific adapters implementing the interface, a router to select providers based on rules (cost, quality, availability), comprehensive error handling for provider-specific failures, and configuration-driven provider selection. I would start with 2-3 main providers and design for extensibility.',
    keyPoints: [
      'Unified client enables vendor flexibility and cost optimization',
      'Disadvantages include complexity and testing overhead',
      'Use interface/adapter pattern for abstraction',
      'Implement router for intelligent provider selection',
      'Design for extensibility but start simple',
    ],
  },
  {
    id: 'q3',
    question:
      'A team member wants to use GPT-4 for all tasks in your application to "maximize quality." Explain why this might not be the best approach, and describe a strategy for selecting the right model for different tasks.',
    sampleAnswer:
      'Using GPT-4 for all tasks is costly and unnecessary because: (1) Most tasks do not require GPT-4 level reasoning - simple extraction, classification, or summarization work fine with GPT-3.5, (2) Cost compounds quickly - at 20x the price, GPT-4 makes the application unsustainable at scale, (3) GPT-3.5 is often faster, improving user experience, and (4) Quality improvement is marginal for simple tasks. A better strategy is task-based model selection: For deterministic tasks (data extraction, classification, simple Q&A) use GPT-3.5 at temperature 0 - quality is nearly identical to GPT-4 but cost is 5% of GPT-4. For creative tasks (writing, brainstorming) use Claude Sonnet - comparable quality at lower cost. For complex reasoning (code generation, multi-step problems, detailed analysis) use GPT-4 - the quality justifies the cost. For very high-volume simple tasks (sentiment analysis) use Claude Haiku - cheapest option at $0.25/1M tokens. Implementation: Create a model selector function that routes based on task type, measure quality metrics for each task-model combination, set up A/B tests to validate cheaper models work, monitor costs by task type, and gradually migrate expensive tasks to cheaper models where quality is acceptable.',
    keyPoints: [
      'Using expensive models for all tasks is wasteful',
      'Simple tasks do not benefit from GPT-4',
      'Route tasks to appropriate models based on complexity',
      'Measure quality vs cost trade-offs empirically',
      'Most applications need a mix of models',
    ],
  },
];
