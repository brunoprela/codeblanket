import { MultipleChoiceQuestion } from '@/lib/types';

export const llmProductionSystemsQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'llmps-mc-1',
    question:
      'Your LLM-based application serves 1000 concurrent users with streaming responses. You notice that GPU memory usage spikes during traffic peaks, causing out-of-memory errors. Which optimization would most effectively address this issue?',
    options: [
      'Implement continuous batching (dynamic batching) to efficiently pack requests with varying generation lengths',
      'Reduce the maximum sequence length for all requests',
      'Add more GPU memory by upgrading to larger instances',
      'Implement request queuing to limit concurrent requests',
    ],
    correctAnswer: 0,
    explanation:
      "Continuous batching (also called dynamic batching or iteration-level batching) is crucial for LLM serving: as sequences in a batch complete, new requests are added, maximizing GPU utilization and throughput. This handles varying response lengths efficiently without wasting memory on padding. PagedAttention (vLLM) implements this effectively. Option B (reduce length) hurts functionality. Option C (more memory) is expensive and doesn't address inefficiency. Option D (queueing) reduces throughput and increases latency unnecessarily if batching is optimized.",
    difficulty: 'advanced',
    topic: 'LLM Production Systems',
  },
  {
    id: 'llmps-mc-2',
    question:
      'Your LLM application uses GPT-4 via API with costs of $0.03 per 1K prompt tokens and $0.06 per 1K completion tokens. Monthly costs are $50K with 60% from prompts. Which cost optimization strategy would provide the largest savings?',
    options: [
      'Implement prompt caching to reuse common prompt prefixes',
      'Reduce completion length by lowering max_tokens parameter',
      'Switch to a smaller, cheaper model for all requests',
      'Implement aggressive rate limiting to reduce usage',
    ],
    correctAnswer: 0,
    explanation:
      "Prompt caching has the highest impact: 60% of costs ($30K) come from prompts, and many applications have repeated prompt prefixes (system prompts, context). Caching can reduce prompt costs by 50-90% ($15K-$27K savings). OpenAI and other providers offer prompt caching features. Option B (reduce completion length) targets 40% of costs and may hurt quality. Option C (smaller model) may significantly degrade quality. Option D (rate limiting) reduces functionality and doesn't address cost efficiency.",
    difficulty: 'advanced',
    topic: 'LLM Production Systems',
  },
  {
    id: 'llmps-mc-3',
    question:
      'Your LLM-powered customer service chatbot occasionally generates inappropriate or factually incorrect responses. Which combination of techniques provides the most robust safeguards?',
    options: [
      'Fine-tune the LLM on curated customer service data only',
      'Multi-layer approach: input validation + output moderation + fallback to human agent + response validation against knowledge base',
      'Use a single high-quality prompt with detailed instructions',
      'Implement strict output filtering with regex patterns',
    ],
    correctAnswer: 1,
    explanation:
      "Multi-layer defense-in-depth is most robust: (1) Input validation blocks malicious/off-topic inputs, (2) Output moderation catches inappropriate content, (3) Factual validation checks against knowledge base, (4) Fallback to human agents for uncertain cases. This catches issues at multiple points. Fine-tuning alone (option A) doesn't prevent all issues. Good prompts (option C) help but aren't sufficient for high-stakes applications. Regex filtering (option D) is brittle and easily bypassed by paraphrasing.",
    difficulty: 'advanced',
    topic: 'LLM Production Systems',
  },
  {
    id: 'llmps-mc-4',
    question:
      'Your LLM application experiences variable latency: sometimes responding in 2 seconds, sometimes 20 seconds for similar requests. Investigation shows this correlates with response length but not prompt length. What is the most likely cause and solution?',
    options: [
      'API rate limiting is causing throttling; implement exponential backoff',
      'Model inference is variable; switch to a different model provider',
      'Generation length is unbounded; implement max_tokens limits and streaming with early stopping',
      'Network latency is causing delays; deploy models closer to users',
    ],
    correctAnswer: 2,
    explanation:
      "Variable latency correlating with response length indicates unbounded generation: some responses generate many tokens (20+ seconds for 1000+ tokens). Solutions include setting appropriate max_tokens limits, implementing streaming with early stopping when sufficient information is provided, and setting stop sequences. This gives predictable latency. Option A (rate limiting) would show error messages or consistent delays. Option B (provider) doesn't address the root cause. Option D (network) doesn't explain the correlation with response length.",
    difficulty: 'intermediate',
    topic: 'LLM Production Systems',
  },
  {
    id: 'llmps-mc-5',
    question:
      "You're deploying an LLM application that must handle API failures gracefully. The LLM provider has 99.9% uptime, but your application requires higher reliability. What is the most effective fallback strategy?",
    options: [
      'Cache previous responses and return cached results when API fails',
      'Implement multi-provider fallback: primary provider → secondary provider → graceful degradation message',
      'Retry failed requests indefinitely with exponential backoff',
      'Return an error message to users when the API fails',
    ],
    correctAnswer: 1,
    explanation:
      "Multi-provider fallback provides the best reliability: primary provider (e.g., OpenAI) → secondary provider (e.g., Anthropic, Azure) → graceful degradation (simplified service or informative message). This combines reliability with cost optimization (use cheaper primary, fall back when needed). Option A (caching) doesn't work for novel queries. Option C (infinite retry) creates poor user experience and may violate API rate limits. Option D (error message) provides no resilience and poor user experience.",
    difficulty: 'advanced',
    topic: 'LLM Production Systems',
  },
];
