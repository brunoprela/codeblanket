import { MultipleChoiceQuestion } from '../../../types';

export const rateLimitingThrottlingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-ratelimit-mc-1',
    question: 'What algorithm is best for rate limiting LLM APIs?',
    options: ['Fixed window', 'Sliding window', 'Token bucket', 'Leaky bucket'],
    correctAnswer: 2,
    explanation:
      'Token bucket allows bursts up to capacity while maintaining steady-state rate, perfect for LLM apps where users occasionally need multiple requests.',
  },
  {
    id: 'pllm-ratelimit-mc-2',
    question: 'What headers should you include in rate-limited responses?',
    options: [
      'None',
      'X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset',
      'Just error message',
      'Retry-After only',
    ],
    correctAnswer: 1,
    explanation:
      'Standard rate limit headers inform clients of limits, remaining capacity, and when limits reset, enabling smart retry logic.',
  },
  {
    id: 'pllm-ratelimit-mc-3',
    question:
      'How should you implement distributed rate limiting across multiple servers?',
    options: [
      'Local counters per server',
      'Redis with atomic increment operations',
      'Database updates',
      'No coordination needed',
    ],
    correctAnswer: 1,
    explanation:
      'Redis with Lua scripts for atomic operations provides shared, consistent rate limiting across all server instances.',
  },
  {
    id: 'pllm-ratelimit-mc-4',
    question: 'What is cost-based rate limiting?',
    options: [
      'Charging for API calls',
      'Limiting based on $ spend rather than request count',
      'Pricing tiers',
      'Free tier limits',
    ],
    correctAnswer: 1,
    explanation:
      'Cost-based rate limiting tracks actual $ spend and limits when budget is reached, more accurate than request counts since costs vary by request.',
  },
  {
    id: 'pllm-ratelimit-mc-5',
    question: 'How should you handle rate limit violations?',
    options: [
      'Block user permanently',
      'Return 429 with Retry-After and helpful error message',
      'Ignore and process anyway',
      'Return 500 error',
    ],
    correctAnswer: 1,
    explanation:
      'Return 429 status with Retry-After header, clear explanation of limit, and suggestions (upgrade plan, reduce usage), maintaining good UX.',
  },
];
