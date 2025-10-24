import { MultipleChoiceQuestion } from '../../../types';

export const cachingStrategiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-cache-mc-1',
    question:
      'What is the main advantage of semantic caching over exact-match caching?',
    options: [
      'Faster lookups',
      'Matches similar prompts even with different wording',
      'Uses less memory',
      'More accurate',
    ],
    correctAnswer: 1,
    explanation:
      'Semantic caching uses embeddings to match prompts by meaning, not exact text. "What is Python?" and "Tell me about Python" can match, greatly increasing hit rate.',
  },
  {
    id: 'pllm-cache-mc-2',
    question: 'What is a good semantic similarity threshold for LLM caching?',
    options: ['0.5-0.6', '0.7-0.8', '0.95-0.97', '0.99-1.0'],
    correctAnswer: 2,
    explanation:
      'A threshold of 0.95-0.97 balances cache hits with accuracy. Lower risks serving wrong responses, higher misses too many valid matches.',
  },
  {
    id: 'pllm-cache-mc-3',
    question: 'Why use Redis instead of in-memory caching for LLM responses?',
    options: [
      'Faster access',
      'Shared across multiple server instances',
      'Cheaper',
      'Better for small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Redis provides shared caching across all server instances, persistence across restarts, and larger capacity than in-memory. Critical for distributed systems.',
  },
  {
    id: 'pllm-cache-mc-4',
    question: 'What is Claude prompt caching?',
    options: [
      'Client-side caching',
      'Native caching of KV cache for common prefixes on Claude servers',
      'Redis integration',
      'Semantic caching',
    ],
    correctAnswer: 1,
    explanation:
      'Claude prompt caching caches the key-value cache of marked text on their servers for 5 minutes, dramatically reducing cost for repeated context.',
  },
  {
    id: 'pllm-cache-mc-5',
    question: 'What cache hit rate should you target for cost optimization?',
    options: ['20-30%', '40-50%', '70-90%', '95-100%'],
    correctAnswer: 2,
    explanation:
      'A 70-90% cache hit rate is achievable with good caching strategies and provides massive cost savings (70-90% reduction in API calls).',
  },
];
