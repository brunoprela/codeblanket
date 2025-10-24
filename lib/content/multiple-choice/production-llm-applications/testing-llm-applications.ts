import { MultipleChoiceQuestion } from '../../../types';

export const testingLlmApplicationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-test-mc-1',
    question:
      'How should you test LLM applications without excessive API costs?',
    options: [
      'Test everything with real API',
      'Mock LLM responses in unit tests, limited real API in integration tests',
      'No testing',
      'Only manual testing',
    ],
    correctAnswer: 1,
    explanation:
      'Mock LLM calls in unit tests (no cost), use real API selectively in integration tests with caching, temperature=0 for reproducibility.',
  },
  {
    id: 'pllm-test-mc-2',
    question: 'How do you make LLM integration tests reproducible?',
    options: [
      'They cant be reproducible',
      'Use temperature=0 for deterministic outputs',
      'Random sampling',
      'Test multiple times',
    ],
    correctAnswer: 1,
    explanation:
      'temperature=0 makes outputs deterministic (always same output for same input). Cache responses for subsequent test runs to avoid costs.',
  },
  {
    id: 'pllm-test-mc-3',
    question: 'What is LLM-as-judge for quality evaluation?',
    options: [
      'Manual review',
      'Using GPT-4 to rate response quality/relevance on 1-5 scale',
      'User ratings',
      'Random scoring',
    ],
    correctAnswer: 1,
    explanation:
      'LLM-as-judge uses GPT-4 to automatically evaluate response quality, relevance, and accuracy on a scale, enabling automated quality testing.',
  },
  {
    id: 'pllm-test-mc-4',
    question: 'How do you test rate limiting?',
    options: [
      'Skip testing',
      'Make requests at and above limit, verify 429 responses and headers',
      'Manual testing',
      'Not possible',
    ],
    correctAnswer: 1,
    explanation:
      'Test rate limiting by making requests at limit (should succeed), exceeding limit (should get 429), verifying X-RateLimit-* headers, testing reset.',
  },
  {
    id: 'pllm-test-mc-5',
    question: 'What is load testing for LLM applications?',
    options: [
      'Testing file uploads',
      'Simulating many concurrent users to identify bottlenecks and capacity limits',
      'Testing API keys',
      'Database backup testing',
    ],
    correctAnswer: 1,
    explanation:
      'Load testing (Locust, k6) simulates realistic user behavior at scale to measure throughput, latency, error rates, and identify bottlenecks before production.',
  },
];
