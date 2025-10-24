import { MultipleChoiceQuestion } from '../../../types';

export const errorHandlingResilienceMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-error-mc-1',
    question: 'Which LLM errors should you retry automatically?',
    options: [
      'All errors',
      'Rate limits, timeouts, and 5xx server errors',
      'Invalid requests (400)',
      'Authentication errors (401)',
    ],
    correctAnswer: 1,
    explanation:
      'Retry transient errors: rate limits (429), timeouts, API errors (500/503). Dont retry permanent errors like invalid requests (400) or auth failures (401).',
  },
  {
    id: 'pllm-error-mc-2',
    question: 'What is the circuit breaker pattern?',
    options: [
      'Electrical safety',
      'Stop calling failing services temporarily to let them recover',
      'Error logging',
      'Rate limiting',
    ],
    correctAnswer: 1,
    explanation:
      'Circuit breaker opens after N failures, blocking calls for timeout period. Prevents cascading failures and gives failing services time to recover.',
  },
  {
    id: 'pllm-error-mc-3',
    question: 'What is exponential backoff with jitter?',
    options: [
      'Linear retry delays',
      'Retry delays that double each time (2^n) with random variation',
      'Fixed delays',
      'No delays',
    ],
    correctAnswer: 1,
    explanation:
      'Exponential backoff (2^n seconds) with random jitter prevents thundering herd when many requests retry simultaneously.',
  },
  {
    id: 'pllm-error-mc-4',
    question: 'How should you handle context length exceeded errors?',
    options: [
      'Retry with same prompt',
      'Truncate conversation history and retry',
      'Give up immediately',
      'Use larger model',
    ],
    correctAnswer: 1,
    explanation:
      'Context length errors are non-retriable. Truncate conversation history (keep system prompt + recent messages) and retry with shorter context.',
  },
  {
    id: 'pllm-error-mc-5',
    question: 'What is graceful degradation?',
    options: [
      'Failing completely',
      'Maintaining partial functionality with reduced quality when services fail',
      'Ignoring errors',
      'Automatic retries',
    ],
    correctAnswer: 1,
    explanation:
      'Graceful degradation maintains service with reduced functionality: use cheaper models, cached responses, or limited features instead of complete failure.',
  },
];
