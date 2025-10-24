import { MultipleChoiceQuestion } from '../../../types';

export const asyncConcurrencyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-async-mc-1',
    question: 'What is the main benefit of async/await for LLM applications?',
    options: [
      'Faster code execution',
      'Simpler syntax',
      'Ability to handle many concurrent I/O operations efficiently',
      'Better error handling',
    ],
    correctAnswer: 2,
    explanation:
      'Async/await allows thousands of concurrent I/O-bound operations (API calls) without blocking. While one request waits for LLM response, others can be processed.',
  },
  {
    id: 'pllm-async-mc-2',
    question:
      'How do you limit concurrency in async Python to avoid overwhelming the LLM API?',
    options: [
      'Thread limits',
      'asyncio.Semaphore',
      'Sleep statements',
      'Process pools',
    ],
    correctAnswer: 1,
    explanation:
      'asyncio.Semaphore(n) limits concurrent operations to n. Use it to ensure you dont exceed API rate limits: async with semaphore: await api_call()',
  },
  {
    id: 'pllm-async-mc-3',
    question:
      'What should you use for CPU-intensive work in an async application?',
    options: [
      'async/await',
      'asyncio.create_task()',
      'loop.run_in_executor() with ThreadPoolExecutor',
      'Just run it normally',
    ],
    correctAnswer: 2,
    explanation:
      'CPU-intensive work blocks the event loop. Use loop.run_in_executor() with ThreadPoolExecutor or ProcessPoolExecutor to run CPU work in threads/processes.',
  },
  {
    id: 'pllm-async-mc-4',
    question: 'How do you handle errors when using asyncio.gather()?',
    options: [
      'Errors propagate automatically',
      'Use return_exceptions=True to get exceptions in results',
      'Wrap each coroutine in try/except',
      'Errors are ignored',
    ],
    correctAnswer: 1,
    explanation:
      'asyncio.gather(*tasks, return_exceptions=True) returns exceptions as results instead of raising immediately, allowing you to handle them per-task.',
  },
  {
    id: 'pllm-async-mc-5',
    question: 'What is backpressure in async processing?',
    options: [
      'Memory leaks',
      'When requests arrive faster than you can process them',
      'Network congestion',
      'Database locks',
    ],
    correctAnswer: 1,
    explanation:
      'Backpressure occurs when incoming requests exceed processing capacity. Handle by: rejecting new requests, queuing with limits, or degrading service quality.',
  },
];
