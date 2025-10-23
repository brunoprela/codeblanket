import { MultipleChoiceQuestion } from '../../../types';

export const toolUseObservabilityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question:
      'What is the primary purpose of observability in tool-using systems?',
    options: [
      'To reduce costs',
      'To monitor, debug, and optimize tool usage',
      'To make tools run faster',
      'To reduce code complexity',
    ],
    correctAnswer: 1,
    explanation:
      'Observability enables monitoring performance, debugging issues, understanding usage patterns, and optimizing tool-using systems.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What key metrics should you track for tool usage?',
    options: [
      'Only execution time',
      'Call counts, success rates, latency, costs, and error types',
      'Just error counts',
      'Only costs',
    ],
    correctAnswer: 1,
    explanation:
      'Comprehensive metrics include call counts, success rates, latency percentiles, costs, and error patterns to understand system health.',
  },
  {
    id: 'fcfc-mc-3',
    question: 'What is distributed tracing used for?',
    options: [
      'Running tools in parallel',
      'Following request flow across multiple services and function calls',
      'Reducing latency',
      'Caching results',
    ],
    correctAnswer: 1,
    explanation:
      'Distributed tracing tracks requests as they flow through multiple services and function calls, helping identify bottlenecks and failures.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'Why is cost tracking important in tool-using systems?',
    options: [
      "It's required by law",
      'To understand expenses, set budgets, and optimize usage',
      'To make tools run faster',
      'To improve accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Cost tracking helps understand expenses per tool/user, set appropriate budgets, and identify optimization opportunities.',
  },
  {
    id: 'fcfc-mc-5',
    question:
      'What should you do when tool error rates exceed acceptable thresholds?',
    options: [
      'Ignore it',
      'Trigger alerts and investigate root causes',
      'Disable the tool',
      'Restart the server',
    ],
    correctAnswer: 1,
    explanation:
      'High error rates should trigger alerts for immediate investigation to identify and fix root causes before they impact users.',
  },
];
