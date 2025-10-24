import { MultipleChoiceQuestion } from '../../../types';

export const apiDesignForLlmAppsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-api-mc-1',
    question:
      'What is the recommended approach for handling long-running LLM requests in an API?',
    options: [
      'Block until completion',
      'Return immediately with task_id and provide status endpoint',
      'Use polling with short timeouts',
      'Reject all slow requests',
    ],
    correctAnswer: 1,
    explanation:
      'For long-running requests, immediately return a task_id and provide a separate endpoint to check status. This prevents timeout issues and provides better UX with progress updates.',
  },
  {
    id: 'pllm-api-mc-2',
    question:
      'Which HTTP status code should you return when a user exceeds their rate limit?',
    options: [
      '400 Bad Request',
      '401 Unauthorized',
      '429 Too Many Requests',
      '500 Internal Server Error',
    ],
    correctAnswer: 2,
    explanation:
      '429 Too Many Requests is the standard status code for rate limiting. Include Retry-After header to tell clients when they can retry.',
  },
  {
    id: 'pllm-api-mc-3',
    question:
      'What is the advantage of Server-Sent Events (SSE) over WebSockets for streaming LLM responses?',
    options: [
      'SSE is bidirectional',
      'SSE is simpler and works over HTTP with automatic reconnection',
      'SSE uses less bandwidth',
      'SSE is faster',
    ],
    correctAnswer: 1,
    explanation:
      'SSE is simpler than WebSockets, works over standard HTTP, has automatic reconnection, and is sufficient for one-way streaming. Browser EventSource API makes it easy to use.',
  },
  {
    id: 'pllm-api-mc-4',
    question: 'How should you version your LLM API?',
    options: [
      'Query parameters (?version=2)',
      'URL path (/v1/, /v2/)',
      'Custom headers',
      'No versioning needed',
    ],
    correctAnswer: 1,
    explanation:
      'URL path versioning (/v1/, /v2/) is the most clear and explicit approach. It makes versions discoverable, easy to route, and simple to deprecate old versions.',
  },
  {
    id: 'pllm-api-mc-5',
    question: 'What information should error responses include?',
    options: [
      'Just error message',
      'Error code, message, request_id, and helpful suggestions',
      'Stack trace',
      'Nothing, just status code',
    ],
    correctAnswer: 1,
    explanation:
      'Error responses should include: error code for programmatic handling, human-readable message, request_id for support, and actionable suggestions for resolution.',
  },
];
