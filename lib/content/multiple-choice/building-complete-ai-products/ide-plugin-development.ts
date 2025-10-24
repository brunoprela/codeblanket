import { MultipleChoiceQuestion } from '../../../types';

export const idePluginDevelopmentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcap-ipd-mc-1',
    question:
      'What is the primary advantage of Server-Sent Events (SSE) over WebSocket for streaming code completions?',
    options: [
      'SSE is faster than WebSocket',
      'SSE is simpler (HTTP-based), has auto-reconnect, and sufficient for unidirectional streaming',
      'SSE can send data bidirectionally',
      'WebSocket cannot stream text',
    ],
    correctAnswer: 1,
    explanation:
      'SSE advantages: (1) HTTP-based (easier auth, works through proxies), (2) Auto-reconnect built-in, (3) Simpler implementation (no handshake), (4) Sufficient for server-to-client streaming. WebSocket is bidirectional (overkill for one-way completions) and more complex. Use SSE unless you need bidirectional communication.',
  },
  {
    id: 'bcap-ipd-mc-2',
    question: 'When should inline completions NOT be shown to the user?',
    options: [
      'After every keystroke',
      'When user is typing rapidly (>200 WPM), mid-word, or inside strings',
      'Only at the end of lines',
      'Never show completions',
    ],
    correctAnswer: 1,
    explanation:
      "Don't show completions when: (1) User typing rapidly (>200 WPM) - indicates flow state, don't interrupt, (2) Cursor mid-word - completion would be jarring, (3) Inside string literals (unless code-related), (4) Recently rejected completion (user doesn't want suggestions). Smart triggering improves acceptance rate from 15% to 30%+.",
  },
  {
    id: 'bcap-ipd-mc-3',
    question:
      'What is the optimal debounce time for triggering code completions after the user stops typing?',
    options: ['Immediate (0ms)', '300-500ms', '2000ms', '5000ms'],
    correctAnswer: 1,
    explanation:
      '300-500ms is optimal: Wait for user to pause (not interrupting typing), but fast enough that completion arrives while context is still fresh. <300ms: Too aggressive, triggers during typing. >500ms: Feels slow, user has moved on. Combined with smart triggers (newline, opening brace), 300ms debounce achieves best balance of responsiveness and non-intrusiveness.',
  },
  {
    id: 'bcap-ipd-mc-4',
    question:
      'What is the recommended approach for handling network failures in an IDE plugin?',
    options: [
      'Show error immediately and give up',
      'Retry 3x with exponential backoff, then fallback to local model or show error',
      'Keep retrying forever',
      'Never show errors to users',
    ],
    correctAnswer: 1,
    explanation:
      'Resilient error handling: (1) Retry 3x with exponential backoff (2s, 4s, 8s), (2) Fallback to local model if available (degraded mode), (3) Show clear error with action ("Check connection" or "Try again"). This handles transient network issues (80% of failures) automatically while gracefully degrading for persistent failures.',
  },
  {
    id: 'bcap-ipd-mc-5',
    question: 'How should you measure the success of inline code completions?',
    options: [
      'Total number of completions shown',
      'Acceptance rate (% accepted), time-to-accept, and characters saved',
      'Number of API calls made',
      'Lines of code in codebase',
    ],
    correctAnswer: 1,
    explanation:
      'Key metrics: (1) Acceptance rate: % of shown completions that user accepts (target: >25%), (2) Time-to-accept: <2s indicates relevant completion, (3) Characters saved: Completion length / characters user would type. Quality over quantity - 100 completions with 5% acceptance is worse than 50 with 30% acceptance. Focus on relevance, not volume.',
  },
];
