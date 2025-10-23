/**
 * Multiple choice questions for Interactive Code Editing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const interactivecodeeditingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcgs-interactive-mc-1',
    question:
      'What is the key benefit of streaming LLM responses in interactive editors?',
    options: [
      'Saves tokens',
      'Provides real-time feedback and perceived faster performance',
      'Reduces errors',
      'Enables offline use',
    ],
    correctAnswer: 1,
    explanation:
      'Streaming provides real-time feedback as LLM generates. Users see progress immediately (perceived faster). Can interrupt if wrong direction. Better UX than waiting for complete response.',
  },
  {
    id: 'bcgs-interactive-mc-2',
    question: 'How should multiple simultaneous edit requests be handled?',
    options: [
      'Process all in parallel',
      'Cancel previous request when new one starts',
      'Queue all requests',
      'Reject new requests',
    ],
    correctAnswer: 1,
    explanation:
      'Cancel previous request when new one arrives. User intent changed. Processing outdated request wastes resources and could conflict with new request. Use AbortController to cancel.',
  },
  {
    id: 'bcgs-interactive-mc-3',
    question: 'What context should be sent with each edit request?',
    options: [
      'Only the selected text',
      'File content, cursor position, recent changes, and user instruction',
      'Just the filename',
      'Entire codebase',
    ],
    correctAnswer: 1,
    explanation:
      'Send: file content (for context), cursor position (where to apply), recent changes (conversation history), and user instruction. Balance context (better edits) with token cost.',
  },
  {
    id: 'bcgs-interactive-mc-4',
    question: 'How should generated edits be applied to the editor?',
    options: [
      'Replace entire file',
      'Apply as diff with undo support',
      'Ask user to copy-paste',
      'Append to end of file',
    ],
    correctAnswer: 1,
    explanation:
      'Apply as diff (minimal changes) with undo support. Preserves cursor position, maintains editor state, and allows easy revert. Use editor API for proper integration.',
  },
  {
    id: 'bcgs-interactive-mc-5',
    question: 'What should happen when LLM generates invalid code?',
    options: [
      'Apply anyway',
      'Show error, keep previous valid state, allow user to retry with feedback',
      'Crash the editor',
      'Silently ignore',
    ],
    correctAnswer: 1,
    explanation:
      "Validate generated code. If invalid: show error with details, keep previous valid state (don't break working code), allow retry with validation feedback to LLM. Never apply invalid code.",
  },
];
