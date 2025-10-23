import { MultipleChoiceQuestion } from '../../../types';

export const structuredToolResponsesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question: 'What are the key elements of a well-structured tool response?',
    options: [
      'Just the data',
      'Status, data, error message, and metadata',
      'Only success or failure',
      'Raw API response',
    ],
    correctAnswer: 1,
    explanation:
      'Well-structured responses include status (success/error), data payload, error information when applicable, and metadata like timestamps and costs.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'Why should tool responses include natural language summaries?',
    options: [
      'To increase token usage',
      'To help LLMs quickly understand and communicate results',
      'To make responses slower',
      'To comply with regulations',
    ],
    correctAnswer: 1,
    explanation:
      'Natural language summaries help LLMs quickly understand the essence of results and communicate them naturally to users.',
  },
  {
    id: 'fcfc-mc-3',
    question: 'How should you handle partial success in tool responses?',
    options: [
      'Report it as a complete failure',
      'Use a "partial" status with details of what succeeded and failed',
      'Ignore the failures',
      'Retry until complete success',
    ],
    correctAnswer: 1,
    explanation:
      'A "partial" status with details allows the LLM to understand that some operations succeeded while others failed, enabling appropriate follow-up.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'What should error responses include?',
    options: [
      'Only error code',
      'Error message, type, suggestions for recovery, and context',
      'Stack trace',
      'Just "error" string',
    ],
    correctAnswer: 1,
    explanation:
      'Comprehensive error responses include messages, types, recovery suggestions, and context to help both the LLM and debugging.',
  },
  {
    id: 'fcfc-mc-5',
    question: 'Why is response consistency important across all tools?',
    options: [
      'To reduce code size',
      'To help LLMs reliably parse and understand responses',
      'To improve execution speed',
      'To reduce costs',
    ],
    correctAnswer: 1,
    explanation:
      'Consistent response formats across all tools make it easier for LLMs to parse and understand results, reducing confusion and errors.',
  },
];
