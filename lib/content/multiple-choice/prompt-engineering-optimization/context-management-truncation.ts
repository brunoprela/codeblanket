/**
 * Multiple choice questions for Context Management & Truncation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const contextmanagementtruncationMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-context-mc-1',
        question:
            'What is the typical context window size for GPT-4?',
        options: [
            '4,000 tokens',
            '8,000 tokens',
            '32,000 tokens',
            '128,000 tokens',
        ],
        correctAnswer: 1,
        explanation:
            'GPT-4 has an 8,000 token context window (standard version). GPT-4 Turbo extends this to 128,000 tokens. Context window determines how much text (prompt + response) fits in one request.',
    },
    {
        id: 'peo-context-mc-2',
        question:
            'For code files, which truncation strategy works best?',
        options: [
            'Keep only the beginning',
            'Keep only the end',
            'Keep beginning and end (smart truncation)',
            'Keep random sections',
        ],
        correctAnswer: 2,
        explanation:
            'Smart truncation keeping beginning (imports, definitions) and end (main logic, conclusions) works best for code. This preserves structure while fitting in context. Cursor uses this approach for large files.',
    },
    {
        id: 'peo-context-mc-3',
        question:
            'What is hierarchical context management?',
        options: [
            'Using multiple models at once',
            'Prioritizing context items by importance and allocating token budgets',
            'Organizing files in folders',
            'Using multiple context windows',
        ],
        correctAnswer: 1,
        explanation:
            'Hierarchical context management assigns priorities to different context types (system > current > recent > relevant) and allocates token budget proportionally. This ensures most important information is always included.',
    },
    {
        id: 'peo-context-mc-4',
        question:
            'What is sliding window processing?',
        options: [
            'Moving your window while coding',
            'Processing long documents in overlapping chunks',
            'Animating responses',
            'Using multiple API keys',
        ],
        correctAnswer: 1,
        explanation:
            'Sliding window processing splits long documents into overlapping chunks, processes each chunk, and aggregates results. The overlap (typically 25%) prevents information loss at chunk boundaries.',
    },
    {
        id: 'peo-context-mc-5',
        question:
            'How much token budget should you reserve for the LLM response?',
        options: [
            '10-50 tokens',
            '100-200 tokens',
            '500-1000 tokens',
            '2000+ tokens',
        ],
        correctAnswer: 2,
        explanation:
            'Reserve 500-1000 tokens for the response to ensure the model has room to generate complete answers. Using the entire context window for input leaves no space for output, causing truncated responses.',
    },
];

