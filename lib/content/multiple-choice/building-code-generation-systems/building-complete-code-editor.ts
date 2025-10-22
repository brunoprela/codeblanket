/**
 * Multiple choice questions for Building a Complete Code Editor section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const buildingcompletecodeeditorMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'bcgs-editor-mc-1',
        question:
            'What architecture should an LLM-powered code editor use?',
        options: [
            'Monolithic single-threaded application',
            'Event-driven with async LLM calls and responsive UI',
            'Blocking synchronous calls',
            'No specific architecture needed',
        ],
        correctAnswer: 1,
        explanation:
            'Use event-driven architecture with async LLM calls (don\'t block UI), responsive UI (stream results), and proper state management. User can continue editing while LLM processes. Better UX.',
    },
    {
        id: 'bcgs-editor-mc-2',
        question:
            'How should code context be managed for large projects?',
        options: [
            'Send entire codebase every time',
            'Use semantic search and dependency analysis to find relevant context',
            'Only send current file',
            'Random selection',
        ],
        correctAnswer: 1,
        explanation:
            'Use semantic search (find related code by meaning) and dependency analysis (find imports, call graph) to identify relevant context. Balances completeness with token limits. Smart context > full codebase.',
    },
    {
        id: 'bcgs-editor-mc-3',
        question:
            'What features distinguish an excellent LLM code editor?',
        options: [
            'Just basic code generation',
            'Generation, refactoring, test creation, review, execution, and multi-file edits',
            'Only autocomplete',
            'Just syntax highlighting',
        ],
        correctAnswer: 1,
        explanation:
            'Excellent editors integrate: code generation, refactoring, test creation, code review, execution validation, multi-file edits, and documentation generation. Comprehensive toolset, not just autocomplete.',
    },
    {
        id: 'bcgs-editor-mc-4',
        question:
            'How should the editor handle conflicting changes?',
        options: [
            'Always use LLM changes',
            'Detect conflicts, show diff, let user merge/choose',
            'Always keep user changes',
            'Randomly pick one',
        ],
        correctAnswer: 1,
        explanation:
            'Detect conflicts (user edited while LLM generated), show clear diff (3-way merge view), let user choose or merge manually. Respect user control. Never overwrite user changes silently.',
    },
    {
        id: 'bcgs-editor-mc-5',
        question:
            'What performance optimization is most critical for editor responsiveness?',
        options: [
            'Faster server hardware',
            'Debouncing user input and caching LLM responses',
            'Reduce code quality',
            'Remove features',
        ],
        correctAnswer: 1,
        explanation:
            'Debounce input (wait 300ms after typing stops before calling LLM), cache responses (same request = same result), and cancel outdated requests. Reduces unnecessary API calls, improves perceived performance.',
    },
];

