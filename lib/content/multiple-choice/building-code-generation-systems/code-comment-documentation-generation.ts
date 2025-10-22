/**
 * Multiple choice questions for Code Comment & Documentation Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codecommentdocumentationgenerationMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'bcgs-docs-mc-1',
        question:
            'What makes a good code comment according to best practices?',
        options: [
            'Explains what the code does',
            'Explains WHY the code exists and non-obvious decisions',
            'Repeats the code in English',
            'Describes every line',
        ],
        correctAnswer: 1,
        explanation:
            'Good comments explain WHY (rationale, design decisions, tradeoffs), not WHAT (code already shows what). "This is O(n²) for simplicity; n is always <100" is valuable. "Loop through array" is not.',
    },
    {
        id: 'bcgs-docs-mc-2',
        question:
            'What should docstrings include for a function?',
        options: [
            'Only the function name',
            'Purpose, parameters, return value, exceptions, and examples',
            'Just parameter names',
            'Random usage notes',
        ],
        correctAnswer: 1,
        explanation:
            'Complete docstrings include: purpose/description, parameter descriptions with types, return value description, raised exceptions, and usage examples. This enables users and tools (like IDEs) to understand usage.',
    },
    {
        id: 'bcgs-docs-mc-3',
        question:
            'When should LLM generate comments?',
        options: [
            'For every line of code',
            'For complex algorithms, non-obvious logic, and public APIs',
            'Never, comments become outdated',
            'Only when code review requests it',
        ],
        correctAnswer: 1,
        explanation:
            'Generate comments for: complex algorithms (explain approach), non-obvious logic (explain WHY), public APIs (explain usage). Simple, self-explanatory code doesn\'t need comments.',
    },
    {
        id: 'bcgs-docs-mc-4',
        question:
            'What documentation format should be used for Python docstrings?',
        options: [
            'Free-form text',
            'Standardized formats like Google, NumPy, or Sphinx',
            'Markdown only',
            'No specific format',
        ],
        correctAnswer: 1,
        explanation:
            'Use standardized formats (Google, NumPy, Sphinx) for consistency and tool support. Tools can parse these formats to generate documentation, provide IDE hints, and validate completeness.',
    },
    {
        id: 'bcgs-docs-mc-5',
        question:
            'How should generated documentation handle complex return types?',
        options: [
            'Just say "returns result"',
            'Ignore complex types',
            'Describe structure, fields, and meaning of complex types',
            'Use "see code" references',
        ],
        correctAnswer: 2,
        explanation:
            'Describe complex return types in detail: structure (dict with keys X, Y), field meanings, possible values. "Returns dict with \'status\': str (success/error), \'data\': list of user objects" is helpful.',
    },
];

