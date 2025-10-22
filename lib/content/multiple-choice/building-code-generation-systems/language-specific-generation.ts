/**
 * Multiple choice questions for Language-Specific Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const languagespecificgenerationMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'bcgs-language-mc-1',
        question:
            'What is the most important consideration when generating language-specific code?',
        options: [
            'Use same patterns for all languages',
            'Follow language idioms and conventions',
            'Always use verbose syntax',
            'Ignore language-specific features',
        ],
        correctAnswer: 1,
        explanation:
            'Follow language idioms and conventions. Python uses list comprehensions, Go uses goroutines, Rust uses ownership. Idiomatic code is more readable, maintainable, and aligns with community expectations.',
    },
    {
        id: 'bcgs-language-mc-2',
        question:
            'How should error handling be generated for Python vs Go?',
        options: [
            'Same approach for both',
            'Python: try/except with specific exceptions; Go: explicit error returns',
            'Always use error codes',
            'Never handle errors',
        ],
        correctAnswer: 1,
        explanation:
            'Python uses exceptions: try/except with specific exception types (ValueError, KeyError). Go uses explicit error returns: "if err != nil { return nil, err }". Respect language conventions.',
    },
    {
        id: 'bcgs-language-mc-3',
        question:
            'What should guide typing approach in generated code?',
        options: [
            'Never use types',
            'Always use dynamic types',
            'Match language\'s type system: Python type hints, TypeScript/Go strong typing, dynamic for Lua',
            'Always use "any" type',
        ],
        correctAnswer: 2,
        explanation:
            'Match language\'s type system. Python: use type hints (helps IDEs, type checkers). TypeScript/Go: use strong typing (compile-time safety). Lua/JavaScript: dynamic typing acceptable. Respect language philosophy.',
    },
    {
        id: 'bcgs-language-mc-4',
        question:
            'How should concurrency be generated for different languages?',
        options: [
            'Always use threads',
            'Language-specific: Python asyncio, Go goroutines, JavaScript Promises, Rust async/await',
            'Never use concurrency',
            'Always use callbacks',
        ],
        correctAnswer: 1,
        explanation:
            'Use language-specific concurrency: Python asyncio (async/await), Go goroutines + channels, JavaScript Promises/async, Rust async/await + futures. Each has different model and best practices.',
    },
    {
        id: 'bcgs-language-mc-5',
        question:
            'What should language-specific prompts include?',
        options: [
            'Generic instructions only',
            'Language name, version, idioms, common libraries, and style guide references',
            'Just the language name',
            'No specific instructions',
        ],
        correctAnswer: 1,
        explanation:
            'Include: language name + version (Python 3.10+), idioms (use context managers), common libraries (use numpy for arrays), style guides (PEP 8, Google Style). More context = more idiomatic code.',
    },
];

