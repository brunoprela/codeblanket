/**
 * Multiple choice questions for Code Refactoring with LLMs section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const coderefactoringllmsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'bcgs-refactor-mc-1',
        question:
            'What is the key principle when refactoring for improved code structure?',
        options: [
            'Make all functions shorter',
            'Preserve external behavior while improving internal structure',
            'Use more classes',
            'Add more comments',
        ],
        correctAnswer: 1,
        explanation:
            'Refactoring means improving code structure WITHOUT changing external behavior. Tests should still pass. Users should see no difference. Only internal organization improves.',
    },
    {
        id: 'bcgs-refactor-mc-2',
        question:
            'How should you validate that a refactoring preserved functionality?',
        options: [
            'Read the code manually',
            'Run existing test suite',
            'Ask LLM if it looks right',
            'Deploy to production',
        ],
        correctAnswer: 1,
        explanation:
            'Run the existing test suite. If all tests pass, behavior is preserved. This is why having tests BEFORE refactoring is critical. No tests = no safety net.',
    },
    {
        id: 'bcgs-refactor-mc-3',
        question:
            'What refactoring pattern extracts duplicated code into reusable function?',
        options: [
            'Extract Variable',
            'Extract Method',
            'Inline Function',
            'Rename Variable',
        ],
        correctAnswer: 1,
        explanation:
            'Extract Method (or Extract Function) takes duplicated code and creates a new function. Call the function instead of duplicating code. Reduces duplication, improves maintainability.',
    },
    {
        id: 'bcgs-refactor-mc-4',
        question:
            'When should you refactor code?',
        options: [
            'Only when adding new features',
            'Only when fixing bugs',
            'Continuously as part of normal development (Boy Scout Rule)',
            'Never, it is too risky',
        ],
        correctAnswer: 2,
        explanation:
            'Refactor continuously (Boy Scout Rule: leave code better than you found it). Small refactorings are safe and compound. Waiting creates large, risky refactorings.',
    },
    {
        id: 'bcgs-refactor-mc-5',
        question:
            'What anti-pattern indicates code might need refactoring?',
        options: [
            'Well-named variables',
            'Functions over 200 lines with many responsibilities',
            'Comprehensive test coverage',
            'Clear documentation',
        ],
        correctAnswer: 1,
        explanation:
            'Long functions with many responsibilities (God Functions) are hard to understand, test, and change. Refactor into smaller, focused functions with single responsibilities.',
    },
];

