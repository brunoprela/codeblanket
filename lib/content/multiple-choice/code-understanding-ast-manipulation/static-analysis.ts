/**
 * Multiple choice questions for Static Analysis & Code Quality section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const staticanalysisMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-staticanalysis-mc-1',
        question:
            'What does "static" analysis mean?',
        options: [
            'Analysis of static variables only',
            'Code analysis without executing the code',
            'Analysis of compiled binaries',
            'Analysis during runtime',
        ],
        correctAnswer: 1,
        explanation:
            'Static analysis examines code without running it, finding bugs, style issues, and potential problems by analyzing structure and logic. Opposite of dynamic (runtime) analysis.',
    },
    {
        id: 'cuam-staticanalysis-mc-2',
        question:
            'How do linters like pylint detect unused variables?\n\nx = 5\ny = 10\nprint(x)',
        options: [
            'By executing the code',
            'By building symbol table and tracking variable reads',
            'By checking variable names',
            'By analyzing comments',
        ],
        correctAnswer: 1,
        explanation:
            'Linters track symbol definitions and uses. Here, y is defined (Name in Store context) but never read (no Name in Load context), so it\'s flagged as unused.',
    },
    {
        id: 'cuam-staticanalysis-mc-3',
        question:
            'What is data flow analysis?',
        options: [
            'Analyzing network data',
            'Tracking how values flow through variables across statements',
            'Measuring code performance',
            'Analyzing database queries',
        ],
        correctAnswer: 1,
        explanation:
            'Data flow analysis tracks value assignment and usage across code paths. Example: detecting "use of uninitialized variable" by verifying all paths define variable before use.',
    },
    {
        id: 'cuam-staticanalysis-mc-4',
        question:
            'What type of bug can static analysis NOT catch?',
        options: [
            'Syntax errors',
            'Logic errors that require runtime values',
            'Unused variables',
            'Type mismatches',
        ],
        correctAnswer: 1,
        explanation:
            'Static analysis cannot predict runtime values or test actual behavior. Example: can\'t catch "if x > 0 but should be x >= 0" without knowing intended logic or runtime tests.',
    },
    {
        id: 'cuam-staticanalysis-mc-5',
        question:
            'What is control flow analysis used for?',
        options: [
            'Code formatting',
            'Tracking possible execution paths and reachability',
            'Measuring code speed',
            'Generating documentation',
        ],
        correctAnswer: 1,
        explanation:
            'Control flow analysis builds a graph of possible execution paths. Detects unreachable code, missing return statements, and validates that all paths handle errors properly.',
    },
];

