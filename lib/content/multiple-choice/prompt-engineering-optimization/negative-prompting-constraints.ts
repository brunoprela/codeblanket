/**
 * Multiple choice questions for Negative Prompting & Constraints section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const negativepromptingconstraintsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-negative-mc-1',
        question:
            'Why are "do NOT" instructions effective?',
        options: [
            'They make prompts longer',
            'They set explicit boundaries preventing specific failure modes',
            'They confuse the model',
            'They work better with older models',
        ],
        correctAnswer: 1,
        explanation:
            '"Do NOT" instructions explicitly prevent specific observed failure modes. They\'re more effective than hoping positive instructions prevent issues. They set clear boundaries on what the AI should not do.',
    },
    {
        id: 'peo-negative-mc-2',
        question:
            'What type of constraint is "Do not apologize more than once"?',
        options: [
            'Format constraint',
            'Behavioral constraint',
            'Safety constraint',
            'Scope constraint',
        ],
        correctAnswer: 1,
        explanation:
            'This is a behavioral constraint controlling how the AI communicates and interacts. Behavioral constraints set tone, style, and interaction patterns, distinct from format (structure), safety (protection), or scope (boundaries).',
    },
    {
        id: 'peo-negative-mc-3',
        question:
            'Which is an example of a scope constraint?',
        options: [
            '"Be concise"',
            '"Only handle billing and account questions, not technical support"',
            '"Output as JSON"',
            '"Do not share personal information"',
        ],
        correctAnswer: 1,
        explanation:
            'Scope constraints define what topics/tasks the AI can and cannot handle. They set the boundaries of expertise and capabilities, helping maintain focus and enabling proper escalation of out-of-scope requests.',
    },
    {
        id: 'peo-negative-mc-4',
        question:
            'What is the best way to test if constraints are being followed?',
        options: [
            'Ask the AI if it will follow them',
            'Create adversarial test cases designed to trigger violations',
            'Trust they work without testing',
            'Only test in production',
        ],
        correctAnswer: 1,
        explanation:
            'Adversarial test cases intentionally try to trigger constraint violations. This reveals whether constraints are truly effective. Examples: testing "do not apologize" by providing frustrating scenarios, testing scope limits by asking out-of-scope questions.',
    },
    {
        id: 'peo-negative-mc-5',
        question:
            'What should you do when constraints are frequently violated?',
        options: [
            'Remove the constraints',
            'Strengthen wording, add examples, or restructure the prompt',
            'Accept violations as normal',
            'Switch models',
        ],
        correctAnswer: 1,
        explanation:
            'High violation rates mean constraints need improvement. Strengthen wording, add examples showing correct behavior, restructure for clarity, or add validation. Track violation rates to measure effectiveness of changes.',
    },
];

