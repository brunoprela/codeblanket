/**
 * Multiple choice questions for Prompt Engineering Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const promptengineeringfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-fundamentals-mc-1',
        question:
            'What is the primary benefit of using few-shot prompting over zero-shot prompting?',
        options: [
            'Lower cost per API call',
            'Faster response times',
            'More consistent and reliable outputs',
            'Works with smaller models',
        ],
        correctAnswer: 2,
        explanation:
            'Few-shot prompting provides examples that demonstrate desired behavior, leading to significantly more consistent and reliable outputs, especially for complex or nuanced tasks. While it costs more tokens, the improved reliability is worth it in production.',
    },
    {
        id: 'peo-fundamentals-mc-2',
        question:
            'Which component is most critical for making LLM outputs parseable in production code?',
        options: [
            'Role definition',
            'Output format specification',
            'Task description',
            'Example selection',
        ],
        correctAnswer: 1,
        explanation:
            'Output format specification is most critical for parseability. It defines the exact structure (JSON, CSV, etc.) that code will parse. Without clear format specification, outputs vary unpredictably, breaking parsers and causing production errors.',
    },
    {
        id: 'peo-fundamentals-mc-3',
        question:
            'What is the main purpose of version controlling prompts in production?',
        options: [
            'To save disk space',
            'To track performance changes and enable rollback',
            'To satisfy compliance requirements',
            'To share prompts with other teams',
        ],
        correctAnswer: 1,
        explanation:
            'Version control tracks which prompt versions correspond to which performance metrics, enables quick rollback if a new version underperforms, and maintains history for debugging. This is critical for maintaining reliable production AI systems.',
    },
    {
        id: 'peo-fundamentals-mc-4',
        question:
            'When testing prompts, what is the minimum recommended number of test cases for production reliability?',
        options: [
            '5-10 test cases',
            '20-30 test cases',
            '50-100 test cases',
            '500+ test cases',
        ],
        correctAnswer: 2,
        explanation:
            '50-100 diverse test cases are recommended for production. This provides good coverage of common cases, edge cases, and failure modes while being manageable. Too few cases miss important scenarios; too many becomes expensive to evaluate.',
    },
    {
        id: 'peo-fundamentals-mc-5',
        question:
            'What is the most effective way to handle prompt failures in production?',
        options: [
            'Return a generic error message',
            'Log the failure and retry with the same prompt',
            'Analyze failure patterns and improve the prompt',
            'Switch to a different model',
        ],
        correctAnswer: 2,
        explanation:
            'Analyzing failure patterns and improving the prompt addresses root causes and prevents future failures. Simply retrying or switching models doesn\'t fix underlying prompt issues. Systematic improvement based on failures is key to production reliability.',
    },
];

