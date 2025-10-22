/**
 * Multiple choice questions for Prompt Optimization Techniques section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const promptoptimizationtechniquesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-optimization-mc-1',
        question:
            'What is the most important first step in prompt optimization?',
        options: [
            'Try many random variations',
            'Ask the LLM to improve itself',
            'Establish baseline measurements with test cases',
            'Use the most expensive model',
        ],
        correctAnswer: 2,
        explanation:
            'Establishing baseline measurements with comprehensive test cases is essential. You need to know current performance before you can measure improvements. Without this, optimization is guesswork.',
    },
    {
        id: 'peo-optimization-mc-2',
        question:
            'What is prompt compression primarily used for?',
        options: [
            'Improving accuracy',
            'Reducing token costs while preserving meaning',
            'Making prompts more readable',
            'Increasing response speed',
        ],
        correctAnswer: 1,
        explanation:
            'Prompt compression reduces token count (and thus costs) while trying to preserve semantic meaning and output quality. It\'s critical for high-volume applications where token costs add up quickly.',
    },
    {
        id: 'peo-optimization-mc-3',
        question:
            'What does LLMLingua do?',
        options: [
            'Translates prompts between languages',
            'Uses a small model to intelligently compress prompts up to 20x',
            'Checks prompts for grammar errors',
            'Generates prompts automatically',
        ],
        correctAnswer: 1,
        explanation:
            'LLMLingua uses a small language model to identify and remove less important tokens, achieving up to 20x compression while preserving semantic meaning with minimal quality loss.',
    },
    {
        id: 'peo-optimization-mc-4',
        question:
            'When should you choose a more expensive model over a cheaper one?',
        options: [
            'Always use the most expensive',
            'When task is high-value and accuracy critical',
            'Never - always use cheapest',
            'Based on response speed only',
        ],
        correctAnswer: 1,
        explanation:
            'Use expensive models (GPT-4) for high-value tasks where accuracy matters and errors are costly. Use cheaper models (GPT-3.5) for high-volume, low-stakes tasks. Balance cost against business value and error cost.',
    },
    {
        id: 'peo-optimization-mc-5',
        question:
            'What statistical measure indicates a prompt improvement is real, not random?',
        options: [
            'Any improvement is good enough',
            'p-value < 0.05',
            'At least 2% improvement',
            'Tested on 5 examples',
        ],
        correctAnswer: 1,
        explanation:
            'p-value < 0.05 indicates statistical significance - the improvement is unlikely to be due to chance. Also need sufficient sample size and proper A/B testing methodology to ensure reliability.',
    },
];

