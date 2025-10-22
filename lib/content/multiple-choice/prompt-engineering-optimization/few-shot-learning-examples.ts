/**
 * Multiple choice questions for Few-Shot Learning & Examples section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const fewshotlearningexamplesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'peo-fewshot-mc-1',
        question:
            'How many examples typically provide the best cost/performance tradeoff for most tasks?',
        options: [
            '1-2 examples',
            '3-5 examples',
            '10-15 examples',
            '20+ examples',
        ],
        correctAnswer: 1,
        explanation:
            '3-5 examples typically provide the best balance. This is enough to demonstrate patterns and edge cases without excessive token costs. More complex tasks might need up to 10, but returns diminish after that.',
    },
    {
        id: 'peo-fewshot-mc-2',
        question:
            'When ordering examples, which sequence is most effective?',
        options: [
            'Random order',
            'Complex to simple',
            'Simple to complex',
            'Order doesn\'t matter',
        ],
        correctAnswer: 2,
        explanation:
            'Simple to complex ordering helps models learn the pattern progressively, starting with clear cases before showing nuanced examples. This mirrors effective teaching and improves pattern recognition.',
    },
    {
        id: 'peo-fewshot-mc-3',
        question:
            'What is dynamic example selection (RAG for examples)?',
        options: [
            'Randomly selecting examples each time',
            'Retrieving most relevant examples based on query similarity',
            'Using the same examples for all queries',
            'Letting users choose examples',
        ],
        correctAnswer: 1,
        explanation:
            'Dynamic example selection uses embeddings to find and retrieve examples most similar to the current query, providing the most relevant demonstrations. This scales better and improves relevance compared to static examples.',
    },
    {
        id: 'peo-fewshot-mc-4',
        question:
            'What is the most important characteristic of good examples?',
        options: [
            'They are very long',
            'They cover diverse cases and edge cases',
            'They use complex vocabulary',
            'They are all the same length',
        ],
        correctAnswer: 1,
        explanation:
            'Diversity and edge case coverage are most important. Examples should represent the variety of inputs the system will encounter, including common cases, edge cases, and different formats to ensure robust performance.',
    },
    {
        id: 'peo-fewshot-mc-5',
        question:
            'When should you use zero-shot instead of few-shot prompting?',
        options: [
            'Always - it\'s simpler',
            'For simple, well-defined tasks with powerful models where consistency is less critical',
            'Never - few-shot is always better',
            'Only when examples are unavailable',
        ],
        correctAnswer: 1,
        explanation:
            'Zero-shot is appropriate for simple tasks with powerful models (like GPT-4) when the task is well-understood and format flexibility is acceptable. Few-shot is better when consistency, format control, or handling nuanced cases is critical.',
    },
];

