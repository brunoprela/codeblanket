/**
 * Multiple choice questions for Cost Tracking & Optimization section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const costtrackingoptimizationMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'Which tokens typically cost more per token?',
        options: [
            'Input tokens',
            'Output tokens',
            'They cost the same',
            'It varies randomly'
        ],
        correctAnswer: 1,
        explanation:
            'Output tokens cost 2-3x more than input tokens because generation requires iterative computation (generating one token at a time), while input processing is a single forward pass. For GPT-3.5: input costs $0.50/1M, output costs $1.50/1M (3x more).'
    },
    {
        id: 'mc2',
        question: 'What is the fastest way to reduce LLM costs by 80%?',
        options: [
            'Use shorter prompts',
            'Implement caching',
            'Route simple tasks to cheaper models (GPT-3.5 instead of GPT-4)',
            'Reduce usage'
        ],
        correctAnswer: 2,
        explanation:
            'Routing tasks by complexity to appropriate models provides the biggest immediate cost reduction - GPT-3.5 is 20x cheaper than GPT-4 and handles simple tasks well. If 40-50% of tasks can use GPT-3.5, costs drop dramatically. Caching helps but depends on query patterns.'
    },
    {
        id: 'mc3',
        question: 'Why is tracking cost per user important?',
        options: [
            'To send users their bill',
            'To identify unprofitable users and potential abuse',
            'To improve model quality',
            'It\'s not important'
        ],
        correctAnswer: 1,
        explanation:
            'User-level cost tracking reveals which users are profitable vs. unprofitable and detects abuse (excessive usage). If a user on a $50/month plan costs $500/month in API calls, that\'s unsustainable. Enable rate limiting, tiered pricing, or usage caps based on this data.'
    },
    {
        id: 'mc4',
        question: 'What percentage cache hit rate provides significant cost savings?',
        options: [
            '5-10%',
            '30-40%',
            '70-80%',
            '95%+'
        ],
        correctAnswer: 1,
        explanation:
            'A 30-40% cache hit rate provides significant savings - reducing API calls by 30-40% at minimal caching cost (Redis ~$50-100/month). Higher hit rates are better but harder to achieve. Even 20-30% hit rates justify caching investment for high-volume applications.'
    },
    {
        id: 'mc5',
        question: 'How can you estimate the cost of an API call before making it?',
        options: [
            'Count tokens with tiktoken and multiply by model pricing',
            'It cannot be estimated',
            'Use response.cost field',
            'Divide prompt length by 1000'
        ],
        correctAnswer: 0,
        explanation:
            'Use tiktoken to count input tokens, estimate output tokens based on typical responses, then multiply by the model\'s pricing (per 1M tokens). This gives accurate cost estimates before making API calls, essential for budget management and cost alerts.'
    }
];

