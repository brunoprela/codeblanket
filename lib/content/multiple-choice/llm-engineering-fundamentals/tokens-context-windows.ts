/**
 * Multiple choice questions for Tokens, Context Windows & Limitations section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const tokenscontextwindowsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'Approximately how many tokens is 1,000 words?',
        options: [
            '500 tokens',
            '1,000 tokens',
            '1,333 tokens',
            '2,000 tokens'
        ],
        correctAnswer: 2,
        explanation:
            'Roughly 1 token ≈ 0.75 words or 4 characters. So 1,000 words ≈ 1,333 tokens. This varies by language and text type, but 1.33 tokens per word is a good rule of thumb for English.'
    },
    {
        id: 'mc2',
        question: 'Why is tiktoken library important for production LLM applications?',
        options: [
            'It makes API calls faster',
            'It provides accurate token counting for cost calculation',
            'It improves model quality',
            'It caches responses automatically'
        ],
        correctAnswer: 1,
        explanation:
            'tiktoken provides accurate token counting using the same tokenizer as OpenAI models. This is essential for calculating costs precisely, checking context limits before API calls, and budgeting tokens. Estimation (chars/4) is too inaccurate for production.'
    },
    {
        id: 'mc3',
        question: 'What happens when you exceed a model\'s context window limit?',
        options: [
            'The model automatically truncates old messages',
            'The API request fails with an error',
            'Processing becomes slower but still works',
            'Only the response is shortened'
        ],
        correctAnswer: 1,
        explanation:
            'Exceeding the context window causes the API request to fail with an error (typically 400 Bad Request). Models do NOT automatically truncate - you must manage context limits in your application by truncating, summarizing, or using RAG before making the API call.'
    },
    {
        id: 'mc4',
        question: 'Which strategy is best for processing a 100,000 token document?',
        options: [
            'Split into chunks and process each independently',
            'Use RAG to retrieve relevant sections',
            'Summarize hierarchically to fit in context',
            'All of the above depending on the use case'
        ],
        correctAnswer: 3,
        explanation:
            'The best strategy depends on the use case: RAG for question-answering (find relevant sections), chunking for sequential analysis (process parts independently), hierarchical summarization for overview (compress to summary). Production apps often combine multiple strategies.'
    },
    {
        id: 'mc5',
        question: 'Why do output tokens typically cost more than input tokens?',
        options: [
            'Output tokens are higher quality',
            'Generation requires more computation than processing',
            'It\'s arbitrary pricing by providers',
            'Output tokens use more memory'
        ],
        correctAnswer: 1,
        explanation:
            'Output token generation requires iterative computation (generating one token at a time with attention over all previous tokens), while input processing is a single forward pass. This makes generation 2-3x more computationally expensive, reflected in 2-3x higher pricing.'
    }
];

