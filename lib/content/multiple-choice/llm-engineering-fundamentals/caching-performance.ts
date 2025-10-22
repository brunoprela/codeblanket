/**
 * Multiple choice questions for Caching & Performance section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const cachingperformanceMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What is a typical cache hit rate that justifies implementing caching?',
        options: [
            '5%',
            '20-30%',
            '80%',
            '95%'
        ],
        correctAnswer: 1,
        explanation:
            'A 20-30% cache hit rate typically justifies caching implementation. At high volumes, saving 20-30% of API costs exceeds caching infrastructure costs (Redis ~$50-100/month). Higher rates are better but even 20% provides significant ROI.'
    },
    {
        id: 'mc2',
        question: 'What is semantic caching?',
        options: [
            'Caching based on word meaning',
            'Caching responses for similar (not identical) queries using embeddings',
            'Caching only important responses',
            'A type of database'
        ],
        correctAnswer: 1,
        explanation:
            'Semantic caching uses embeddings to find similar queries and return cached responses even when queries are not identical. "What is Python?" and "Tell me about Python" are semantically similar (0.95+ similarity) and can share a cached response, increasing hit rates significantly.'
    },
    {
        id: 'mc3',
        question: 'Where should you implement caching for maximum benefit?',
        options: [
            'Only client-side',
            'Only server-side',
            'Multiple layers: client, application, and provider-level',
            'Caching does not help LLM applications'
        ],
        correctAnswer: 2,
        explanation:
            'Implement multiple caching layers: Client-side for user-specific repeated requests, application-level (Redis) for shared requests across users, provider-level (Claude Prompt Caching) for large repeated context. Each layer catches different patterns, maximizing savings.'
    },
    {
        id: 'mc4',
        question: 'Why might your cache hit rate be only 15%?',
        options: [
            'Cache key includes varying fields (timestamp, request_id)',
            'User input variations not normalized ("Python" vs "python")',
            'Cache size too small (evicting before reuse)',
            'All of the above'
        ],
        correctAnswer: 3,
        explanation:
            'Low hit rates result from: overly specific cache keys (including metadata that should not be there), input variations not normalized, or cache size/TTL too small causing premature eviction. Fix by normalizing inputs, cleaning cache keys, and increasing size/TTL appropriately.'
    },
    {
        id: 'mc5',
        question: 'What is Claude Prompt Caching?',
        options: [
            'A Redis plugin',
            'Provider-level caching of repeated context (90% cost reduction)',
            'A type of semantic caching',
            'Client-side caching'
        ],
        correctAnswer: 1,
        explanation:
            'Claude Prompt Caching is a provider-level feature where Claude caches your repeated prompt context (e.g., large documentation sent with every request). First request pays full price, subsequent requests within 5 minutes pay 10% for cached portions - 90% cost reduction on cached content.'
    }
];

