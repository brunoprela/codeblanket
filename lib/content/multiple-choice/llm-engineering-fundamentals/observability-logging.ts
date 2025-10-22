/**
 * Multiple choice questions for LLM Observability & Logging section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const observabilityloggingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What is the difference between monitoring and observability?',
        options: [
            'They are the same thing',
            'Monitoring tracks predefined metrics; observability explores arbitrary questions',
            'Monitoring is for production; observability is for development',
            'Observability is more expensive'
        ],
        correctAnswer: 1,
        explanation:
            'Monitoring tracks predefined metrics (error rate, latency, cost) to answer "is there a problem?". Observability uses rich logs to explore arbitrary questions like "why did costs spike Tuesday at 2pm?" Production needs both - monitoring for alerts, observability for debugging.'
    },
    {
        id: 'mc2',
        question: 'What should ALWAYS be logged for every LLM API request?',
        options: [
            'Only errors',
            'Request ID, model, tokens, cost, latency, status',
            'Full prompts and responses',
            'Nothing, to save storage'
        ],
        correctAnswer: 1,
        explanation:
            'Always log: request_id (for tracing), model, token counts, cost, latency, and status. This enables cost tracking, performance monitoring, and debugging. Full prompts/responses can be sampled (1-10%) to reduce storage while maintaining visibility.'
    },
    {
        id: 'mc3',
        question: 'How should you reduce logging storage costs for 100GB/day of logs?',
        options: [
            'Stop logging entirely',
            'Sample detailed logs (1-10%), use tiered storage (hot/warm/cold)',
            'Only log errors',
            'Reduce log retention to 1 day'
        ],
        correctAnswer: 1,
        explanation:
            'Sample detailed logs (full prompts/responses) at 1-10% while logging all requests at basic level (metadata, tokens, cost). Use tiered storage: Elasticsearch for 7 days (fast queries), S3 for 8-90 days (cheap storage), Glacier for 90+ days (archive). This cuts costs 80-90% while maintaining visibility.'
    },
    {
        id: 'mc4',
        question: 'What log format is best for production LLM applications?',
        options: [
            'Plain text',
            'Structured JSON',
            'CSV',
            'XML'
        ],
        correctAnswer: 1,
        explanation:
            'Structured JSON logging enables easy querying, parsing, and analysis. You can filter by user_id, model, cost range, etc. Plain text logs are hard to parse and query. JSON works seamlessly with log aggregation tools (Elasticsearch, BigQuery).'
    },
    {
        id: 'mc5',
        question: 'Which metric indicates your LLM API is unhealthy?',
        options: [
            'Error rate >5%',
            'p95 latency >5 seconds',
            'Cost spike >2x baseline',
            'All of the above'
        ],
        correctAnswer: 3,
        explanation:
            'All indicate potential issues: Error rate >5% suggests systemic failures (rate limits, bugs), p95 latency >5s indicates poor user experience, cost spike >2x suggests abuse or bugs. Set alerts on all three metrics for production health monitoring.'
    }
];

