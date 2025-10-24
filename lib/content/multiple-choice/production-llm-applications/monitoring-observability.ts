import { MultipleChoiceQuestion } from '../../../types';

export const monitoringObservabilityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-monitor-mc-1',
    question:
      'What are the most important metrics to track for LLM applications?',
    options: [
      'Only response time',
      'Only cost',
      'Latency, cost, error rate, token usage, and cache hit rate',
      'Just error logs',
    ],
    correctAnswer: 2,
    explanation:
      'Track multiple dimensions: latency (p50/p95/p99), cost per request and total, error rate by type, tokens used, cache effectiveness for comprehensive visibility.',
  },
  {
    id: 'pllm-monitor-mc-2',
    question: 'What is distributed tracing used for?',
    options: [
      'Error logging',
      'Tracking requests across multiple services with trace IDs',
      'Cost calculation',
      'Rate limiting',
    ],
    correctAnswer: 1,
    explanation:
      'Distributed tracing uses trace IDs to follow requests across services (API → cache → LLM → database), helping identify bottlenecks and debug issues.',
  },
  {
    id: 'pllm-monitor-mc-3',
    question: 'Why use structured logging (JSON) instead of plain text?',
    options: [
      'Looks better',
      'Easier to search, filter, and analyze programmatically',
      'Saves space',
      'Required by law',
    ],
    correctAnswer: 1,
    explanation:
      'Structured JSON logs are easily searchable by field (user_id, error_type), analyzable in tools like Elasticsearch, and machine-readable for automated alerts.',
  },
  {
    id: 'pllm-monitor-mc-4',
    question: 'What alerts should you set for production LLM applications?',
    options: [
      'None, manual monitoring only',
      'Only when system is completely down',
      'Error rate >5%, latency spikes, cost spikes, queue backup',
      'Alert on every error',
    ],
    correctAnswer: 2,
    explanation:
      'Set threshold-based alerts: error rate >5% for 5min, p99 latency >30s, hourly cost >$X, queue depth >500. Balance signal vs noise.',
  },
  {
    id: 'pllm-monitor-mc-5',
    question: 'How should you track LLM costs in production?',
    options: [
      'Guess at end of month',
      'Track every API call with tokens and cost in database/metrics',
      'Only track total spend',
      'Ignore costs',
    ],
    correctAnswer: 1,
    explanation:
      'Track every API call with model, tokens used, calculated cost. Store in database for analysis by user/model/feature. Real-time dashboards prevent surprises.',
  },
];
