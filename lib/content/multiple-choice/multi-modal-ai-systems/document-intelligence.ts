export const documentintelligenceMultipleChoice = [
  {
    id: 'mmas-document-intelligence-mc-1',
    question:
      'What is the recommended approach for optimizing costs in document intelligence systems?',
    options: [
      'Always use the highest quality models for best results',
      'Cache results, batch similar requests, and use appropriate model quality settings based on use case',
      'Process everything in real-time to minimize latency',
      'Store raw inputs and reprocess on demand',
    ],
    correctAnswer: 1,
    explanation:
      'Cost optimization requires caching frequently requested results, batching similar operations for efficiency, and intelligently selecting model quality settings (high quality for critical tasks, standard for routine operations). This balanced approach maintains quality while controlling costs.',
  },
  {
    id: 'mmas-document-intelligence-mc-2',
    question:
      'Which strategy best reduces latency in document intelligence applications?',
    options: [
      'Always wait for highest accuracy results',
      'Preprocess inputs, use parallel processing, implement caching, and stream responses when possible',
      'Send all requests to the fastest model regardless of quality',
      'Queue all requests and process in order',
    ],
    correctAnswer: 1,
    explanation:
      'Latency reduction requires multiple strategies: preprocessing inputs to reduce size, parallel processing of independent operations, caching common results, and streaming responses for long-running operations. This comprehensive approach provides the best user experience.',
  },
  {
    id: 'mmas-document-intelligence-mc-3',
    question:
      'What is the best way to ensure quality in production document intelligence systems?',
    options: [
      'Manually review every output before returning to users',
      'Use confidence scoring, validate outputs programmatically, implement human review for low-confidence cases, and monitor quality metrics continuously',
      'Always use the most expensive model available',
      'Accept all outputs without validation',
    ],
    correctAnswer: 1,
    explanation:
      'Quality assurance requires a multi-layered approach: confidence scoring to identify uncertain outputs, programmatic validation against expected formats, human review for low-confidence or critical cases, and continuous monitoring of quality metrics. This ensures high quality while remaining scalable.',
  },
  {
    id: 'mmas-document-intelligence-mc-4',
    question: 'How should you handle errors in document intelligence systems?',
    options: [
      'Return generic error messages to users',
      'Implement exponential backoff for retries, use fallback strategies, log comprehensively, and provide meaningful error messages',
      'Fail immediately without retrying',
      'Ignore errors and return cached responses',
    ],
    correctAnswer: 1,
    explanation:
      'Robust error handling includes exponential backoff for transient failures, fallback strategies (alternative models or degraded functionality), comprehensive logging for debugging, and meaningful error messages for users. This approach maximizes reliability and aids troubleshooting.',
  },
  {
    id: 'mmas-document-intelligence-mc-5',
    question:
      'What monitoring metrics are most critical for document intelligence systems?',
    options: [
      'Only track total number of requests processed',
      'Monitor latency (p50, p95, p99), error rate, cost per operation, quality metrics, and user satisfaction',
      'Only monitor costs to optimize spending',
      'Track only successful requests',
    ],
    correctAnswer: 1,
    explanation:
      'Comprehensive monitoring requires tracking multiple dimensions: latency percentiles to understand performance distribution, error rates to identify reliability issues, cost per operation for optimization, quality metrics to ensure accuracy, and user satisfaction to measure overall success. Together, these provide complete visibility into system health.',
  },
];
