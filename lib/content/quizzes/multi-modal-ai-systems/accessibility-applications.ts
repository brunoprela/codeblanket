export const accessibilityapplicationsQuiz = [
  {
    id: 'mmas-accessibility-applications-q-1',
    question:
      'Design a production system for accessibility applications that handles 1000+ requests per hour. What are the key architectural components, optimization strategies, and quality assurance measures you would implement?',
    hint: 'Consider caching, batch processing, error handling, and monitoring.',
    sampleAnswer: `A production accessibility applications system requires: **Architecture:** API gateway, processing queue, worker pool, caching layer, database. **Optimization:** Batch similar requests, cache common results, use appropriate model quality settings, parallel processing. **Quality:** Input validation, output verification, confidence scoring, human review for low-confidence cases. **Monitoring:** Latency tracking, cost per request, error rates, quality metrics. **Scaling:** Horizontal scaling of workers, rate limiting, load balancing, auto-scaling based on queue depth.`,
    keyPoints: [
      'Queue-based architecture with worker pool',
      'Aggressive caching and batch processing',
      'Comprehensive monitoring and alerting',
      'Quality assurance with confidence scoring',
    ],
  },
  {
    id: 'mmas-accessibility-applications-q-2',
    question:
      'Explain the main challenges and solutions for accessibility applications in production. Include cost optimization, latency reduction, and accuracy improvement strategies.',
    hint: 'Think about preprocessing, model selection, caching, and iterative improvement.',
    sampleAnswer: `**Challenges:** High computational cost, variable latency, quality consistency, error handling. **Solutions:** **Cost:** Optimize inputs (resize, compress), cache results, use cheaper models for simple cases, batch processing. **Latency:** Preprocess when possible, parallel execution, streaming responses, edge caching. **Accuracy:** Validate inputs, use appropriate detail levels, ensemble methods for critical tasks, human review loop. **Reliability:** Retry logic, fallback models, graceful degradation, comprehensive error logging.`,
    keyPoints: [
      'Cost optimization through caching and model selection',
      'Latency reduction via preprocessing and parallelization',
      'Accuracy improvements through validation and ensembles',
      'Reliability via retries and fallbacks',
    ],
  },
  {
    id: 'mmas-accessibility-applications-q-3',
    question:
      'Design an evaluation framework for accessibility applications systems. What metrics would you track, how would you collect ground truth data, and how would you continuously improve the system?',
    hint: 'Consider automated metrics, human evaluation, A/B testing, and feedback loops.',
    sampleAnswer: `**Metrics:** Accuracy/precision/recall, latency (p50, p95, p99), cost per operation, error rate, user satisfaction. **Ground Truth:** Human annotation, expert review, user feedback, automated validation where possible. **Evaluation:** Offline evaluation on test set, online A/B testing, shadow mode for new changes. **Improvement:** Analyze failure cases, retrain on corrections, adjust prompts based on results, monitor distribution shift. **Continuous:** Automated daily evaluation, weekly human review, monthly model updates, regular A/B tests of improvements.`,
    keyPoints: [
      'Multi-dimensional metrics: accuracy, latency, cost, satisfaction',
      'Combination of automated and human evaluation',
      'Continuous monitoring and iterative improvement',
      'A/B testing for validating changes',
    ],
  },
];
