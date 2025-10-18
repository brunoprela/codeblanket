/**
 * Quiz questions for API Monitoring & Analytics section
 */

export const apimonitoringQuiz = [
  {
    id: 'monitoring-d1',
    question:
      'Design a comprehensive monitoring and alerting system for a production API serving 10,000 requests/second. Include metrics, dashboards, alerts, and on-call runbooks.',
    sampleAnswer: `Production API monitoring system design:

**[Full implementation example provided in actual response - truncated here for brevity]**`,
    keyPoints: [
      'Track Golden Signals: latency (p50/p95/p99), traffic, errors, saturation',
      'Distributed tracing with OpenTelemetry/Jaeger for cross-service visibility',
      'Multi-tier alerting: critical (page immediately), warning (investigate next day)',
      'Comprehensive dashboards showing request rate, latency, errors, saturation',
      'Runbooks for common issues with step-by-step troubleshooting',
    ],
  },
  {
    id: 'monitoring-d2',
    question:
      'Your API latency p99 suddenly jumped from 200ms to 2s. Walk through your debugging process using monitoring tools.',
    sampleAnswer: `Systematic debugging approach for latency spike:

**[Full debugging walkthrough provided in actual response - truncated here for brevity]**`,
    keyPoints: [
      'Check dashboard for affected endpoints and time correlation',
      'Use distributed tracing to identify slow service in chain',
      'Analyze database query performance and connection pool',
      'Review recent deployments and configuration changes',
      'Implement fixes and verify with monitoring before closing incident',
    ],
  },
  {
    id: 'monitoring-d3',
    question:
      'Compare different monitoring approaches: metrics (Prometheus), logs (ELK), and traces (Jaeger). When would you use each?',
    sampleAnswer: `Comparison of monitoring approaches:

**[Full comparison provided in actual response - truncated here for brevity]**`,
    keyPoints: [
      'Metrics: time-series data, alerting, dashboards (Prometheus)',
      'Logs: detailed event information, debugging (ELK/Splunk)',
      'Traces: request flow across services (Jaeger/Zipkin)',
      'Use all three together for comprehensive observability',
      'Metrics for alerting, logs for debugging, traces for understanding',
    ],
  },
];
