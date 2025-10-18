/**
 * Quiz questions for Microservices Monitoring section
 */

export const microservicesmonitoringQuiz = [
  {
    id: 'q1-monitoring',
    question:
      'Explain the three pillars of observability and how they complement each other in microservices.',
    sampleAnswer:
      "Three pillars: (1) Metrics - numeric measurements over time (request rate, latency, errors). Good for: alerting, dashboards, trends. Tools: Prometheus/Grafana. (2) Logs - text records of events. Good for: debugging specific issues, audit trails. Tools: ELK stack. (3) Distributed Tracing - tracks requests across services showing full call path and timing. Good for: finding bottlenecks, understanding dependencies. Tools: Jaeger. They complement: Metrics alert you THAT there's a problem (P95 latency spiked), Logs tell you WHAT happened (specific errors), Traces show you WHERE the problem is (which service is slow). Example: Alert fires for high latency → check traces to find slow service → check logs for that service to see errors. Correlation IDs tie them together.",
    keyPoints: [
      'Metrics: numeric measurements, alerting, trends (Prometheus)',
      'Logs: text records, debugging, audit (ELK)',
      'Traces: request flow across services, bottlenecks (Jaeger)',
      'Complement: Metrics=THAT, Logs=WHAT, Traces=WHERE',
      'Correlation IDs tie them together across services',
    ],
  },
  {
    id: 'q2-monitoring',
    question:
      'What are RED metrics? Why are they important for microservices monitoring?',
    sampleAnswer:
      "RED metrics: (1) Rate - requests per second, (2) Errors - error rate percentage, (3) Duration - latency percentiles (P50, P95, P99). Important because they measure user-facing impact directly. Rate shows traffic patterns and helps capacity planning. Errors show user-facing failures (alert if > 1%). Duration shows user experience (alert if P95 > 500ms). These are SYMPTOMS that users experience, not causes. Better than monitoring CPU/memory (causes) which don't always correlate with user impact. Example: CPU at 80% but P95 latency is 100ms → no problem. CPU at 50% but P95 latency is 2s → big problem. Monitor every service with RED metrics, aggregate for system view.",
    keyPoints: [
      'Rate: requests/second (traffic patterns, capacity)',
      'Errors: error rate % (user-facing failures)',
      'Duration: latency P50/P95/P99 (user experience)',
      'Measure user-facing symptoms, not causes',
      "Better than CPU/memory (causes that don't always affect users)",
    ],
  },
  {
    id: 'q3-monitoring',
    question:
      'How do correlation IDs help with debugging in microservices? How do you implement them?',
    sampleAnswer:
      'Correlation ID is unique identifier attached to every request as it flows through multiple services. Helps debug by: (1) Trace single request across all services, (2) Correlate logs from different services, (3) Find which service failed. Implementation: (1) API Gateway generates UUID when request enters system, (2) Add to HTTP header X-Correlation-ID, (3) Each service extracts correlation ID, includes in all logs, passes to downstream services in headers, (4) Store in thread-local storage or request context. Querying: Search logs/traces by correlation ID to see full request flow. Example: User reports "order failed" → get correlation ID from their request → query logs for that ID → see Order Service succeeded, Payment Service failed with "insufficient funds".',
    keyPoints: [
      'Correlation ID: unique identifier per request',
      'Traces request flow across all services',
      'Generated at API Gateway, passed via headers',
      'Included in all logs and traces',
      'Query by correlation ID to debug specific request',
    ],
  },
];
