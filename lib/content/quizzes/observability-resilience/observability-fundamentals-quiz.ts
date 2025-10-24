/**
 * Quiz questions for Observability Fundamentals section
 */

export const observabilityFundamentalsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between observability and monitoring. Why is observability increasingly important in microservices architectures?',
    sampleAnswer:
      'Monitoring answers questions you know to ask—it involves predefined dashboards, known metrics, and specific alerts (e.g., "Is CPU > 80%?"). You set up monitors for anticipated problems. Observability, however, answers questions you didn\'t know you\'d need to ask. It enables exploratory investigation and ad-hoc queries to understand system behavior (e.g., "Why is user X experiencing 5-second latency only on Tuesdays?"). The key difference: monitoring tells you THAT something is wrong, while observability helps you understand WHY. In microservices architectures, observability becomes critical because: (1) Complexity is exponentially higher with 100+ services instead of one monolith. (2) Failure modes are emergent and unpredictable—you can\'t anticipate every possible combination of service interactions. (3) Issues often span multiple services, requiring correlation across distributed components. (4) Traditional debugging methods (SSH into server, grep logs) don\'t scale. (5) The "unknown unknowns" dominate—most production issues are scenarios you never tested. Modern observability tools with rich telemetry data, correlation IDs, and powerful querying enable engineers to investigate novel issues in production that couldn\'t have been anticipated during development.',
    keyPoints: [
      'Monitoring = known questions, observability = unknown questions',
      'Monitoring shows THAT something is wrong, observability shows WHY',
      'Microservices create emergent complexity with unpredictable failure modes',
      'Issues span multiple services, requiring distributed correlation',
      "Traditional debugging (SSH, grep) doesn't scale to 100+ services",
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the three pillars of observability (logs, metrics, traces). For each pillar, explain its characteristics, use cases, and trade-offs. How do they complement each other?',
    sampleAnswer:
      'The three pillars work together to provide comprehensive system visibility: **LOGS**: Discrete, timestamped events that happened at specific points in time. Characteristics: High cardinality (unique events), immutable records, often unstructured. Use cases: Debugging specific errors, audit trails, understanding request flow. Trade-offs: High volume (expensive to store), difficult to query at scale, finding signal in noise is challenging. Example: "User 123 login failed: invalid password" tells you exactly what happened. **METRICS**: Numerical measurements aggregated over time. Characteristics: Time-series data, low cardinality (bounded dimensions), efficient storage, pre-aggregated. Use cases: Real-time dashboards, alerting, trend analysis, system health. Trade-offs: Lose detail through aggregation, high-cardinality dimensions cause storage explosion. Example: "API error rate: 5%" tells you system health but not which specific requests failed. **TRACES**: The journey of a request through distributed system, showing service dependencies and timing. Characteristics: Spans in parent-child relationships, includes timing and metadata. Use cases: Understanding request flow, finding bottlenecks, debugging distributed systems. Trade-offs: High overhead if not sampled, complex to set up properly. Example: Shows that a 2-second API call spent 1.8 seconds in database query. They complement each other: Metrics alert you to a problem (error rate spike), traces show you where in the request flow it\'s happening (which service), and logs tell you exactly what went wrong (specific error message). Together they answer: WHAT is happening (metrics), WHERE it\'s happening (traces), and WHY it\'s happening (logs).',
    keyPoints: [
      'Logs = discrete events, metrics = aggregated numbers, traces = request journeys',
      'Logs have high cardinality and detail, metrics are efficient but lose detail',
      'Traces show distributed request flow and timing',
      'Use together: metrics alert → traces narrow down → logs explain',
      'Trade-offs: logs (expensive volume), metrics (lose detail), traces (require sampling)',
    ],
  },
  {
    id: 'q3',
    question:
      'What is OpenTelemetry and why has it become the industry standard? What problems does it solve, and what are the benefits of adopting it over proprietary instrumentation?',
    sampleAnswer:
      'OpenTelemetry (OTel) is a vendor-neutral, open-source standard for collecting observability telemetry data (logs, metrics, and traces). It solves the problem of vendor lock-in and instrumentation fragmentation. **Problems it solves**: (1) Vendor Lock-in: Before OTel, switching from Datadog to New Relic meant rewriting all instrumentation code. (2) Inconsistent Instrumentation: Different libraries and vendors used different formats and conventions, making correlation difficult. (3) Duplication: Teams needed separate instrumentation for different backends (Jaeger for traces, Prometheus for metrics). (4) Maintenance Burden: Proprietary SDKs changed frequently, breaking instrumentation. **Benefits of OpenTelemetry**: (1) Vendor Neutrality: Instrument once, export to any backend (Jaeger, Datadog, Prometheus, New Relic). Change vendors without code changes. (2) Standardization: Semantic conventions ensure consistency (http.status_code, db.system), enabling cross-service correlation and better tooling. (3) Auto-Instrumentation: Pre-built instrumentation for popular frameworks (Express, Django, Spring Boot) works out of the box. (4) Single SDK: One SDK for all three pillars (logs, metrics, traces) instead of three separate libraries. (5) Industry Support: Backed by CNCF, supported by all major vendors, large ecosystem. (6) Future-Proof: As standard evolves, all tools update together. **Adoption Strategy**: Start with auto-instrumentation for quick wins, then add custom instrumentation for business-specific operations. Use OTel Collector for flexible data routing and processing. The investment in OTel pays off through flexibility, standardization, and avoiding vendor lock-in.',
    keyPoints: [
      'Vendor-neutral standard for telemetry collection (logs, metrics, traces)',
      'Solves vendor lock-in: instrument once, export to any backend',
      'Provides auto-instrumentation for common frameworks',
      'Semantic conventions ensure consistency and correlation',
      'Backed by CNCF, supported by all major vendors',
    ],
  },
];
