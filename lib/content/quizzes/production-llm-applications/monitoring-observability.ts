export const monitoringObservabilityQuiz = [
  {
    id: 'pllm-q-8-1',
    question:
      'Design a comprehensive monitoring and observability system for a production LLM application serving 100K requests/day. What metrics would you track, how would you visualize them, and what alerts would you set up?',
    sampleAnswer:
      'Metrics categories: 1) Performance: p50/p95/p99 latency, requests per second, time to first token, error rate by type. 2) Cost: total spend per hour/day, cost per request, tokens used by model, cache savings. 3) Quality: user satisfaction scores, response relevance, retry rates. 4) Reliability: success rate, timeout rate, rate limit hits, queue depth. 5) Business: active users, usage by tier, revenue per user. Tools: Prometheus for metrics collection, Grafana for visualization, ELK stack for logs, Jaeger for distributed tracing. Dashboards: Real-time overview (requests/sec, error rate, current spend), Model performance (latency by model, cost comparison), User metrics (top spenders, usage by tier), Cost analysis (daily spend trends, projected monthly). Alerts: Error rate >5% for 5min (PagerDuty), p99 latency >30s (Slack), hourly cost >$20 (Email), queue depth >500 (Slack), any model 100% error rate (PagerDuty immediately). Implementation: Prometheus exporters in app, custom metrics with @monitor_llm_call decorator, structured logging with request IDs, trace context propagation. On-call rotation with runbooks for common issues.',
    keyPoints: [
      'Comprehensive metrics across performance, cost, quality, and reliability',
      'Multi-tool stack: Prometheus, Grafana, ELK, Jaeger',
      'Tiered alerting with appropriate urgency levels',
    ],
  },
  {
    id: 'pllm-q-8-2',
    question:
      'Explain how distributed tracing works for LLM applications. Provide a specific example of tracing a request through API gateway, cache, LLM service, and database. How do you debug performance issues using traces?',
    sampleAnswer:
      'Distributed tracing tracks request flow across services using trace IDs and span IDs. Flow: 1) API gateway receives request, creates trace_id=abc123, span_id=1, 2) Checks cache service (span_id=2, parent=1), cache miss takes 5ms, 3) Calls LLM service (span_id=3, parent=1), LLM API call (span_id=4, parent=3) takes 2500ms, 4) Stores in database (span_id=5, parent=3) takes 50ms, 5) Returns response, total 2560ms. Implementation: Use OpenTelemetry, create tracer = trace.get_tracer(__name__), wrap operations with @tracer.start_as_current_span("operation"), add attributes (model, tokens, cost), propagate context with headers (trace_id in X-Trace-Id). Each service extracts trace_id, creates child spans, exports to Jaeger. Debugging: View trace in Jaeger UI, see waterfall of all operations, identify bottleneck (LLM API call 97% of time), check span attributes for details (model=gpt-4, tokens=3500), compare similar requests to find anomalies (some take 10s while others 2s). Use traces to: Find slow dependencies, identify unnecessary sequential operations (could be parallel), detect memory leaks (span duration increasing over time), track error propagation (which service failed first).',
    keyPoints: [
      'Trace ID propagation through all services with parent-child span relationships',
      'OpenTelemetry implementation with rich span attributes',
      'Waterfall visualization to identify bottlenecks and optimization opportunities',
    ],
  },
  {
    id: 'pllm-q-8-3',
    question:
      'How would you implement structured logging for an LLM application that enables efficient debugging and audit trails? Include specific log formats, retention policies, and how to correlate logs across services.',
    sampleAnswer:
      'Structured logging format (JSON): {event: "llm_request", request_id: "uuid", trace_id: "abc", user_id: "user123", model: "gpt-4", prompt_length: 500, timestamp: "2024-01-01T10:00:00Z", ip: "1.2.3.4"}. Log levels: DEBUG (development only), INFO (business events), WARNING (degraded performance), ERROR (failures), CRITICAL (service down). Key events: llm_request (incoming), llm_response (completion with duration, tokens, cost), llm_error (failures with stack trace), cache_hit/miss, auth_attempt, rate_limit_exceeded. Implementation: Custom logger class, log.info(json.dumps({...})), include context automatically (request_id, user_id, trace_id from thread local or async context). Correlation: Use request_id for single request tracking, trace_id for distributed requests, session_id for user journey. Store in Elasticsearch: Index per day (logs-2024-01-01), retain hot data 7 days, warm data 30 days, cold data 90 days, delete after 90 days (or archive to S3). Searching: By request_id for debugging single request, by user_id for user behavior analysis, by error_type for pattern identification, by model for performance comparison. Kibana dashboards: Error frequency over time, most common errors, slowest requests, user activity timeline. Security: Redact sensitive data (PII, API keys), encrypt logs at rest, restrict access by role, audit log access.',
    keyPoints: [
      'JSON structured logs with comprehensive context and correlation IDs',
      'Tiered retention: hot/warm/cold storage for cost optimization',
      'Searchable in Elasticsearch with PII redaction and access control',
    ],
  },
];
