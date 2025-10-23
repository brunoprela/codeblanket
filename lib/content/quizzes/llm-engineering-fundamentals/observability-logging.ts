/**
 * Quiz questions for LLM Observability & Logging section
 */

export const observabilityloggingQuiz = [
  {
    id: 'q1',
    question:
      'Design a comprehensive logging strategy for a production LLM application that processes 10,000 requests per day. What would you log, how would you structure it, and what tools would you use?',
    sampleAnswer:
      'Comprehensive logging strategy: What to log per request: (1) Request metadata: request_id (unique), timestamp, user_id, session_id, endpoint/feature, environment (prod/staging). (2) Input data: model name, full messages array (or hash if PII concerns), parameters (temperature, max_tokens, etc.), estimated input tokens. (3) Output data: response content (or hash), actual token usage (input/output/total), finish_reason, latency breakdown (time to first token, total time). (4) Cost data: calculated cost per request, cumulative user cost, cumulative daily cost. (5) Quality metrics: user feedback if available, parse success/failure, validation errors. (6) Error data: exception type and message, stack trace for errors, retry attempts and outcomes. Structure (JSON for easy querying): {"request_id": "uuid", "timestamp": "ISO8601", "user_id": "user123", "model": "gpt-3.5-turbo", "input_tokens": 150, "output_tokens": 300, "cost": 0.00023, "latency_ms": 1250, "status": "success", "environment": "prod"}. Storage strategy: (1) Hot data (last 7 days): Elasticsearch for fast querying, full-text search on requests/responses, real-time alerting. (2) Warm data (7-90 days): S3/GCS for cost-effective storage, queryable via Athena/BigQuery for analysis. (3) Cold data (90+ days): Compressed in cheap storage, archived for compliance. Tools: (1) Structured logging library (Python logging with JSON formatter), (2) Log aggregation (Fluentd/Filebeat → Elasticsearch), (3) Dashboards (Grafana/Kibana) showing: request volume, error rates, cost trends, latency percentiles (p50, p95, p99), model usage distribution. (4) Alerting (PagerDuty/Opsgenie): Error rate >5%, Cost spike >2x baseline, Latency p95 >5s, Failed requests for specific users. (5) Specialized LLM tools (LangSmith or Helicone): Full request/response traces, automatic prompt versioning, cost analytics by prompt. Implementation: Log at INFO level for successes, ERROR for failures, never log PII in plaintext (hash or mask), use structured logging (JSON) for parsing, include correlation IDs across distributed services, sample detailed logs (e.g., full response) at 1-10% for debugging without storage explosion. Result: Complete visibility into costs, quality, performance, and usage patterns - essential for production LLM ops.',
    keyPoints: [
      'Log request metadata, input/output, costs, and errors',
      'Use structured JSON logging for easy querying',
      'Store with tiered strategy (hot/warm/cold)',
      'Use Elasticsearch for recent data, S3 for archives',
      'Alert on error rates, cost spikes, and latency',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between observability and monitoring for LLM applications. Provide examples of questions each approach helps you answer.',
    sampleAnswer:
      'Monitoring is about known knowns - tracking predefined metrics to answer expected questions. Observability is about unknown unknowns - exploring arbitrary questions about system behavior to debug novel issues. Monitoring examples for LLMs: (1) "What is current error rate?" - dashboard showing 2.3% errors. (2) "What is average cost per request?" - metric showing $0.002. (3) "What is p95 latency?" - 3.2 seconds. (4) "How many requests per minute?" - 150 rpm. (5) "What percentage using GPT-4 vs GPT-3.5?" - 20% vs 80%. These are predefined metrics with dashboards. Monitoring answers "is the system healthy?" Observability examples for LLMs: (1) "Why did costs spike 300% on Tuesday at 2pm?" - query logs to find specific user/prompt pattern causing issue. (2) "Which prompts produce most parse failures?" - aggregate by prompt template to find problematic patterns. (3) "Why is user X seeing slow responses?" - trace their requests across services to find bottleneck. (4) "What changed between yesterday (working) and today (broken)?" - compare logs to find prompt version change. (5) "Which model performs best for summarization tasks?" - query logs filtered by task type, analyze quality metrics. These require exploring raw data. Observability answers "why is this specific thing happening?" Key differences: Monitoring uses aggregates and predefined metrics - fast but limited. Observability uses raw event data with rich context - slower but unlimited questions. Monitoring uses dashboards and alerts. Observability uses query interfaces and traces. For LLM applications specifically: Monitoring tracks: uptime, throughput, error rates, costs, latency. Observability requires: full request/response logs, user-level traces, prompt versions, model choices, parameter values. Production needs BOTH: Monitoring for alerts and health checks (is anything wrong?), Observability for debugging and optimization (why and how to fix?). Example: Monitoring alerts "error rate is 10%" - problem detected. Observability answers "errors happen on requests with >5K tokens using GPT-4 when temperature >0.8" - root cause found, solution clear (reduce temperature or switch model for long prompts). Without observability, you know there\'s a problem but not why. Without monitoring, you might not know there\'s a problem at all.',
    keyPoints: [
      'Monitoring tracks predefined metrics (error rate, cost)',
      'Observability explores arbitrary questions from raw logs',
      'Monitoring answers "is there a problem?"',
      'Observability answers "why and how to fix?"',
      'Production needs both for complete visibility',
    ],
  },
  {
    id: 'q3',
    question:
      "You've implemented logging, but your logs are growing to 100GB/day and becoming expensive to store. Describe strategies to reduce logging costs while maintaining visibility.",
    sampleAnswer:
      'Cost reduction strategies without losing visibility: (1) Sampling: Log every request at basic level (metadata, tokens, cost, latency), sample detailed logs (full prompt/response) at 1-10% of requests, always log errors and slow requests (>95th percentile), sample rate can be dynamic based on health (more sampling when errors high). Impact: 90%+ storage reduction with minimal visibility loss. (2) Tiered storage: Hot (7 days): Elasticsearch - full logs, fast queries, alerts. Warm (8-90 days): S3 + Athena - compressed, queryable, slower. Cold (90+ days): Glacier - compressed, rarely accessed. Impact: 80% cost reduction by using cheaper storage tiers. (3) Log level management: Production: INFO for successes (minimal), ERROR for failures (detailed). Never DEBUG in production continuously. Can enable DEBUG temporarily per user/request for debugging. (4) Content truncation: Hash or truncate long prompts/responses - store first 500 chars + length + hash. Store full content only for: errors, sampled requests, flagged by importance. Can retrieve full content from client cache if needed for debugging. (5) Structured compression: Use columnar formats (Parquet) instead of raw JSON - 5-10x compression. Store logs in batches, not individual files. Remove redundant fields (common values like "status": "success" on most logs). (6) Retention policies: Keep detailed logs 7 days, aggregate metrics 90 days, summary statistics 1 year+. Delete raw logs after retention period. (7) PII handling: Never log full PII (emails, names, addresses). Use hashes or anonymization. Reduce legal liability and storage. Implementation: Configure log sampling per environment/feature, set up automatic archival to cheaper storage after 7 days, implement log aggregation and summarization jobs, monitor storage costs as metric (should be <1% of LLM API costs). Result: 100GB/day → 10GB/day hot + 30GB/day warm → $50/month storage instead of $500/month. Still have full visibility for debugging, just sample and tier intelligently.',
    keyPoints: [
      'Sample detailed logs at 1-10%, always log errors',
      'Use tiered storage (hot/warm/cold)',
      'Truncate or hash long content',
      'Use compressed columnar formats',
      'Implement retention policies and auto-archival',
    ],
  },
];
