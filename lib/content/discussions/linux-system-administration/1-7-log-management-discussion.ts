export const logManagementDiscussion = [
  {
    id: 1,
    question:
      'Application logs are consuming 50GB/day, causing high CloudWatch Logs costs. Design a comprehensive strategy to reduce costs while maintaining operational visibility.',
    answer:
      '**Strategy:** 1) Implement log levels properly (reduce DEBUG in prod). 2) Use log sampling for high-volume endpoints (10% sample rate). 3) Filter noisy logs before shipping (e.g., health checks). 4) Aggregate similar log lines. 5) Reduce retention (30d for INFO, 90d for ERROR). 6) Use S3 export for long-term archival. 7) Local log aggregation before CloudWatch. 8) Structured logging for efficient querying. **Expected savings:** 60-80%.',
  },
  {
    id: 2,
    question:
      'Design a structured logging format for a microservices architecture that enables distributed tracing and correlation across services.',
    answer:
      '**JSON format with:** `timestamp`, `service_name`, `trace_id` (globally unique per request), `span_id` (unique per service call), `parent_span_id`, `log_level`, `message`, `user_id`, `request_method`, `request_path`, `duration_ms`, `error_type`. **Trace_id** propagated via HTTP headers (X-Trace-Id). Enables CloudWatch Insights queries to trace requests across all services. Include correlation IDs for async operations.',
  },
  {
    id: 3,
    question:
      'How would you implement a log rotation strategy for a high-traffic application generating 10GB of logs daily, with compliance requirements to retain logs for 7 years?',
    answer:
      '**Multi-tier approach:** 1) **Hot storage (0-30 days):** Local disk with hourly rotation, compress immediately, ship to CloudWatch Logs. 2) **Warm storage (31-365 days):** Export to S3 Standard, compressed, with CloudWatch Logs Insights access. 3) **Cold storage (1-7 years):** S3 Glacier Deep Archive, lifecycle policy. 4) **Local retention:** Keep 7 days on disk (rotate daily, compress, gzip). 5) **Cost:** ~$50/TB/month S3 Standard → ~$1/TB/month Glacier Deep Archive.',
  },
];
