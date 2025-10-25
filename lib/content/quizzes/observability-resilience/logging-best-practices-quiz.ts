/**
 * Quiz questions for Logging Best Practices section
 */

export const loggingBestPracticesQuiz = [
  {
    id: 'q1',
    question:
      'Compare structured logging (JSON) versus unstructured logging (plain text). What are the advantages of structured logging, and in what scenarios might it not be worth the overhead?',
    sampleAnswer:
      'Structured logging uses machine-readable formats (typically JSON) where each log entry is a set of key-value pairs, while unstructured logging uses plain text messages. **Advantages of Structured Logging**: (1) Machine-Readable: Can query logs with SQL-like syntax ("amount > 100", "user_id = 123") instead of regex parsing. (2) Consistent Fields: Every log entry has predictable structure, enabling aggregation and analysis. (3) Easy Correlation: Include trace_id, user_id consistently, making distributed tracing trivial. (4) Better Tooling: Log aggregation tools (Elasticsearch, Loki) can index and search efficiently. (5) Aggregation: Count errors by type, calculate p99 latency from logs, group by dimensions. Example: Instead of "User john purchased laptop for $999", use {event: "purchase", user_id: "john", product: "laptop", amount: 999, timestamp: "..."}. **Scenarios Where Unstructured Might Be Acceptable**: (1) Local Development: Quick printf debugging where you won\'t query logs. (2) Low-Volume Systems: If you only generate 100 logs/day, parsing overhead is negligible. (3) Human-Only Consumption: Logs only read by developers SSH\'d into servers. (4) Legacy Systems: Retrofitting structured logging into 10-year-old codebase might not be worth it. **Trade-offs**: Structured logging has higher storage cost (JSON is verbose), requires more discipline from developers (must define schema), and can be harder to read raw. However, for any production system with centralized logging, the benefits far outweigh costs. Modern systems should default to structured logging.',
    keyPoints: [
      'Structured logging (JSON) enables machine queries and aggregation',
      'Consistent fields allow correlation across services (trace_id)',
      'Log tools can index and search structured logs efficiently',
      'Trade-off: More verbose storage, requires discipline',
      'Worth it for production systems, maybe not for local dev or legacy',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the concept of log sampling and when it should be used. How do you balance between completeness (logging everything) and cost/performance?',
    sampleAnswer:
      'Log sampling means intentionally logging only a subset of events rather than every occurrence. At scale, logging everything becomes prohibitively expensive in storage, network bandwidth, and I/O overhead. **Sampling Strategies**: (1) Level-Based: ERROR/FATAL: 100% (log everything), WARN: 100%, INFO: 10% (sample 1 in 10), DEBUG: 1% (sample 1 in 100). Rationale: Errors are rare and critical, debug logs are common and less important. (2) Adaptive Sampling: Normal: 1% sampling. Error spike detected: 100% sampling for 5 minutes. Return to normal: 1%. This ensures you have full data during incidents. (3) Tail-Based Sampling: Buffer logs for entire request. If request errors: Keep all logs. If request succeeds and fast: Sample at 1%. This keeps interesting requests while discarding boring ones. (4) Rate Limiting: Maximum 1000 logs/second per service to prevent log storms. **Balancing Completeness vs Cost**: At Netflix scale (billions of logs/day), 100% logging would cost millions monthly and impact performance. Sample aggressively in production (1-10% for success paths), but always log errors at 100%. Use tail-based sampling to keep all logs for problematic requests. Monitor "logs dropped" counter to ensure you\'re not sampling too aggressively. **Red Flags**: If you\'re frequently saying "I wish I had logs for that incident," you\'re under-sampling. If log storage costs exceed compute costs, you\'re over-logging. **Best Practice**: Start with level-based sampling, add adaptive sampling for incidents, graduate to tail-based sampling at scale.',
    keyPoints: [
      'Sampling reduces cost and overhead at scale (storage, I/O, network)',
      'Level-based: ERROR 100%, INFO 10%, DEBUG 1%',
      'Adaptive: Increase sampling during incidents',
      'Tail-based: Keep all logs for errors, sample successes',
      'Always log errors at 100%, sample success paths',
    ],
  },
  {
    id: 'q3',
    question:
      'What data should NEVER be logged, and why? How do you prevent sensitive data from accidentally appearing in logs?',
    sampleAnswer:
      'Certain data must never appear in logs due to security, privacy, and compliance requirements. **NEVER Log**: (1) Passwords: Even hashed passwords shouldn\'t be logged (rainbow tables, dictionary attacks possible). (2) API Keys/Tokens: Logging authentication tokens allows anyone with log access to impersonate users. (3) Credit Card Numbers: PCI-DSS compliance violation, severe penalties. (4) Personal Identifiable Information (PII): Full names, SSN, addresses, phone numbers (GDPR, CCPA violations). (5) Session IDs: Can be used for session hijacking. (6) Private Keys: Cryptographic keys, certificates. (7) Raw Request Bodies: Often contain sensitive data. **Why This Matters**: (1) Security: Logs are often stored insecurely, backed up to S3, sent to third-party services (Datadog, Splunk). (2) Compliance: GDPR fines up to 4% of revenue, PCI-DSS violations can cost millions. (3) Insider Threats: Not everyone who can read logs should access customer data. (4) Data Breaches: If logs leak, you\'ve leaked customer data. **Prevention Strategies**: (1) Redaction at Source: logger.info("User login", {email: redact (email), user_id: user.id}). Function redacts email to "j***@example.com". (2) Log Scrubbers: Regex patterns in log forwarders (Filebeat, Fluentd) remove sensitive patterns. (3) Type System: Use TypeScript/type hints to mark sensitive fields, lint against logging them. (4) Code Review: Flag logs that contain user input or raw request/response. (5) Testing: Automated tests scan logs for patterns like credit card numbers, SSNs. (6) Production Audits: Periodically grep production logs for sensitive patterns. (7) Access Control: Role-based access to logs (not everyone can read all logs). **Example Redaction**: Instead of logging full credit card "4532 1234 5678 9010", log "****-****-****-9010" or just don\'t log it at all. Use user_id for correlation, not personally identifiable details.',
    keyPoints: [
      'Never log: passwords, API keys, credit cards, PII, session IDs',
      'Logs often stored insecurely and backed up, increasing breach risk',
      'Compliance violations (GDPR, PCI-DSS) can cost millions',
      'Prevention: redact at source, log scrubbers, code review, testing',
      'Use user_id for correlation, not personally identifiable details',
    ],
  },
];
