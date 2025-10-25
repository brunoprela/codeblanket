/**
 * Quiz questions for Audit Logging & Compliance section
 */

export const auditloggingcomplianceQuiz = [
  {
    id: 'audit-log-q-1',
    question:
      'Design a comprehensive audit logging system for an LLM application that must comply with GDPR, HIPAA, and SOC 2. What events do you log, what data do you include/exclude, and how do you handle data retention and user rights (right to deletion)?',
    hint: 'Consider comprehensive event logging, PII handling, and compliance requirements.',
    sampleAnswer:
      '**Requirements:** GDPR: User consent, right to deletion, data minimization. HIPAA: Audit trails, access logging, PHI protection. SOC 2: Security monitoring, access control, incident response. **Events to Log:** (1) User Actions: Authentication (login, logout, MFA). Requests (prompt, timestamp, user_id). Responses (output, model used). Rate limit hits, blocks. (2) Security Events: Failed login attempts. Prompt injection attempts. Content moderation blocks. PII detection triggers. (3) System Events: Model changes. Configuration updates. Access control changes. (4) Admin Actions: User data access. Data exports/deletions. System configuration. **Log Structure:** {timestamp: "2025-01-15T10:30:00Z", event_type: "llm_request", user_id: "user_12345", session_id: "session_abc", request_id: "req_xyz", prompt: "***REDACTED***",  # Never log actual prompt (may contain PII), prompt_hash: "sha256_hash",  # For deduplication, response_length: 150, model: "gpt-4", tokens_used: 200, blocked: false, block_reason: null, ip_address: "***HASHED***",  # Hash IP for privacy, user_agent: "***REDACTED***"}. **What to Exclude:** Never log: Full prompts (may contain PII/PHI). Full responses (same reason). User PII directly (hash/pseudonymize). Passwords, API keys. **Data Retention:** GDPR: 30 days for debugging logs, 1 year for security logs, 7 years for financial/legal logs. HIPAA: 6 years minimum. SOC 2: As defined in security policy (typically 1 year). Implementation: def log_ttl (event_type): if event_type == "debugging": return timedelta (days=30). elif event_type == "security": return timedelta (days=365). elif event_type == "financial": return timedelta (days=7*365). **User Rights (GDPR):** Right to Access: def export_user_data (user_id): logs = get_logs (user_id). anonymized = redact_pii (logs). return logs. Right to Deletion: def delete_user_data (user_id): # Mark for deletion (not immediate), mark_for_deletion (user_id). # Actual deletion after retention period, schedule_deletion (user_id, after=timedelta (days=30)). # Keep aggregated stats, keep_anonymized_statistics(). Right to Rectification: Allow users to update incorrect data. **HIPAA Compliance:** Log all PHI access: {event: "phi_access", user: "doctor_id", patient: "patient_id", reason: "treatment", timestamp: "..."}. Encrypt logs at rest and in transit. Access controls: Only authorized personnel. Audit log access: Track who views logs. **SOC 2 Compliance:** Monitoring: Alerts on suspicious patterns. Incident response: Log all security incidents. Access control: Logs show who has access to what. Change management: Log all system changes. **Result:** Comprehensive, compliant audit system.',
    keyPoints: [
      'Log security events, user actions, system changes',
      'Never log full prompts/responses (PII risk)',
      'Different retention periods per regulation',
      'Support user rights: access, deletion, rectification',
    ],
  },
  {
    id: 'audit-log-q-2',
    question:
      'Your audit logs are growing at 100GB/day, costing $3,000/month in storage. Design a cost-effective logging strategy that maintains compliance while reducing costs by 70%. What can be summarized, what must be retained, and how do you balance cost with auditability?',
    hint: 'Consider log levels, sampling, aggregation, and tiered storage.',
    sampleAnswer:
      '**Current: 100GB/day = 3TB/month.** Cost: $3,000/month (cloud storage). Target: <$900/month (70% reduction). **Analysis:** What\'s in logs? Debug logs: 60GB/day (60%), Security logs: 20GB/day (20%), User activity: 15GB/day (15%), System metrics: 5GB/day (5%). **Strategy 1: Appropriate Log Levels** - Production: WARN and above (not DEBUG). Development: DEBUG. Current: Everything logged at DEBUG level. Remove: Stack traces, verbose debug info, redundant logs. Result: 60GB → 10GB (83% reduction in debug logs). **Strategy 2: Sampling** - Not all requests need full logging. Sample: 100% of security events (blocks, attacks). 10% of normal requests. 100% of errors. def should_log_full (event): if event.is_security_event: return True. if event.is_error: return True. if random.random() < 0.1:  # 10% sample, return True. return False  # Log minimal info. Result: 15GB → 3GB (80% reduction in user activity logs). **Strategy 3: Aggregation** - Instead of logging every request individually: Aggregate metrics per hour: {hour: "2025-01-15 10:00", total_requests: 50000, avg_latency: 120ms, error_rate: 0.2%, blocked_requests: 50}. Store aggregated data, keep individual logs for errors/security. Result: 5GB → 0.5GB (90% reduction in metrics). **Strategy 4: Tiered Storage** - Hot storage (immediate access): Last 7 days. Cold storage (slower access): 8-30 days. Archive (glacier): 31+ days. Cost: Hot: $0.03/GB/month. Cold: $0.01/GB/month. Archive: $0.004/GB/month. Implementation: def move_to_appropriate_tier (log): age = now() - log.timestamp. if age < 7 days: keep in hot. elif age < 30 days: move to cold. else: move to archive. **Strategy 5: Compression** - Compress logs (gzip): 5-10x reduction. 100GB → 15GB compressed. Cost: Negligible CPU. **Strategy 6: Smart Retention** - Debug logs: 7 days (then delete). Security logs: 1 year. Compliance logs: 7 years (archive). Aggregated metrics: Forever (minimal size). **Final Calculation:** Debug: 10GB/day × 7 days × $0.03 = $2.10. Security: 20GB/day × 365 days × $0.01 (cold) = $73. User activity: 3GB/day × 30 days × $0.01 = $0.90. Metrics: 0.5GB/day × 365 days × $0.004 = $0.73. Compliance archive: 5GB/day × 2555 days (7 years) × $0.004 = $51. Total: ~$128/month (96% reduction!) ✓. **Trade-offs:** Can\'t debug every request (only sampled 10%). Aggregated data loses individual request details. Archived data has slower retrieval. **Result:** Cost: $3,000 → $130 (96% reduction, better than target).',
    keyPoints: [
      'Reduce log levels in production',
      'Sample normal requests, keep 100% of security events',
      'Aggregate metrics instead of individual logs',
      'Tiered storage: hot → cold → archive',
    ],
  },
  {
    id: 'audit-log-q-3',
    question:
      'A security incident requires you to analyze 6 months of audit logs to find all affected users. Your logs are distributed across 1,000 files. Design a log querying system that can find relevant entries in <5 minutes. What indexing, storage, and querying strategies do you use?',
    hint: 'Consider centralized logging, indexing, and efficient query tools.',
    sampleAnswer:
      '**Problem:** 6 months of logs, 1,000 files, need results in <5 minutes. **Current: File-Based Logs** - Logs stored as: logs/2025-01-15.log, logs/2025-01-16.log, .... Query: grep "user_12345" logs/*.log (takes 30 minutes). **Solution: Centralized Logging System** - Options: Elasticsearch (ELK stack), Splunk, Datadog, CloudWatch Insights. **Elasticsearch Architecture:** (1) Ingest: Logs → Logstash → Elasticsearch. (2) Index: Structured JSON documents with indices. (3) Query: Fast full-text search and aggregations. **Indexing Strategy:** Index structure: Index per day: logs-2025-01-15. Automatic rollover. Fields: timestamp (indexed), user_id (keyword, indexed), event_type (keyword, indexed), ip_address (keyword, indexed), request_id (keyword, indexed), Full text search on: error_message, block_reason. **Example Query (Elasticsearch):** GET /logs-*/_search { query: { bool: { must: [ {range: {timestamp: {gte: "2024-07-01", lte: "2025-01-15"}}}, {term: {event_type: "security_event"}}, {term: {blocked: true}} ], should: [ {match: {block_reason: "injection"}} ] } }, size: 1000 }. Result: <5 seconds for 6 months of data. **Aggregation for Affected Users:** GET /logs-*/_search { query: {...}, aggs: { affected_users: { terms: { field: "user_id", size: 10000 } } } }. Returns: List of all affected users with counts. **Optimization:** Partitioning: Partition by: date (for time-range queries), user_id (for user-specific queries). Retention: Auto-delete old indices. Shard allocation: 1 shard per 50GB. Replica shards: For reliability. **Alternative: Data Warehouse** - For very large scale: Logs → S3 → AWS Athena or BigQuery. Query using SQL: SELECT DISTINCT user_id FROM logs WHERE date >= "2024-07-01" AND event_type = "security_event". Result: ~30 seconds for 6 months. **Cost Comparison:** Elasticsearch: $500/month for 100GB/day. Athena: $5 per TB queried (~$3/query for 6 months). Trade-off: Elasticsearch: Fast, expensive. Athena: Slower, cheaper, better for rare queries. **Result:** Elasticsearch: <5 seconds. Athena: ~30 seconds. Both meet <5 minute requirement.',
    keyPoints: [
      'Centralized logging (Elasticsearch, Splunk)',
      'Index key fields: user_id, timestamp, event_type',
      'Partition by date and user for fast queries',
      'Consider data warehouse (Athena/BigQuery) for large scale',
    ],
  },
];
