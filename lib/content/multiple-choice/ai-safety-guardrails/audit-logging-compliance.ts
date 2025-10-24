/**
 * Multiple choice questions for Audit Logging & Compliance section
 */

export const auditloggingcomplianceMultipleChoice = [
  {
    id: 'audit-log-mc-1',
    question:
      'Under GDPR, a user requests deletion of their data. Your audit logs contain their user_id. What should you do?',
    options: [
      'Delete all logs containing their user_id immediately',
      'Anonymize/pseudonymize their user_id in logs',
      'Keep logs unchanged—audit logs are exempt from GDPR',
      'Deny the request—logs are required for security',
    ],
    correctAnswer: 1,
    explanation:
      "The best practice is to anonymize/pseudonymize the user_id in logs (e.g., hash it or replace with a pseudonym). This balances the user's right to deletion with security/audit requirements. Option A (immediate deletion) may violate security regulations. Option C is false—GDPR applies to logs. Option D—you cannot simply deny; you must find a compliant solution.",
  },
  {
    id: 'audit-log-mc-2',
    question:
      'You should log the full prompt and response for every LLM request for debugging. True or false?',
    options: [
      'True—debugging requires full context',
      'False—prompts may contain PII/sensitive data',
      'True—but only in development',
      'False—logs are too expensive',
    ],
    correctAnswer: 1,
    explanation:
      'False—prompts and responses may contain PII, PHI, or sensitive information that should not be logged. Instead: log metadata (timestamp, user_id, token count), hash prompts for deduplication, sample a small percentage for debugging. Option D is a concern but not the primary reason.',
  },
  {
    id: 'audit-log-mc-3',
    question: 'Your logs are 100GB/day. What gives the LARGEST cost reduction?',
    options: [
      'Compress logs with gzip',
      'Use tiered storage (hot → cold → archive)',
      'Reduce log levels (DEBUG → WARN in production)',
      'Sample normal requests, keep all errors',
    ],
    correctAnswer: 2,
    explanation:
      "Reducing log levels from DEBUG to WARN typically eliminates 60-80% of log volume (debug logs are very verbose). This is the largest single reduction. Tiered storage (B) saves cost but doesn't reduce volume. Compression (A) saves 5-10x but log generation is still expensive. Sampling (D) is effective but typically less impactful than removing debug logs.",
  },
  {
    id: 'audit-log-mc-4',
    question: 'For SOC 2 compliance, you must:',
    options: [
      'Encrypt logs at rest only',
      'Log all user actions but not system events',
      'Have audit trails showing who accessed what data',
      'Delete logs after 30 days',
    ],
    correctAnswer: 2,
    explanation:
      'SOC 2 requires comprehensive audit trails showing access controls—who accessed what data, when, and why. Option A is incomplete (need encryption in transit too). Option B is backwards (need both user and system events). Option D violates SOC 2 (need longer retention, typically 1 year).',
  },
  {
    id: 'audit-log-mc-5',
    question:
      'You need to query 6 months of logs to find affected users. Best tool?',
    options: [
      'grep on log files',
      'Elasticsearch or Splunk',
      'Manual review',
      'SQL database',
    ],
    correctAnswer: 1,
    explanation:
      'Elasticsearch or Splunk are designed for log analysis at scale. They provide indexed search, aggregations, and can query 6 months of logs in seconds. Option A (grep) takes hours for large logs. Option C (manual) is infeasible. Option D (SQL database) can work but is not optimized for log-style data.',
  },
];
