/**
 * Multiple choice questions for Log Management
 */

import { MultipleChoiceQuestion } from '../../../types';

export const logManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'log-mc-1',
    question: 'Which logrotate option compresses old log files?',
    options: ['compress', 'delaycompress', 'gzip', 'archive'],
    correctAnswer: 0,
    explanation:
      'The "compress" directive tells logrotate to compress rotated logs using gzip by default. "delaycompress" delays compression until the next rotation cycle. "gzip" and "archive" are not valid logrotate directives.',
    difficulty: 'easy',
    topic: 'Logrotate',
  },
  {
    id: 'log-mc-2',
    question: 'What does "journalctl --vacuum-time=30d" do?',
    options: [
      'Rotate logs every 30 days',
      'Delete logs older than 30 days',
      'Compress logs after 30 days',
      'Archive logs for 30 days',
    ],
    correctAnswer: 1,
    explanation:
      'The --vacuum-time option deletes journal files older than the specified time period. This helps manage disk space by removing old logs while keeping recent ones for troubleshooting.',
    difficulty: 'medium',
    topic: 'Journald',
  },
  {
    id: 'log-mc-3',
    question:
      'Which CloudWatch Logs feature allows you to search and analyze log data?',
    options: [
      'CloudWatch Alarms',
      'CloudWatch Insights',
      'CloudWatch Metrics',
      'CloudWatch Events',
    ],
    correctAnswer: 1,
    explanation:
      'CloudWatch Logs Insights is a query language for searching and analyzing log data. It allows filtering, aggregating, and visualizing logs with SQL-like queries. CloudWatch Alarms are for alerting, Metrics for monitoring, and Events for automation.',
    difficulty: 'easy',
    topic: 'CloudWatch',
  },
  {
    id: 'log-mc-4',
    question: 'Why use structured logging (JSON) instead of plain text?',
    options: [
      'Smaller file size',
      'Easier parsing and querying',
      'Better compression',
      'Faster writes',
    ],
    correctAnswer: 1,
    explanation:
      'Structured logging with JSON enables easy parsing, querying, and filtering. Log analysis tools can extract fields automatically without regex parsing. It supports complex data types and makes correlation across services much easier. File size is actually larger than plain text.',
    difficulty: 'medium',
    topic: 'Logging Best Practices',
  },
  {
    id: 'log-mc-5',
    question:
      'What is the most cost-effective AWS storage for long-term log archival (7+ years)?',
    options: [
      'S3 Standard',
      'S3 Intelligent-Tiering',
      'S3 Glacier',
      'S3 Glacier Deep Archive',
    ],
    correctAnswer: 3,
    explanation:
      'S3 Glacier Deep Archive is the most cost-effective option for long-term archival (~$1/TB/month). It is designed for data accessed less than once per year with retrieval times of 12+ hours. Perfect for compliance requirements needing multi-year retention.',
    difficulty: 'advanced',
    topic: 'Cost Optimization',
  },
];
