/**
 * Multiple choice questions for Logging Best Practices section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const loggingBestPracticesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of structured logging (JSON) over unstructured logging (plain text)?',
    options: [
      'Structured logs are easier for humans to read',
      'Structured logs enable machine queries, aggregation, and consistent correlation',
      'Structured logs take less storage space',
      'Structured logs are faster to write',
    ],
    correctAnswer: 1,
    explanation:
      'Structured logging (JSON) enables machine-readable queries (e.g., "amount > 100", "user_id = 123"), aggregation (count errors by type), and consistent correlation (trace_id in every log). While JSON is more verbose (larger storage) and harder to read raw than plain text, the ability to query and analyze logs programmatically far outweighs these costs for production systems.',
  },
  {
    id: 'mc2',
    question:
      'Which log level should be used for the event: "User authentication failed due to invalid password"?',
    options: ['DEBUG', 'INFO', 'WARN', 'ERROR'],
    correctAnswer: 2,
    explanation:
      "This should be WARN because: (1) It\'s expected behavior (users sometimes enter wrong passwords), (2) The system handled it gracefully (no system error), (3) But it's worth tracking for security (potential attack patterns). ERROR would be if the authentication system itself failed. INFO might be for successful login. DEBUG would be too verbose for production. WARN strikes the right balance for security-relevant but expected events.",
  },
  {
    id: 'mc3',
    question: 'Which of the following should NEVER be logged in production?',
    options: [
      'User IDs',
      'Request trace IDs',
      'Credit card numbers',
      'HTTP status codes',
    ],
    correctAnswer: 2,
    explanation:
      "Credit card numbers should NEVER be logged due to: (1) PCI-DSS compliance requirements (severe penalties for violations), (2) Security risk (logs often stored insecurely, backed up, sent to third parties), (3) Data breach liability (if logs leak, you've leaked payment data). User IDs, trace IDs, and status codes are safe to log. Other never-log items include: passwords, API keys, SSNs, and full names (PII).",
  },
  {
    id: 'mc4',
    question: 'What is log sampling, and when should it be used?',
    options: [
      'Recording only ERROR logs and ignoring all other levels',
      'Logging only a percentage of events (e.g., 10% of INFO logs) to reduce volume and cost',
      'Logging events from a sample of users rather than all users',
      'Taking random samples of log files for analysis',
    ],
    correctAnswer: 1,
    explanation:
      'Log sampling means logging only a percentage of events (e.g., log 100% of errors, 10% of INFO logs, 1% of DEBUG logs) to reduce storage costs and I/O overhead at scale. At high volumes (billions of logs/day), logging everything becomes prohibitively expensive. Sampling strategies include: level-based (ERROR 100%, INFO 10%), adaptive (increase during incidents), and tail-based (keep all logs for failed requests, sample successful ones).',
  },
  {
    id: 'mc5',
    question:
      'What is a correlation ID (trace ID) in logging, and why is it important?',
    options: [
      'A unique ID for each log entry',
      'A unique ID for the server writing logs',
      'A unique ID that flows through all services for a single request, enabling correlation',
      'A unique ID for each user',
    ],
    correctAnswer: 2,
    explanation:
      'A correlation ID (trace_id) is a unique identifier generated when a request enters the system and propagated through all services handling that request. It enables: (1) Tracking a single user request across multiple microservices, (2) Filtering logs to see only entries for a specific request, (3) Debugging distributed issues by correlating logs from different services, (4) Connecting logs to traces and metrics. Without correlation IDs, debugging distributed systems is nearly impossible.',
  },
];
