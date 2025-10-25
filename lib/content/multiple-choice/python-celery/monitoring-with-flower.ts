/**
 * Multiple choice questions for Task Monitoring with Flower section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const monitoringWithFlowerMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is Flower?',
    options: [
      'A message broker alternative to Redis',
      'Real-time web-based monitoring tool for Celery',
      'A Python web framework',
      'A task queue alternative to Celery',
    ],
    correctAnswer: 1,
    explanation:
      'Flower is real-time web-based monitoring dashboard for Celery. Shows workers, tasks, queues, statistics. Access at http://localhost:5555 after running: celery -A tasks flower. Not a broker, framework, or queue. Essential for production Celery monitoring.',
  },
  {
    id: 'mc2',
    question: 'How do you start Flower with authentication?',
    options: [
      'celery -A tasks flower --auth=admin:password',
      'celery -A tasks flower --basic_auth=admin:password',
      'celery -A tasks flower --login=admin:password',
      'Authentication not supported',
    ],
    correctAnswer: 1,
    explanation:
      'Start Flower with auth: celery -A tasks flower --basic_auth=admin:password. Or multiple users: --basic_auth=admin:pass1,user:pass2. Or use flowerconfig.py: basic_auth = ["admin:hash"]. Essential for production (don\'t expose Flower publicly without auth!).',
  },
  {
    id: 'mc3',
    question: 'What can you monitor in Flower?',
    options: [
      'Only active tasks',
      'Only worker status',
      'Workers, tasks, queues, execution statistics, task history',
      'Only queue depth',
    ],
    correctAnswer: 2,
    explanation:
      'Flower monitors: (1) Workers (CPU, memory, pool type), (2) Tasks (active, succeeded, failed, execution time), (3) Queues (depth, routing), (4) Statistics (success/failure rates, time-series graphs), (5) Task history (arguments, results, state). Comprehensive real-time monitoring.',
  },
  {
    id: 'mc4',
    question: 'How do you access Flower API programmatically?',
    options: [
      'GET http://localhost:5555/api/workers',
      'Flower has no API',
      'Only through web UI',
      'Via Celery inspect()',
    ],
    correctAnswer: 0,
    explanation:
      'Flower provides REST API: GET /api/workers (worker list), GET /api/tasks (task list), GET /api/queues/length (queue depths). Use for automation, monitoring, auto-scaling. Example: requests.get("http://localhost:5555/api/workers").json(). Useful for programmatic access.',
  },
  {
    id: 'mc5',
    question: 'In production, how should you secure Flower?',
    options: [
      'No security needed',
      'Basic auth + NGINX reverse proxy + SSL + IP whitelist',
      'Just basic auth is enough',
      'Flower is automatically secure',
    ],
    correctAnswer: 1,
    explanation:
      "Production Flower security: (1) Basic auth --basic_auth (2) NGINX reverse proxy, (3) SSL/TLS encryption, (4) IP whitelist (internal network only), (5) OAuth for SSO, (6) Read-only mode (disable controls). Don't expose Flower publicly without security!",
  },
];
