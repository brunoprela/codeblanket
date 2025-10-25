/**
 * Multiple choice questions for Alternative Task Queues section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const alternativeTaskQueuesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is RQ (Redis Queue)?',
    options: [
      'Alternative to Celery, simple task queue using Redis',
      'A database',
      'A web framework',
      'A monitoring tool',
    ],
    correctAnswer: 0,
    explanation:
      'RQ is simple, lightweight task queue for Python using Redis. Simpler than Celery but fewer features (no RabbitMQ, no periodic tasks, limited monitoring). Good for MVPs and small projects. Example: q.enqueue (send_email, "user@example.com"). When project grows, migrate to Celery.',
  },
  {
    id: 'mc2',
    question: 'What is the main advantage of RQ over Celery?',
    options: [
      'More features',
      'Simplicity - much easier to set up and use',
      'Better performance',
      'Larger community',
    ],
    correctAnswer: 1,
    explanation:
      "RQ's main advantage is simplicity. Setup: 5 lines of code. No complex config. Good for quick MVPs. Celery more powerful but requires more setup (broker config, workers, monitoring). RQ trade-off: Simple (✅) vs Features (❌). For small projects, RQ simplicity wins.",
  },
  {
    id: 'mc3',
    question: 'Which task queue supports both Redis and RabbitMQ?',
    options: [
      'RQ (Redis only)',
      'Huey (Redis only)',
      'Celery and Dramatiq',
      'None of them',
    ],
    correctAnswer: 2,
    explanation:
      'Celery and Dramatiq support both Redis and RabbitMQ. RQ supports only Redis. Huey supports only Redis. Multi-broker support important for: (1) Start with Redis (simple), (2) Scale to RabbitMQ (reliability). Celery most flexible broker support.',
  },
  {
    id: 'mc4',
    question:
      'Which lightweight task queue has built-in periodic task support?',
    options: [
      'RQ (no periodic tasks)',
      'Dramatiq (no built-in periodic)',
      'Huey (has periodic tasks)',
      'None of them',
    ],
    correctAnswer: 2,
    explanation:
      'Huey has built-in periodic task support via @huey.periodic_task (crontab(...)). RQ has no periodic tasks. Dramatiq needs APScheduler. Celery has Celery Beat. If you need lightweight + periodic tasks, Huey is best choice. Perfect for Flask/Django projects.',
  },
  {
    id: 'mc5',
    question: 'When should you migrate from RQ to Celery?',
    options: [
      'Never, always use RQ',
      'When you need complex workflows, better monitoring, >100K tasks/day',
      'Immediately for all projects',
      'RQ is always better',
    ],
    correctAnswer: 1,
    explanation:
      "Migrate RQ → Celery when: (1) Need complex workflows (chains, chords), (2) Need better monitoring (Flower), (3) Scale >100K tasks/day, (4) Need RabbitMQ reliability, (5) Need Celery Beat. Start with RQ (simple), migrate to Celery when requirements grow. Don't use Celery for simple projects.",
  },
];
