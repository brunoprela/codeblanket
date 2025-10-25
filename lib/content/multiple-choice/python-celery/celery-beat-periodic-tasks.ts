/**
 * Multiple choice questions for Celery Beat (Periodic Tasks) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const celeryBeatPeriodicTasksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How many Celery Beat instances should you run in production?',
    options: [
      'One per worker for load balancing',
      'Exactly ONE instance (multiple Beats = duplicate tasks)',
      'As many as you want for redundancy',
      'One per queue',
    ],
    correctAnswer: 1,
    explanation:
      'Run exactly ONE Beat instance in production. Multiple Beats independently schedule tasks → duplicate execution. Example: 3 Beats = each task runs 3×! Prevention: Kubernetes replicas: 1, systemd single service, Redis lock. Beat is scheduler (queues tasks), workers are executors (process tasks). Multiple workers OK, but only 1 Beat!',
  },
  {
    id: 'mc2',
    question:
      'What crontab expression runs a task every weekday (Mon-Fri) at 9 AM?',
    options: [
      'crontab (hour=9, minute=0)',
      "crontab (hour=9, minute=0, day_of_week='1-5')",
      "crontab (hour=9, minute=0, day_of_week='mon-sun')",
      "crontab (minute=0, hour='9-17')",
    ],
    correctAnswer: 1,
    explanation:
      'crontab (hour=9, minute=0, day_of_week="1-5") runs every weekday at 9 AM. day_of_week: 0=Sunday, 1=Monday, ..., 6=Saturday. "1-5" = Mon-Fri. Option 1: Every day (not just weekdays). Option 3: "mon-sun" is every day. Option 4: Every hour 9 AM-5 PM (not 9 AM only). Alternatives: day_of_week="mon-fri" also works.',
  },
  {
    id: 'mc3',
    question:
      'What is the purpose of the Beat schedule file (celerybeat-schedule)?',
    options: [
      'It stores task code for execution',
      'It stores last_run_at timestamps to prevent duplicate execution after restarts',
      'It replaces the need for a message broker',
      'It stores task results',
    ],
    correctAnswer: 1,
    explanation:
      'Schedule file stores last_run_at timestamps for each periodic task. Purpose: Remember when tasks last ran across Beat restarts. Without it: Every Beat restart would run all tasks immediately. With it: Beat knows "daily backup already ran at 2 AM, skip until tomorrow". Binary file, located at celerybeat-schedule or custom path. Corruption → tasks run wrong schedule. Prevention: Use database scheduler (django-celery-beat).',
  },
  {
    id: 'mc4',
    question: 'How do you run a task every 15 minutes?',
    options: [
      'schedule: 15.0',
      "schedule: crontab (minute='*/15')",
      "schedule: crontab (hour='*/15')",
      'schedule: 900.0',
    ],
    correctAnswer: 1,
    explanation:
      'crontab (minute="*/15") runs every 15 minutes (0, 15, 30, 45). Option 1 (15.0): Runs every 15 seconds (not minutes). Option 3: Invalid - */15 on hour. Option 4 (900.0): Every 900 seconds = 15 minutes (works but crontab clearer). Both crontab (minute="*/15") and schedule: 900.0 work, but crontab is more readable for minute-based schedules.',
  },
  {
    id: 'mc5',
    question:
      'What happens if a periodic task is still running when the next scheduled time arrives?',
    options: [
      'Beat waits for the current execution to finish',
      'Beat queues another instance of the task (can run simultaneously)',
      'Beat skips the next execution',
      'Beat kills the current execution and starts a new one',
    ],
    correctAnswer: 1,
    explanation:
      'Beat queues tasks independently of execution status. If task scheduled every 5 minutes but takes 10 minutes, Beat queues at T+0, T+5, T+10 → multiple instances run simultaneously (workers execute concurrently). Prevention: Set expires option (task expires if not executed within X seconds) or ensure task completes faster than schedule interval. Make tasks idempotent to handle overlapping executions safely.',
  },
];
