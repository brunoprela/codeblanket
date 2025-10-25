export const CeleryDjangoIntegrationMultipleChoice = {
  title: 'Celery + Django Integration - Multiple Choice Questions',
  questions: [
    {
      question:
        'Which decorator should you use to define a Celery task in Django?',
      options: [
        'A) @celery_task',
        'B) @shared_task',
        'C) @async_task',
        'D) @background_task',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) @shared_task**

\`\`\`python
from celery import shared_task

@shared_task
def send_email(user_id):
    # Task code
    pass
\`\`\`

@shared_task creates app-independent tasks that work with Django's Celery setup.
      `,
    },
    {
      question: 'How do you execute a Celery task asynchronously?',
      options: [
        'A) task.run()',
        'B) task.execute()',
        'C) task.delay()',
        'D) task.async()',
      ],
      correctAnswer: 2,
      explanation: `
**Correct Answer: C) task.delay()**

\`\`\`python
# Asynchronous execution
send_email.delay(user_id=1)

# Alternative with more options
send_email.apply_async(args=[user_id], countdown=60)
\`\`\`

delay() is a shortcut for apply_async() with simple arguments.
      `,
    },
    {
      question: 'What is Celery Beat used for?',
      options: [
        'A) Monitoring task performance',
        'B) Scheduling periodic tasks',
        'C) Load balancing workers',
        'D) Task result storage',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Scheduling periodic tasks**

\`\`\`python
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'daily-report': {
        'task': 'myapp.tasks.send_report',
        'schedule': crontab(hour=8, minute=0),
    },
}

# Run beat scheduler
celery -A myproject beat
\`\`\`

Beat is Celery's scheduler for recurring tasks.
      `,
    },
    {
      question: 'How do you retry a failed task in Celery?',
      options: [
        'A) return retry()',
        'B) raise self.retry()',
        'C) task.retry()',
        'D) retry_task()',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) raise self.retry()**

\`\`\`python
@shared_task(bind=True, max_retries=3)
def my_task(self):
    try:
        # Task logic
        pass
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)
\`\`\`

bind=True gives access to self for retry functionality.
      `,
    },
    {
      question: 'What is the purpose of CELERY_TASK_ACKS_LATE setting?',
      options: [
        'A) Speeds up task execution',
        'B) Acknowledges task after completion (not before)',
        'C) Delays task start',
        'D) Enables task logging',
      ],
      correctAnswer: 1,
      explanation: `
**Correct Answer: B) Acknowledges task after completion (not before)**

\`\`\`python
CELERY_TASK_ACKS_LATE = True
\`\`\`

With acks_late, if worker crashes before completing task, the task is redelivered to another worker. Important for critical tasks.
      `,
    },
  ],
};
