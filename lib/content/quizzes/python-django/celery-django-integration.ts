export const celeryDjangoIntegrationQuiz = [
  {
    id: 1,
    question:
      'Explain how to integrate Celery with Django for background task processing. Include configuration, task definition, and best practices for production deployments.',
    answer: `
**Celery Django Setup:**

\`\`\`python
# requirements.txt
celery[redis]==5.3.0
django-celery-beat==2.5.0
django-celery-results==2.5.0

# myproject/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

app = Celery('myproject')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# myproject/__init__.py
from .celery import app as celery_app
__all__ = ('celery_app',)

# settings.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'django-db'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
\`\`\`

**Task Definition:**

\`\`\`python
# tasks.py
from celery import shared_task
from django.core.mail import send_mail

@shared_task
def send_email_task (user_id, subject, message):
    user = User.objects.get (id=user_id)
    send_mail(
        subject,
        message,
        'noreply@example.com',
        [user.email],
    )
    return f'Email sent to {user.email}'

@shared_task (bind=True, max_retries=3)
def process_article (self, article_id):
    try:
        article = Article.objects.get (id=article_id)
        # Process article
        article.status = 'processed'
        article.save()
    except Exception as exc:
        raise self.retry (exc=exc, countdown=60)
\`\`\`

**Using Tasks:**

\`\`\`python
# Async execution
send_email_task.delay (user_id=1, subject='Hello', message='World')

# With countdown
send_email_task.apply_async(
    args=[user_id, subject, message],
    countdown=300  # 5 minutes delay
)

# Schedule at specific time
from datetime import datetime, timedelta
eta = datetime.now() + timedelta (hours=1)
send_email_task.apply_async(
    args=[user_id, subject, message],
    eta=eta
)
\`\`\`

**Production Best Practices:**1. **Task Routing:**
\`\`\`python
CELERY_TASK_ROUTES = {
    'myapp.tasks.send_email_task': {'queue': 'emails'},
    'myapp.tasks.process_video': {'queue': 'heavy'},
}
\`\`\`

2. **Task Rate Limiting:**
\`\`\`python
@shared_task (rate_limit='10/m')  # 10 per minute
def api_call_task():
    pass
\`\`\`

3. **Task Monitoring:**
\`\`\`python
# Install flower
pip install flower

# Run
celery -A myproject flower
\`\`\`

4. **Graceful Handling:**
\`\`\`python
@shared_task (bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 5})
def robust_task (self, data):
    # Process with automatic retry
    pass
\`\`\`
      `,
  },
  {
    question:
      'Describe periodic tasks with Celery Beat in Django. Include scheduling strategies, dynamic task creation, and handling task failures.',
    answer: `
**Celery Beat Setup:**

\`\`\`python
# settings.py
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'send-daily-report': {
        'task': 'myapp.tasks.send_daily_report',
        'schedule': crontab (hour=8, minute=0),  # 8 AM daily
    },
    'cleanup-old-data': {
        'task': 'myapp.tasks.cleanup_old_data',
        'schedule': crontab (hour=2, minute=0, day_of_week=0),  # Sunday 2 AM
    },
    'check-status-every-5min': {
        'task': 'myapp.tasks.check_status',
        'schedule': 300.0,  # Every 5 minutes
    },
}
\`\`\`

**Dynamic Scheduling (django-celery-beat):**

\`\`\`python
from django_celery_beat.models import PeriodicTask, IntervalSchedule, CrontabSchedule
import json

# Create interval schedule
schedule, _ = IntervalSchedule.objects.get_or_create(
    every=10,
    period=IntervalSchedule.MINUTES,
)

# Create periodic task
PeriodicTask.objects.create(
    interval=schedule,
    name='Check user status',
    task='myapp.tasks.check_user_status',
    args=json.dumps([user_id]),
)

# Crontab schedule
crontab_schedule, _ = CrontabSchedule.objects.get_or_create(
    minute='0',
    hour='*/4',  # Every 4 hours
)

PeriodicTask.objects.create(
    crontab=crontab_schedule,
    name='Backup database',
    task='myapp.tasks.backup_database',
)
\`\`\`

**Task Failure Handling:**

\`\`\`python
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@shared_task (bind=True, max_retries=3, default_retry_delay=60)
def resilient_task (self, data):
    try:
        # Process data
        result = process_data (data)
        return result
    except TemporaryError as exc:
        # Retry on temporary errors
        logger.warning (f'Task failed, retrying: {exc}')
        raise self.retry (exc=exc, countdown=60 * (self.request.retries + 1))
    except PermanentError as exc:
        # Don't retry on permanent errors
        logger.error (f'Task failed permanently: {exc}')
        raise
    finally:
        # Always cleanup
        cleanup_resources()

# Callbacks
@shared_task
def on_success (result):
    logger.info (f'Task succeeded: {result}')

@shared_task
def on_failure (exc):
    logger.error (f'Task failed: {exc}')
    send_alert_email (exc)

# Chain tasks
from celery import chain
chain(
    process_task.s (data),
    on_success.s()
).apply_async (link_error=on_failure.s())
\`\`\`

**Monitoring & Management:**

\`\`\`python
# Check task status
from celery.result import AsyncResult

result = send_email_task.delay (user_id)
task = AsyncResult (result.id)

if task.ready():
    print(task.result)
elif task.failed():
    print(task.traceback)
\`\`\`
      `,
  },
  {
    question:
      'Explain task queues, routing, and priority in Celery. How do you optimize task execution for different workload types?',
    answer: `
**Task Queues & Routing:**

\`\`\`python
# settings.py
CELERY_TASK_ROUTES = {
    'myapp.tasks.send_email': {
        'queue': 'emails',
        'routing_key': 'email.send',
    },
    'myapp.tasks.process_video': {
        'queue': 'video',
        'routing_key': 'video.process',
    },
    'myapp.tasks.generate_report': {
        'queue': 'reports',
        'routing_key': 'report.generate',
    },
}

# Define queues
CELERY_TASK_DEFAULT_QUEUE = 'default'
CELERY_TASK_QUEUES = (
    Queue('default', routing_key='default'),
    Queue('emails', routing_key='email.#'),
    Queue('video', routing_key='video.#'),
    Queue('reports', routing_key='report.#'),
)
\`\`\`

**Starting Workers for Different Queues:**

\`\`\`bash
# Email worker (many concurrent tasks)
celery -A myproject worker -Q emails -c 20 -n email_worker

# Video worker (resource intensive, few concurrent)
celery -A myproject worker -Q video -c 2 -n video_worker

# Report worker
celery -A myproject worker -Q reports -c 5 -n report_worker
\`\`\`

**Task Priorities:**

\`\`\`python
# Kombu priority support
from kombu import Queue

CELERY_TASK_QUEUES = (
    Queue('high_priority', routing_key='high', priority=10),
    Queue('normal', routing_key='normal', priority=5),
    Queue('low_priority', routing_key='low', priority=1),
)

# Send high priority task
urgent_task.apply_async (priority=10, queue='high_priority')
\`\`\`

**Workload Optimization:**

\`\`\`python
# 1. I/O Bound Tasks (emails, API calls)
@shared_task (bind=True, rate_limit='100/m')
def io_bound_task (self):
    # Many concurrent workers
    pass

# 2. CPU Bound Tasks (video processing, ML)
@shared_task (bind=True, rate_limit='10/h')
def cpu_bound_task (self):
    # Few concurrent workers, more CPU
    pass

# 3. Memory Intensive Tasks
@shared_task (bind=True, soft_time_limit=300, time_limit=600)
def memory_intensive_task (self):
    # Separate worker with more RAM
    pass

# 4. Long Running Tasks
@shared_task (bind=True, acks_late=True, reject_on_worker_lost=True)
def long_running_task (self):
    # Acknowledge after completion
    pass
\`\`\`

**Advanced Routing:**

\`\`\`python
from celery.app.task import Task

class PriorityTask(Task):
    def apply_async (self, *args, **kwargs):
        if kwargs.get('priority') == 'high':
            kwargs['queue'] = 'high_priority'
        return super().apply_async(*args, **kwargs)

@shared_task (base=PriorityTask)
def flexible_task():
    pass

# Use
flexible_task.apply_async (priority='high')
\`\`\`

**Production Configuration:**

\`\`\`python
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # For long tasks
CELERY_TASK_ACKS_LATE = True  # Acknowledge after completion
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000  # Restart workers
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 min hard limit
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60  # 25 min soft limit
\`\`\`
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
