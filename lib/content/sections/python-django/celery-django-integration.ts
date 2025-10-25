export const celeryDjangoIntegration = {
  title: 'Celery + Django Integration',
  id: 'celery-django-integration',
  content: `
# Celery + Django Integration

## Introduction

**Celery** is the de facto standard for handling background tasks in Django. It allows you to run time-consuming operations asynchronously, improving application responsiveness and enabling scheduled periodic tasks.

### Why Celery + Django?

- **Async Task Processing**: Send emails, generate reports without blocking requests
- **Scheduled Tasks**: Periodic cleanup, daily reports, scheduled notifications
- **Scalability**: Distribute work across multiple workers/servers
- **Reliability**: Retry failed tasks, persist task state
- **Monitoring**: Track task progress and failures

**Real-World Examples:**
- Instagram: Image processing, feed generation
- Pinterest: Email notifications, recommendation engine
- Robinhood: Trade executions, portfolio updates

By the end of this section, you'll master:
- Celery setup with Django
- Task definition and execution
- Periodic tasks with Beat
- Result backends
- Error handling and retries
- Production deployment

---

## Installation and Setup

\`\`\`bash
# Install Celery and Redis
pip install celery redis

# Or with RabbitMQ
pip install celery amqp
\`\`\`

### Project Structure

\`\`\`
myproject/
├── myproject/
│   ├── __init__.py
│   ├── celery.py        # Celery configuration
│   ├── settings.py
│   └── urls.py
├── articles/
│   ├── tasks.py         # Task definitions
│   └── views.py
└── manage.py
\`\`\`

### Celery Configuration

\`\`\`python
# myproject/celery.py
import os
from celery import Celery

# Set default Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

# Create Celery app
app = Celery('myproject')

# Load config from Django settings (CELERY_ prefix)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps
app.autodiscover_tasks()

@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
\`\`\`

\`\`\`python
# myproject/__init__.py
# Ensure Celery app is loaded when Django starts
from .celery import app as celery_app

__all__ = ('celery_app',)
\`\`\`

\`\`\`python
# settings.py
# Celery Configuration
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# Task execution settings
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60  # 25 minutes
\`\`\`

---

## Defining Tasks

### Basic Task

\`\`\`python
# articles/tasks.py
from celery import shared_task
from django.core.mail import send_mail
from .models import Article

@shared_task
def send_article_notification(article_id):
    """Send email notification when article is published"""
    article = Article.objects.get(id=article_id)
    
    send_mail(
        subject=f'New Article: {article.title}',
        message=article.content,
        from_email='noreply@example.com',
        recipient_list=['subscribers@example.com'],
    )
    
    return f'Notification sent for article {article_id}'
\`\`\`

### Calling Tasks

\`\`\`python
# In views or signals
from .tasks import send_article_notification

# Method 1: delay() - simple
send_article_notification.delay(article_id)

# Method 2: apply_async() - with options
send_article_notification.apply_async(
    args=[article_id],
    countdown=60,  # Execute in 60 seconds
    expires=3600,  # Expire after 1 hour
    retry=True,
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)

# Method 3: apply() - execute synchronously (for testing)
result = send_article_notification.apply(args=[article_id])
\`\`\`

### Task with Return Value

\`\`\`python
@shared_task
def generate_report(user_id):
    """Generate user activity report"""
    user = User.objects.get(id=user_id)
    
    report_data = {
        'total_articles': user.articles.count(),
        'total_views': user.articles.aggregate(Sum('view_count'))['view_count__sum'],
        'top_articles': list(user.articles.order_by('-view_count')[:5].values('title', 'view_count')),
    }
    
    return report_data

# In view
from celery.result import AsyncResult

def start_report(request):
    task = generate_report.delay(request.user.id)
    return JsonResponse({'task_id': task.id})

def check_report(request, task_id):
    result = AsyncResult(task_id)
    
    if result.ready():
        return JsonResponse({
            'status': 'complete',
            'result': result.get()
        })
    else:
        return JsonResponse({
            'status': 'pending',
            'state': result.state
        })
\`\`\`

---

## Periodic Tasks with Celery Beat

### Install Beat

\`\`\`bash
pip install django-celery-beat
\`\`\`

\`\`\`python
# settings.py
INSTALLED_APPS = [
    ...
    'django_celery_beat',
]

# Migrate
python manage.py migrate django_celery_beat
\`\`\`

### Define Periodic Tasks

\`\`\`python
# settings.py
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    'send-daily-report': {
        'task': 'articles.tasks.send_daily_report',
        'schedule': crontab(hour=9, minute=0),  # Every day at 9 AM
    },
    'cleanup-old-sessions': {
        'task': 'core.tasks.cleanup_sessions',
        'schedule': crontab(hour=2, minute=0),  # Every day at 2 AM
    },
    'update-trending-articles': {
        'task': 'articles.tasks.update_trending',
        'schedule': 300.0,  # Every 5 minutes
    },
    'weekly-summary': {
        'task': 'articles.tasks.weekly_summary',
        'schedule': crontab(day_of_week=1, hour=10, minute=0),  # Monday at 10 AM
    },
}
\`\`\`

### Periodic Task Example

\`\`\`python
# articles/tasks.py
@shared_task
def cleanup_old_drafts():
    """Delete draft articles older than 30 days"""
    from datetime import timedelta
    from django.utils import timezone
    
    threshold = timezone.now() - timedelta(days=30)
    deleted_count, _ = Article.objects.filter(
        status='draft',
        created_at__lt=threshold
    ).delete()
    
    return f'Deleted {deleted_count} old drafts'

@shared_task
def send_daily_report():
    """Send daily stats to admins"""
    from datetime import timedelta
    from django.utils import timezone
    from django.core.mail import mail_admins
    
    yesterday = timezone.now() - timedelta(days=1)
    
    stats = {
        'articles_published': Article.objects.filter(
            published_at__gte=yesterday
        ).count(),
        'total_views': Article.objects.filter(
            published_at__gte=yesterday
        ).aggregate(Sum('view_count'))['view_count__sum'] or 0,
    }
    
    mail_admins(
        subject='Daily Report',
        message=f"Published: {stats['articles_published']}, Views: {stats['total_views']}"
    )
\`\`\`

### Run Celery Beat

\`\`\`bash
# Start Celery worker
celery -A myproject worker -l info

# Start Celery Beat (in separate terminal)
celery -A myproject beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
\`\`\`

---

## Error Handling and Retries

### Automatic Retries

\`\`\`python
@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def send_email_task(self, email, subject, message):
    """Send email with automatic retry on failure"""
    try:
        send_mail(subject, message, 'noreply@example.com', [email])
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
\`\`\`

### Custom Retry Logic

\`\`\`python
@shared_task(bind=True)
def process_payment(self, order_id):
    """Process payment with custom retry logic"""
    try:
        order = Order.objects.get(id=order_id)
        payment_gateway.charge(order)
        order.status = 'paid'
        order.save()
    except PaymentError as exc:
        # Retry only for specific errors
        if exc.retriable:
            raise self.retry(exc=exc, max_retries=5, countdown=120)
        else:
            # Don't retry, mark as failed
            order.status = 'failed'
            order.save()
            raise
    except Exception as exc:
        # Log unexpected errors
        logger.error(f'Payment processing error: {exc}', exc_info=True)
        raise self.retry(exc=exc, max_retries=3)
\`\`\`

### Error Callbacks

\`\`\`python
@shared_task
def process_article(article_id):
    """Process article"""
    article = Article.objects.get(id=article_id)
    # Processing logic
    return article_id

@shared_task
def on_process_error(request, exc, traceback):
    """Called when process_article fails"""
    logger.error(f'Task {request.id} failed: {exc}')
    # Send notification to admin
    mail_admins('Task Failed', f'Task {request.id} failed with error: {exc}')

# Link error callback
process_article.apply_async(
    args=[article_id],
    link_error=on_process_error.s()
)
\`\`\`

---

## Task Chaining and Workflows

### Sequential Tasks (Chains)

\`\`\`python
from celery import chain

@shared_task
def download_image(url):
    """Download image from URL"""
    # Download logic
    return image_path

@shared_task
def process_image(image_path):
    """Process image (resize, optimize)"""
    # Processing logic
    return processed_path

@shared_task
def upload_to_s3(processed_path):
    """Upload to S3"""
    # Upload logic
    return s3_url

# Chain tasks together
workflow = chain(
    download_image.s('http://example.com/image.jpg'),
    process_image.s(),
    upload_to_s3.s()
)

# Execute chain
result = workflow.apply_async()
\`\`\`

### Parallel Tasks (Groups)

\`\`\`python
from celery import group

@shared_task
def resize_image(image_path, size):
    """Resize image to specific size"""
    # Resize logic
    return resized_path

# Execute multiple tasks in parallel
job = group(
    resize_image.s('/path/to/image.jpg', '100x100'),
    resize_image.s('/path/to/image.jpg', '300x300'),
    resize_image.s('/path/to/image.jpg', '600x600'),
)

result = job.apply_async()
\`\`\`

### Chord (Group with Callback)

\`\`\`python
from celery import chord

@shared_task
def process_article_chunk(article_ids):
    """Process a chunk of articles"""
    return len(article_ids)

@shared_task
def summarize_results(results):
    """Called after all chunks are processed"""
    total_processed = sum(results)
    print(f'Processed {total_processed} articles')
    return total_processed

# Process in parallel, then summarize
article_ids = list(Article.objects.values_list('id', flat=True))
chunks = [article_ids[i:i+100] for i in range(0, len(article_ids), 100)]

workflow = chord(
    (process_article_chunk.s(chunk) for chunk in chunks),
    summarize_results.s()
)

workflow.apply_async()
\`\`\`

---

## Monitoring and Debugging

### Flower (Web-based Monitoring)

\`\`\`bash
# Install Flower
pip install flower

# Run Flower
celery -A myproject flower

# Access at http://localhost:5555
\`\`\`

### Task Logging

\`\`\`python
import logging
logger = logging.getLogger(__name__)

@shared_task(bind=True)
def complex_task(self):
    """Task with logging"""
    logger.info(f'Task {self.request.id} started')
    
    try:
        # Task logic
        logger.info('Processing step 1')
        # ...
        logger.info('Processing step 2')
        # ...
        logger.info(f'Task {self.request.id} completed')
    except Exception as e:
        logger.error(f'Task {self.request.id} failed: {e}', exc_info=True)
        raise
\`\`\`

---

## Production Configuration

### Supervisor Configuration

\`\`\`ini
; /etc/supervisor/conf.d/celery.conf
[program:celery-worker]
command=/path/to/venv/bin/celery -A myproject worker -l info
directory=/path/to/project
user=www-data
numprocs=1
stdout_logfile=/var/log/celery/worker.log
stderr_logfile=/var/log/celery/worker-error.log
autostart=true
autorestart=true
startsecs=10
stopwaitsecs=600
stopasgroup=true
killasgroup=true

[program:celery-beat]
command=/path/to/venv/bin/celery -A myproject beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
directory=/path/to/project
user=www-data
numprocs=1
stdout_logfile=/var/log/celery/beat.log
stderr_logfile=/var/log/celery/beat-error.log
autostart=true
autorestart=true
startsecs=10
\`\`\`

### Systemd Configuration

\`\`\`ini
# /etc/systemd/system/celery.service
[Unit]
Description=Celery Service
After=network.target

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/celery multi start worker -A myproject --pidfile=/var/run/celery/%n.pid --logfile=/var/log/celery/%n%I.log --loglevel=INFO
ExecStop=/path/to/venv/bin/celery multi stopwait worker --pidfile=/var/run/celery/%n.pid
ExecReload=/path/to/venv/bin/celery multi restart worker -A myproject --pidfile=/var/run/celery/%n.pid --logfile=/var/log/celery/%n%I.log --loglevel=INFO

[Install]
WantedBy=multi-user.target
\`\`\`

---

## Summary

**Key Concepts:**
- **shared_task**: Define reusable tasks
- **delay()**: Simple async execution
- **apply_async()**: Async with options
- **Beat**: Periodic task scheduling
- **Chains**: Sequential task execution
- **Groups**: Parallel task execution
- **Retries**: Automatic failure recovery

**Production Checklist:**
- ✅ Use Redis/RabbitMQ as broker
- ✅ Set task time limits
- ✅ Implement retry logic
- ✅ Monitor with Flower
- ✅ Use Supervisor/Systemd
- ✅ Log task execution
- ✅ Handle task failures gracefully
- ✅ Use separate queues for different task types
- ✅ Set up result backend for task results
- ✅ Test tasks thoroughly

Celery transforms Django applications by enabling background processing, scheduled tasks, and distributed computing. It's essential for building scalable, responsive applications.
`,
};
