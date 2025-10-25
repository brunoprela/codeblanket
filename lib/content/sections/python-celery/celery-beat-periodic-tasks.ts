export const celeryBeatPeriodicTasks = {
  title: 'Celery Beat (Periodic Tasks)',
  id: 'celery-beat-periodic-tasks',
  content: `
# Celery Beat (Periodic Tasks)

## Introduction

**Celery Beat** is Celery's scheduler for periodic tasks - think of it as "cron for Celery". It enables running tasks at regular intervals (every 5 minutes, daily at midnight, every Monday, etc.) without manual triggering.

**Common Use Cases:**
- Daily database backups
- Hourly report generation
- Weekly email digests
- Cleanup old data every night
- Health checks every 5 minutes
- Stock price updates every minute

---

## How Celery Beat Works

\`\`\`
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Celery Beat  │────────▶│ Message      │────────▶│ Celery       │
│ (Scheduler)  │ Queue   │ Broker       │ Execute │ Worker       │
│              │ tasks   │ (Redis)      │         │              │
└──────────────┘         └──────────────┘         └──────────────┘
     Timer                    Queue                   Process
     
Every N seconds/minutes/hours:
1. Beat checks schedule
2. If task due, Beat queues task to broker
3. Worker picks up and executes task
\`\`\`

**Key Points:**
- Beat is a **scheduler**, not an executor
- Beat queues tasks to broker (just like manual \`.delay()\`)
- Workers execute scheduled tasks (same workers as manual tasks)
- Only run **ONE** Beat instance (multiple Beats = duplicate tasks!)

---

## Basic Periodic Task Configuration

\`\`\`python
"""
Celery Beat Configuration
"""

from celery import Celery
from celery.schedules import crontab

app = Celery('myapp', broker='redis://localhost:6379/0')

# Configure periodic tasks
app.conf.beat_schedule = {
    # Task 1: Run every 30 seconds
    'add-every-30-seconds': {
        'task': 'tasks.add',
        'schedule': 30.0,  # seconds
        'args': (16, 16)
    },
    
    # Task 2: Run every 5 minutes
    'cleanup-every-5-minutes': {
        'task': 'tasks.cleanup',
        'schedule': 300.0,  # 5 * 60 seconds
    },
    
    # Task 3: Run every day at midnight
    'daily-backup': {
        'task': 'tasks.backup_database',
        'schedule': crontab(hour=0, minute=0),
    },
    
    # Task 4: Run every Monday at 9 AM
    'weekly-report': {
        'task': 'tasks.generate_weekly_report',
        'schedule': crontab(hour=9, minute=0, day_of_week=1),
    },
    
    # Task 5: Run every hour
    'hourly-sync': {
        'task': 'tasks.sync_data',
        'schedule': crontab(minute=0),  # Every hour at :00
    },
}

# Define tasks
@app.task
def add(x, y):
    return x + y

@app.task
def cleanup():
    """Clean up old data"""
    print("Cleaning up...")

@app.task
def backup_database():
    """Daily database backup"""
    print("Backing up database...")

@app.task
def generate_weekly_report():
    """Weekly report"""
    print("Generating weekly report...")

@app.task
def sync_data():
    """Hourly data sync"""
    print("Syncing data...")
\`\`\`

### Running Celery Beat

\`\`\`bash
# Terminal 1: Start Beat scheduler
celery -A tasks beat --loglevel=info

# Terminal 2: Start worker (executes scheduled tasks)
celery -A tasks worker --loglevel=info

# Or run Beat + Worker in same process (development only!)
celery -A tasks worker --beat --loglevel=info
\`\`\`

**⚠️ Important:** Only run ONE Beat instance in production (multiple Beats = duplicate tasks)

---

## Schedule Types

### 1. Interval Schedule (Seconds)

\`\`\`python
from celery import Celery

app = Celery('myapp', broker='redis://localhost:6379/0')

app.conf.beat_schedule = {
    'every-10-seconds': {
        'task': 'tasks.every_10_seconds',
        'schedule': 10.0,  # Run every 10 seconds
    },
    'every-minute': {
        'task': 'tasks.every_minute',
        'schedule': 60.0,  # Run every 60 seconds
    },
    'every-hour': {
        'task': 'tasks.every_hour',
        'schedule': 3600.0,  # Run every 3600 seconds
    },
}
\`\`\`

### 2. Crontab Schedule (Cron-like)

\`\`\`python
from celery.schedules import crontab

app.conf.beat_schedule = {
    # Every midnight
    'midnight-task': {
        'task': 'tasks.midnight_task',
        'schedule': crontab(hour=0, minute=0),
    },
    
    # Every day at 9:30 AM
    'morning-task': {
        'task': 'tasks.morning_task',
        'schedule': crontab(hour=9, minute=30),
    },
    
    # Every Monday at 8 AM
    'monday-morning': {
        'task': 'tasks.monday_task',
        'schedule': crontab(hour=8, minute=0, day_of_week=1),
    },
    
    # Every weekday at 5 PM
    'weekday-evening': {
        'task': 'tasks.weekday_task',
        'schedule': crontab(hour=17, minute=0, day_of_week='mon-fri'),
    },
    
    # Every 15 minutes
    'every-15-minutes': {
        'task': 'tasks.quarter_hourly',
        'schedule': crontab(minute='*/15'),  # 0, 15, 30, 45
    },
    
    # First day of every month at midnight
    'monthly-task': {
        'task': 'tasks.monthly_task',
        'schedule': crontab(hour=0, minute=0, day_of_month=1),
    },
    
    # Every hour at :30 (1:30, 2:30, 3:30, ...)
    'half-past-every-hour': {
        'task': 'tasks.half_hourly',
        'schedule': crontab(minute=30),
    },
}
\`\`\`

### 3. Solar Schedule (Sunrise/Sunset)

\`\`\`python
from celery.schedules import solar

app.conf.beat_schedule = {
    # At sunrise
    'sunrise-task': {
        'task': 'tasks.at_sunrise',
        'schedule': solar('sunrise', -37.81, 144.96),  # Melbourne coordinates
    },
    
    # At sunset
    'sunset-task': {
        'task': 'tasks.at_sunset',
        'schedule': solar('sunset', 40.71, -74.01),  # New York coordinates
    },
}
\`\`\`

---

## Crontab Expression Examples

\`\`\`python
from celery.schedules import crontab

# Every minute
crontab()

# Every 5 minutes
crontab(minute='*/5')

# Every hour
crontab(minute=0)

# Every day at midnight
crontab(hour=0, minute=0)

# Every day at 3:30 AM
crontab(hour=3, minute=30)

# Every Monday at 9 AM
crontab(hour=9, minute=0, day_of_week=1)

# Every weekday (Mon-Fri) at 5 PM
crontab(hour=17, minute=0, day_of_week='1-5')
crontab(hour=17, minute=0, day_of_week='mon-fri')

# Every weekend (Sat-Sun) at 10 AM
crontab(hour=10, minute=0, day_of_week='6-0')

# First day of every month at midnight
crontab(hour=0, minute=0, day_of_month=1)

# Last day of every month (use day_of_month=-1)
crontab(hour=0, minute=0, day_of_month=-1)

# Every quarter hour (0, 15, 30, 45)
crontab(minute='*/15')

# Specific times: 9 AM, 12 PM, 3 PM
crontab(hour='9,12,15', minute=0)

# Every 2 hours
crontab(minute=0, hour='*/2')

# Every 4 hours starting at 1 AM (1, 5, 9, 13, 17, 21)
crontab(minute=0, hour='1-23/4')
\`\`\`

---

## Dynamic Periodic Tasks (Database Backend)

Store schedules in database for runtime modification:

\`\`\`python
"""
Django Celery Beat - Database-backed periodic tasks
"""

# Install
pip install django-celery-beat

# settings.py
INSTALLED_APPS = [
    'django_celery_beat',
]

# Run migrations
python manage.py migrate django_celery_beat

# Celery config
from celery import Celery

app = Celery('myapp')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Use database scheduler
app.conf.beat_scheduler = 'django_celery_beat.schedulers:DatabaseScheduler'

# Now create schedules via Django admin or code
from django_celery_beat.models import PeriodicTask, IntervalSchedule, CrontabSchedule

# Create interval schedule (every 10 seconds)
schedule, created = IntervalSchedule.objects.get_or_create(
    every=10,
    period=IntervalSchedule.SECONDS,
)

# Create periodic task
PeriodicTask.objects.create(
    interval=schedule,
    name='Cleanup every 10 seconds',
    task='tasks.cleanup',
)

# Create crontab schedule (daily at midnight)
crontab_schedule, created = CrontabSchedule.objects.get_or_create(
    minute='0',
    hour='0',
    day_of_week='*',
    day_of_month='*',
    month_of_year='*',
)

PeriodicTask.objects.create(
    crontab=crontab_schedule,
    name='Daily backup',
    task='tasks.backup_database',
)

# Enable/disable task dynamically
task = PeriodicTask.objects.get(name='Daily backup')
task.enabled = False  # Disable
task.save()

# Modify schedule dynamically
task = PeriodicTask.objects.get(name='Cleanup every 10 seconds')
task.interval.every = 60  # Change to 60 seconds
task.interval.save()
\`\`\`

---

## Production Configuration

\`\`\`python
"""
Production Celery Beat Configuration
"""

from celery import Celery
from celery.schedules import crontab

app = Celery('myapp', broker='redis://localhost:6379/0')

# Timezone configuration
app.conf.timezone = 'UTC'
app.conf.enable_utc = True

# Beat schedule
app.conf.beat_schedule = {
    # Health check every 5 minutes
    'health-check': {
        'task': 'tasks.health_check',
        'schedule': 300.0,
        'options': {'expires': 240}  # Expire if not run within 4 minutes
    },
    
    # Daily backup at 2 AM UTC
    'daily-backup': {
        'task': 'tasks.backup_database',
        'schedule': crontab(hour=2, minute=0),
        'kwargs': {'full': True}
    },
    
    # Hourly report generation
    'hourly-reports': {
        'task': 'tasks.generate_reports',
        'schedule': crontab(minute=0),
        'options': {
            'queue': 'reports',
            'priority': 5
        }
    },
    
    # Cleanup old data every night at 3 AM
    'nightly-cleanup': {
        'task': 'tasks.cleanup_old_data',
        'schedule': crontab(hour=3, minute=0),
        'kwargs': {'days_old': 90}
    },
    
    # Weekly analytics every Sunday at midnight
    'weekly-analytics': {
        'task': 'tasks.compute_weekly_analytics',
        'schedule': crontab(hour=0, minute=0, day_of_week=0),
    },
    
    # Monthly billing on 1st of month
    'monthly-billing': {
        'task': 'tasks.process_monthly_billing',
        'schedule': crontab(hour=0, minute=0, day_of_month=1),
    },
}

# Beat configuration
app.conf.beat_schedule_filename = '/var/run/celerybeat-schedule'  # Schedule file location
app.conf.beat_max_loop_interval = 5  # Check schedule every 5 seconds (default)
\`\`\`

---

## Running Beat in Production

### Option 1: Separate Process (Recommended)

\`\`\`bash
# Beat process (ONE instance only!)
celery -A tasks beat \\
    --loglevel=info \\
    --pidfile=/var/run/celerybeat.pid \\
    --schedule=/var/run/celerybeat-schedule

# Worker processes (multiple OK)
celery -A tasks worker --loglevel=info --concurrency=4
\`\`\`

### Option 2: Systemd Service

\`\`\`ini
# /etc/systemd/system/celerybeat.service
[Unit]
Description=Celery Beat Service
After=network.target

[Service]
Type=simple
User=celery
Group=celery
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/venv/bin/celery -A tasks beat \\
    --loglevel=info \\
    --pidfile=/var/run/celery/beat.pid \\
    --schedule=/var/run/celery/beat-schedule
Restart=always

[Install]
WantedBy=multi-user.target
\`\`\`

\`\`\`bash
# Enable and start
sudo systemctl enable celerybeat
sudo systemctl start celerybeat

# Check status
sudo systemctl status celerybeat

# View logs
sudo journalctl -u celerybeat -f
\`\`\`

### Option 3: Docker Compose

\`\`\`yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  celery-worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    volumes:
      - .:/app
    depends_on:
      - redis

  celery-beat:
    build: .
    command: celery -A tasks beat --loglevel=info
    volumes:
      - .:/app
    depends_on:
      - redis
    # IMPORTANT: Only 1 instance!
    deploy:
      replicas: 1
\`\`\`

### Option 4: Kubernetes

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-beat
spec:
  replicas: 1  # MUST be 1!
  selector:
    matchLabels:
      app: celery-beat
  template:
    metadata:
      labels:
        app: celery-beat
    spec:
      containers:
      - name: celery-beat
        image: myapp:latest
        command: ["celery", "-A", "tasks", "beat", "--loglevel=info"]
        env:
        - name: CELERY_BROKER_URL
          value: "redis://redis:6379/0"
\`\`\`

---

## Monitoring Periodic Tasks

\`\`\`python
"""
Monitoring Beat Tasks
"""

from celery import Celery
from celery.signals import beat_init, celeryd_init
import logging

app = Celery('myapp', broker='redis://localhost:6379/0')
logger = logging.getLogger(__name__)

@beat_init.connect
def on_beat_init(sender, **kwargs):
    """Called when Beat starts"""
    logger.info("Celery Beat started")

# Track task execution
from celery.signals import task_success, task_failure

@task_success.connect
def task_succeeded(sender=None, result=None, **kwargs):
    """Log successful periodic task"""
    if sender.request.id:
        logger.info(f"Periodic task succeeded: {sender.name}")

@task_failure.connect
def task_failed(sender=None, exception=None, **kwargs):
    """Alert on periodic task failure"""
    logger.error(f"Periodic task failed: {sender.name}, Error: {exception}")
    # Send alert to ops team
    alert_ops_team(f"Periodic task {sender.name} failed")

# Health check periodic task
@app.task
def health_check():
    """Periodic health check"""
    try:
        # Check database
        db.execute("SELECT 1")
        
        # Check external services
        response = requests.get("https://api.example.com/health", timeout=5)
        response.raise_for_status()
        
        # Check queue depth
        queue_depth = get_queue_depth()
        if queue_depth > 10000:
            alert_ops_team(f"Queue depth high: {queue_depth}")
        
        logger.info("Health check passed")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        alert_ops_team(f"Health check failed: {e}")
        raise
\`\`\`

---

## Best Practices

\`\`\`python
"""
Celery Beat Best Practices
"""

from celery import Celery
from celery.schedules import crontab

app = Celery('myapp', broker='redis://localhost:6379/0')

app.conf.beat_schedule = {
    # ✅ GOOD: Set expiration to prevent task pile-up
    'health-check': {
        'task': 'tasks.health_check',
        'schedule': 300.0,  # Every 5 minutes
        'options': {
            'expires': 240  # Expire if not run within 4 minutes
        }
    },
    
    # ✅ GOOD: Route to specific queue
    'generate-reports': {
        'task': 'tasks.generate_reports',
        'schedule': crontab(hour=0, minute=0),
        'options': {
            'queue': 'reports',  # Dedicated queue
            'priority': 9  # High priority
        }
    },
    
    # ✅ GOOD: Add task arguments
    'cleanup-old-data': {
        'task': 'tasks.cleanup',
        'schedule': crontab(hour=3, minute=0),
        'kwargs': {'days_old': 90, 'dry_run': False}
    },
    
    # ✅ GOOD: Make tasks idempotent
    'daily-sync': {
        'task': 'tasks.sync_data',  # Idempotent task
        'schedule': crontab(hour=2, minute=0),
    },
}

# Idempotent periodic task
@app.task
def sync_data():
    """Idempotent: Safe to run multiple times"""
    # Check if already synced today
    if is_synced_today():
        logger.info("Already synced today, skipping")
        return
    
    # Perform sync
    perform_sync()
    
    # Mark as synced
    mark_synced_today()

# ❌ BAD: Non-idempotent task
@app.task
def send_daily_email():
    """Non-idempotent: Sends duplicate emails if run multiple times"""
    send_email_to_all_users("Daily digest")  # ❌ Sends duplicates!

# ✅ GOOD: Idempotent version
@app.task
def send_daily_email_idempotent():
    """Idempotent: Checks if already sent"""
    if email_sent_today("daily_digest"):
        return  # Already sent
    
    send_email_to_all_users("Daily digest")
    mark_email_sent_today("daily_digest")
\`\`\`

---

## Common Patterns

### 1. Time-Zone Aware Schedules

\`\`\`python
app.conf.timezone = 'America/New_York'
app.conf.enable_utc = True

app.conf.beat_schedule = {
    'morning-report': {
        'task': 'tasks.morning_report',
        'schedule': crontab(hour=9, minute=0),  # 9 AM EST
    },
}
\`\`\`

### 2. One-Time Scheduled Task

\`\`\`python
from datetime import datetime, timedelta

# Schedule task to run once in future
eta = datetime.utcnow() + timedelta(hours=1)
send_reminder.apply_async(args=['user@example.com'], eta=eta)
\`\`\`

### 3. Dynamic Schedule Modification

\`\`\`python
# With django-celery-beat
from django_celery_beat.models import PeriodicTask

task = PeriodicTask.objects.get(name='health-check')
task.enabled = False  # Disable during maintenance
task.save()

# Re-enable after maintenance
task.enabled = True
task.save()
\`\`\`

---

## Summary

**Key Concepts:**
- **Celery Beat**: Scheduler for periodic tasks
- **Schedule Types**: Interval (seconds), Crontab (cron-like), Solar (sunrise/sunset)
- **ONE Beat instance**: Multiple Beats = duplicate tasks
- **Database Backend**: django-celery-beat for dynamic schedules

**Best Practices:**
- Run ONE Beat instance in production
- Set task expiration (\`expires\`) to prevent pile-up
- Make periodic tasks idempotent
- Route to specific queues
- Use database backend for dynamic schedules
- Monitor Beat health
- Use systemd/Docker for production deployment

**Next Section:** Task results and state management! 📊
`,
};
