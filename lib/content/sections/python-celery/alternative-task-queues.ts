export const alternativeTaskQueues = {
  title: 'Alternative Task Queues (RQ, Dramatiq, Huey)',
  id: 'alternative-task-queues',
  content: `
# Alternative Task Queues

## Introduction

While Celery is powerful, it's not the only option. **Alternative task queues** offer simpler APIs and different trade-offs. This section compares **RQ** (Redis Queue), **Dramatiq**, and **Huey** as alternatives to Celery.

**Decision Factors:**
- Simplicity vs features
- Setup time vs scalability
- Community size vs ease of use

---

## RQ (Redis Queue)

**RQ** is a simple, lightweight task queue for Python using Redis.

\`\`\`python
# Installation
pip install rq

# Basic usage
from redis import Redis
from rq import Queue

redis_conn = Redis()
q = Queue (connection=redis_conn)

def send_email (email):
    print(f"Sending email to {email}")
    return "sent"

# Enqueue task
job = q.enqueue (send_email, "user@example.com")

# Check status
print(job.is_finished)
print(job.result)
\`\`\`

**Pros:**
✅ Very simple API
✅ Quick setup (5 minutes)
✅ Good for MVPs

**Cons:**
❌ Redis only
❌ No periodic tasks
❌ Limited features

---

## Dramatiq

**Dramatiq** focuses on fast, reliable processing.

\`\`\`python
# Installation
pip install dramatiq[redis]

# Basic usage
import dramatiq
from dramatiq.brokers.redis import RedisBroker

broker = RedisBroker (host="localhost")
dramatiq.set_broker (broker)

@dramatiq.actor
def send_email (email):
    print(f"Sending email to {email}")

# Send task
send_email.send("user@example.com")
\`\`\`

**Pros:**
✅ Fast and reliable
✅ Automatic retries
✅ Redis + RabbitMQ support

**Cons:**
❌ Smaller community
❌ Fewer integrations

---

## Huey

**Huey** is lightweight with Flask/Django integration.

\`\`\`python
# Installation
pip install huey

# Basic usage
from huey import RedisHuey

huey = RedisHuey('myapp')

@huey.task()
def send_email (email):
    print(f"Sending email to {email}")

# Execute
send_email("user@example.com")

# Periodic tasks
from huey import crontab

@huey.periodic_task (crontab (minute='0', hour='*/3'))
def backup():
    perform_backup()
\`\`\`

**Pros:**
✅ Lightweight
✅ Periodic tasks built-in
✅ Good for Flask/Django

**Cons:**
❌ Limited scalability
❌ Smaller ecosystem

---

## Comparison Summary

| Feature | Celery | RQ | Dramatiq | Huey |
|---------|--------|-----|----------|------|
| Setup | Complex | Simple | Medium | Simple |
| Periodic Tasks | ✅ | ❌ | ❌ | ✅ |
| Brokers | Multiple | Redis only | Redis + RabbitMQ | Redis |
| Best For | Enterprise | MVPs | Fast processing | Flask/Django |

---

## When to Migrate to Celery

**Migrate from RQ/Huey/Dramatiq to Celery when:**
- Need complex workflows (chains, chords)
- Scale exceeds 100K tasks/day
- Need better monitoring (Flower)
- Require RabbitMQ reliability

**Migration Strategy:**
1. Install Celery
2. Dual-run (old + new)
3. Migrate gradually
4. Deprecate old queue

---

## Summary

**RQ**: Simple, fast, good for MVPs
**Dramatiq**: Reliable, auto-retries, fast
**Huey**: Lightweight, periodic tasks, Flask/Django
**Celery**: Enterprise, complex workflows, scalable

**Recommendation**: Start simple (RQ/Huey), migrate to Celery when needed.

**Congratulations! You've completed Module 7: Celery & Distributed Task Processing! 🎉**
`,
};
