export const queueSystemsBackgroundJobsContent = `
# Queue Systems & Background Jobs

## Introduction

Queue systems are the backbone of scalable LLM applications. They decouple request handling from processing, enable asynchronous workflows, provide reliability through retries, and allow you to scale processing independently from your API layer. In this section, we'll explore how to implement production-ready queue systems for LLM workloads.

When you're building an LLM application that needs to process thousands of documents, generate content for many users, or handle long-running multi-step workflows, queue systems become essential. They transform your architecture from "make the user wait for AI to finish" to "accept the request immediately, process in the background, and notify when complete."

We'll cover Celery (Python\'s most popular task queue), Redis Queue (RQ), RabbitMQ patterns, job priorities, monitoring, and failure handling. By the end, you'll know how to build robust background processing systems that can handle production workloads reliably.

## Why Queue Systems Matter for LLM Apps

LLM API calls are slow and expensive. A single GPT-4 API call can take 10-30 seconds and cost several cents. When you're building production applications, you need to:

**Accept Requests Immediately**: Users shouldn't wait 30 seconds for a response. Accept their request, return a task ID, and process in the background.

**Handle Failures Gracefully**: If an LLM API call fails (rate limit, timeout, API error), you need to retry without bothering the user.

**Process at Scale**: When you have 1000 documents to process, you want to process them concurrently across multiple workers, not sequentially.

**Prioritize Work**: Premium users or urgent tasks should be processed first, while batch jobs can wait.

**Track Progress**: Users need to know "your document is 45% processed" rather than just waiting in silence.

**Control Costs**: Rate limit API calls to avoid surprise bills, and batch operations during off-peak hours for cheaper pricing.

### Synchronous vs Queue-Based Architecture

\`\`\`python
# SYNCHRONOUS: User waits for LLM to finish
@app.post("/analyze-document")
def analyze_sync (document_id: str):
    # User\'s HTTP request is open, waiting...
    document = get_document (document_id)
    
    # This blocks for 30 seconds
    analysis = call_llm_api (document.content)
    
    # Finally return after 30 seconds
    return {"analysis": analysis}


# QUEUE-BASED: User gets immediate response
@app.post("/analyze-document")
def analyze_async (document_id: str):
    # Submit job to queue
    task = analyze_document_task.delay (document_id)
    
    # Return immediately (< 100ms)
    return {
        "task_id": task.id,
        "status": "queued",
        "check_url": f"/tasks/{task.id}"
    }

# Worker processes the job in background
@celery_app.task
def analyze_document_task (document_id: str):
    document = get_document (document_id)
    analysis = call_llm_api (document.content)  # Takes 30 seconds, but user isn't waiting
    save_analysis (document_id, analysis)
    send_notification (document.user_id, "Analysis complete!")
\`\`\`

## Celery: Python's Distributed Task Queue

Celery is the most popular task queue for Python. It\'s battle-tested, feature-rich, and integrates well with Django, Flask, and FastAPI.

### Core Concepts

**Broker**: Message queue that stores tasks (Redis, RabbitMQ, AWS SQS)
**Workers**: Processes that pull tasks from the queue and execute them
**Backend**: Storage for task results (Redis, PostgreSQL, MongoDB)
**Tasks**: Functions decorated with @app.task that can be executed asynchronously
**Beat**: Scheduler for periodic tasks (cron-like functionality)

### Basic Celery Setup

\`\`\`python
# celery_app.py
from celery import Celery
import openai

# Initialize Celery
app = Celery(
    'llm_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minute hard limit
    task_soft_time_limit=570,  # 9.5 minute soft limit
    worker_prefetch_multiplier=1,  # Workers take 1 task at a time
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
)

@app.task (bind=True, max_retries=3)
def generate_content (self, prompt: str, model: str = "gpt-3.5-turbo"):
    """
    Generate content with retry logic.
    
    Args:
        self: Task instance (bind=True provides this)
        prompt: Text prompt for generation
        model: LLM model to use
    """
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Calling LLM API...'}
        )
        
        # Call LLM
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        
        # Update state
        self.update_state(
            state='SUCCESS',
            meta={'status': 'Complete', 'result': result}
        )
        
        return {
            'result': result,
            'tokens_used': response.usage.total_tokens,
            'model': model
        }
    
    except openai.error.RateLimitError as exc:
        # Retry with exponential backoff
        countdown = 60 * (2 ** self.request.retries)
        raise self.retry (exc=exc, countdown=countdown)
    
    except openai.error.APIError as exc:
        # Retry on API errors
        raise self.retry (exc=exc, countdown=30)
    
    except Exception as exc:
        # Log unexpected errors but don't retry
        logging.error (f"Task {self.request.id} failed: {str (exc)}")
        raise


# Start worker with:
# celery -A celery_app worker --loglevel=info --concurrency=4
\`\`\`

### Using Celery Tasks

\`\`\`python
# In your API or application code
from celery_app import generate_content

# Submit task
task = generate_content.delay("Write a haiku about Python", model="gpt-4")

# Get task ID
task_id = task.id

# Check task status
result = generate_content.AsyncResult (task_id)
print(result.state)  # PENDING, PROCESSING, SUCCESS, FAILURE

# Get result (blocks until complete)
if result.ready():
    if result.successful():
        data = result.get()
        print(data['result'])
    else:
        print(f"Task failed: {result.info}")

# Check progress for long tasks
if result.state == 'PROCESSING':
    print(result.info['status'])  # "Calling LLM API..."
\`\`\`

### FastAPI Integration

\`\`\`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery_app import generate_content, app as celery_app

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"

@app.post("/generate")
async def create_generation (request: GenerateRequest):
    """Submit generation task to queue."""
    # Submit to Celery
    task = generate_content.delay (request.prompt, request.model)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "check_url": f"/tasks/{task.id}"
    }

@app.get("/tasks/{task_id}")
async def get_task_status (task_id: str):
    """Check task status and get result."""
    task = celery_app.AsyncResult (task_id)
    
    if task.state == 'PENDING':
        return {
            "status": "pending",
            "message": "Task is waiting to be processed"
        }
    
    elif task.state == 'PROCESSING':
        return {
            "status": "processing",
            "progress": task.info.get('status', ')
        }
    
    elif task.state == 'SUCCESS':
        return {
            "status": "success",
            "result": task.info
        }
    
    elif task.state == 'FAILURE':
        return {
            "status": "failed",
            "error": str (task.info)
        }
    
    return {
        "status": task.state.lower()
    }

@app.delete("/tasks/{task_id}")
async def cancel_task (task_id: str):
    """Cancel a running task."""
    celery_app.control.revoke (task_id, terminate=True)
    return {"status": "cancelled"}
\`\`\`

## Task Chains and Workflows

Complex LLM workflows often require multiple steps. Celery provides patterns for chaining tasks.

### Sequential Chains

\`\`\`python
from celery import chain

@app.task
def extract_text (document_id: str):
    """Step 1: Extract text from document."""
    document = get_document (document_id)
    text = extract_text_from_pdf (document.path)
    return {'document_id': document_id, 'text': text}

@app.task
def summarize_text (data: dict):
    """Step 2: Summarize extracted text."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Summarize this text:\\n\\n{data['text']}"
        }]
    )
    data['summary'] = response.choices[0].message.content
    return data

@app.task
def save_results (data: dict):
    """Step 3: Save results to database."""
    update_document(
        data['document_id'],
        summary=data['summary']
    )
    return data

# Create a chain of tasks
workflow = chain(
    extract_text.s("doc_123"),
    summarize_text.s(),
    save_results.s()
)

# Execute the chain
result = workflow.apply_async()
\`\`\`

### Parallel Groups

\`\`\`python
from celery import group

@app.task
def analyze_sentiment (text: str):
    """Analyze sentiment of text."""
    # Call LLM
    return {"sentiment": "positive", "score": 0.8}

@app.task
def extract_entities (text: str):
    """Extract named entities."""
    # Call LLM
    return {"entities": ["Python", "Celery", "Redis"]}

@app.task
def categorize_text (text: str):
    """Categorize text."""
    # Call LLM
    return {"category": "Technology"}

# Run multiple analyses in parallel
job = group(
    analyze_sentiment.s("Python is great!"),
    extract_entities.s("Python is great!"),
    categorize_text.s("Python is great!")
)

result = job.apply_async()

# Wait for all to complete
results = result.get()
# [{"sentiment": "positive"}, {"entities": [...]}, {"category": "Technology"}]
\`\`\`

### Chords: Parallel then Callback

\`\`\`python
from celery import chord

@app.task
def process_chunk (text_chunk: str):
    """Process a single text chunk."""
    # Analyze chunk
    return analysis_result

@app.task
def combine_results (results: list):
    """Combine results from all chunks."""
    # Aggregate all chunk analyses
    return final_result

# Split text into chunks, process in parallel, then combine
chunks = split_text_into_chunks (long_text)

workflow = chord(
    group([process_chunk.s (chunk) for chunk in chunks]),
    combine_results.s()
)

result = workflow.apply_async()
\`\`\`

## Redis Queue (RQ): Simpler Alternative

RQ is a simpler queue system, perfect for smaller applications or when you don't need Celery\'s complexity.

\`\`\`python
# rq_tasks.py
from redis import Redis
from rq import Queue
import openai

# Connect to Redis
redis_conn = Redis (host='localhost', port=6379)

# Create queue
q = Queue (connection=redis_conn)

# Define task function (no decorator needed!)
def generate_content (prompt: str, model: str = "gpt-3.5-turbo"):
    """Simple RQ task."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Submit task
job = q.enqueue(
    generate_content,
    "Write a poem",
    model="gpt-4",
    job_timeout='5m'  # Task timeout
)

# Check status
print(job.get_status())  # queued, started, finished, failed

# Get result (blocks)
if job.is_finished:
    result = job.result
    print(result)

# Start worker with:
# rq worker --with-scheduler
\`\`\`

## Priority Queues

Handle urgent tasks before normal ones using priority queues.

\`\`\`python
from celery import Celery

app = Celery('app', broker='redis://localhost:6379/0')

# Configure multiple queues with priorities
app.conf.task_routes = {
    'tasks.urgent_task': {'queue': 'urgent'},
    'tasks.normal_task': {'queue': 'default'},
    'tasks.batch_task': {'queue': 'batch'},
}

@app.task
def urgent_task (data):
    """High priority task for premium users."""
    return process_urgently (data)

@app.task
def normal_task (data):
    """Normal priority task."""
    return process_normally (data)

@app.task
def batch_task (data):
    """Low priority batch processing."""
    return process_in_batch (data)

# Submit to specific queues
urgent_task.apply_async (args=["important"], queue='urgent')
normal_task.apply_async (args=["regular"], queue='default')
batch_task.apply_async (args=["can_wait"], queue='batch')

# Start workers with different queue priorities
# High priority worker (processes urgent first):
# celery -A app worker -Q urgent,default --loglevel=info

# Low priority worker (only processes batch):
# celery -A app worker -Q batch --loglevel=info
\`\`\`

## Job Retries and Dead Letter Queues

Handle failures gracefully with automatic retries and dead letter queues for permanently failed jobs.

\`\`\`python
from celery import Task
import openai
import logging

class LLMTask(Task):
    """Custom task class with LLM-specific error handling."""
    
    autoretry_for = (
        openai.error.RateLimitError,
        openai.error.APIError,
        openai.error.Timeout
    )
    retry_kwargs = {'max_retries': 5}
    retry_backoff = True  # Exponential backoff
    retry_backoff_max = 600  # Max 10 minutes between retries
    retry_jitter = True  # Add randomness to prevent thundering herd
    
    def on_failure (self, exc, task_id, args, kwargs, einfo):
        """Called when task fails after all retries."""
        logging.error (f"Task {task_id} permanently failed: {exc}")
        
        # Move to dead letter queue
        move_to_dead_letter_queue (task_id, exc, args, kwargs)
        
        # Notify admin
        send_alert (f"Task {task_id} failed permanently")
    
    def on_retry (self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        retry_count = self.request.retries
        logging.warning (f"Task {task_id} retry {retry_count}: {exc}")


@app.task (base=LLMTask)
def robust_generation (prompt: str):
    """Generation with comprehensive error handling."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    except openai.error.InvalidRequestError as e:
        # Don't retry invalid requests
        logging.error (f"Invalid request: {str (e)}")
        raise  # Fail immediately
    
    except Exception as e:
        # Retry other errors
        logging.error (f"Error in generation: {str (e)}")
        raise


def move_to_dead_letter_queue (task_id, error, args, kwargs):
    """Store permanently failed tasks for manual review."""
    dead_letter_db.insert({
        'task_id': task_id,
        'error': str (error),
        'args': args,
        'kwargs': kwargs,
        'failed_at': datetime.utcnow(),
        'status': 'dead'
    })
\`\`\`

## Progress Tracking

Track progress for long-running tasks so users know what's happening.

\`\`\`python
from celery import Task

@app.task (bind=True)
def process_large_document (self, document_id: str):
    """
    Process large document with progress updates.
    """
    document = get_document (document_id)
    pages = document.pages
    total_pages = len (pages)
    
    results = []
    
    for i, page in enumerate (pages):
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'current': i + 1,
                'total': total_pages,
                'percent': int((i + 1) / total_pages * 100),
                'status': f'Processing page {i + 1} of {total_pages}'
            }
        )
        
        # Process page
        page_result = process_page_with_llm (page)
        results.append (page_result)
    
    # Complete
    return {
        'document_id': document_id,
        'pages_processed': total_pages,
        'results': results
    }


# Check progress from API
@app.get("/tasks/{task_id}/progress")
def get_progress (task_id: str):
    """Get task progress."""
    task = app.AsyncResult (task_id)
    
    if task.state == 'PROCESSING':
        return {
            "status": "processing",
            "current": task.info.get('current', 0),
            "total": task.info.get('total', 0),
            "percent": task.info.get('percent', 0),
            "message": task.info.get('status', ')
        }
    
    return {"status": task.state.lower()}
\`\`\`

## Monitoring and Observability

Monitor queue health, worker performance, and task statistics.

\`\`\`python
from celery import Celery
from celery.events import EventReceiver
from kombu import Connection

def monitor_celery_events():
    """
    Monitor Celery events in real-time.
    """
    app = Celery('app', broker='redis://localhost:6379/0')
    
    def on_task_sent (event):
        """Task was sent to worker."""
        print(f"Task {event['uuid']} sent to worker")
    
    def on_task_started (event):
        """Task started processing."""
        print(f"Task {event['uuid']} started")
    
    def on_task_succeeded (event):
        """Task completed successfully."""
        print(f"Task {event['uuid']} succeeded in {event['runtime']}s")
    
    def on_task_failed (event):
        """Task failed."""
        print(f"Task {event['uuid']} failed: {event['exception']}")
    
    with Connection (app.broker_connection()) as connection:
        receiver = EventReceiver(
            connection,
            handlers={
                'task-sent': on_task_sent,
                'task-started': on_task_started,
                'task-succeeded': on_task_succeeded,
                'task-failed': on_task_failed,
            }
        )
        receiver.capture (limit=None, timeout=None, wakeup=True)


# Get queue stats
@app.get("/queue/stats")
def get_queue_stats():
    """Get statistics about queues and workers."""
    inspector = celery_app.control.inspect()
    
    stats = {
        'active_tasks': inspector.active(),
        'scheduled_tasks': inspector.scheduled(),
        'reserved_tasks': inspector.reserved(),
        'registered_tasks': inspector.registered(),
        'stats': inspector.stats(),
    }
    
    # Count tasks in each queue
    with redis_conn.pipeline() as pipe:
        pipe.llen('celery')  # default queue
        pipe.llen('urgent')
        pipe.llen('batch')
        lengths = pipe.execute()
    
    stats['queue_lengths'] = {
        'default': lengths[0],
        'urgent': lengths[1],
        'batch': lengths[2]
    }
    
    return stats
\`\`\`

## RabbitMQ for Advanced Routing

For complex routing and guaranteed delivery, use RabbitMQ as the broker.

\`\`\`python
# Configure Celery with RabbitMQ
app = Celery(
    'app',
    broker='amqp://guest:guest@localhost:5672//',
    backend='redis://localhost:6379/0'
)

# Advanced routing with exchanges
app.conf.task_routes = {
    'tasks.llm.*': {
        'exchange': 'llm',
        'exchange_type': 'topic',
        'routing_key': 'llm.generation'
    },
    'tasks.processing.*': {
        'exchange': 'processing',
        'exchange_type': 'direct',
        'routing_key': 'processing.pdf'
    }
}

# Priority support
@app.task (priority=9)  # 0-9, higher = more priority
def high_priority_task():
    pass

@app.task (priority=0)
def low_priority_task():
    pass
\`\`\`

## Best Practices

1. **Always set task timeouts** to prevent hung tasks from blocking workers

2. **Use \`task_acks_late=True\`** to ensure tasks aren't lost if worker crashes

3. **Implement idempotent tasks** that can be safely retried

4. **Monitor queue depth** and scale workers based on load

5. **Use separate queues** for different priorities and task types

6. **Set appropriate retry policies** with exponential backoff

7. **Track costs** by logging tokens used per task

8. **Implement dead letter queues** for permanently failed tasks

9. **Use progress updates** for long-running tasks

10. **Monitor worker health** and auto-restart failed workers

Queue systems transform your LLM application from a synchronous bottleneck to a scalable, resilient production system. Master these patterns to handle any workload.
`;
