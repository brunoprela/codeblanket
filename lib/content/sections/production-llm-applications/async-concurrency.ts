export const asyncConcurrencyContent = `
# Async & Concurrency

## Introduction

Large Language Model APIs are inherently slow compared to traditional operations - a single API call can take anywhere from 1 to 30+ seconds depending on the model and response length. When building production applications that serve multiple users or process multiple requests, handling these slow operations efficiently becomes critical. This is where async programming and concurrency patterns become essential.

In this section, we'll explore how to use Python's async/await syntax, handle concurrent LLM requests efficiently, manage backpressure, and implement patterns that maximize throughput while maintaining responsiveness. We'll cover everything from basic asyncio concepts to advanced patterns for production LLM applications.

## Why Async Matters for LLM Applications

Traditional synchronous programming blocks on I/O operations. When you make an LLM API call synchronously, your entire process waits for the response. This is fine for single requests, but becomes a bottleneck when:

**Multiple Users**: You need to handle requests from many users simultaneously

**Batch Processing**: You're processing hundreds or thousands of documents

**Multi-Step Workflows**: Your application makes multiple sequential LLM calls

**Real-Time Requirements**: Users expect responsive interfaces despite slow AI operations

**Cost Optimization**: You want to maximize throughput per server instance

### Synchronous vs Asynchronous Example

\`\`\`python
import time
import openai
from typing import List

# SYNCHRONOUS (Slow): Processes one request at a time
def generate_sync(prompts: List[str]) -> List[str]:
    """
    Synchronous generation - processes sequentially.
    For 10 prompts @ 3 seconds each = 30 seconds total.
    """
    results = []
    
    for prompt in prompts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        results.append(response.choices[0].message.content)
    
    return results


# ASYNCHRONOUS (Fast): Processes many requests concurrently
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_async(prompts: List[str]) -> List[str]:
    """
    Asynchronous generation - processes concurrently.
    For 10 prompts @ 3 seconds each = ~3 seconds total (limited by max concurrency).
    """
    async def generate_one(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    # Run all prompts concurrently
    tasks = [generate_one(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    return results


# Test the difference
prompts = [f"Tell me a fact about {topic}" 
           for topic in ["Python", "JavaScript", "Go", "Rust", "Java"]]

# Synchronous: ~15 seconds
start = time.time()
results_sync = generate_sync(prompts)
print(f"Sync took: {time.time() - start:.2f}s")

# Asynchronous: ~3 seconds (5x faster!)
start = time.time()
results_async = asyncio.run(generate_async(prompts))
print(f"Async took: {time.time() - start:.2f}s")
\`\`\`

## Python Asyncio Fundamentals

Before diving into LLM-specific patterns, let's review asyncio basics.

### Async/Await Syntax

\`\`\`python
import asyncio

# Define an async function with 'async def'
async def async_function():
    """An asynchronous function."""
    print("Starting async operation")
    
    # 'await' pauses execution until the operation completes
    # Other tasks can run during this pause
    await asyncio.sleep(1)
    
    print("Async operation complete")
    return "Result"


# Run an async function
result = asyncio.run(async_function())
\`\`\`

### Running Multiple Tasks Concurrently

\`\`\`python
async def task_a():
    await asyncio.sleep(2)
    return "Task A complete"

async def task_b():
    await asyncio.sleep(1)
    return "Task B complete"

async def run_concurrent():
    # Method 1: asyncio.gather (preserves order)
    results = await asyncio.gather(
        task_a(),
        task_b()
    )
    print(results)  # ['Task A complete', 'Task B complete']
    
    # Method 2: asyncio.create_task (more control)
    task_a_obj = asyncio.create_task(task_a())
    task_b_obj = asyncio.create_task(task_b())
    
    result_a = await task_a_obj
    result_b = await task_b_obj

asyncio.run(run_concurrent())
\`\`\`

### Error Handling in Async

\`\`\`python
async def safe_gather(*tasks):
    """
    Gather with error handling - one failure doesn't cancel others.
    """
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate successful results from exceptions
    successes = []
    failures = []
    
    for result in results:
        if isinstance(result, Exception):
            failures.append(result)
        else:
            successes.append(result)
    
    return successes, failures


async def unreliable_task(task_id: int):
    """Simulates a task that might fail."""
    await asyncio.sleep(0.1)
    if task_id == 3:
        raise ValueError(f"Task {task_id} failed!")
    return f"Task {task_id} success"


async def main():
    tasks = [unreliable_task(i) for i in range(5)]
    successes, failures = await safe_gather(*tasks)
    
    print(f"Successful: {len(successes)}")  # 4
    print(f"Failed: {len(failures)}")        # 1

asyncio.run(main())
\`\`\`

## Async LLM API Calls

Most LLM providers offer async clients for efficient concurrent operations.

### OpenAI Async Client

\`\`\`python
import asyncio
from openai import AsyncOpenAI
import time
from typing import List, Dict

client = AsyncOpenAI()

async def generate_completion(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Generate a single completion asynchronously.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return None


async def generate_batch(prompts: List[str]) -> List[Dict]:
    """
    Generate completions for multiple prompts concurrently.
    """
    start_time = time.time()
    
    # Create tasks for all prompts
    tasks = [generate_completion(prompt) for prompt in prompts]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    
    # Format results
    formatted_results = []
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        formatted_results.append({
            'prompt': prompt,
            'result': result if not isinstance(result, Exception) else None,
            'error': str(result) if isinstance(result, Exception) else None
        })
    
    print(f"Processed {len(prompts)} prompts in {elapsed:.2f}s")
    print(f"Average time per prompt: {elapsed/len(prompts):.2f}s")
    
    return formatted_results


# Example usage
async def main():
    prompts = [
        "Write a haiku about programming",
        "Explain quantum computing in one sentence",
        "What is the capital of France?",
        "Generate a random interesting fact"
    ]
    
    results = await generate_batch(prompts)
    
    for item in results:
        print(f"\\nPrompt: {item['prompt']}")
        print(f"Result: {item['result']}")

asyncio.run(main())
\`\`\`

### Anthropic Claude Async

\`\`\`python
import anthropic
import asyncio

client = anthropic.AsyncAnthropic()

async def generate_with_claude(prompt: str, max_tokens: int = 1024):
    """Generate with Claude asynchronously."""
    message = await client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text


async def parallel_claude_calls(prompts: List[str]):
    """Make multiple Claude API calls in parallel."""
    tasks = [generate_with_claude(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return results
\`\`\`

## Controlled Concurrency with Semaphores

When processing many requests, you don't want unlimited concurrency - this can:
- Overwhelm the LLM provider and trigger rate limits
- Consume too much memory
- Make debugging difficult

Use semaphores to limit concurrency:

\`\`\`python
import asyncio
from openai import AsyncOpenAI
from typing import List

client = AsyncOpenAI()

class ConcurrencyLimiter:
    """Limit concurrent LLM API calls."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'in_progress': 0
        }
    
    async def generate(self, prompt: str) -> dict:
        """
        Generate completion with concurrency limiting.
        """
        async with self.semaphore:  # Blocks if max concurrent reached
            self.stats['in_progress'] += 1
            self.stats['total'] += 1
            
            try:
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                self.stats['successful'] += 1
                result = response.choices[0].message.content
                
                return {
                    'prompt': prompt,
                    'result': result,
                    'success': True
                }
            
            except Exception as e:
                self.stats['failed'] += 1
                return {
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                }
            
            finally:
                self.stats['in_progress'] -= 1
    
    async def process_batch(self, prompts: List[str]) -> List[dict]:
        """
        Process a batch of prompts with controlled concurrency.
        """
        print(f"Processing {len(prompts)} prompts with max {self.semaphore._value} concurrent")
        
        tasks = [self.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        print(f"\\nStats:")
        print(f"  Total: {self.stats['total']}")
        print(f"  Successful: {self.stats['successful']}")
        print(f"  Failed: {self.stats['failed']}")
        
        return results


async def main():
    # Process 100 prompts but max 10 at a time
    limiter = ConcurrencyLimiter(max_concurrent=10)
    
    prompts = [f"Tell me about topic {i}" for i in range(100)]
    
    results = await limiter.process_batch(prompts)
    
    print(f"\\nCompleted {len(results)} generations")

asyncio.run(main())
\`\`\`

## Rate Limiting with Token Bucket

Implement sophisticated rate limiting to avoid hitting provider limits:

\`\`\`python
import asyncio
import time
from typing import Optional

class TokenBucket:
    """
    Token bucket rate limiter for async operations.
    Allows burst traffic while maintaining average rate.
    """
    
    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from bucket.
        Waits if not enough tokens available.
        """
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                
                # Refill tokens based on elapsed time
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                # Release lock while waiting
                await asyncio.sleep(wait_time)


class RateLimitedClient:
    """LLM client with rate limiting."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.client = AsyncOpenAI()
        # Convert to requests per second
        rate = requests_per_minute / 60
        self.rate_limiter = TokenBucket(rate=rate, capacity=rate * 10)
    
    async def generate(self, prompt: str) -> str:
        """Generate with rate limiting."""
        # Acquire token (waits if rate limit reached)
        await self.rate_limiter.acquire()
        
        # Make API call
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content


async def test_rate_limiting():
    """Test rate limiting with 60 requests per minute."""
    client = RateLimitedClient(requests_per_minute=60)
    
    start = time.time()
    
    # Try to make 10 requests
    tasks = [client.generate(f"Prompt {i}") for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"Made {len(results)} requests in {elapsed:.2f}s")
    print(f"Rate: {len(results)/elapsed:.2f} req/s")

asyncio.run(test_rate_limiting())
\`\`\`

## Async Queue Processing

Use async queues for producer-consumer patterns:

\`\`\`python
import asyncio
from asyncio import Queue
from openai import AsyncOpenAI
import logging

client = AsyncOpenAI()

async def producer(queue: Queue, prompts: List[str]):
    """
    Producer: Add prompts to queue.
    """
    for prompt in prompts:
        await queue.put(prompt)
        logging.info(f"Added prompt to queue: {prompt[:50]}...")
    
    # Signal completion
    await queue.put(None)


async def consumer(
    queue: Queue,
    consumer_id: int,
    results: List
):
    """
    Consumer: Process prompts from queue.
    """
    while True:
        prompt = await queue.get()
        
        # Check for completion signal
        if prompt is None:
            queue.task_done()
            await queue.put(None)  # Pass signal to other consumers
            break
        
        try:
            logging.info(f"Consumer {consumer_id} processing: {prompt[:50]}...")
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = {
                'prompt': prompt,
                'result': response.choices[0].message.content,
                'consumer_id': consumer_id
            }
            
            results.append(result)
            logging.info(f"Consumer {consumer_id} completed processing")
        
        except Exception as e:
            logging.error(f"Consumer {consumer_id} error: {str(e)}")
            results.append({
                'prompt': prompt,
                'error': str(e),
                'consumer_id': consumer_id
            })
        
        finally:
            queue.task_done()


async def process_with_queue(
    prompts: List[str],
    num_consumers: int = 3
):
    """
    Process prompts using producer-consumer pattern.
    """
    queue = Queue(maxsize=100)
    results = []
    
    # Start producer
    producer_task = asyncio.create_task(producer(queue, prompts))
    
    # Start consumers
    consumer_tasks = [
        asyncio.create_task(consumer(queue, i, results))
        for i in range(num_consumers)
    ]
    
    # Wait for producer to finish
    await producer_task
    
    # Wait for all items to be processed
    await queue.join()
    
    # Wait for all consumers to finish
    await asyncio.gather(*consumer_tasks)
    
    return results


async def main():
    logging.basicConfig(level=logging.INFO)
    
    prompts = [f"Generate content about topic {i}" for i in range(20)]
    
    results = await process_with_queue(prompts, num_consumers=5)
    
    print(f"\\nProcessed {len(results)} prompts")
    
    # Show distribution across consumers
    from collections import Counter
    consumer_distribution = Counter(r.get('consumer_id') for r in results)
    print(f"Consumer distribution: {dict(consumer_distribution)}")

asyncio.run(main())
\`\`\`

## Handling Backpressure

Backpressure occurs when requests arrive faster than you can process them. Handle it gracefully:

\`\`\`python
import asyncio
from asyncio import Queue, QueueFull
from typing import Optional
import logging

class BackpressureHandler:
    """Handle backpressure in async processing."""
    
    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 10
    ):
        self.queue = Queue(maxsize=max_queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            'accepted': 0,
            'rejected': 0,
            'processed': 0,
            'failed': 0
        }
    
    async def submit(self, prompt: str, timeout: float = 1.0) -> dict:
        """
        Submit a request with backpressure handling.
        
        Returns immediately if queue is full (rather than waiting).
        """
        try:
            # Try to add to queue with timeout
            await asyncio.wait_for(
                self.queue.put(prompt),
                timeout=timeout
            )
            
            self.stats['accepted'] += 1
            
            return {
                'accepted': True,
                'position': self.queue.qsize()
            }
        
        except asyncio.TimeoutError:
            # Queue is full and timeout reached
            self.stats['rejected'] += 1
            
            return {
                'accepted': False,
                'reason': 'queue_full',
                'queue_size': self.queue.qsize()
            }
    
    async def process_worker(self):
        """Worker that processes queue items."""
        while True:
            prompt = await self.queue.get()
            
            if prompt is None:  # Shutdown signal
                self.queue.task_done()
                break
            
            async with self.semaphore:
                try:
                    # Process the prompt
                    result = await self._generate(prompt)
                    self.stats['processed'] += 1
                
                except Exception as e:
                    logging.error(f"Processing failed: {str(e)}")
                    self.stats['failed'] += 1
                
                finally:
                    self.queue.task_done()
    
    async def _generate(self, prompt: str):
        """Generate completion."""
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            **self.stats,
            'queue_size': self.queue.qsize(),
            'rejection_rate': self.stats['rejected'] / max(1, self.stats['accepted'] + self.stats['rejected'])
        }


async def test_backpressure():
    """Test backpressure handling."""
    handler = BackpressureHandler(max_queue_size=10, max_concurrent=3)
    
    # Start workers
    workers = [
        asyncio.create_task(handler.process_worker())
        for _ in range(3)
    ]
    
    # Simulate high request rate
    submit_results = []
    for i in range(50):
        result = await handler.submit(f"Prompt {i}")
        submit_results.append(result)
        
        # Small delay between requests
        await asyncio.sleep(0.01)
    
    # Wait for processing to complete
    await handler.queue.join()
    
    # Shutdown workers
    for _ in workers:
        await handler.queue.put(None)
    await asyncio.gather(*workers)
    
    # Print stats
    stats = handler.get_stats()
    print(f"\\nBackpressure Test Results:")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Rejection Rate: {stats['rejection_rate']:.2%}")

asyncio.run(test_backpressure())
\`\`\`

## Timeouts and Cancellation

Handle long-running operations with timeouts:

\`\`\`python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_with_timeout(
    prompt: str,
    timeout: float = 30.0
) -> Optional[str]:
    """
    Generate with timeout.
    Cancels if generation takes too long.
    """
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            ),
            timeout=timeout
        )
        
        return response.choices[0].message.content
    
    except asyncio.TimeoutError:
        logging.warning(f"Generation timed out after {timeout}s")
        return None
    
    except asyncio.CancelledError:
        logging.info("Generation was cancelled")
        raise  # Re-raise to propagate cancellation


async def cancellable_generation(prompt: str):
    """Generation that can be cancelled externally."""
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    except asyncio.CancelledError:
        # Cleanup on cancellation
        logging.info("Cleaning up cancelled generation")
        raise


async def test_cancellation():
    """Test task cancellation."""
    task = asyncio.create_task(
        cancellable_generation("Write a very long story...")
    )
    
    # Let it run for 2 seconds then cancel
    await asyncio.sleep(2)
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled successfully")

asyncio.run(test_cancellation())
\`\`\`

## Thread Pool for CPU-Bound Tasks

For CPU-intensive preprocessing or postprocessing:

\`\`\`python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

def cpu_intensive_task(text: str) -> str:
    """
    Simulates CPU-intensive preprocessing.
    (e.g., tokenization, embedding calculation, etc.)
    """
    # Simulate heavy computation
    time.sleep(0.1)
    return text.upper()


async def process_with_threadpool(texts: List[str]):
    """
    Process CPU-intensive tasks in thread pool
    while keeping async code non-blocking.
    """
    loop = asyncio.get_event_loop()
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Run CPU-intensive tasks in threads
        tasks = [
            loop.run_in_executor(executor, cpu_intensive_task, text)
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks)
    
    return results


async def main():
    texts = [f"text {i}" for i in range(20)]
    
    start = time.time()
    results = await process_with_threadpool(texts)
    elapsed = time.time() - start
    
    print(f"Processed {len(results)} texts in {elapsed:.2f}s")

asyncio.run(main())
\`\`\`

## FastAPI Integration

Integrate async patterns with FastAPI for production APIs:

\`\`\`python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio

app = FastAPI()
client = AsyncOpenAI()

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"

# Async endpoint
@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """
    Async endpoint for generation.
    Handles request asynchronously without blocking.
    """
    response = await client.chat.completions.create(
        model=request.model,
        messages=[{"role": "user", "content": request.prompt}]
    )
    
    return {
        "result": response.choices[0].message.content,
        "model": request.model
    }


# Background task pattern
@app.post("/generate-background")
async def generate_background(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit generation as background task.
    Returns immediately with task ID.
    """
    task_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        process_generation,
        task_id,
        request.prompt,
        request.model
    )
    
    return {
        "task_id": task_id,
        "status": "processing"
    }


async def process_generation(task_id: str, prompt: str, model: str):
    """Process generation in background."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        
        # Store result in database
        save_result(task_id, result)
    
    except Exception as e:
        logging.error(f"Background task {task_id} failed: {str(e)}")
        save_error(task_id, str(e))
\`\`\`

## Best Practices

1. **Use async for I/O-bound operations** (API calls, database queries)

2. **Limit concurrency** with semaphores to avoid overwhelming services

3. **Implement rate limiting** to stay within provider limits

4. **Handle backpressure** gracefully when load exceeds capacity

5. **Set timeouts** on all async operations to prevent hanging

6. **Use connection pools** for database and HTTP clients

7. **Monitor queue depth** and adjust worker count dynamically

8. **Implement graceful shutdown** to complete in-flight requests

9. **Use structured logging** with request IDs for debugging async flows

10. **Test under load** to find concurrency limits and bottlenecks

Async programming enables your LLM application to handle many users efficiently while maintaining responsiveness. Master these patterns to build high-performance production systems.
`;
