export const batchProcessing = {
  title: 'Batch Processing for LLM Applications',
  content: `

# Batch Processing for LLM Applications

## Introduction

Batch processing is a powerful cost optimization strategy for LLM applications where requests don't need immediate responses. By accumulating requests and processing them together, you can:

- Reduce API costs by 40-60%
- Better manage rate limits
- Optimize resource utilization
- Handle traffic spikes more gracefully
- Enable cost-effective background processing

This section covers:
- When to use batch processing vs real-time
- Batch accumulation strategies
- Parallel batch execution
- Priority queues for mixed workloads
- Cost optimization through batching
- Production batch systems

---

## Real-Time vs Batch Processing

### When to Use Each

**Real-Time** (immediate response required):
- Chat applications
- Interactive code completion
- Live customer support
- Real-time translation

**Batch** (delayed response acceptable):
- Email summarization (process overnight)
- Content moderation (5-minute delay OK)
- Document analysis (hours OK)
- Report generation (scheduled)
- Data enrichment pipelines

\`\`\`python
from enum import Enum

class ProcessingMode(Enum):
    REALTIME = "realtime"     # <1s response
    NEARTIME = "neartime"     # 1-60s response
    BATCH = "batch"           # Minutes to hours

def choose_processing_mode(use_case: str) -> ProcessingMode:
    """Choose appropriate processing mode"""
    
    realtime_usecases = {
        "chat", "autocomplete", "live_support",
        "translation", "search"
    }
    
    neartime_usecases = {
        "content_moderation", "sentiment_analysis",
        "classification", "extraction"
    }
    
    batch_usecases = {
        "email_summary", "report_generation",
        "bulk_analysis", "data_enrichment",
        "scheduled_tasks"
    }
    
    if use_case in realtime_usecases:
        return ProcessingMode.REALTIME
    elif use_case in neartime_usecases:
        return ProcessingMode.NEARTIME
    else:
        return ProcessingMode.BATCH

# Examples
print(choose_processing_mode("chat"))                # REALTIME
print(choose_processing_mode("content_moderation"))  # NEARTIME  
print(choose_processing_mode("email_summary"))       # BATCH
\`\`\`

---

## Basic Batch Accumulation

\`\`\`python
import asyncio
from typing import List, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

@dataclass
class BatchRequest:
    id: str
    prompt: str
    model: str
    submitted_at: datetime = field(default_factory=datetime.now)
    priority: int = 0
    callback: Callable = None

class SimpleBatchProcessor:
    """Accumulate requests and process in batches"""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_wait_seconds: int = 60
    ):
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.queue = []
        self.processing = False
    
    async def submit(self, request: BatchRequest):
        """Submit request to batch queue"""
        self.queue.append(request)
        print(f"📝 Queued request {request.id} (queue size: {len(self.queue)})")
        
        # Check if we should process now
        should_process = (
            len(self.queue) >= self.batch_size or
            self._oldest_request_age() > self.max_wait_seconds
        )
        
        if should_process and not self.processing:
            await self.process_batch()
    
    def _oldest_request_age(self) -> float:
        """Get age of oldest request in seconds"""
        if not self.queue:
            return 0
        oldest = min(self.queue, key=lambda r: r.submitted_at)
        return (datetime.now() - oldest.submitted_at).total_seconds()
    
    async def process_batch(self):
        """Process accumulated batch"""
        if not self.queue or self.processing:
            return
        
        self.processing = True
        
        # Take requests for this batch
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]
        
        print(f"\n🔄 Processing batch of {len(batch)} requests")
        start_time = time.time()
        
        # Process all requests in parallel
        tasks = [self._process_single(req) for req in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        print(f"✅ Batch completed in {elapsed:.2f}s")
        print(f"   Avg time per request: {elapsed/len(batch):.2f}s\n")
        
        # Call callbacks
        for request, result in zip(batch, results):
            if request.callback and not isinstance(result, Exception):
                request.callback(result)
        
        self.processing = False
        
        # Process next batch if queue has items
        if self.queue:
            await self.process_batch()
        
        return results
    
    async def _process_single(self, request: BatchRequest):
        """Process single request"""
        print(f"  Processing {request.id}...")
        
        # Call LLM API
        response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        return {
            "id": request.id,
            "response": response.choices[0].message.content,
            "usage": response.usage._asdict()
        }

# Usage
processor = SimpleBatchProcessor(
    batch_size=10,
    max_wait_seconds=60
)

# Submit requests
for i in range(25):
    await processor.submit(BatchRequest(
        id=f"req_{i}",
        prompt=f"Summarize document {i}",
        model="gpt-3.5-turbo"
    ))

# Output:
# Queued request req_0 (queue size: 1)
# Queued request req_1 (queue size: 2)
# ...
# Queued request req_9 (queue size: 10)
#
# 🔄 Processing batch of 10 requests
#   Processing req_0...
#   Processing req_1...
#   ...
# ✅ Batch completed in 3.2s
#    Avg time per request: 0.32s
#
# (Batch 2 automatically starts)
\`\`\`

**Cost Savings**: Processing 10 requests in parallel vs serially saves time and allows for optimizations.

---

## Intelligent Batch Sizing

\`\`\`python
import statistics
from typing import List
import time

class AdaptiveBatchProcessor:
    """Dynamically adjust batch size based on performance"""
    
    def __init__(
        self,
        initial_batch_size: int = 10,
        min_batch_size: int = 5,
        max_batch_size: int = 50,
        target_latency_seconds: float = 5.0
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency_seconds
        
        # Performance tracking
        self.batch_latencies = []
        self.batch_sizes = []
    
    async def process_batch(self, requests: List[BatchRequest]):
        """Process batch and adjust size"""
        
        start_time = time.time()
        
        # Process requests in parallel
        tasks = [self._process_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        latency = time.time() - start_time
        
        # Record metrics
        self.batch_latencies.append(latency)
        self.batch_sizes.append(len(requests))
        
        # Keep only recent history
        if len(self.batch_latencies) > 20:
            self.batch_latencies = self.batch_latencies[-20:]
            self.batch_sizes = self.batch_sizes[-20:]
        
        # Adjust batch size
        self._adjust_batch_size(latency, len(requests))
        
        return results
    
    def _adjust_batch_size(self, latency: float, batch_size: int):
        """Dynamically adjust batch size based on performance"""
        
        # If we're consistently under target latency, increase batch size
        if len(self.batch_latencies) >= 3:
            recent_avg_latency = statistics.mean(self.batch_latencies[-3:])
            
            if recent_avg_latency < self.target_latency * 0.8:
                # We have headroom, increase batch size
                new_size = min(
                    int(self.current_batch_size * 1.2),
                    self.max_batch_size
                )
                if new_size > self.current_batch_size:
                    print(f"📈 Increasing batch size: {self.current_batch_size} → {new_size}")
                    self.current_batch_size = new_size
            
            elif recent_avg_latency > self.target_latency:
                # We're too slow, decrease batch size
                new_size = max(
                    int(self.current_batch_size * 0.8),
                    self.min_batch_size
                )
                if new_size < self.current_batch_size:
                    print(f"📉 Decreasing batch size: {self.current_batch_size} → {new_size}")
                    self.current_batch_size = new_size
    
    def get_recommended_batch_size(self) -> int:
        """Get current recommended batch size"""
        return self.current_batch_size
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        if not self.batch_latencies:
            return "No batches processed yet"
        
        avg_latency = statistics.mean(self.batch_latencies)
        avg_batch_size = statistics.mean(self.batch_sizes)
        total_requests = sum(self.batch_sizes)
        
        return f"""
Adaptive Batch Performance:
  Total Batches: {len(self.batch_latencies)}
  Total Requests: {total_requests}
  Avg Batch Size: {avg_batch_size:.1f}
  Avg Latency: {avg_latency:.2f}s
  Current Batch Size: {self.current_batch_size}
  Target Latency: {self.target_latency}s
"""

# Usage
processor = AdaptiveBatchProcessor(
    initial_batch_size=10,
    target_latency_seconds=5.0
)

# Process multiple batches - size auto-adjusts
for batch_num in range(10):
    batch_size = processor.get_recommended_batch_size()
    requests = [BatchRequest(f"req_{i}", f"prompt_{i}", "gpt-3.5-turbo") 
                for i in range(batch_size)]
    
    await processor.process_batch(requests)

print(processor.get_performance_report())

# Output might show:
# 📈 Increasing batch size: 10 → 12
# 📈 Increasing batch size: 12 → 14
# (finds optimal size around 15-20)
\`\`\`

---

## Priority Queue for Mixed Workloads

\`\`\`python
import heapq
from typing import List, Tuple
from enum import IntEnum

class Priority(IntEnum):
    CRITICAL = 0  # Process first
    HIGH = 1
    NORMAL = 2
    LOW = 3       # Process last

class PriorityBatchProcessor:
    """Batch processor with priority queues"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        # Min-heap for priority (lower number = higher priority)
        self.priority_queue = []
        self.counter = 0  # For stable sorting
    
    async def submit(self, request: BatchRequest):
        """Submit request with priority"""
        
        # Push to heap: (priority, counter, request)
        # Counter ensures FIFO for same priority
        heapq.heappush(
            self.priority_queue,
            (request.priority, self.counter, request)
        )
        self.counter += 1
        
        print(f"📝 Queued {request.id} (priority: {Priority(request.priority).name})")
        
        # Process if we have enough items
        if len(self.priority_queue) >= self.batch_size:
            await self.process_next_batch()
    
    async def process_next_batch(self):
        """Process next batch in priority order"""
        
        if not self.priority_queue:
            return
        
        # Take highest priority items
        batch = []
        for _ in range(min(self.batch_size, len(self.priority_queue))):
            priority, counter, request = heapq.heappop(self.priority_queue)
            batch.append(request)
        
        print(f"\n🔄 Processing batch of {len(batch)} requests")
        print(f"   Priorities: {[Priority(r.priority).name for r in batch]}")
        
        # Process in parallel
        tasks = [self._process_single(req) for req in batch]
        results = await asyncio.gather(*tasks)
        
        print(f"✅ Batch completed\n")
        
        return results
    
    async def _process_single(self, request: BatchRequest):
        """Process single request"""
        response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        return response.choices[0].message.content
    
    def get_queue_stats(self) -> Dict:
        """Get queue statistics by priority"""
        priority_counts = {p.name: 0 for p in Priority}
        
        for priority, _, _ in self.priority_queue:
            priority_counts[Priority(priority).name] += 1
        
        return {
            "total": len(self.priority_queue),
            "by_priority": priority_counts
        }

# Usage
processor = PriorityBatchProcessor(batch_size=5)

# Submit mixed priority requests
requests = [
    BatchRequest("req_1", "Critical task", "gpt-4", priority=Priority.CRITICAL),
    BatchRequest("req_2", "Low priority", "gpt-3.5-turbo", priority=Priority.LOW),
    BatchRequest("req_3", "Normal task", "gpt-3.5-turbo", priority=Priority.NORMAL),
    BatchRequest("req_4", "Critical task 2", "gpt-4", priority=Priority.CRITICAL),
    BatchRequest("req_5", "High priority", "gpt-3.5-turbo", priority=Priority.HIGH),
    BatchRequest("req_6", "Low priority 2", "gpt-3.5-turbo", priority=Priority.LOW),
]

for req in requests:
    await processor.submit(req)

# Output:
# 🔄 Processing batch of 5 requests
#    Priorities: ['CRITICAL', 'CRITICAL', 'HIGH', 'NORMAL', 'LOW']
# 
# (CRITICAL tasks processed first, LOW tasks last)
\`\`\`

---

## Cost-Optimized Batch Processing

\`\`\`python
from typing import List, Dict
import asyncio

class CostOptimizedBatchProcessor:
    """Optimize batching for minimum cost"""
    
    def __init__(
        self,
        batch_size: int = 20,
        use_cheaper_models_for_batch: bool = True
    ):
        self.batch_size = batch_size
        self.use_cheaper_models = use_cheaper_models_for_batch
        
        # Track costs
        self.total_cost = 0.0
        self.requests_processed = 0
    
    async def process_batch(self, requests: List[BatchRequest]):
        """Process batch with cost optimization"""
        
        print(f"\n💰 Cost-optimized batch processing ({len(requests)} requests)")
        
        # Strategy 1: Group by model for efficiency
        by_model = self._group_by_model(requests)
        
        # Strategy 2: Downgrade to cheaper models if acceptable
        if self.use_cheaper_models:
            by_model = self._optimize_model_selection(by_model)
        
        # Strategy 3: Process each model group in parallel
        all_results = []
        for model, model_requests in by_model.items():
            print(f"  Processing {len(model_requests)} requests with {model}")
            results = await self._process_model_batch(model, model_requests)
            all_results.extend(results)
        
        # Calculate cost
        batch_cost = self._calculate_batch_cost(all_results)
        self.total_cost += batch_cost
        self.requests_processed += len(requests)
        
        avg_cost = batch_cost / len(requests)
        print(f"  Batch cost: \${batch_cost:.4f} (\${avg_cost:.4f}/request)")
        print(f"  Total saved: \${self._estimate_savings():.2f}\n")
        
        return all_results
    
    def _group_by_model(self, requests: List[BatchRequest]) -> Dict[str, List[BatchRequest]]:
        """Group requests by model"""
        by_model = {}
        for req in requests:
            if req.model not in by_model:
                by_model[req.model] = []
            by_model[req.model].append(req)
        return by_model
    
    def _optimize_model_selection(
        self,
        by_model: Dict[str, List[BatchRequest]]
    ) -> Dict[str, List[BatchRequest]]:
        """Downgrade to cheaper models where appropriate"""
        
        optimized = {}
        
        for model, requests in by_model.items():
            # For batch processing, simpler tasks can use cheaper models
            if model == "gpt-4" and self._can_use_cheaper_model(requests):
                print(f"  ⚡ Downgrading {len(requests)} requests: gpt-4 → gpt-3.5-turbo")
                model = "gpt-3.5-turbo"
            
            if model not in optimized:
                optimized[model] = []
            optimized[model].extend(requests)
        
        return optimized
    
    def _can_use_cheaper_model(self, requests: List[BatchRequest]) -> bool:
        """Check if cheaper model is acceptable"""
        # Heuristic: if all prompts are short, use cheaper model
        avg_prompt_length = sum(len(r.prompt) for r in requests) / len(requests)
        return avg_prompt_length < 200  # Simplified heuristic
    
    async def _process_model_batch(
        self,
        model: str,
        requests: List[BatchRequest]
    ) -> List[Dict]:
        """Process all requests for a specific model in parallel"""
        
        tasks = []
        for req in requests:
            task = openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": req.prompt}]
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        results = []
        for req, response in zip(requests, responses):
            results.append({
                "id": req.id,
                "model": model,
                "response": response.choices[0].message.content,
                "usage": response.usage._asdict()
            })
        
        return results
    
    def _calculate_batch_cost(self, results: List[Dict]) -> float:
        """Calculate total cost for batch"""
        
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        total_cost = 0.0
        for result in results:
            model = result["model"]
            usage = result["usage"]
            
            input_cost = (usage["prompt_tokens"] / 1000) * pricing[model]["input"]
            output_cost = (usage["completion_tokens"] / 1000) * pricing[model]["output"]
            total_cost += input_cost + output_cost
        
        return total_cost
    
    def _estimate_savings(self) -> float:
        """Estimate cost savings vs sequential processing"""
        # Sequential processing might not benefit from model optimization
        sequential_cost = self.requests_processed * 0.001  # Rough estimate
        return sequential_cost - self.total_cost

# Usage
processor = CostOptimizedBatchProcessor(
    batch_size=20,
    use_cheaper_models_for_batch=True
)

# Submit mix of requests
requests = [
    BatchRequest(f"req_{i}", f"Short prompt {i}", "gpt-4")
    for i in range(20)
]

results = await processor.process_batch(requests)

# Output:
# 💰 Cost-optimized batch processing (20 requests)
#   ⚡ Downgrading 20 requests: gpt-4 → gpt-3.5-turbo
#   Processing 20 requests with gpt-3.5-turbo
#   Batch cost: $0.0150 (vs $0.6000 if using GPT-4)
#   Total saved: $0.58
\`\`\`

**Savings**: 95%+ cost reduction by batching + model optimization!

---

## Production Batch System with Celery

\`\`\`python
from celery import Celery, group
from kombu import Queue
import redis

# Initialize Celery
app = Celery('llm_batch_processor')
app.config_from_object({
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'task_routes': {
        'tasks.process_llm_batch': {'queue': 'batch'},
        'tasks.process_llm_realtime': {'queue': 'realtime'}
    }
})

# Define queues with priorities
app.conf.task_queues = (
    Queue('realtime', priority=10),
    Queue('batch', priority=1),
)

@app.task(bind=True, max_retries=3)
def process_llm_request(self, request_id: str, prompt: str, model: str):
    """Process individual LLM request"""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "request_id": request_id,
            "response": response.choices[0].message.content,
            "usage": response.usage._asdict()
        }
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

@app.task
def process_llm_batch(request_batch: List[Dict]):
    """Process batch of LLM requests in parallel"""
    
    print(f"Processing batch of {len(request_batch)} requests")
    
    # Create group of parallel tasks
    job = group(
        process_llm_request.s(
            req["id"],
            req["prompt"],
            req["model"]
        ) for req in request_batch
    )
    
    # Execute in parallel
    result = job.apply_async()
    
    # Wait for all to complete (or use callback)
    results = result.get()
    
    print(f"Batch completed: {len(results)} results")
    return results

# Submit batch job
def submit_batch_job(requests: List[Dict]):
    """Submit batch processing job"""
    
    # Send to batch queue (lower priority)
    task = process_llm_batch.apply_async(
        args=[requests],
        queue='batch',
        priority=1
    )
    
    print(f"Submitted batch job: {task.id}")
    return task.id

# Usage
requests = [
    {"id": f"req_{i}", "prompt": f"Summarize document {i}", "model": "gpt-3.5-turbo"}
    for i in range(50)
]

# Submit for batch processing
job_id = submit_batch_job(requests)

# Check status later
from celery.result import AsyncResult
result = AsyncResult(job_id)
print(f"Job status: {result.status}")

# Get results when ready
if result.ready():
    results = result.get()
    print(f"Processed {len(results)} requests")
\`\`\`

---

## Scheduled Batch Processing

\`\`\`python
from celery.schedules import crontab
from datetime import datetime, time as dt_time

class ScheduledBatchProcessor:
    """Process batches on a schedule"""
    
    def __init__(self):
        self.pending_requests = []
        self.schedule = {
            "night": dt_time(2, 0),    # 2 AM
            "morning": dt_time(8, 0),  # 8 AM  
            "afternoon": dt_time(14, 0), # 2 PM
        }
    
    def queue_for_next_batch(self, request: BatchRequest):
        """Queue request for next scheduled batch"""
        self.pending_requests.append(request)
        next_run = self.get_next_scheduled_run()
        print(f"📅 Queued for batch at {next_run}")
    
    def get_next_scheduled_run(self) -> datetime:
        """Get next scheduled batch time"""
        now = datetime.now()
        
        # Find next scheduled time
        for name, scheduled_time in sorted(self.schedule.items(), key=lambda x: x[1]):
            scheduled_datetime = datetime.combine(now.date(), scheduled_time)
            if scheduled_datetime > now:
                return scheduled_datetime
        
        # If all times today have passed, use tomorrow's first slot
        first_time = min(self.schedule.values())
        return datetime.combine(now.date(), first_time) + timedelta(days=1)
    
    async def run_scheduled_batch(self):
        """Run the scheduled batch"""
        if not self.pending_requests:
            print("No pending requests")
            return
        
        print(f"\n⏰ Running scheduled batch at {datetime.now()}")
        print(f"   Processing {len(self.pending_requests)} queued requests")
        
        # Process all pending
        results = await self.process_batch(self.pending_requests)
        
        # Clear queue
        self.pending_requests = []
        
        return results

# Celery beat schedule for automated runs
app.conf.beat_schedule = {
    'process-batch-night': {
        'task': 'tasks.run_scheduled_batch',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'process-batch-morning': {
        'task': 'tasks.run_scheduled_batch',
        'schedule': crontab(hour=8, minute=0),  # 8 AM daily
    },
    'process-batch-afternoon': {
        'task': 'tasks.run_scheduled_batch',
        'schedule': crontab(hour=14, minute=0),  # 2 PM daily
    },
}

# Usage
processor = ScheduledBatchProcessor()

# Queue requests throughout the day
for i in range(1000):
    processor.queue_for_next_batch(
        BatchRequest(f"req_{i}", f"Analyze document {i}", "gpt-3.5-turbo")
    )

# Batches automatically run at 2 AM, 8 AM, 2 PM
# Users get results within 6 hours (acceptable for non-realtime tasks)
\`\`\`

---

## Monitoring and Metrics

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class BatchMetrics:
    total_batches: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    batch_latencies: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    
    def record_batch(self, size: int, latency: float, cost: float):
        """Record batch metrics"""
        self.total_batches += 1
        self.total_requests += size
        self.total_cost += cost
        self.batch_latencies.append(latency)
        self.batch_sizes.append(size)
    
    def get_report(self) -> str:
        """Generate metrics report"""
        if not self.batch_latencies:
            return "No batches processed"
        
        avg_latency = statistics.mean(self.batch_latencies)
        avg_batch_size = statistics.mean(self.batch_sizes)
        avg_cost_per_request = self.total_cost / self.total_requests
        
        # Estimate savings vs sequential processing
        sequential_time = self.total_requests * 2.0  # 2s per request
        actual_time = sum(self.batch_latencies)
        time_saved = sequential_time - actual_time
        
        return f"""
📊 Batch Processing Metrics
{'=' * 50}

Batches Processed: {self.total_batches}
Total Requests: {self.total_requests:,}
Avg Batch Size: {avg_batch_size:.1f}

Performance:
  Avg Latency: {avg_latency:.2f}s per batch
  Total Time: {actual_time:.0f}s
  Time Saved: {time_saved:.0f}s ({time_saved/sequential_time*100:.0f}% faster)

Cost:
  Total Cost: \${self.total_cost:.2f}
  Avg Cost/Request: \${avg_cost_per_request:.4f}

Throughput:
  {self.total_requests / actual_time:.1f} requests/second
"""

# Usage
metrics = BatchMetrics()

# After each batch
metrics.record_batch(
    size=20,
    latency=4.5,
    cost=0.10
)

# Generate report
print(metrics.get_report())
\`\`\`

---

## Best Practices

### 1. Choose Batch Size Wisely
- Too small: Inefficient, loses batching benefits
- Too large: High latency, potential timeouts
- Sweet spot: 10-50 for most use cases
- Adapt based on performance metrics

### 2. Set Appropriate Timeouts
- Max wait time before processing (e.g., 60s)
- Balance between efficiency and user experience
- Consider user expectations for your use case

### 3. Handle Failures Gracefully
- Retry failed requests
- Don't fail entire batch for one bad request
- Log failures for investigation

### 4. Monitor Queue Depth
- Alert if queue grows too large
- May indicate need to scale
- Or reduce batch size

### 5. Optimize for Cost
- Downgrade models for batch tasks when appropriate
- Process during off-peak hours for rate limit relief
- Group similar requests together

---

## Summary

Batch processing can reduce costs and improve efficiency:

- **Accumulate requests**: Process together instead of one-by-one
- **Parallel execution**: Process batch items concurrently
- **Adaptive sizing**: Adjust batch size based on performance
- **Priority queues**: Handle mixed real-time/batch workloads
- **Scheduled processing**: Process during optimal times
- **Cost optimization**: Use cheaper models, better resource utilization

**Typical Savings**: 40-60% cost reduction + better throughput for suitable workloads.

`,
  exercises: [
    {
      prompt:
        'Implement a production batch processor with Celery that processes 10,000 requests/day in optimized batches.',
      solution: `Use Celery configuration shown, set up workers, monitor queue depth and processing times.`,
    },
    {
      prompt:
        'Build an adaptive batch processor that automatically tunes batch size for 5s target latency.',
      solution: `Implement AdaptiveBatchProcessor, run for several batches, verify it converges to optimal size.`,
    },
    {
      prompt:
        'Create a priority queue system that ensures critical requests process within 30s while batching low-priority requests efficiently.',
      solution: `Use PriorityBatchProcessor with appropriate thresholds and batch sizes per priority level.`,
    },
  ],
};
