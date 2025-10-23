/**
 * LLM Observability & Logging Section
 * Module 1: LLM Engineering Fundamentals
 */

export const observabilityloggingSection = {
  id: 'observability-logging',
  title: 'LLM Observability & Logging',
  content: `# LLM Observability & Logging

Master monitoring, logging, and debugging LLM applications for reliable production systems.

## Why Observability Matters

You can't fix what you can't see. LLM apps need comprehensive observability.

### What to Monitor

\`\`\`python
"""
KEY METRICS TO TRACK:

1. PERFORMANCE
   - Latency (time to first token, total time)
   - Throughput (requests per second)
   - Token usage

2. COST
   - Cost per request
   - Daily/monthly spend
   - Cost by model/user

3. QUALITY
   - Success/failure rate
   - Parse errors
   - User feedback

4. ERRORS
   - API errors (rate limits, timeouts)
   - Parse failures
   - Validation errors

5. USAGE PATTERNS
   - Popular prompts
   - User behavior
   - Peak times
"""
\`\`\`

## Basic Logging

Start with structured logging for every request.

### Simple Logger

\`\`\`python
import logging
from datetime import datetime
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_llm_request(
    model: str,
    prompt: str,
    response: str,
    latency: float,
    tokens_used: int,
    cost: float
):
    """Log LLM request with all relevant data."""
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'prompt_preview': prompt[:100],  # First 100 chars
        'response_preview': response[:100],
        'latency_seconds': latency,
        'tokens_used': tokens_used,
        'cost_usd': cost
    }
    
    logger.info(f"LLM Request: {json.dumps(log_data)}")

# Usage
import time
from openai import OpenAI

client = OpenAI()

start = time.time()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is Python?"}]
)
latency = time.time() - start

log_llm_request(
    model="gpt-3.5-turbo",
    prompt="What is Python?",
    response=response.choices[0].message.content,
    latency=latency,
    tokens_used=response.usage.total_tokens,
    cost=0.00125  # Calculate actual cost
)
\`\`\`

### Structured Logging with Context

\`\`\`python
import logging
import json
from typing import Dict, Any, Optional
from contextvars import ContextVar

# Request ID context variable
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class StructuredLogger:
    """
    Structured logger with context.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _add_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add context to log data."""
        request_id = request_id_var.get()
        if request_id:
            data['request_id'] = request_id
        return data
    
    def log_request(
        self,
        model: str,
        messages: list,
        **kwargs
    ):
        """Log LLM request."""
        data = self._add_context({
            'event': 'llm_request',
            'model': model,
            'message_count': len(messages),
            **kwargs
        })
        self.logger.info(json.dumps(data))
    
    def log_response(
        self,
        response_text: str,
        tokens: int,
        latency: float,
        cost: float,
        **kwargs
    ):
        """Log LLM response."""
        data = self._add_context({
            'event': 'llm_response',
            'response_length': len(response_text),
            'tokens': tokens,
            'latency_ms': latency * 1000,
            'cost': cost,
            **kwargs
        })
        self.logger.info(json.dumps(data))
    
    def log_error(
        self,
        error: Exception,
        **kwargs
    ):
        """Log error."""
        data = self._add_context({
            'event': 'llm_error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            **kwargs
        })
        self.logger.error(json.dumps(data))

# Usage
import uuid

logger = StructuredLogger(__name__)

# Set request ID for this request
request_id_var.set(str(uuid.uuid4()))

# Log request
logger.log_request(
    model='gpt-3.5-turbo',
    messages=[{"role": "user", "content": "Hello"}],
    user_id='user123'
)

# Log response
logger.log_response(
    response_text="Hi! How can I help?",
    tokens=25,
    latency=0.5,
    cost=0.00005
)

# All logs will have the same request_id
\`\`\`

## LangSmith Integration

LangSmith provides comprehensive LLM observability.

### Basic LangSmith Setup

\`\`\`python
# pip install langsmith

import os
from langsmith import Client
from openai import OpenAI

# Set up LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# LangChain automatically logs to LangSmith
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This chain will automatically log to LangSmith
llm = LangChainOpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run and it logs automatically!
result = chain.run(topic="quantum computing")
print(result)

# View traces at: https://smith.langchain.com
\`\`\`

### Manual LangSmith Logging

\`\`\`python
from langsmith import Client
from langsmith.run_trees import RunTree
from openai import OpenAI
import time

client = OpenAI()
ls_client = Client()

def call_with_langsmith(prompt: str, run_name: str = "llm_call"):
    """
    Call LLM with manual LangSmith logging.
    """
    
    # Create run
    run = RunTree(
        name=run_name,
        run_type="llm",
        inputs={"prompt": prompt},
        extra={"model": "gpt-3.5-turbo"}
    )
    
    try:
        start = time.time()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency = time.time() - start
        
        # Log outputs
        run.end(
            outputs={"response": response.choices[0].message.content},
            extra={
                "tokens": response.usage.total_tokens,
                "latency": latency
            }
        )
        
        # Post to LangSmith
        run.post()
        
        return response.choices[0].message.content
    
    except Exception as e:
        # Log error
        run.end(error=str(e))
        run.post()
        raise

# Usage
result = call_with_langsmith("What is machine learning?")
print(result)
\`\`\`

## Custom Observability Platform

Build a simple observability system.

### Request Tracker

\`\`\`python
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

@dataclass
class LLMRequest:
    """Single LLM request record."""
    id: str
    timestamp: datetime
    model: str
    prompt: str
    response: str
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    tokens_total: int
    cost_usd: float
    success: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None

class RequestTracker:
    """
    Track all LLM requests for analysis.
    """
    
    def __init__(self, storage_path: str = "llm_requests.jsonl"):
        self.storage_path = Path(storage_path)
        self.requests: List[LLMRequest] = []
    
    def track(self, request: LLMRequest):
        """Track a request."""
        self.requests.append(request)
        
        # Append to file
        with open(self.storage_path, 'a') as f:
            # Convert datetime to ISO format
            data = asdict(request)
            data['timestamp'] = request.timestamp.isoformat()
            f.write(json.dumps(data) + '\\n')
    
    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        if not self.requests:
            return {}
        
        successful = [r for r in self.requests if r.success]
        failed = [r for r in self.requests if not r.success]
        
        return {
            'total_requests': len(self.requests),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.requests) * 100,
            
            'total_tokens': sum(r.tokens_total for r in self.requests),
            'total_cost': sum(r.cost_usd for r in self.requests),
            
            'avg_latency_ms': sum(r.latency_ms for r in self.requests) / len(self.requests),
            'avg_tokens': sum(r.tokens_total for r in self.requests) / len(self.requests),
            'avg_cost': sum(r.cost_usd for r in self.requests) / len(self.requests),
            
            'models_used': list(set(r.model for r in self.requests)),
        }
    
    def get_recent(self, n: int = 10) -> List[LLMRequest]:
        """Get N most recent requests."""
        return sorted(self.requests, key=lambda r: r.timestamp, reverse=True)[:n]
    
    def get_failed(self) -> List[LLMRequest]:
        """Get failed requests."""
        return [r for r in self.requests if not r.success]
    
    def get_expensive(self, n: int = 10) -> List[LLMRequest]:
        """Get most expensive requests."""
        return sorted(self.requests, key=lambda r: r.cost_usd, reverse=True)[:n]

# Usage
import uuid

tracker = RequestTracker()

# Track a request
request = LLMRequest(
    id=str(uuid.uuid4()),
    timestamp=datetime.now(),
    model='gpt-3.5-turbo',
    prompt='What is Python?',
    response='Python is a programming language...',
    latency_ms=500,
    tokens_prompt=10,
    tokens_completion=50,
    tokens_total=60,
    cost_usd=0.00012,
    success=True
)

tracker.track(request)

# Get stats
stats = tracker.get_stats()
print(json.dumps(stats, indent=2))
\`\`\`

### Real-time Dashboard

\`\`\`python
from collections import deque
from datetime import datetime, timedelta
from typing import Deque

class RealtimeDashboard:
    """
    Real-time metrics dashboard.
    """
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.requests: Deque[LLMRequest] = deque()
    
    def add_request(self, request: LLMRequest):
        """Add request to dashboard."""
        self.requests.append(request)
        
        # Remove old requests outside window
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        while self.requests and self.requests[0].timestamp < cutoff:
            self.requests.popleft()
    
    def get_metrics(self) -> Dict:
        """Get real-time metrics."""
        if not self.requests:
            return {'requests_in_window': 0}
        
        requests_list = list(self.requests)
        
        total_time = (
            requests_list[-1].timestamp - requests_list[0].timestamp
        ).total_seconds()
        
        return {
            'window_minutes': self.window_minutes,
            'requests_in_window': len(requests_list),
            'requests_per_minute': len(requests_list) / self.window_minutes if total_time > 0 else 0,
            
            'success_rate': sum(1 for r in requests_list if r.success) / len(requests_list) * 100,
            
            'avg_latency_ms': sum(r.latency_ms for r in requests_list) / len(requests_list),
            'p95_latency_ms': sorted([r.latency_ms for r in requests_list])[int(len(requests_list) * 0.95)],
            
            'tokens_per_minute': sum(r.tokens_total for r in requests_list) / self.window_minutes,
            'cost_per_minute': sum(r.cost_usd for r in requests_list) / self.window_minutes,
        }
    
    def print_dashboard(self):
        """Print formatted dashboard."""
        metrics = self.get_metrics()
        
        if metrics['requests_in_window'] == 0:
            print("No requests in window")
            return
        
        print("\\n" + "="*60)
        print(f"REAL-TIME DASHBOARD (Last {self.window_minutes} minutes)")
        print("="*60)
        
        print(f"\\nVOLUME:")
        print(f"  Requests: {metrics['requests_in_window']}")
        print(f"  Requests/min: {metrics['requests_per_minute']:.2f}")
        
        print(f"\\nQUALITY:")
        print(f"  Success rate: {metrics['success_rate']:.1f}%")
        
        print(f"\\nPERFORMANCE:")
        print(f"  Avg latency: {metrics['avg_latency_ms']:.0f}ms")
        print(f"  P95 latency: {metrics['p95_latency_ms']:.0f}ms")
        
        print(f"\\nUSAGE:")
        print(f"  Tokens/min: {metrics['tokens_per_minute']:.0f}")
        print(f"  Cost/min: \${metrics['cost_per_minute']: .6f
}")

print("=" * 60 + "\\n")

# Usage
dashboard = RealtimeDashboard(window_minutes = 5)

# Simulate requests
import random
for i in range(20):
    request = LLMRequest(
        id = str(uuid.uuid4()),
        timestamp = datetime.now() - timedelta(minutes = random.randint(0, 4)),
        model = 'gpt-3.5-turbo',
        prompt = f'Query {i}',
        response = f'Response {i}',
        latency_ms = random.uniform(200, 1000),
        tokens_prompt = random.randint(10, 100),
        tokens_completion = random.randint(50, 500),
        tokens_total = random.randint(60, 600),
        cost_usd = random.uniform(0.0001, 0.01),
        success = random.random() > 0.1  # 90 % success rate
    )
dashboard.add_request(request)

# View dashboard
dashboard.print_dashboard()
\`\`\`

## Helicone Integration

Helicone provides LLM observability with minimal code changes.

### Setup Helicone

\`\`\`python
# pip install helicone

from openai import OpenAI

# Add Helicone proxy
client = OpenAI(
    api_key="your-openai-key",
    base_url="https://oai.hconeai.com/v1",
    default_headers={
        "Helicone-Auth": "Bearer your-helicone-key"
    }
)

# Use normally - automatically logs to Helicone!
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# View at: https://helicone.ai
\`\`\`

### Helicone with Custom Properties

\`\`\`python
from openai import OpenAI

client = OpenAI(
    base_url="https://oai.hconeai.com/v1",
    default_headers={
        "Helicone-Auth": "Bearer your-key"
    }
)

# Add custom properties for filtering/grouping
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={
        "Helicone-Property-User": "user123",
        "Helicone-Property-Feature": "chat",
        "Helicone-Property-Environment": "production"
    }
)

# Now can filter by user, feature, or environment in Helicone dashboard
\`\`\`

## Debugging LLM Applications

Tools and techniques for debugging.

### Detailed Request Logging

\`\`\`python
import logging
import json
from typing import List, Dict

class LLMDebugger:
    """
    Debug LLM interactions with detailed logging.
    """
    
    def __init__(self, log_file: str = "llm_debug.log"):
        self.logger = logging.getLogger("llm_debugger")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s\\n%(message)s\\n'
        )
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
    
    def log_request(
        self,
        model: str,
        messages: List[Dict],
        **kwargs
    ):
        """Log request details."""
        self.logger.debug(
            f"REQUEST:\\n"
            f"Model: {model}\\n"
            f"Messages:\\n{json.dumps(messages, indent=2)}\\n"
            f"Parameters:\\n{json.dumps(kwargs, indent=2)}"
        )
    
    def log_response(
        self,
        response_obj: Any
    ):
        """Log full response object."""
        # Convert response to dict for logging
        response_dict = {
            'content': response_obj.choices[0].message.content,
            'model': response_obj.model,
            'usage': {
                'prompt_tokens': response_obj.usage.prompt_tokens,
                'completion_tokens': response_obj.usage.completion_tokens,
                'total_tokens': response_obj.usage.total_tokens
            },
            'finish_reason': response_obj.choices[0].finish_reason
        }
        
        self.logger.debug(
            f"RESPONSE:\\n{json.dumps(response_dict, indent=2)}"
        )
    
    def log_error(self, error: Exception):
        """Log error with full traceback."""
        import traceback
        self.logger.error(
            f"ERROR:\\n"
            f"Type: {type(error).__name__}\\n"
            f"Message: {str(error)}\\n"
            f"Traceback:\\n{traceback.format_exc()}"
        )

# Usage
from openai import OpenAI

debugger = LLMDebugger()
client = OpenAI()

messages = [{"role": "user", "content": "Explain Python"}]

# Log request
debugger.log_request("gpt-3.5-turbo", messages, temperature=0.7)

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7
    )
    
    # Log response
    debugger.log_response(response)
    
except Exception as e:
    debugger.log_error(e)

# Check llm_debug.log for detailed logs
\`\`\`

## Production Observability System

\`\`\`python
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging

@dataclass
class Observation:
    """Single observation event."""
    timestamp: datetime
    event_type: str
    data: Dict

class ObservabilitySystem:
    """
    Complete observability system for production.
    """
    
    def __init__(
        self,
        service_name: str,
        log_file: str = "llm_observability.log",
        enable_console: bool = True
    ):
        self.service_name = service_name
        self.observations: List[Observation] = []
        
        # Setup logging
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        if enable_console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def observe(self, event_type: str, data: Dict):
        """Record an observation."""
        obs = Observation(
            timestamp=datetime.now(),
            event_type=event_type,
            data=data
        )
        
        self.observations.append(obs)
        
        # Log it
        log_data = {
            'service': self.service_name,
            'event': event_type,
            **data
        }
        self.logger.info(json.dumps(log_data))
    
    def get_metrics(self, event_type: Optional[str] = None) -> Dict:
        """Get metrics for event type."""
        if event_type:
            events = [o for o in self.observations if o.event_type == event_type]
        else:
            events = self.observations
        
        return {
            'total_events': len(events),
            'event_types': list(set(o.event_type for o in self.observations)),
            'time_range': {
                'start': min(o.timestamp for o in events).isoformat() if events else None,
                'end': max(o.timestamp for o in events).isoformat() if events else None
            }
        }

# Usage
obs = ObservabilitySystem("llm-service")

# Observe request
obs.observe('request_started', {
    'model': 'gpt-3.5-turbo',
    'user_id': 'user123'
})

# Observe response
obs.observe('request_completed', {
    'latency_ms': 500,
    'tokens': 100,
    'cost': 0.001
})

# Observe error
obs.observe('request_failed', {
    'error': 'Rate limit exceeded',
    'retry_count': 3
})

# Get metrics
metrics = obs.get_metrics()
print(json.dumps(metrics, indent=2))
\`\`\`

## Key Takeaways

1. **Log everything** - requests, responses, errors
2. **Structured logging** - use JSON for easy parsing
3. **Track metrics** - latency, cost, success rate
4. **Use request IDs** - trace requests across systems
5. **Set up dashboards** - real-time visibility
6. **Monitor costs** - track spending per request
7. **Debug with detail** - capture full context
8. **Use observability tools** - LangSmith, Helicone
9. **Alert on anomalies** - high error rates, costs
10. **Analyze patterns** - learn from production data

## Next Steps

Now you can monitor LLM apps. Next: **Caching & Performance** - learning to optimize performance and reduce costs through intelligent caching.`,
};
