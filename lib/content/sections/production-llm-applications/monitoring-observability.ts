export const monitoringObservabilityContent = `
# Monitoring & Observability

## Introduction

Monitoring and observability are essential for production LLM applications. You need to know when things go wrong, understand why they're failing, track costs in real-time, and optimize performance. This section covers metrics collection, logging, distributed tracing, and building dashboards for LLM applications.

## Key Metrics for LLM Applications

**Latency Metrics**:
- P50, P95, P99 response times
- Time to first token (streaming)
- End-to-end request duration
- API call latency vs total processing time

**Cost Metrics**:
- Tokens used per request/user/day
- Cost per request/user/day
- Cost by model
- Cache hit rate (cost savings)

**Quality Metrics**:
- User satisfaction ratings
- Response relevance scores
- Error rates by type
- Retry rates

**Throughput Metrics**:
- Requests per second
- Tokens processed per second
- Queue depth
- Worker utilization

**Reliability Metrics**:
- Success rate
- Error rate by category
- Timeout rate
- Rate limit hits

\`\`\`python
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps

# Define metrics
request_count = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status', 'user_tier']
)

request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

tokens_used = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: input/output
)

cost_total = Counter(
    'llm_cost_dollars_total',
    'Total cost in dollars',
    ['model', 'user_tier']
)

cache_hits = Counter(
    'llm_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

queue_depth = Gauge(
    'llm_queue_depth',
    'Current queue depth',
    ['queue_name']
)


def monitor_llm_call(func):
    """
    Decorator to monitor LLM API calls.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        model = kwargs.get('model', 'unknown')
        user_tier = kwargs.get('user_tier', 'free')
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            request_count.labels(
                model=model,
                status='success',
                user_tier=user_tier
            ).inc()
            
            # Record tokens
            if 'usage' in result:
                tokens_used.labels(
                    model=model,
                    type='input'
                ).inc(result['usage'].get('prompt_tokens', 0))
                
                tokens_used.labels(
                    model=model,
                    type='output'
                ).inc(result['usage'].get('completion_tokens', 0))
                
                # Calculate and record cost
                cost = calculate_cost(result['usage'], model)
                cost_total.labels(
                    model=model,
                    user_tier=user_tier
                ).inc(cost)
            
            return result
        
        except Exception as e:
            # Record failure
            request_count.labels(
                model=model,
                status=type(e).__name__,
                user_tier=user_tier
            ).inc()
            
            raise
        
        finally:
            # Record duration
            duration = time.time() - start_time
            request_duration.labels(model=model).observe(duration)
    
    return wrapper


@monitor_llm_call
async def generate_completion(prompt: str, model: str, user_tier: str):
    """Generate completion with automatic monitoring."""
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        'result': response.choices[0].message.content,
        'usage': response.usage
    }
\`\`\`

## Structured Logging

\`\`\`python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """
    Structured logging for LLM applications.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
    
    def log_request(
        self,
        request_id: str,
        user_id: str,
        prompt: str,
        model: str,
        **kwargs
    ):
        """Log incoming request."""
        self.logger.info(json.dumps({
            'event': 'llm_request',
            'request_id': request_id,
            'user_id': user_id,
            'model': model,
            'prompt_length': len(prompt),
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }))
    
    def log_response(
        self,
        request_id: str,
        duration: float,
        tokens_used: int,
        cost: float,
        success: bool,
        **kwargs
    ):
        """Log response."""
        self.logger.info(json.dumps({
            'event': 'llm_response',
            'request_id': request_id,
            'duration_seconds': duration,
            'tokens_used': tokens_used,
            'cost_dollars': cost,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }))
    
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        **kwargs
    ):
        """Log error."""
        self.logger.error(json.dumps({
            'event': 'llm_error',
            'request_id': request_id,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }))


logger = StructuredLogger('llm-service')

# Usage
request_id = str(uuid.uuid4())
logger.log_request(request_id, user_id, prompt, model)

try:
    result = generate(prompt, model)
    logger.log_response(request_id, duration, tokens, cost, True)
except Exception as e:
    logger.log_error(request_id, type(e).__name__, str(e))
\`\`\`

## Distributed Tracing

\`\`\`python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)


async def traced_generation(prompt: str, model: str):
    """
    Generate with distributed tracing.
    """
    with tracer.start_as_current_span("generate_completion") as span:
        span.set_attribute("model", model)
        span.set_attribute("prompt_length", len(prompt))
        
        # Check cache
        with tracer.start_as_current_span("check_cache"):
            cached = cache.get(prompt, model)
            if cached:
                span.set_attribute("cache_hit", True)
                return cached
            span.set_attribute("cache_hit", False)
        
        # Call LLM
        with tracer.start_as_current_span("llm_api_call") as api_span:
            api_span.set_attribute("provider", "openai")
            
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            api_span.set_attribute("tokens_used", response.usage.total_tokens)
        
        # Store in cache
        with tracer.start_as_current_span("store_cache"):
            cache.set(prompt, model, response.choices[0].message.content)
        
        span.set_attribute("success", True)
        return response.choices[0].message.content
\`\`\`

## Real-Time Dashboards

\`\`\`python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/metrics/realtime")
async def get_realtime_metrics():
    """
    Get real-time metrics for dashboard.
    """
    return {
        'requests_per_minute': get_requests_per_minute(),
        'avg_latency_ms': get_avg_latency(),
        'error_rate': get_error_rate(),
        'cache_hit_rate': get_cache_hit_rate(),
        'cost_per_hour': get_cost_per_hour(),
        'active_users': get_active_users(),
        'queue_depth': get_queue_depth()
    }

@app.get("/metrics/costs")
async def get_cost_metrics():
    """Cost breakdown metrics."""
    return {
        'total_cost_today': get_daily_cost(),
        'cost_by_model': get_cost_by_model(),
        'cost_by_user_tier': get_cost_by_tier(),
        'projected_monthly_cost': get_projected_cost()
    }
\`\`\`

## Alerting

\`\`\`python
class AlertManager:
    """Manage alerts for LLM application."""
    
    def __init__(self):
        self.thresholds = {
            'error_rate': 0.05,  # 5%
            'latency_p99': 30.0,  # seconds
            'cost_per_hour': 100.0,  # dollars
            'queue_depth': 1000
        }
    
    def check_alerts(self):
        """Check all alert conditions."""
        if get_error_rate() > self.thresholds['error_rate']:
            self.send_alert(
                'high_error_rate',
                f"Error rate: {get_error_rate():.2%}"
            )
        
        if get_latency_p99() > self.thresholds['latency_p99']:
            self.send_alert(
                'high_latency',
                f"P99 latency: {get_latency_p99():.1f}s"
            )
    
    def send_alert(self, alert_type: str, message: str):
        """Send alert to team."""
        # Send to Slack, PagerDuty, etc.
        pass
\`\`\`

## Best Practices

1. **Monitor everything**: Latency, costs, errors, cache hits
2. **Use structured logging** for easy querying
3. **Implement distributed tracing** for complex workflows
4. **Set up real-time dashboards** for visibility
5. **Alert on important metrics** (error rates, costs, latency)
6. **Track costs per user/tier** for profitability analysis
7. **Monitor cache performance** for cost optimization
8. **Log request IDs** for debugging
9. **Track queue depth** to scale workers
10. **Review metrics regularly** to optimize

Comprehensive monitoring enables you to catch issues before users notice them and optimize for cost and performance.
`;
