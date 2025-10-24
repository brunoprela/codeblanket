export const errorHandlingResilienceContent = `
# Error Handling & Resilience

## Introduction

LLM applications face unique error scenarios: API rate limits, model timeouts, context length exceeded, transient network failures, and provider outages. Production systems must handle these gracefully, retry intelligently, and maintain service even when dependencies fail.

In this section, we'll explore comprehensive error handling strategies for LLM applications, including retry patterns, circuit breakers, fallback strategies, and graceful degradation. We'll build resilient systems that users can depend on even when things go wrong.

## Common LLM Error Types

Understanding error types helps you handle them appropriately:

**Rate Limit Errors (429)**: You've exceeded the provider's rate limit
- Retry with exponential backoff
- Queue for later processing
- Switch to alternative provider

**Timeout Errors**: Request took too long
- Retry with shorter timeout
- Use streaming to detect early failures
- Split large requests

**Invalid Request Errors (400)**: Token limit exceeded, invalid parameters
- Don't retry (won't succeed)
- Reduce prompt size
- Provide helpful error message to user

**Authentication Errors (401)**: Invalid API key
- Don't retry
- Check key configuration
- Alert administrators

**Server Errors (500/503)**: Provider infrastructure issue
- Retry with backoff
- Switch to backup provider
- Use cached responses

**Network Errors**: Connection failed, DNS issues
- Retry with exponential backoff
- Check connectivity
- Use circuit breaker

\`\`\`python
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import logging

class LLMErrorHandler:
    """Comprehensive error handling for LLM API calls."""
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.APIConnectionError
        )),
        before_sleep=lambda retry_state: logging.warning(
            f"Retrying after error: {retry_state.outcome.exception()}"
        )
    )
    def call_with_retry(prompt: str, model: str = "gpt-3.5-turbo"):
        """
        Call LLM with automatic retry for transient errors.
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response.choices[0].message.content
        
        except openai.error.InvalidRequestError as e:
            # Don't retry invalid requests
            if "maximum context length" in str(e):
                raise TokenLimitExceeded(
                    "Prompt exceeds token limit. Try a shorter prompt."
                ) from e
            raise
        
        except openai.error.AuthenticationError as e:
            # Don't retry auth errors
            raise ConfigurationError("Invalid API key") from e


# Custom exceptions
class TokenLimitExceeded(Exception):
    """Prompt exceeds model's token limit."""
    pass

class ConfigurationError(Exception):
    """Configuration issue (API key, etc.)."""
    pass
\`\`\`

## Circuit Breaker Pattern

Prevent cascading failures by "opening the circuit" when a service is unhealthy:

\`\`\`python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any
import logging

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Circuit breaker for LLM API calls.
    
    Prevents calling failing services repeatedly,
    giving them time to recover.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logging.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. "
                    f"Will retry after {self.recovery_timeout}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logging.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logging.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again."""
        if not self.last_failure_time:
            return True
        
        return (
            datetime.utcnow() - self.last_failure_time
            > timedelta(seconds=self.recovery_timeout)
        )


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


# Usage
openai_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=openai.error.APIError
)

def call_openai_with_breaker(prompt: str):
    """Call OpenAI with circuit breaker protection."""
    def _call():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    try:
        return openai_breaker.call(_call)
    except CircuitBreakerOpen as e:
        # Return cached response or error message
        return get_cached_or_fallback(prompt)
\`\`\`

## Fallback Strategies

Provide alternative responses when primary service fails:

\`\`\`python
from typing import List, Optional, Callable
import logging

class FallbackChain:
    """
    Chain of fallback strategies for LLM calls.
    
    Tries each strategy in order until one succeeds.
    """
    
    def __init__(self):
        self.strategies: List[Callable] = []
    
    def add_strategy(self, strategy: Callable, name: str):
        """Add a fallback strategy."""
        self.strategies.append((name, strategy))
    
    def execute(self, prompt: str, **kwargs) -> dict:
        """
        Execute with fallback chain.
        
        Returns:
            dict with 'result', 'strategy_used', and 'fallback_level'
        """
        for i, (name, strategy) in enumerate(self.strategies):
            try:
                logging.info(f"Trying strategy: {name}")
                result = strategy(prompt, **kwargs)
                
                return {
                    'result': result,
                    'strategy_used': name,
                    'fallback_level': i,
                    'success': True
                }
            
            except Exception as e:
                logging.warning(f"Strategy '{name}' failed: {str(e)}")
                
                if i == len(self.strategies) - 1:
                    # Last strategy failed
                    return {
                        'result': None,
                        'strategy_used': None,
                        'fallback_level': len(self.strategies),
                        'success': False,
                        'error': str(e)
                    }
                
                # Try next strategy
                continue


# Define fallback strategies
def primary_strategy(prompt: str, **kwargs):
    """Primary: Use GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def secondary_strategy(prompt: str, **kwargs):
    """Secondary: Use GPT-3.5 Turbo."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def tertiary_strategy(prompt: str, **kwargs):
    """Tertiary: Use Claude."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def cache_strategy(prompt: str, cache, **kwargs):
    """Quaternary: Return cached similar response."""
    cached = cache.get_semantic_match(prompt, threshold=0.85)
    if cached:
        return cached
    raise ValueError("No cached response available")


def error_strategy(prompt: str, **kwargs):
    """Final fallback: Return helpful error message."""
    return (
        "I'm experiencing technical difficulties. "
        "Please try again in a few moments."
    )


# Build fallback chain
fallback_chain = FallbackChain()
fallback_chain.add_strategy(primary_strategy, "GPT-4")
fallback_chain.add_strategy(secondary_strategy, "GPT-3.5-Turbo")
fallback_chain.add_strategy(tertiary_strategy, "Claude")
fallback_chain.add_strategy(
    lambda p, **k: cache_strategy(p, cache=semantic_cache, **k),
    "Semantic Cache"
)
fallback_chain.add_strategy(error_strategy, "Error Message")

# Use fallback chain
result = fallback_chain.execute("What is Python?")
print(f"Result: {result['result']}")
print(f"Strategy used: {result['strategy_used']}")
print(f"Fallback level: {result['fallback_level']}")
\`\`\`

## Graceful Degradation

Maintain service with reduced functionality rather than complete failure:

\`\`\`python
from enum import Enum

class ServiceMode(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class GracefulService:
    """
    LLM service with graceful degradation.
    
    Automatically switches to degraded mode when experiencing issues.
    """
    
    def __init__(self):
        self.mode = ServiceMode.NORMAL
        self.error_count = 0
        self.error_threshold = 10
        self.error_window = 60  # seconds
        self.recent_errors = []
    
    def generate(self, prompt: str, require_quality: bool = False):
        """
        Generate response with graceful degradation.
        
        Args:
            prompt: User prompt
            require_quality: If True, fail rather than degrade
        """
        if self.mode == ServiceMode.MAINTENANCE:
            raise ServiceUnavailable("Service is in maintenance mode")
        
        # Check if we should enter degraded mode
        self._check_health()
        
        if self.mode == ServiceMode.NORMAL:
            try:
                return self._normal_generation(prompt)
            except Exception as e:
                self._record_error(e)
                
                if require_quality:
                    raise
                
                # Try degraded mode
                return self._degraded_generation(prompt)
        
        else:  # DEGRADED mode
            if require_quality:
                raise ServiceDegraded("Service is in degraded mode")
            
            return self._degraded_generation(prompt)
    
    def _normal_generation(self, prompt: str) -> dict:
        """Normal mode: Use best model."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return {
            'result': response.choices[0].message.content,
            'mode': 'normal',
            'model': 'gpt-4',
            'quality': 'high'
        }
    
    def _degraded_generation(self, prompt: str) -> dict:
        """
        Degraded mode: Use faster/cheaper model or cache.
        """
        # Try cache first
        cached = cache.get(prompt)
        if cached:
            return {
                'result': cached,
                'mode': 'degraded',
                'source': 'cache',
                'quality': 'medium'
            }
        
        # Use cheaper model
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500  # Limit for speed
            )
            
            return {
                'result': response.choices[0].message.content,
                'mode': 'degraded',
                'model': 'gpt-3.5-turbo',
                'quality': 'medium'
            }
        
        except Exception as e:
            # Last resort: template response
            return {
                'result': self._template_response(prompt),
                'mode': 'degraded',
                'source': 'template',
                'quality': 'low'
            }
    
    def _check_health(self):
        """Check service health and update mode."""
        now = time.time()
        
        # Remove old errors
        self.recent_errors = [
            err_time for err_time in self.recent_errors
            if now - err_time < self.error_window
        ]
        
        error_rate = len(self.recent_errors) / self.error_window
        
        # Switch modes based on error rate
        if error_rate > 0.1:  # 10% error rate
            if self.mode == ServiceMode.NORMAL:
                self.mode = ServiceMode.DEGRADED
                logging.warning("Entering DEGRADED mode")
                alert_team("Service entered degraded mode")
        
        elif error_rate < 0.01:  # 1% error rate
            if self.mode == ServiceMode.DEGRADED:
                self.mode = ServiceMode.NORMAL
                logging.info("Returning to NORMAL mode")
    
    def _record_error(self, error: Exception):
        """Record an error for health tracking."""
        self.recent_errors.append(time.time())
        logging.error(f"Error in generation: {str(error)}")
    
    def _template_response(self, prompt: str) -> str:
        """Generate template response when all else fails."""
        return (
            "I'm currently experiencing high load. "
            "Your request has been queued and you'll be notified when complete."
        )


class ServiceUnavailable(Exception):
    """Service is unavailable."""
    pass

class ServiceDegraded(Exception):
    """Service is in degraded mode."""
    pass


# Usage
service = GracefulService()

# Normal request (may degrade automatically)
result = service.generate("Explain quantum computing")
print(f"Mode: {result['mode']}, Quality: {result['quality']}")

# Request requiring high quality (fails if degraded)
try:
    result = service.generate(
        "Write production code",
        require_quality=True
    )
except ServiceDegraded:
    print("Service degraded, cannot provide high-quality response")
\`\`\`

## Timeout Handling

Set appropriate timeouts to prevent hanging requests:

\`\`\`python
import asyncio
from typing import Optional
import openai

class TimeoutHandler:
    """Handle timeouts for LLM API calls."""
    
    @staticmethod
    async def call_with_timeout(
        prompt: str,
        timeout: float = 30.0,
        model: str = "gpt-3.5-turbo"
    ) -> Optional[str]:
        """
        Call LLM with timeout.
        
        Returns None if timeout is reached.
        """
        try:
            response = await asyncio.wait_for(
                openai.ChatCompletion.acreate(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                ),
                timeout=timeout
            )
            
            return response.choices[0].message.content
        
        except asyncio.TimeoutError:
            logging.warning(f"Request timed out after {timeout}s")
            return None
    
    @staticmethod
    async def call_with_progressive_timeout(
        prompt: str,
        timeouts: List[int] = [10, 30, 60],
        models: List[str] = ["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4"]
    ) -> Optional[str]:
        """
        Try with progressively longer timeouts and different models.
        """
        for timeout, model in zip(timeouts, models):
            try:
                result = await asyncio.wait_for(
                    openai.ChatCompletion.acreate(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=timeout
                )
                
                return result.choices[0].message.content
            
            except asyncio.TimeoutError:
                logging.warning(f"Timeout with {model} after {timeout}s, trying next")
                continue
        
        return None


# Usage
result = await TimeoutHandler.call_with_timeout(
    "Very complex question...",
    timeout=30.0
)

if result is None:
    print("Request timed out, try again or simplify the request")
\`\`\`

## Health Checks

Implement health checks for monitoring and load balancing:

\`\`\`python
from fastapi import FastAPI
from datetime import datetime, timedelta
import openai

app = FastAPI()

class HealthChecker:
    """Monitor service health."""
    
    def __init__(self):
        self.last_successful_call = None
        self.consecutive_failures = 0
        self.total_calls = 0
        self.failed_calls = 0
    
    def record_success(self):
        """Record successful API call."""
        self.last_successful_call = datetime.utcnow()
        self.consecutive_failures = 0
        self.total_calls += 1
    
    def record_failure(self):
        """Record failed API call."""
        self.consecutive_failures += 1
        self.total_calls += 1
        self.failed_calls += 1
    
    def is_healthy(self) -> tuple[bool, dict]:
        """
        Check if service is healthy.
        
        Returns:
            (is_healthy, details)
        """
        now = datetime.utcnow()
        
        # Check if we've had successful calls recently
        if self.last_successful_call:
            time_since_success = (now - self.last_successful_call).seconds
        else:
            time_since_success = float('inf')
        
        # Health criteria
        healthy = (
            self.consecutive_failures < 5 and
            time_since_success < 300 and  # Success within 5 minutes
            (self.failed_calls / max(1, self.total_calls)) < 0.5  # < 50% failure rate
        )
        
        details = {
            'healthy': healthy,
            'consecutive_failures': self.consecutive_failures,
            'time_since_last_success': time_since_success,
            'failure_rate': self.failed_calls / max(1, self.total_calls),
            'total_calls': self.total_calls
        }
        
        return healthy, details


health_checker = HealthChecker()


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers.
    
    Returns 200 if healthy, 503 if unhealthy.
    """
    healthy, details = health_checker.is_healthy()
    
    if healthy:
        return {
            "status": "healthy",
            **details
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                **details
            }
        )


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check - can this instance serve traffic?
    """
    # Check if we can reach LLM provider
    try:
        openai.Model.list(timeout=5)
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


@app.get("/health/live")
async def liveness_check():
    """
    Liveness check - is this instance alive?
    """
    return {"status": "alive"}
\`\`\`

## Error Monitoring and Alerting

Track and alert on errors:

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import List
import logging

@dataclass
class ErrorMetrics:
    """Track error metrics."""
    error_type: str
    count: int
    last_occurrence: datetime
    sample_message: str

class ErrorMonitor:
    """Monitor and alert on errors."""
    
    def __init__(self, alert_threshold: int = 10):
        self.errors = {}
        self.alert_threshold = alert_threshold
    
    def record_error(self, error: Exception, context: dict = None):
        """
        Record an error and alert if threshold reached.
        """
        error_type = type(error).__name__
        
        if error_type not in self.errors:
            self.errors[error_type] = ErrorMetrics(
                error_type=error_type,
                count=0,
                last_occurrence=datetime.utcnow(),
                sample_message=str(error)
            )
        
        metrics = self.errors[error_type]
        metrics.count += 1
        metrics.last_occurrence = datetime.utcnow()
        
        # Log error with context
        logging.error(
            f"Error: {error_type}",
            extra={
                'error_message': str(error),
                'context': context,
                'count': metrics.count
            }
        )
        
        # Alert if threshold reached
        if metrics.count >= self.alert_threshold:
            self._send_alert(metrics, context)
    
    def _send_alert(self, metrics: ErrorMetrics, context: dict):
        """Send alert to team."""
        message = (
            f"🚨 Alert: {metrics.error_type}\\n"
            f"Count: {metrics.count}\\n"
            f"Last: {metrics.last_occurrence}\\n"
            f"Sample: {metrics.sample_message}"
        )
        
        # Send to Slack, PagerDuty, etc.
        send_to_slack(message)
    
    def get_metrics(self) -> List[ErrorMetrics]:
        """Get error metrics for dashboard."""
        return list(self.errors.values())


error_monitor = ErrorMonitor(alert_threshold=10)

# Record errors
try:
    result = call_llm()
except Exception as e:
    error_monitor.record_error(e, context={'prompt': prompt, 'model': model})
\`\`\`

## Best Practices

1. **Retry transient errors** (rate limits, timeouts) but not permanent errors (invalid requests)

2. **Use exponential backoff** with jitter to avoid thundering herd

3. **Implement circuit breakers** to prevent cascading failures

4. **Provide fallback strategies** for when primary service fails

5. **Set appropriate timeouts** for all external calls

6. **Implement graceful degradation** rather than complete failure

7. **Monitor error rates** and alert when thresholds are exceeded

8. **Use health checks** for load balancer integration

9. **Log errors with context** for debugging

10. **Test failure scenarios** regularly to ensure resilience works

Resilient systems are the difference between a demo and a production application users can depend on.
`;
