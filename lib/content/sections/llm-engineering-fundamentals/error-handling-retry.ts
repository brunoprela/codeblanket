/**
 * Error Handling & Retry Logic Section
 * Module 1: LLM Engineering Fundamentals
 */

export const errorhandlingretrySection = {
    id: 'error-handling-retry',
    title: 'Error Handling & Retry Logic',
    content: `# Error Handling & Retry Logic

Master production-grade error handling to build reliable LLM applications that gracefully handle failures.

## Common LLM API Errors

Understanding error types is the first step to handling them properly.

### Error Categories

\`\`\`python
"""
COMMON LLM API ERRORS:

1. RATE LIMIT (429)
   - Too many requests
   - Exceeded tokens per minute
   - Temporary - retry with backoff

2. TIMEOUT (Timeout)
   - Request took too long
   - Network issues
   - Retry immediately

3. SERVER ERROR (500, 503)
   - API provider issues
   - Model overloaded
   - Retry with backoff

4. AUTHENTICATION (401)
   - Invalid API key
   - Expired key
   - DON'T RETRY - fix the key

5. INVALID REQUEST (400)
   - Malformed request
   - Invalid parameters
   - DON'T RETRY - fix the request

6. CONTEXT LENGTH (400)
   - Too many tokens
   - Exceeds context window
   - DON'T RETRY - reduce tokens

7. CONTENT POLICY (400)
   - Violates usage policies
   - Harmful content
   - DON'T RETRY - modify content
"""
\`\`\`

### Examining Real Errors

\`\`\`python
from openai import OpenAI, OpenAIError
import time

client = OpenAI()

# Example 1: Rate Limit Error
try:
    # Send too many requests quickly
    for i in range(100):
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}]
        )
except OpenAIError as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}")

# Example 2: Invalid Request
try:
    client.chat.completions.create(
        model="invalid-model-name",
        messages=[{"role": "user", "content": "Hi"}]
    )
except OpenAIError as e:
    print(f"Invalid model error: {e}")

# Example 3: Context Length
try:
    very_long_message = "word " * 100_000
    client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": very_long_message}]
    )
except OpenAIError as e:
    print(f"Context length error: {e}")
\`\`\`

## Basic Error Handling

Start with simple try-except blocks.

### Simple Error Handler

\`\`\`python
from openai import OpenAI, OpenAIError

client = OpenAI()

def call_llm_with_basic_handling(prompt: str) -> str:
    """
    Call LLM with basic error handling.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
result = call_llm_with_basic_handling("What is Python?")

if result:
    print(result)
else:
    print("Failed to get response")
\`\`\`

### Specific Error Types

\`\`\`python
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError
)

client = OpenAI()

def call_with_specific_handling(prompt: str) -> str:
    """
    Handle specific error types differently.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        print("Wait and retry...")
        return None
    
    except APITimeoutError as e:
        print(f"Request timed out: {e}")
        print("Retry immediately...")
        return None
    
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        print("Check your API key!")
        return None
    
    except BadRequestError as e:
        print(f"Bad request: {e}")
        print("Fix your request parameters")
        return None
    
    except APIError as e:
        print(f"API error: {e}")
        print("Server issue - retry with backoff")
        return None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
result = call_with_specific_handling("Hello!")
\`\`\`

## Retry Logic

Automatically retry failed requests with proper strategy.

### Simple Retry

\`\`\`python
import time
from openai import OpenAI, OpenAIError

client = OpenAI()

def call_with_retry(
    prompt: str,
    max_retries: int = 3
) -> str:
    """
    Retry failed requests up to max_retries times.
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
        
        except OpenAIError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying...")
                time.sleep(1)  # Wait 1 second
            else:
                print("Max retries exceeded")
                raise
    
    return None

# Usage
try:
    result = call_with_retry("What is AI?", max_retries=3)
    print(result)
except OpenAIError as e:
    print(f"Failed after retries: {e}")
\`\`\`

### Exponential Backoff

Best practice: increase wait time exponentially between retries.

\`\`\`python
import time
from openai import OpenAI, OpenAIError, RateLimitError

client = OpenAI()

def call_with_exponential_backoff(
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 1.0
) -> str:
    """
    Retry with exponential backoff: wait 1s, 2s, 4s, 8s, 16s...
    
    This prevents hammering the API and respects rate limits.
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
        
        except RateLimitError as e:
            if attempt < max_retries - 1:
                # Calculate wait time: 1s, 2s, 4s, 8s, 16s
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded")
                raise
        
        except OpenAIError as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                print(f"API error. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    return None

# Usage
result = call_with_exponential_backoff("Explain neural networks")
print(result)
\`\`\`

### Exponential Backoff with Jitter

Add randomness to prevent thundering herd problem.

\`\`\`python
import time
import random
from openai import OpenAI, OpenAIError

client = OpenAI()

def call_with_backoff_and_jitter(
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> str:
    """
    Exponential backoff with jitter.
    
    Jitter prevents many clients from retrying simultaneously.
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
        
        except OpenAIError as e:
            if attempt < max_retries - 1:
                # Exponential: 1, 2, 4, 8, 16, ...
                exponential_delay = base_delay * (2 ** attempt)
                
                # Cap at max_delay
                exponential_delay = min(exponential_delay, max_delay)
                
                # Add jitter: random value between 0 and exponential_delay
                jitter = random.uniform(0, exponential_delay)
                wait_time = exponential_delay + jitter
                
                print(f"Error on attempt {attempt + 1}: {type(e).__name__}")
                print(f"Waiting {wait_time:.2f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise
    
    return None

# Usage
result = call_with_backoff_and_jitter(
    "What is machine learning?",
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0
)
\`\`\`

## Using the tenacity Library

tenacity provides powerful, declarative retry logic.

### Basic tenacity Usage

\`\`\`python
# pip install tenacity

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import OpenAI, OpenAIError, RateLimitError

client = OpenAI()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True
)
def call_with_tenacity(prompt: str) -> str:
    """
    Automatically retry with tenacity decorator.
    
    Configuration:
    - Stop after 5 attempts
    - Exponential backoff: 1s, 2s, 4s, 8s, 16s (max 60s)
    - Only retry on RateLimitError and APIError
    - Reraise exception if all retries fail
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Usage
try:
    result = call_with_tenacity("Explain Python decorators")
    print(result)
except OpenAIError as e:
    print(f"Failed after retries: {e}")
\`\`\`

### Advanced tenacity Configuration

\`\`\`python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    # Stop conditions
    stop=stop_after_attempt(5),
    
    # Wait strategy with jitter
    wait=wait_exponential_jitter(initial=1, max=60),
    
    # Only retry specific errors
    retry=retry_if_exception_type((
        RateLimitError,
        APITimeoutError,
        APIError
    )),
    
    # Logging
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
    
    # Reraise final exception
    reraise=True
)
def call_with_advanced_retry(prompt: str) -> str:
    """
    Production-ready retry with logging.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Usage
result = call_with_advanced_retry("What is deep learning?")
print(result)
\`\`\`

## Circuit Breaker Pattern

Prevent cascading failures by "opening the circuit" after repeated failures.

### Circuit Breaker Implementation

\`\`\`python
import time
from typing import Callable, Any
from enum import Enum

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Circuit breaker for LLM API calls.
    
    When too many failures occur:
    1. CLOSED → OPEN (stop making requests)
    2. Wait for timeout period
    3. OPEN → HALF_OPEN (try one request)
    4. If success: HALF_OPEN → CLOSED
    5. If fail: HALF_OPEN → OPEN
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        """
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time >= self.timeout:
                print("Circuit breaker: Trying to recover (HALF_OPEN)...")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN. Blocked for {self.timeout}s after failures.")
        
        try:
            result = func(*args, **kwargs)
            
            # Success!
            if self.state == CircuitState.HALF_OPEN:
                print("Circuit breaker: Recovered! (CLOSED)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
        
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            print(f"Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}")
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                print(f"Circuit breaker: OPEN! Blocking requests for {self.timeout}s")
            
            raise

# Usage
from openai import OpenAI, OpenAIError

client = OpenAI()
breaker = CircuitBreaker(
    failure_threshold=3,
    timeout=30.0,
    expected_exception=OpenAIError
)

def make_api_call(prompt: str):
    """Wrapped API call."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Test circuit breaker
for i in range(10):
    try:
        result = breaker.call(make_api_call, "Hello")
        print(f"Request {i+1}: Success")
    except Exception as e:
        print(f"Request {i+1}: {e}")
    
    time.sleep(1)
\`\`\`

## Production Error Handler

Combine all strategies into production-ready handler.

### Complete Error Handler

\`\`\`python
from openai import (
    OpenAI,
    OpenAIError,
    RateLimitError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    APIError
)
import time
import random
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

class ProductionLLMClient:
    """
    Production-ready LLM client with comprehensive error handling.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        retry_config: Optional[RetryConfig] = None
    ):
        self.client = OpenAI()
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[str]:
        """
        Make chat completion with full error handling.
        """
        self.total_requests += 1
        
        for attempt in range(self.retry_config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                
                return response.choices[0].message.content
            
            except AuthenticationError as e:
                # Don't retry auth errors
                logger.error(f"Authentication error: {e}")
                self.failed_requests += 1
                raise
            
            except BadRequestError as e:
                # Don't retry bad requests
                logger.error(f"Bad request: {e}")
                self.failed_requests += 1
                raise
            
            except RateLimitError as e:
                # Retry with backoff
                if attempt < self.retry_config.max_retries - 1:
                    wait_time = self._calculate_wait_time(attempt)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                    self.retried_requests += 1
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded for rate limit")
                    self.failed_requests += 1
                    raise
            
            except APITimeoutError as e:
                # Retry immediately
                if attempt < self.retry_config.max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}. Retrying...")
                    self.retried_requests += 1
                    time.sleep(0.5)  # Brief pause
                else:
                    logger.error("Max retries exceeded for timeout")
                    self.failed_requests += 1
                    raise
            
            except APIError as e:
                # Server error - retry with backoff
                if attempt < self.retry_config.max_retries - 1:
                    wait_time = self._calculate_wait_time(attempt)
                    logger.warning(f"API error. Retrying in {wait_time:.2f}s...")
                    self.retried_requests += 1
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded for API error")
                    self.failed_requests += 1
                    raise
            
            except OpenAIError as e:
                # Generic OpenAI error
                logger.error(f"OpenAI error: {e}")
                self.failed_requests += 1
                raise
            
            except Exception as e:
                # Unexpected error
                logger.error(f"Unexpected error: {e}")
                self.failed_requests += 1
                raise
        
        return None
    
    def _calculate_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time with exponential backoff and jitter.
        """
        # Exponential backoff
        exponential = self.retry_config.base_delay * (
            self.retry_config.exponential_base ** attempt
        )
        
        # Cap at max
        exponential = min(exponential, self.retry_config.max_delay)
        
        # Add jitter if enabled
        if self.retry_config.jitter:
            jitter = random.uniform(0, exponential * 0.1)  # 10% jitter
            return exponential + jitter
        
        return exponential
    
    def get_metrics(self) -> Dict:
        """Get client metrics."""
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0
            else 0
        )
        
        return {
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'retried_requests': self.retried_requests,
            'success_rate': success_rate
        }

# Usage
client = ProductionLLMClient(
    model="gpt-3.5-turbo",
    retry_config=RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        jitter=True
    )
)

# Make requests
messages = [{"role": "user", "content": "What is Python?"}]

try:
    response = client.chat(messages)
    print(response)
except Exception as e:
    print(f"Failed: {e}")

# Check metrics
metrics = client.get_metrics()
print(f"\\nMetrics:")
print(f"  Total requests: {metrics['total_requests']}")
print(f"  Failed: {metrics['failed_requests']}")
print(f"  Retried: {metrics['retried_requests']}")
print(f"  Success rate: {metrics['success_rate']:.1f}%")
\`\`\`

## Error Handling Best Practices

\`\`\`python
"""
ERROR HANDLING BEST PRACTICES:

1. DIFFERENTIATE ERROR TYPES
   ✅ Handle specific exceptions differently
   ✅ Don't retry auth/bad request errors
   ✅ Retry transient errors only

2. USE EXPONENTIAL BACKOFF
   ✅ Start with 1s delay
   ✅ Double each retry
   ✅ Cap at reasonable max (60s)
   ✅ Add jitter to prevent thundering herd

3. LIMIT RETRIES
   ✅ Set max retry attempts (3-5)
   ✅ Don't retry forever
   ✅ Fail fast when appropriate

4. LOG EVERYTHING
   ✅ Log every error
   ✅ Log retry attempts
   ✅ Include context (prompt, params)
   ✅ Track metrics

5. USE CIRCUIT BREAKERS
   ✅ Prevent cascading failures
   ✅ Give API time to recover
   ✅ Fail fast when circuit open

6. PROVIDE FALLBACKS
   ✅ Cache previous responses
   ✅ Use simpler models
   ✅ Return graceful errors to users

7. MONITOR AND ALERT
   ✅ Track error rates
   ✅ Alert on high failure rate
   ✅ Monitor retry metrics
   ✅ Analyze error patterns

8. TEST ERROR SCENARIOS
   ✅ Test rate limits
   ✅ Test timeouts
   ✅ Test network failures
   ✅ Verify retry logic
"""
\`\`\`

## Key Takeaways

1. **Understand error types** - different errors need different handling
2. **Use exponential backoff** - wait 1s, 2s, 4s, 8s between retries
3. **Add jitter** - prevents thundering herd
4. **Limit retries** - don't retry forever (3-5 attempts)
5. **Don't retry everything** - auth and bad request errors shouldn't retry
6. **Use tenacity** - powerful declarative retry library
7. **Implement circuit breakers** - prevent cascading failures
8. **Log everything** - track errors, retries, and successes
9. **Monitor metrics** - success rate, retry rate, error patterns
10. **Test failure scenarios** - ensure your error handling actually works

## Next Steps

Now you have robust error handling. Next: **Cost Tracking & Optimization** - learning to monitor and optimize LLM costs in production.`,
};

