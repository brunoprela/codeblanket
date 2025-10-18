/**
 * Retry Decorator with Exponential Backoff
 * Problem ID: intermediate-retry-decorator
 * Order: 17
 */

import { Problem } from '../../../types';

export const intermediate_retry_decoratorProblem: Problem = {
  id: 'intermediate-retry-decorator',
  title: 'Retry Decorator with Exponential Backoff',
  difficulty: 'Medium',
  description: `Create a decorator that retries failed function calls with exponential backoff.

**Features:**
- Retry on exception
- Exponential backoff (wait time doubles each retry)
- Maximum retry limit
- Log retry attempts
- Specify which exceptions to retry

**Exponential Backoff:** 1s, 2s, 4s, 8s...

**Example:**
\`\`\`python
@retry(max_attempts=3, delay=1, backoff=2, exceptions=(ConnectionError,))
def fetch_data():
    # Network call that might fail
    response = requests.get(url)
    return response.json()
\`\`\``,
  examples: [
    {
      input: '@retry(max_attempts=3)\\ndef flaky(): ...',
      output: 'Retries up to 3 times',
    },
  ],
  constraints: [
    'Use exponential backoff',
    'Log each retry',
    'Re-raise if max attempts exceeded',
  ],
  hints: [
    'time.sleep() for delays',
    'Multiply delay by backoff factor',
    'Use isinstance() to check exceptions',
  ],
  starterCode: `import time
import functools
import random

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Decorator to retry function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay (exponential backoff)
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
        
    Examples:
        >>> @retry(max_attempts=3, delay=1, backoff=2)
        ... def flaky_function():
        ...     if random.random() < 0.7:
        ...         raise ConnectionError("Failed")
        ...     return "Success"
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass
        return wrapper
    return decorator


# Test with simulated flaky function
@retry(max_attempts=5, delay=0.5, backoff=2, exceptions=(ConnectionError, TimeoutError))
def flaky_network_call(success_rate=0.3):
    """
    Simulate a flaky network call.
    
    Args:
        success_rate: Probability of success (0-1)
        
    Returns:
        Success message
        
    Raises:
        ConnectionError: If call fails
    """
    print(f"  Attempting network call...")
    if random.random() > success_rate:
        raise ConnectionError("Network error")
    return "Data fetched successfully"


@retry(max_attempts=3, delay=1)
def divide(a, b):
    """Division with retry (will fail permanently on ZeroDivisionError)."""
    print(f"  Attempting {a} / {b}")
    return a / b


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Test basic division (should work immediately)
        result = divide(10, 5)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 2:
            return f"FAIL: Wrong result: {result}"
        
        # Verify retry prints attempt info
        if "attempt" not in output.lower() and "retry" not in output.lower():
            return "FAIL: Decorator should log retry attempts"
        
        return 2
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
  testCases: [
    {
      input: [],
      expected: 2,
      functionName: 'test_retry',
    },
  ],
  solution: `import time
import functools
import random

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        print(f"Max attempts ({max_attempts}) reached. Giving up.")
                        raise
                    
                    print(f"Attempt {attempt} failed: {e}")
                    print(f"Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should not reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


@retry(max_attempts=3, delay=1)
def divide(a, b):
    """Division with retry."""
    return a / b

result = divide(10, 5)`,
  timeComplexity: 'O(2^n) for backoff delays where n is attempts',
  spaceComplexity: 'O(1)',
  order: 17,
  topic: 'Python Intermediate',
};
