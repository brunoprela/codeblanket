/**
 * Retry Decorator
 * Problem ID: decorator-retry
 * Order: 1
 */

import { Problem } from '../../../types';

export const decorator_retryProblem: Problem = {
  id: 'decorator-retry',
  title: 'Retry Decorator',
  difficulty: 'Medium',
  description: `Create a decorator retry that automatically retries a function if it raises an exception.

The decorator should:
- Accept a parameter max_attempts (number of retry attempts)
- Retry the function if it raises an exception
- Raise the last exception if all attempts fail
- Print attempt numbers for debugging

**Example:**
python
@retry(max_attempts=3)
def flaky_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Failed!")
    return "Success!"
`,
  examples: [
    {
      input: '@retry(max_attempts=3), function that fails twice then succeeds',
      output: '"Success!" after 3 attempts',
      explanation: 'Function retries until success or max attempts reached.',
    },
  ],
  constraints: [
    'max_attempts >= 1',
    'Preserve function metadata',
    'Must work with any callable',
  ],
  hints: [
    'Use a closure to capture max_attempts',
    'Use functools.wraps to preserve metadata',
    'Use a loop for retry logic',
  ],
  starterCode: `from functools import wraps

def retry(max_attempts):
    """
    Decorator that retries a function on exception.
    
    Args:
        max_attempts: Maximum number of attempts
        
    Returns:
        Decorated function
    """
    # Your code here
    pass


# Test code
attempt_count = 0

@retry(max_attempts=3)
def failing_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError(f"Attempt {attempt_count} failed")
    return "Success!"


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Reset attempt count
    global attempt_count
    attempt_count = 0
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = failing_function()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != "Success!":
            return "FAIL: Wrong return value"
        
        # Verify retry happened (should have 3 attempts)
        if attempt_count != 3:
            return f"FAIL: Expected 3 attempts, got {attempt_count}"
        
        # Verify decorator printed attempt info
        if "attempt" not in output.lower():
            return "FAIL: Decorator should print attempt numbers"
        
        return "Success!"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
  testCases: [
    {
      input: [],
      expected: 'Success!',
      functionName: 'test_retry',
    },
  ],
  solution: `from functools import wraps

def retry(max_attempts):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"Attempt {attempt}/{max_attempts}")
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
            return None
        return wrapper
    return decorator


# Test code
attempt_count = 0

@retry(max_attempts=3)
def failing_function():
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ValueError(f"Attempt {attempt_count} failed")
    return "Success!"


def test_retry():
    """Test function that validates retry decorator"""
    import sys
    from io import StringIO
    
    # Reset attempt count
    global attempt_count
    attempt_count = 0
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = failing_function()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != "Success!":
            return "FAIL: Wrong return value"
        
        # Verify retry happened (should have 3 attempts)
        if attempt_count != 3:
            return f"FAIL: Expected 3 attempts, got {attempt_count}"
        
        # Verify decorator printed attempt info
        if "attempt" not in output.lower():
            return "FAIL: Decorator should print attempt numbers"
        
        return "Success!"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 1,
  topic: 'Python Advanced',
};
