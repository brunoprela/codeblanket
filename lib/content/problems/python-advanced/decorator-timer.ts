/**
 * Function Timer Decorator
 * Problem ID: decorator-timer
 * Order: 3
 */

import { Problem } from '../../../types';

export const decorator_timerProblem: Problem = {
  id: 'decorator-timer',
  title: 'Function Timer Decorator',
  difficulty: 'Easy',
  description: `Create a decorator that measures and prints the execution time of a function.

The decorator should:
- Measure time before and after function execution
- Print the elapsed time in seconds
- Return the function result unchanged
- Preserve function metadata

**Use Case:** Performance profiling and optimization.`,
  examples: [
    {
      input: 'Function that sleeps for 1 second',
      output: 'Prints "Execution time: 1.00s"',
    },
  ],
  constraints: [
    'Use time.time() for measurement',
    'Print with 2 decimal places',
    'Must work with any function',
  ],
  hints: [
    'Import time module',
    'Record time before and after function call',
    'Use f-string for formatting',
  ],
  starterCode: `import time
from functools import wraps

def timer(func):
    """
    Decorator that times function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    # Your code here
    pass


@timer
def slow_function():
    time.sleep(1)
    return "Done"


# Test helper function (for automated testing)
def test_timer():
    """Test function that verifies decorator works correctly"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the decorated function
        result = slow_function()
        
        # Get the printed output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify the function returned the correct value
        if result != "Done":
            return "FAIL: Wrong return value"
        
        # Verify decorator printed timing information
        if not output or "time" not in output.lower():
            return "FAIL: Decorator didn't print timing info"
        
        # Verify the output contains a number (the timing)
        import re
        if not re.search(r'\\d+\\.?\\d*', output):
            return "FAIL: No timing value found in output"
        
        return "Done"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
  testCases: [
    {
      input: [],
      expected: 'Done',
      functionName: 'test_timer',
    },
  ],
  solution: `import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper


@timer
def slow_function():
    time.sleep(1)
    return "Done"


# Test helper function (for automated testing)
def test_timer():
    """Test function that verifies decorator works correctly"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the decorated function
        result = slow_function()
        
        # Get the printed output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify the function returned the correct value
        if result != "Done":
            return "FAIL: Wrong return value"
        
        # Verify decorator printed timing information
        if not output or "time" not in output.lower():
            return "FAIL: Decorator didn't print timing info"
        
        # Verify the output contains a number (the timing)
        import re
        if not re.search(r'\\d+\\.?\\d*', output):
            return "FAIL: No timing value found in output"
        
        return "Done"
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
  timeComplexity: 'O(1) overhead',
  spaceComplexity: 'O(1)',
  order: 3,
  topic: 'Python Advanced',
};
