/**
 * Function Timer Decorator
 * Problem ID: intermediate-decorator-timer
 * Order: 11
 */

import { Problem } from '../../../types';

export const intermediate_decorator_timerProblem: Problem = {
  id: 'intermediate-decorator-timer',
  title: 'Function Timer Decorator',
  difficulty: 'Medium',
  description: `Create a decorator that measures and logs function execution time.

**Requirements:**
- Measure execution time in milliseconds
- Log function name and arguments
- Support both sync functions
- Optionally repeat execution N times and average

**Example:**
\`\`\`python
@timer(repeat=3)
def slow_function(n):
    time.sleep(n)
    return n * 2
\`\`\``,
  examples: [
    {
      input: '@timer(repeat=1)\\ndef add(a, b): return a + b',
      output: 'Function add(2, 3) took 0.02ms',
    },
  ],
  constraints: [
    'Use functools.wraps',
    'Preserve function signature',
    'Handle exceptions',
  ],
  hints: [
    'Use time.perf_counter() for precision',
    'functools.wraps preserves metadata',
    'Decorator with arguments needs nested functions',
  ],
  starterCode: `import time
import functools

def timer(repeat=1):
    """
    Decorator to measure function execution time.
    
    Args:
        repeat: Number of times to execute and average
        
    Returns:
        Decorated function
        
    Examples:
        >>> @timer(repeat=3)
        ... def add(a, b):
        ...     return a + b
        >>> add(2, 3)
        5
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass
        return wrapper
    return decorator


# Test
@timer(repeat=1)
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@timer(repeat=5)
def quick_math(x, y):
    """Perform quick calculation."""
    return x ** 2 + y ** 2


def test_timer():
    """Test function that validates timer decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call decorated function
        result = fibonacci(10)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 55:
            return f"FAIL: Wrong result: {result}"
        
        # Verify decorator printed timing information
        if not output or "took" not in output.lower() and "time" not in output.lower():
            return "FAIL: Decorator should print timing info"
        
        return 55
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"
`,
  testCases: [
    {
      input: [],
      expected: 55,
      functionName: 'test_timer',
    },
  ],
  solution: `import time
import functools

def timer(repeat=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Format arguments for display
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            # Run function multiple times
            times = []
            result = None
            
            for _ in range(repeat):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate average
            avg_time = sum(times) / len(times)
            
            # Log execution
            if repeat > 1:
                print(f"Function {func.__name__}({signature}) "
                      f"took {avg_time:.4f}ms (avg of {repeat} runs)")
            else:
                print(f"Function {func.__name__}({signature}) "
                      f"took {avg_time:.4f}ms")
            
            return result
        return wrapper
    return decorator


# Test
@timer(repeat=1)
def fibonacci(n):
    """Calculate nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@timer(repeat=5)
def quick_math(x, y):
    """Perform quick calculation."""
    return x ** 2 + y ** 2


def test_timer():
    """Test function that validates timer decorator"""
    import sys
    from io import StringIO
    
    # Capture printed output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call decorated function
        result = fibonacci(10)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verify function returned correct value
        if result != 55:
            return f"FAIL: Wrong result: {result}"
        
        # Verify decorator printed timing information
        if not output or "took" not in output.lower() and "time" not in output.lower():
            return "FAIL: Decorator should print timing info"
        
        return 55
    except Exception as e:
        sys.stdout = old_stdout
        return f"FAIL: {str(e)}"`,
  timeComplexity: 'O(r*f) where r is repeats, f is function complexity',
  spaceComplexity: 'O(r) for storing times',
  order: 11,
  topic: 'Python Intermediate',
};
