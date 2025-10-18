/**
 * Custom Cache Decorator
 * Problem ID: decorator-cache
 * Order: 2
 */

import { Problem } from '../../../types';

export const decorator_cacheProblem: Problem = {
  id: 'decorator-cache',
  title: 'Custom Cache Decorator',
  difficulty: 'Medium',
  description: `Implement a caching decorator similar to functools.lru_cache but simpler.

The decorator should:
- Cache function results based on arguments
- Return cached result if arguments match
- Work with both args and kwargs
- Handle unhashable arguments gracefully

**Note:** This tests understanding of closures, dictionaries, and argument handling.`,
  examples: [
    {
      input: '@cache, fibonacci(5) called twice',
      output: 'Second call returns cached result instantly',
    },
  ],
  constraints: [
    'Cache must be a dictionary',
    'Handle both positional and keyword arguments',
    'Arguments must be hashable',
  ],
  hints: [
    'Store cache in closure',
    'Use tuple of args and frozenset of kwargs as key',
    'Check cache before calling function',
  ],
  starterCode: `from functools import wraps

def cache(func):
    """
    Simple caching decorator.
    
    Args:
        func: Function to cache
        
    Returns:
        Decorated function with caching
    """
    # Your code here
    pass


@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


call_count = 0

@cache
def counting_function(x):
    """Function that counts how many times it's actually called"""
    global call_count
    call_count += 1
    return x * 2


def test_cache():
    """Test function that validates cache decorator"""
    global call_count
    
    # First, test fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Test caching by calling same function multiple times
    call_count = 0
    r1 = counting_function(5)
    r2 = counting_function(5)
    r3 = counting_function(5)
    
    # Verify result is correct
    if r1 != 10 or r2 != 10 or r3 != 10:
        return "FAIL: Wrong cached result"
    
    # Verify caching happened (should only call once)
    if call_count != 1:
        return f"FAIL: Cache not working, function called {call_count} times instead of 1"
    
    return 55
`,
  testCases: [
    {
      input: [],
      expected: 55,
      functionName: 'test_cache',
    },
  ],
  solution: `from functools import wraps

def cache(func):
    _cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from args and kwargs
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    
    return wrapper


@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


call_count = 0

@cache
def counting_function(x):
    """Function that counts how many times it's actually called"""
    global call_count
    call_count += 1
    return x * 2


def test_cache():
    """Test function that validates cache decorator"""
    global call_count
    
    # First, test fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Test caching by calling same function multiple times
    call_count = 0
    r1 = counting_function(5)
    r2 = counting_function(5)
    r3 = counting_function(5)
    
    # Verify result is correct
    if r1 != 10 or r2 != 10 or r3 != 10:
        return "FAIL: Wrong cached result"
    
    # Verify caching happened (should only call once)
    if call_count != 1:
        return f"FAIL: Cache not working, function called {call_count} times instead of 1"
    
    return 55`,
  timeComplexity: 'O(1) for cached calls',
  spaceComplexity: 'O(n) for n unique calls',
  order: 2,
  topic: 'Python Advanced',
};
