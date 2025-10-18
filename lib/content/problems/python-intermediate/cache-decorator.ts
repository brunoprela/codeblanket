/**
 * LRU Cache Decorator
 * Problem ID: intermediate-cache-decorator
 * Order: 14
 */

import { Problem } from '../../../types';

export const intermediate_cache_decoratorProblem: Problem = {
  id: 'intermediate-cache-decorator',
  title: 'LRU Cache Decorator',
  difficulty: 'Hard',
  description: `Implement a Least Recently Used (LRU) cache decorator.

**Features:**
- Cache function results
- Limit cache size
- Evict least recently used items when full
- Track cache hits/misses
- Provide cache statistics

**LRU means:** When cache is full, remove the item that was accessed longest ago.

**Example:**
\`\`\`python
@lru_cache(maxsize=3)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
\`\`\``,
  examples: [
    {
      input: 'Cached fibonacci(10)',
      output: 'Much faster than uncached',
    },
  ],
  constraints: [
    'Implement LRU eviction',
    'Track access order',
    'Support cache stats',
  ],
  hints: [
    'Use OrderedDict for LRU tracking',
    'Move accessed items to end',
    'Check size before adding',
  ],
  starterCode: `from functools import wraps
from collections import OrderedDict

def lru_cache(maxsize=128):
    """
    LRU cache decorator.
    
    Args:
        maxsize: Maximum number of cached items
        
    Returns:
        Decorated function with caching
        
    Examples:
        >>> @lru_cache(maxsize=3)
        ... def add(a, b):
        ...     return a + b
    """
    def decorator(func):
        cache = OrderedDict()
        hits = 0
        misses = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            # Create cache key from arguments
            # Implement LRU logic here
            pass
        
        def cache_info():
            """Return cache statistics."""
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache)
            }
        
        def cache_clear():
            """Clear the cache."""
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


# Test
@lru_cache(maxsize=3)
def fibonacci(n):
    """Calculate Fibonacci number."""
    print(f"Computing fibonacci({n})")
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@lru_cache(maxsize=5)
def expensive_operation(x, y):
    """Simulate expensive computation."""
    print(f"Computing expensive_operation({x}, {y})")
    import time
    time.sleep(0.1)  # Simulate slow operation
    return x ** y

def test_cache():
    """Test function that validates LRU cache decorator"""
    # First test: verify fibonacci works
    result = fibonacci(10)
    if result != 55:
        return f"FAIL: Wrong fibonacci result: {result}"
    
    # Second test: verify caching works by checking cache hits
    fibonacci.cache_clear()  # Clear any existing cache
    
    # Call fibonacci(5) multiple times
    fibonacci(5)  # Miss
    fibonacci(5)  # Should be a hit
    fibonacci(5)  # Should be a hit
    
    info = fibonacci.cache_info()
    
    # Verify cache stats exist
    if 'hits' not in info or 'misses' not in info:
        return "FAIL: Cache stats not available"
    
    # Verify we have at least 1 cache hit (from repeated calls)
    if info['hits'] < 1:
        return f"FAIL: No cache hits detected. Got {info}"
    
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
from collections import OrderedDict

def lru_cache(maxsize=128):
    def decorator(func):
        cache = OrderedDict()
        hits = 0
        misses = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if in cache
            if key in cache:
                hits += 1
                # Move to end (most recently used)
                cache.move_to_end(key)
                return cache[key]
            
            # Not in cache - compute result
            misses += 1
            result = func(*args, **kwargs)
            
            # Add to cache
            cache[key] = result
            cache.move_to_end(key)
            
            # Evict least recently used if over size
            if len(cache) > maxsize:
                cache.popitem(last=False)  # Remove first (oldest)
            
            return result
        
        def cache_info():
            return {
                'hits': hits,
                'misses': misses,
                'maxsize': maxsize,
                'currsize': len(cache),
                'hit_rate': hits / (hits + misses) if (hits + misses) > 0 else 0
            }
        
        def cache_clear():
            nonlocal hits, misses
            cache.clear()
            hits = misses = 0
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


@lru_cache(maxsize=3)
def fibonacci(n):
    """Calculate Fibonacci number."""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)`,
  timeComplexity: 'O(1) for cache lookup, O(n) for function execution',
  spaceComplexity: 'O(maxsize)',
  order: 14,
  topic: 'Python Intermediate',
};
