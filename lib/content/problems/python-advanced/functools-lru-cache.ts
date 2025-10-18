/**
 * LRU Cache for Memoization
 * Problem ID: advanced-functools-lru-cache
 * Order: 27
 */

import { Problem } from '../../../types';

export const functools_lru_cacheProblem: Problem = {
  id: 'advanced-functools-lru-cache',
  title: 'LRU Cache for Memoization',
  difficulty: 'Medium',
  description: `Use @functools.lru_cache decorator for automatic memoization with LRU eviction.

Apply LRU cache to:
- Fibonacci calculation
- Expensive recursive functions
- API calls simulation
- Dynamic programming problems

**Benefit:** Automatic caching with bounded memory using LRU policy.`,
  examples: [
    {
      input: 'fibonacci(100)',
      output: 'Fast result with memoization',
    },
  ],
  constraints: [
    'Use @lru_cache decorator',
    'Specify maxsize if needed',
    'Arguments must be hashable',
  ],
  hints: [
    '@lru_cache(maxsize=128)',
    'maxsize=None for unlimited cache',
    'Use cache_info() to see stats',
  ],
  starterCode: `from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    """Calculate nth Fibonacci number with memoization.
    
    Args:
        n: Position in sequence
        
    Returns:
        nth Fibonacci number
    """
    pass


@lru_cache(maxsize=128)
def count_ways_to_climb(n, steps=(1, 2)):
    """Count ways to climb n stairs with given step sizes.
    
    Args:
        n: Number of stairs
        steps: Tuple of allowed step sizes
        
    Returns:
        Number of ways to climb
    """
    pass


# Test
print(fibonacci(100))
print(count_ways_to_climb(10))
print(fibonacci.cache_info())  # View cache statistics
`,
  testCases: [
    {
      input: [10],
      expected: 55,
    },
  ],
  solution: `from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


@lru_cache(maxsize=128)
def count_ways_to_climb(n, steps=(1, 2)):
    if n == 0:
        return 1
    if n < 0:
        return 0
    return sum(count_ways_to_climb(n - step, steps) for step in steps)`,
  timeComplexity: 'O(n) with memoization vs O(2^n) without',
  spaceComplexity: 'O(n) for cache',
  order: 27,
  topic: 'Python Advanced',
};
