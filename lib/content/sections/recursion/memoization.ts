/**
 * Memoization: Optimizing Recursion Section
 */

export const memoizationSection = {
  id: 'memoization',
  title: 'Memoization: Optimizing Recursion',
  content: `## Making Recursion Efficient with Memoization

Memoization is caching recursive results to avoid redundant calculations. It transforms exponential algorithms into polynomial time.

---

## The Problem: Redundant Calculations

### Naive Fibonacci - Terrible Performance

\`\`\`python
def fibonacci_naive(n):
    """Extremely slow for n > 35"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

# Time complexity: O(2^n) - EXPONENTIAL!
# Space complexity: O(n) - call stack depth

# Why is it so slow?
# fibonacci(5) calls:
#                    fib(5)
#                   /      \\
#              fib(4)      fib(3)
#             /     \\      /     \\
#         fib(3)  fib(2) fib(2) fib(1)
#        /    \\    / \\    / \\
#    fib(2) fib(1) ... ...
#    /  \\
# fib(1) fib(0)
#
# fib(3) is calculated TWICE
# fib(2) is calculated THREE times
# fib(1) is calculated FIVE times!
\`\`\`

**Counting calls:**
\`\`\`python
call_count = 0

def fibonacci_count(n):
    global call_count
    call_count += 1
    if n <= 1:
        return n
    return fibonacci_count(n - 1) + fibonacci_count(n - 2)

call_count = 0
print(fibonacci_count(10))  # 55
print(f"Calls: {call_count}")  # 177 calls for just n=10!

call_count = 0
print(fibonacci_count(20))  # 6765
print(f"Calls: {call_count}")  # 21,891 calls for n=20! 
\`\`\`

---

## Solution 1: Manual Memoization

**Use a dictionary to cache results:**

\`\`\`python
def fibonacci_memo(n, cache=None):
    """Fast Fibonacci with memoization"""
    # Initialize cache on first call
    if cache is None:
        cache = {}
    
    # Check if result is already computed
    if n in cache:
        return cache[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and store result
    result = fibonacci_memo(n - 1, cache) + fibonacci_memo(n - 2, cache)
    cache[n] = result
    
    return result

# Time complexity: O(n) - each number computed once
# Space complexity: O(n) - cache + call stack

print(fibonacci_memo(100))  # Works instantly!
# 354224848179261915075 (100th Fibonacci number)
\`\`\`

**Visualization of memoized calls:**
\`\`\`
fib(5) → compute
├─ fib(4) → compute
│  ├─ fib(3) → compute
│  │  ├─ fib(2) → compute
│  │  │  ├─ fib(1) → base case
│  │  │  └─ fib(0) → base case
│  │  └─ fib(1) → base case
│  └─ fib(2) → CACHED! (no recursion)
└─ fib(3) → CACHED! (no recursion)

Only 9 calls instead of 15!
\`\`\`

---

## Solution 2: Python's @lru_cache Decorator

**Easiest way to add memoization:**

\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=None)  # Unlimited cache size
def fibonacci_cached(n):
    """Memoized Fibonacci using decorator"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

print(fibonacci_cached(100))  # Instant!
print(fibonacci_cached(500))  # Still instant!

# Check cache statistics
print(fibonacci_cached.cache_info())
# CacheInfo(hits=X, misses=Y, maxsize=None, currsize=Z)
\`\`\`

**How it works:**
- \`@lru_cache\` automatically caches return values
- Uses function arguments as cache key
- LRU = Least Recently Used (evicts old entries when full)
- \`maxsize = None\` means unlimited cache

---

## More Memoization Examples

### Example 1: Climbing Stairs
\`\`\`python
@lru_cache(maxsize=None)
def climb_stairs(n):
    """
    You can climb 1 or 2 steps at a time.
    How many distinct ways to climb n stairs?
    """
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    # Recursive case: either came from n-1 or n-2
    return climb_stairs(n - 1) + climb_stairs(n - 2)

print(climb_stairs(10))  # 89
print(climb_stairs(50))  # 20365011074 (instant with cache!)
\`\`\`

### Example 2: Longest Common Subsequence
\`\`\`python
def lcs(s1, s2, i=0, j=0, cache=None):
    """
    Find length of longest common subsequence.
    Example: "ABCD" and "ACDF" → "ACD" (length 3)
    """
    if cache is None:
        cache = {}
    
    # Create cache key from current positions
    key = (i, j)
    if key in cache:
        return cache[key]
    
    # Base case: reached end of either string
    if i >= len(s1) or j >= len(s2):
        return 0
    
    # If characters match, include it
    if s1[i] == s2[j]:
        result = 1 + lcs(s1, s2, i + 1, j + 1, cache)
    else:
        # Try skipping character in either string
        result = max(
            lcs(s1, s2, i + 1, j, cache),  # Skip s1[i]
            lcs(s1, s2, i, j + 1, cache)   # Skip s2[j]
        )
    
    cache[key] = result
    return result

print(lcs("ABCDEF", "ACDF"))  # 4 ("ACDF")
\`\`\`

---

## When to Use Memoization

✅ **Use memoization when:**

1. **Overlapping Subproblems:**
   - Same inputs computed multiple times
   - Example: Fibonacci, DP problems

2. **Pure Functions:**
   - Same input always gives same output
   - No side effects

3. **Expensive Computations:**
   - Complex calculations
   - Worth the memory cost

4. **Reasonable Input Space:**
   - Limited number of unique inputs
   - Cache won't grow too large

❌ **Don't use memoization when:**

1. **No Overlapping Subproblems:**
   - Each input computed once
   - Example: simple array traversal

2. **Impure Functions:**
   - Results depend on external state
   - Side effects exist

3. **Unlimited Input Space:**
   - Cache grows indefinitely
   - Memory concerns

---

## Memoization vs Dynamic Programming

**Memoization (Top-Down):**
- Recursive approach
- Cache results as needed
- Easier to write initially
- May compute unnecessary subproblems

**Dynamic Programming (Bottom-Up):**
- Iterative approach
- Build table systematically
- Usually more efficient
- Computes all subproblems

**Both solve same problems - different approaches!**

\`\`\`python
# Memoization (Top-Down)
@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

# Dynamic Programming (Bottom-Up)
def fib_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# Both O(n) time and space!
\`\`\`

---

## Best Practices

**1. Use @lru_cache for simplicity:**
\`\`\`python
from functools import lru_cache

@lru_cache(maxsize=128)  # Or maxsize=None
def my_function(n):
    # Your recursive code
    pass
\`\`\`

**2. Or implement manual caching for control:**
\`\`\`python
def my_function(n, cache=None):
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    # Compute result
    cache[n] = result
    return result
\`\`\`

**3. Consider DP for better space optimization:**
\`\`\`python
# Fibonacci with O(1) space
def fib_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(n - 1):
        prev, curr = curr, prev + curr
    return curr
\`\`\`

**Remember:** Memoization turns exponential algorithms into polynomial time - it's recursion's superpower!`,
};
