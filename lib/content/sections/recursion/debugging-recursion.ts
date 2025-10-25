/**
 * Debugging & Visualizing Recursion Section
 */

export const debuggingrecursionSection = {
  id: 'debugging-recursion',
  title: 'Debugging & Visualizing Recursion',
  content: `## Mastering Recursive Thinking

Recursion can be tricky to debug. These techniques help you understand what's happening.

---

## Technique 1: Print Debugging with Indentation

**Track recursion depth visually:**

\`\`\`python
def factorial (n, depth=0):
    """Factorial with visual call stack"""
    indent = "  " * depth
    print(f"{indent}→ factorial({n})")
    
    # Base case
    if n <= 1:
        print(f"{indent}← returning 1")
        return 1
    
    # Recursive case
    result = n * factorial (n - 1, depth + 1)
    print(f"{indent}← returning {result}")
    return result

factorial(4)
\`\`\`

**Output:**
\`\`\`
→ factorial(4)
  → factorial(3)
    → factorial(2)
      → factorial(1)
      ← returning 1
    ← returning 2
  ← returning 6
← returning 24
\`\`\`

---

## Technique 2: Trace with Call Stack

**Manually track the call stack:**

\`\`\`python
def fibonacci_trace (n, depth=0):
    """Fibonacci with detailed trace"""
    indent = "  " * depth
    print(f"{indent}fib({n})")
    
    if n <= 1:
        print(f"{indent}→ {n}")
        return n
    
    print(f"{indent}Computing fib({n-1}) + fib({n-2})")
    
    left = fibonacci_trace (n - 1, depth + 1)
    right = fibonacci_trace (n - 2, depth + 1)
    
    result = left + right
    print(f"{indent}→ fib({n}) = {left} + {right} = {result}")
    
    return result

fibonacci_trace(4)
\`\`\`

**Output shows the recursive tree:**
\`\`\`
fib(4)
Computing fib(3) + fib(2)
  fib(3)
  Computing fib(2) + fib(1)
    fib(2)
    Computing fib(1) + fib(0)
      fib(1)
      → 1
      fib(0)
      → 0
    → fib(2) = 1 + 0 = 1
    fib(1)
    → 1
  → fib(3) = 1 + 1 = 2
  fib(2)
  Computing fib(1) + fib(0)
    fib(1)
    → 1
    fib(0)
    → 0
  → fib(2) = 1 + 0 = 1
→ fib(4) = 2 + 1 = 3
\`\`\`

---

## Technique 3: Count Calls (Performance Check)

**Measure how many times function is called:**

\`\`\`python
def count_calls (func):
    """Decorator to count function calls"""
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@count_calls
def fibonacci_naive (n):
    if n <= 1:
        return n
    return fibonacci_naive (n - 1) + fibonacci_naive (n - 2)

result = fibonacci_naive(10)
print(f"Result: {result}")
print(f"Calls: {fibonacci_naive.calls}")
# Result: 55
# Calls: 177

# Compare with memoized version
from functools import lru_cache

@count_calls
@lru_cache (maxsize=None)
def fibonacci_memo (n):
    if n <= 1:
        return n
    return fibonacci_memo (n - 1) + fibonacci_memo (n - 2)

result = fibonacci_memo(10)
print(f"Result: {result}")
print(f"Calls: {fibonacci_memo.calls}")
# Result: 55
# Calls: 11 (much better!)
\`\`\`

---

## Technique 4: Visualize as a Tree

**Draw the recursion tree on paper:**

\`\`\`python
def print_tree (n, prefix="", is_last=True):
    """Visualize recursion as tree structure"""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}fib({n})")
    
    if n <= 1:
        return
    
    extension = "    " if is_last else "│   "
    new_prefix = prefix + extension
    
    print_tree (n - 1, new_prefix, False)
    print_tree (n - 2, new_prefix, True)

print_tree(5)
\`\`\`

**Output:**
\`\`\`
└── fib(5)
    ├── fib(4)
    │   ├── fib(3)
    │   │   ├── fib(2)
    │   │   │   ├── fib(1)
    │   │   │   └── fib(0)
    │   │   └── fib(1)
    │   └── fib(2)
    │       ├── fib(1)
    │       └── fib(0)
    └── fib(3)
        ├── fib(2)
        │   ├── fib(1)
        │   └── fib(0)
        └── fib(1)
\`\`\`

---

## Technique 5: Step Through with Debugger

**Use Python debugger (pdb):**

\`\`\`python
import pdb

def factorial (n):
    pdb.set_trace()  # Debugger will stop here
    
    if n <= 1:
        return 1
    return n * factorial (n - 1)

# Run and use debugger commands:
# n - next line
# s - step into function
# c - continue
# p variable - print variable
# l - list source code
\`\`\`

---

## Common Recursion Bugs & How to Fix

### Bug 1: Infinite Recursion
\`\`\`python
# ❌ BAD: No base case
def bad_function (n):
    return bad_function (n - 1)  # RecursionError!

# ✅ GOOD: Always have base case
def good_function (n):
    if n <= 0:  # Base case
        return 0
    return good_function (n - 1)
\`\`\`

### Bug 2: Base Case Never Reached
\`\`\`python
# ❌ BAD: Progress in wrong direction
def countdown (n):
    if n == 0:
        return
    print(n)
    countdown (n + 1)  # Goes up, not down!

# ✅ GOOD: Make progress toward base case
def countdown (n):
    if n == 0:
        return
    print(n)
    countdown (n - 1)  # Correctly decreases
\`\`\`

### Bug 3: Wrong Return Value
\`\`\`python
# ❌ BAD: Forgetting to return
def sum_array (arr, index=0):
    if index >= len (arr):
        return 0
    arr[index] + sum_array (arr, index + 1)  # Missing return!

# ✅ GOOD: Always return
def sum_array (arr, index=0):
    if index >= len (arr):
        return 0
    return arr[index] + sum_array (arr, index + 1)
\`\`\`

### Bug 4: Modifying Shared State
\`\`\`python
# ❌ BAD: Mutable default argument
def collect_numbers (n, result=[]):  # Shared across calls!
    if n <= 0:
        return result
    result.append (n)
    return collect_numbers (n - 1, result)

# ✅ GOOD: Use None and create new list
def collect_numbers (n, result=None):
    if result is None:
        result = []
    if n <= 0:
        return result
    result.append (n)
    return collect_numbers (n - 1, result)
\`\`\`

---

## Debugging Checklist

When your recursion doesn't work, check:

**1. Base Case (s):**
- [ ] Do I have a base case?
- [ ] Is it correct?
- [ ] Will it definitely be reached?
- [ ] Does it return the right value?

**2. Recursive Case:**
- [ ] Am I making progress toward base case?
- [ ] Am I returning the result (not just computing it)?
- [ ] Am I correctly combining results?

**3. Function Signature:**
- [ ] Are my parameters being modified correctly?
- [ ] Am I avoiding mutable default arguments?
- [ ] Do I need helper parameters?

**4. Testing:**
- [ ] Test with smallest inputs (n=0, n=1, empty array)
- [ ] Test with small inputs (n=2, n=3)
- [ ] Trace execution by hand
- [ ] Add print statements

**5. Performance:**
- [ ] Am I recalculating the same values?
- [ ] Should I add memoization?
- [ ] Is recursion the right approach?

---

## Pro Tips for Thinking Recursively

**1. Start with the base case:**
   - What\'s the simplest input I can handle?
   - What should I return for that case?

**2. Assume recursion works:**
   - "If I had the answer for n-1, how do I get n?"
   - Don't trace through all the calls

**3. Test with small examples:**
   - n=0, n=1, n=2, n=3
   - Verify each step manually

**4. Draw it out:**
   - Sketch the recursion tree
   - See the pattern visually

**5. Add tracing temporarily:**
   - Print statements show what's happening
   - Remove after debugging

**Remember:** Recursion is a skill that improves with practice. Start simple, test thoroughly, and trust the process!`,
};
