/**
 * Recursion problems
 */

import { Problem } from '../types';

export const recursionProblems: Problem[] = [
  {
    id: 'recursion-factorial',
    title: 'Factorial',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Calculate the factorial of a non-negative integer n.

The factorial of n (written as n!) is the product of all positive integers less than or equal to n.

**Definition:**
- 0! = 1
- n! = n × (n-1)!

**Examples:**
- 5! = 5 × 4 × 3 × 2 × 1 = 120
- 3! = 3 × 2 × 1 = 6

This is the classic introduction to recursion problem.`,
    examples: [
      { input: 'n = 5', output: '120' },
      { input: 'n = 0', output: '1' },
      { input: 'n = 1', output: '1' },
    ],
    constraints: ['0 <= n <= 12', 'Result will fit in a 32-bit integer'],
    hints: [
      'Base case: factorial(0) = 1 and factorial(1) = 1',
      'Recursive case: factorial(n) = n * factorial(n-1)',
      'Make sure n decreases with each recursive call',
    ],
    starterCode: `def factorial(n):
    """
    Calculate factorial of n using recursion.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! (factorial of n)
        
    Examples:
        >>> factorial(5)
        120
        >>> factorial(0)
        1
    """
    pass


# Test cases
print(factorial(5))  # Expected: 120
print(factorial(0))  # Expected: 1
`,
    testCases: [
      { input: [5], expected: 120 },
      { input: [0], expected: 1 },
      { input: [1], expected: 1 },
      { input: [3], expected: 6 },
      { input: [10], expected: 3628800 },
    ],
    solution: `def factorial(n):
    """Calculate factorial using recursion"""
    # Base cases
    if n <= 1:
        return 1
    
    # Recursive case
    return n * factorial(n - 1)


# Time Complexity: O(n) - makes n recursive calls
# Space Complexity: O(n) - call stack depth is n`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n) - call stack',
    followUp: [
      'Can you implement this iteratively?',
      'How would you handle very large numbers?',
      'What happens if n is negative?',
    ],
  },
  {
    id: 'recursion-power',
    title: 'Power Function',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Implement pow(x, n), which calculates x raised to the power n.

**Note:** Don't use the built-in ** operator or pow() function.

**Examples:**
- pow(2, 3) = 2 × 2 × 2 = 8
- pow(5, 0) = 1
- pow(2, -2) = 1/(2²) = 0.25`,
    examples: [
      { input: 'x = 2.0, n = 10', output: '1024.0' },
      { input: 'x = 2.0, n = -2', output: '0.25' },
      { input: 'x = 2.0, n = 0', output: '1.0' },
    ],
    constraints: [
      '-100.0 < x < 100.0',
      '-2³¹ <= n <= 2³¹-1',
      'Result will fit in double',
    ],
    hints: [
      'Base case: n = 0 returns 1',
      'For negative n, calculate 1 / pow(x, -n)',
      'Recursive case: pow(x, n) = x * pow(x, n-1)',
      'BONUS: Can you optimize to O(log n) using divide and conquer?',
    ],
    starterCode: `def power(x, n):
    """
    Calculate x^n using recursion.
    
    Args:
        x: Base (float)
        n: Exponent (integer, can be negative)
        
    Returns:
        x raised to power n
        
    Examples:
        >>> power(2.0, 10)
        1024.0
        >>> power(2.0, -2)
        0.25
    """
    pass


# Test cases
print(power(2.0, 10))  # Expected: 1024.0
print(power(2.0, -2))  # Expected: 0.25
`,
    testCases: [
      { input: [2.0, 10], expected: 1024.0 },
      { input: [2.0, -2], expected: 0.25 },
      { input: [2.0, 0], expected: 1.0 },
      { input: [3.0, 3], expected: 27.0 },
    ],
    solution: `def power(x, n):
    """Calculate power using recursion"""
    # Base case
    if n == 0:
        return 1.0
    
    # Handle negative exponent
    if n < 0:
        return 1.0 / power(x, -n)
    
    # Recursive case
    return x * power(x, n - 1)


# Time Complexity: O(n) - makes n recursive calls
# Space Complexity: O(n) - call stack depth

# OPTIMIZED VERSION - O(log n):
def power_optimized(x, n):
    """Fast power using divide and conquer"""
    if n == 0:
        return 1.0
    
    if n < 0:
        return 1.0 / power_optimized(x, -n)
    
    # Divide and conquer
    half = power_optimized(x, n // 2)
    
    if n % 2 == 0:
        return half * half  # Even: x^n = (x^(n/2))^2
    else:
        return half * half * x  # Odd: x^n = (x^(n/2))^2 * x

# Optimized Time: O(log n)
# Optimized Space: O(log n)`,
    timeComplexity: 'O(n) - O(log n) optimized',
    spaceComplexity: 'O(n) - O(log n) optimized',
    followUp: [
      'Can you optimize this to O(log n)?',
      'How does the optimized version work?',
      'What if n is very large?',
    ],
  },
  {
    id: 'recursion-sum-array',
    title: 'Sum of Array',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Calculate the sum of all elements in an array using recursion.

You cannot use loops or built-in sum() function - must use recursion!

**Approach:**
- Process the array element by element
- Base case: empty array returns 0
- Recursive case: first element + sum of rest`,
    examples: [
      { input: 'arr = [1, 2, 3, 4, 5]', output: '15' },
      { input: 'arr = []', output: '0' },
      { input: 'arr = [10]', output: '10' },
    ],
    constraints: ['0 <= arr.length <= 1000', '-1000 <= arr[i] <= 1000'],
    hints: [
      'Base case: if array is empty, return 0',
      'Recursive case: first element + sum(rest of array)',
      'You can use array slicing: arr[1:] for "rest of array"',
      'Alternative: use index parameter to avoid creating new arrays',
    ],
    starterCode: `def sum_array(arr):
    """
    Calculate sum of array using recursion.
    
    Args:
        arr: List of integers
        
    Returns:
        Sum of all elements
        
    Examples:
        >>> sum_array([1, 2, 3, 4, 5])
        15
        >>> sum_array([])
        0
    """
    pass


# Test cases
print(sum_array([1, 2, 3, 4, 5]))  # Expected: 15
print(sum_array([]))  # Expected: 0
`,
    testCases: [
      { input: [[1, 2, 3, 4, 5]], expected: 15 },
      { input: [[]], expected: 0 },
      { input: [[10]], expected: 10 },
      { input: [[-1, -2, -3]], expected: -6 },
      { input: [[100, -50, 25]], expected: 75 },
    ],
    solution: `def sum_array(arr):
    """Sum array using recursion - Method 1: Array slicing"""
    # Base case: empty array
    if len(arr) == 0:
        return 0
    
    # Recursive case: first element + sum of rest
    return arr[0] + sum_array(arr[1:])


# More efficient method using index (avoids array copying):
def sum_array_index(arr, index=0):
    """Sum array using recursion - Method 2: Index tracking"""
    # Base case: reached end of array
    if index >= len(arr):
        return 0
    
    # Recursive case: current element + sum of rest
    return arr[index] + sum_array_index(arr, index + 1)


# Time Complexity: O(n) - processes each element once
# Space Complexity: O(n) - call stack depth
# Note: Method 1 also has O(n) for array slicing`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    followUp: [
      'Which approach is more efficient - slicing or index?',
      'How would you sum a 2D array recursively?',
      'Can you make it tail recursive?',
    ],
  },
  {
    id: 'recursion-reverse-string',
    title: 'Reverse String',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Reverse a string using recursion.

You cannot use built-in reverse functions or slicing [::-1] - must use recursion!

**Approach:**
- Base case: empty or single character returns itself
- Recursive case: last character + reverse of rest`,
    examples: [
      { input: 's = "hello"', output: '"olleh"' },
      { input: 's = "a"', output: '"a"' },
      { input: 's = ""', output: '""' },
    ],
    constraints: [
      '0 <= s.length <= 1000',
      's consists of printable ASCII characters',
    ],
    hints: [
      'Base case: string of length 0 or 1 is already reversed',
      'Recursive case: last char + reverse(all but last char)',
      'Or: reverse(all but first char) + first char',
      'String slicing: s[:-1] is all but last, s[-1] is last char',
    ],
    starterCode: `def reverse_string(s):
    """
    Reverse string using recursion.
    
    Args:
        s: String to reverse
        
    Returns:
        Reversed string
        
    Examples:
        >>> reverse_string("hello")
        "olleh"
        >>> reverse_string("a")
        "a"
    """
    pass


# Test cases
print(reverse_string("hello"))  # Expected: "olleh"
print(reverse_string(""))  # Expected: ""
`,
    testCases: [
      { input: ['hello'], expected: 'olleh' },
      { input: ['a'], expected: 'a' },
      { input: [''], expected: '' },
      { input: ['racecar'], expected: 'racecar' },
      { input: ['Python'], expected: 'nohtyP' },
    ],
    solution: `def reverse_string(s):
    """Reverse string using recursion"""
    # Base case: empty or single character
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])


# Alternative approach (first + rest):
def reverse_string_alt(s):
    """Reverse string using recursion - alternative"""
    if len(s) <= 1:
        return s
    
    # Recursive case: reverse of rest + first char
    return reverse_string_alt(s[1:]) + s[0]


# Time Complexity: O(n²) due to string concatenation
# Space Complexity: O(n²) - call stack + string copies
# Note: In Python, strings are immutable, so each + creates new string`,
    timeComplexity: 'O(n²) due to string immutability',
    spaceComplexity: 'O(n²)',
    followUp: [
      'How can you make this more efficient?',
      'What if you use a list instead of strings?',
      'Can you reverse it in-place (for mutable structures)?',
    ],
  },
  {
    id: 'recursion-palindrome',
    title: 'Check Palindrome',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Check if a string is a palindrome using recursion.

A palindrome reads the same forward and backward (e.g., "racecar", "madam").

**Approach:**
- Base case: string of length 0 or 1 is a palindrome
- Recursive case: first and last characters must match, AND middle must be palindrome

**Note:** Ignore spaces and case for this problem.`,
    examples: [
      { input: 's = "racecar"', output: 'true' },
      { input: 's = "hello"', output: 'false' },
      { input: 's = "A man a plan a canal Panama"', output: 'true' },
    ],
    constraints: [
      '1 <= s.length <= 1000',
      's consists of printable ASCII characters',
    ],
    hints: [
      'First clean the string: remove spaces and convert to lowercase',
      'Base case: length 0 or 1 is a palindrome',
      'Check if first and last characters match',
      'Recursively check the middle substring',
      'Use helper function with left and right pointers',
    ],
    starterCode: `def is_palindrome(s):
    """
    Check if string is palindrome using recursion.
    
    Args:
        s: String to check (ignore spaces and case)
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    pass


# Test cases
print(is_palindrome("racecar"))  # Expected: True
print(is_palindrome("hello"))  # Expected: False
`,
    testCases: [
      { input: ['racecar'], expected: true },
      { input: ['hello'], expected: false },
      { input: ['a'], expected: true },
      { input: ['ab'], expected: false },
      { input: ['aba'], expected: true },
    ],
    solution: `def is_palindrome(s):
    """Check if string is palindrome using recursion"""
    # Clean string: remove spaces and lowercase
    s = s.replace(' ', '').lower()
    
    def helper(left, right):
        # Base case: pointers met or crossed
        if left >= right:
            return True
        
        # Check if characters match
        if s[left] != s[right]:
            return False
        
        # Recursively check middle
        return helper(left + 1, right - 1)
    
    return helper(0, len(s) - 1)


# Alternative without helper function:
def is_palindrome_alt(s):
    """Check palindrome - alternative approach"""
    # Clean string
    s = s.replace(' ', '').lower()
    
    # Base case
    if len(s) <= 1:
        return True
    
    # Check first and last, recurse on middle
    if s[0] != s[-1]:
        return False
    
    return is_palindrome_alt(s[1:-1])


# Time Complexity: O(n) - checks each character once
# Space Complexity: O(n) - call stack depth`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    followUp: [
      'How would you handle special characters?',
      'Can you do this with O(1) space using iteration?',
      'What about checking if a linked list is a palindrome?',
    ],
  },
  {
    id: 'recursion-fibonacci',
    title: 'Fibonacci Number',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Calculate the nth Fibonacci number using recursion.

The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

**Definition:**
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

**Important:** Naive recursion is very slow (O(2^n)). You'll need to optimize with memoization!`,
    examples: [
      { input: 'n = 5', output: '5 (sequence: 0, 1, 1, 2, 3, 5)' },
      { input: 'n = 10', output: '55' },
      { input: 'n = 0', output: '0' },
    ],
    constraints: ['0 <= n <= 30'],
    hints: [
      'Base cases: F(0) = 0, F(1) = 1',
      'Recursive case: F(n) = F(n-1) + F(n-2)',
      'WARNING: This is exponential without optimization!',
      'Use memoization (caching) to make it O(n)',
      'Python: @lru_cache decorator or manual dictionary',
    ],
    starterCode: `def fibonacci(n):
    """
    Calculate nth Fibonacci number using recursion.
    
    Args:
        n: Index in Fibonacci sequence (0-indexed)
        
    Returns:
        nth Fibonacci number
        
    Examples:
        >>> fibonacci(5)
        5
        >>> fibonacci(10)
        55
    """
    pass


# Test cases
print(fibonacci(5))   # Expected: 5
print(fibonacci(10))  # Expected: 55
`,
    testCases: [
      { input: [0], expected: 0 },
      { input: [1], expected: 1 },
      { input: [2], expected: 1 },
      { input: [5], expected: 5 },
      { input: [10], expected: 55 },
    ],
    solution: `# NAIVE SOLUTION - VERY SLOW (don't use!)
def fibonacci_naive(n):
    """Naive recursion - O(2^n) time!"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


# OPTIMIZED WITH MEMOIZATION
def fibonacci(n, cache=None):
    """Fibonacci with memoization"""
    if cache is None:
        cache = {}
    
    # Check cache
    if n in cache:
        return cache[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and cache
    result = fibonacci(n - 1, cache) + fibonacci(n - 2, cache)
    cache[n] = result
    return result


# USING PYTHON DECORATOR
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    """Fibonacci with @lru_cache"""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# Naive: O(2^n) time, O(n) space
# Memoized: O(n) time, O(n) space`,
    timeComplexity: 'O(2^n) naive, O(n) memoized',
    spaceComplexity: 'O(n)',
    followUp: [
      'Why is naive Fibonacci so slow?',
      'How does memoization help?',
      'Can you implement this with DP (bottom-up)?',
      'Can you optimize space to O(1)?',
    ],
  },
  {
    id: 'recursion-binary-search',
    title: 'Binary Search (Recursive)',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Implement binary search using recursion.

Given a sorted array and a target value, return the index where target is found, or -1 if not found.

**Binary Search Algorithm:**
1. Compare target with middle element
2. If equal, return middle index
3. If target is less, search left half
4. If target is greater, search right half

This is a classic divide-and-conquer algorithm.`,
    examples: [
      { input: 'arr = [1,2,3,4,5,6,7,8,9], target = 5', output: '4' },
      { input: 'arr = [1,2,3,4,5,6,7,8,9], target = 10', output: '-1' },
      { input: 'arr = [1], target = 1', output: '0' },
    ],
    constraints: [
      '0 <= arr.length <= 10⁴',
      'arr is sorted in ascending order',
      'All elements are unique',
      '-10⁴ <= arr[i], target <= 10⁴',
    ],
    hints: [
      'Use left and right pointers to define search range',
      'Base case: left > right means target not found',
      'Calculate middle: mid = (left + right) // 2',
      'Compare arr[mid] with target and recurse on appropriate half',
    ],
    starterCode: `def binary_search(arr, target):
    """
    Binary search using recursion.
    
    Args:
        arr: Sorted array of integers
        target: Value to search for
        
    Returns:
        Index of target, or -1 if not found
        
    Examples:
        >>> binary_search([1,2,3,4,5], 3)
        2
        >>> binary_search([1,2,3,4,5], 6)
        -1
    """
    pass


# Test cases
print(binary_search([1,2,3,4,5,6,7,8,9], 5))   # Expected: 4
print(binary_search([1,2,3,4,5,6,7,8,9], 10))  # Expected: -1
`,
    testCases: [
      { input: [[1, 2, 3, 4, 5, 6, 7, 8, 9], 5], expected: 4 },
      { input: [[1, 2, 3, 4, 5, 6, 7, 8, 9], 10], expected: -1 },
      { input: [[1], 1], expected: 0 },
      { input: [[1], 2], expected: -1 },
      { input: [[], 1], expected: -1 },
    ],
    solution: `def binary_search(arr, target, left=0, right=None):
    """Binary search using recursion"""
    # Initialize right on first call
    if right is None:
        right = len(arr) - 1
    
    # Base case: search space exhausted
    if left > right:
        return -1
    
    # Calculate middle
    mid = (left + right) // 2
    
    # Found target
    if arr[mid] == target:
        return mid
    
    # Target is in left half
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    
    # Target is in right half
    else:
        return binary_search(arr, target, mid + 1, right)


# Time Complexity: O(log n) - halves search space each time
# Space Complexity: O(log n) - call stack depth`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',
    followUp: [
      'How is this different from iterative binary search?',
      'Which is better - recursive or iterative?',
      'Can you find the first/last occurrence of a repeated element?',
    ],
  },
  {
    id: 'recursion-gcd',
    title: 'Greatest Common Divisor (GCD)',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Find the greatest common divisor (GCD) of two positive integers using Euclidean algorithm with recursion.

**Euclidean Algorithm:**
- GCD(a, b) = GCD(b, a mod b)
- Base case: GCD(a, 0) = a

**Examples:**
- GCD(48, 18) = 6
- GCD(100, 50) = 50
- GCD(7, 3) = 1

This is one of the oldest and most elegant recursive algorithms!`,
    examples: [
      { input: 'a = 48, b = 18', output: '6' },
      { input: 'a = 100, b = 50', output: '50' },
      { input: 'a = 7, b = 3', output: '1' },
    ],
    constraints: ['1 <= a, b <= 10⁹'],
    hints: [
      'Base case: if b is 0, return a',
      'Recursive case: gcd(a, b) = gcd(b, a % b)',
      'The algorithm always terminates because remainder gets smaller',
      'Works because GCD(a,b) = GCD(b, a mod b)',
    ],
    starterCode: `def gcd(a, b):
    """
    Calculate GCD using Euclidean algorithm with recursion.
    
    Args:
        a: First positive integer
        b: Second positive integer
        
    Returns:
        Greatest common divisor of a and b
        
    Examples:
        >>> gcd(48, 18)
        6
        >>> gcd(100, 50)
        50
    """
    pass


# Test cases
print(gcd(48, 18))   # Expected: 6
print(gcd(100, 50))  # Expected: 50
`,
    testCases: [
      { input: [48, 18], expected: 6 },
      { input: [100, 50], expected: 50 },
      { input: [7, 3], expected: 1 },
      { input: [1, 1], expected: 1 },
      { input: [20, 30], expected: 10 },
    ],
    solution: `def gcd(a, b):
    """GCD using Euclidean algorithm"""
    # Base case: when b is 0, GCD is a
    if b == 0:
        return a
    
    # Recursive case: GCD(a, b) = GCD(b, a mod b)
    return gcd(b, a % b)


# Example trace for gcd(48, 18):
# gcd(48, 18) = gcd(18, 48 % 18) = gcd(18, 12)
# gcd(18, 12) = gcd(12, 18 % 12) = gcd(12, 6)
# gcd(12, 6)  = gcd(6, 12 % 6)   = gcd(6, 0)
# gcd(6, 0)   = 6 ✓

# Time Complexity: O(log(min(a,b)))
# Space Complexity: O(log(min(a,b))) - call stack`,
    timeComplexity: 'O(log(min(a,b)))',
    spaceComplexity: 'O(log(min(a,b)))',
    followUp: [
      'How would you calculate LCM using GCD?',
      'Why does this algorithm work?',
      'Can you implement this iteratively?',
    ],
  },
  {
    id: 'recursion-count-digits',
    title: 'Count Digits',
    difficulty: 'Easy',
    topic: 'Recursion',
    description: `Count the number of digits in a positive integer using recursion.

You cannot convert to string or use logarithms - must use recursion!

**Approach:**
- Base case: single digit number (n < 10) has 1 digit
- Recursive case: remove last digit (n // 10) and add 1`,
    examples: [
      { input: 'n = 12345', output: '5' },
      { input: 'n = 7', output: '1' },
      { input: 'n = 1000', output: '4' },
    ],
    constraints: ['0 <= n <= 10⁹'],
    hints: [
      'Base case: if n < 10, it has 1 digit',
      'Remove last digit: n // 10',
      'Recursive case: 1 + count_digits(n // 10)',
      'What about n = 0? Should it have 0 or 1 digit?',
    ],
    starterCode: `def count_digits(n):
    """
    Count number of digits using recursion.
    
    Args:
        n: Positive integer
        
    Returns:
        Number of digits in n
        
    Examples:
        >>> count_digits(12345)
        5
        >>> count_digits(7)
        1
    """
    pass


# Test cases
print(count_digits(12345))  # Expected: 5
print(count_digits(7))      # Expected: 1
`,
    testCases: [
      { input: [12345], expected: 5 },
      { input: [7], expected: 1 },
      { input: [0], expected: 1 },
      { input: [1000], expected: 4 },
      { input: [999999], expected: 6 },
    ],
    solution: `def count_digits(n):
    """Count digits using recursion"""
    # Base case: single digit (including 0)
    if n < 10:
        return 1
    
    # Recursive case: remove last digit and count rest
    return 1 + count_digits(n // 10)


# Alternative handling 0 as special case:
def count_digits_alt(n):
    """Count digits - alternative"""
    if n == 0:
        return 1  # 0 has 1 digit
    if n < 10:
        return 1
    return 1 + count_digits_alt(n // 10)


# Time Complexity: O(log₁₀ n) - number of digits
# Space Complexity: O(log₁₀ n) - call stack depth`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',
    followUp: [
      'How would you sum all digits instead of counting them?',
      'Can you find the largest digit recursively?',
      'What about counting specific digits (e.g., count 7s)?',
    ],
  },
  {
    id: 'recursion-flatten-list',
    title: 'Flatten Nested List',
    difficulty: 'Medium',
    topic: 'Recursion',
    description: `Flatten a nested list structure using recursion.

Given a list that may contain integers or other nested lists, flatten it to a single-level list.

**Example:**
- Input: [1, [2, 3, [4, 5]], 6]
- Output: [1, 2, 3, 4, 5, 6]

This tests understanding of recursion with varying data types and depths.`,
    examples: [
      { input: '[[1, 2], [3, [4, 5]]]', output: '[1, 2, 3, 4, 5]' },
      { input: '[1, [2, [3, [4]]]]', output: '[1, 2, 3, 4]' },
      { input: '[]', output: '[]' },
    ],
    constraints: [
      'List can be arbitrarily nested',
      'Elements are integers or lists',
      'Total elements <= 1000',
    ],
    hints: [
      'Base case: empty list returns empty list',
      'Check if first element is a list or integer',
      'If integer, add to result and recurse on rest',
      'If list, flatten it recursively and combine with rest',
      'Use isinstance(x, list) to check if x is a list',
    ],
    starterCode: `def flatten(nested_list):
    """
    Flatten nested list structure recursively.
    
    Args:
        nested_list: List that may contain ints or nested lists
        
    Returns:
        Flattened list containing only integers
        
    Examples:
        >>> flatten([1, [2, 3], 4])
        [1, 2, 3, 4]
        >>> flatten([1, [2, [3, [4]]]])
        [1, 2, 3, 4]
    """
    pass


# Test cases
print(flatten([1, [2, 3], 4]))  # Expected: [1, 2, 3, 4]
print(flatten([1, [2, [3, [4]]]]))  # Expected: [1, 2, 3, 4]
`,
    testCases: [
      {
        input: [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        expected: [1, 2, 3, 4],
      },
      { input: [[[1, [2, [3, [4]]]]]], expected: [1, 2, 3, 4] },
      { input: [[[]]], expected: [] },
      { input: [[[1]]], expected: [1] },
      { input: [[[1, 2, 3]]], expected: [1, 2, 3] },
    ],
    solution: `def flatten(nested_list):
    """Flatten nested list recursively"""
    result = []
    
    for item in nested_list:
        if isinstance(item, list):
            # Item is a list - flatten it recursively
            result.extend(flatten(item))
        else:
            # Item is an integer - add directly
            result.append(item)
    
    return result


# Alternative approach without loop in recursion:
def flatten_pure_recursion(nested_list):
    """Flatten using pure recursion (no explicit loops)"""
    # Base case: empty list
    if not nested_list:
        return []
    
    # Get first element
    first = nested_list[0]
    
    # Recursively flatten rest
    rest = flatten_pure_recursion(nested_list[1:])
    
    # If first is list, flatten it and combine with rest
    if isinstance(first, list):
        return flatten_pure_recursion(first) + rest
    else:
        # First is integer, add to front of rest
        return [first] + rest


# Time Complexity: O(n) where n is total number of integers
# Space Complexity: O(d) where d is maximum nesting depth`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(d) where d is nesting depth',
    followUp: [
      'How would you handle other data types (strings, etc.)?',
      'Can you flatten without creating intermediate lists?',
      'What if you want to preserve one level of nesting?',
    ],
  },
  {
    id: 'recursion-tower-hanoi',
    title: 'Tower of Hanoi',
    difficulty: 'Medium',
    topic: 'Recursion',
    description: `Solve the Tower of Hanoi puzzle using recursion.

**Problem:**
- 3 rods: source, auxiliary, target
- n disks of different sizes on source rod
- Move all disks to target rod
- Rules: Only move one disk at a time, never place larger disk on smaller disk

**Solution Pattern:**
1. Move n-1 disks from source to auxiliary (using target)
2. Move largest disk from source to target
3. Move n-1 disks from auxiliary to target (using source)

Return the sequence of moves as a list of tuples (from_rod, to_rod).

This is a classic recursive problem that elegantly demonstrates the power of recursion!`,
    examples: [
      {
        input: 'n = 2, source = "A", aux = "B", target = "C"',
        output: '[("A","B"), ("A","C"), ("B","C")]',
      },
      {
        input: 'n = 3',
        output: '7 moves total',
      },
    ],
    constraints: ['1 <= n <= 15', 'Number of moves = 2^n - 1'],
    hints: [
      'Base case: n = 1, just move disk from source to target',
      'Recursive case: break into 3 steps',
      'Step 1: Move n-1 disks from source to auxiliary',
      'Step 2: Move disk n from source to target',
      'Step 3: Move n-1 disks from auxiliary to target',
      'Notice how the "helper" rod changes in each recursive call',
    ],
    starterCode: `def tower_of_hanoi(n, source='A', auxiliary='B', target='C'):
    """
    Solve Tower of Hanoi puzzle.
    
    Args:
        n: Number of disks
        source: Source rod name
        auxiliary: Auxiliary (helper) rod name
        target: Target rod name
        
    Returns:
        List of moves as tuples (from_rod, to_rod)
        
    Examples:
        >>> tower_of_hanoi(2)
        [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    pass


# Test cases
moves = tower_of_hanoi(2)
print(f"Moves for n=2: {moves}")
print(f"Total moves: {len(moves)}")  # Should be 3 (2^2 - 1)
`,
    testCases: [
      {
        input: [1, 'A', 'B', 'C'],
        expected: [['A', 'C']],
      },
      {
        input: [2, 'A', 'B', 'C'],
        expected: [
          ['A', 'B'],
          ['A', 'C'],
          ['B', 'C'],
        ],
      },
      {
        input: [3, 'A', 'B', 'C'],
        expected: [
          ['A', 'C'],
          ['A', 'B'],
          ['C', 'B'],
          ['A', 'C'],
          ['B', 'A'],
          ['B', 'C'],
          ['A', 'C'],
        ],
      },
    ],
    solution: `def tower_of_hanoi(n, source='A', auxiliary='B', target='C'):
    """
    Solve Tower of Hanoi puzzle recursively.
    
    The key insight: To move n disks from source to target:
    1. Move n-1 disks from source to auxiliary (using target as helper)
    2. Move largest disk from source to target
    3. Move n-1 disks from auxiliary to target (using source as helper)
    """
    # Base case: only one disk
    if n == 1:
        return [(source, target)]
    
    moves = []
    
    # Step 1: Move n-1 disks from source to auxiliary (using target)
    moves.extend(tower_of_hanoi(n - 1, source, target, auxiliary))
    
    # Step 2: Move largest disk from source to target
    moves.append((source, target))
    
    # Step 3: Move n-1 disks from auxiliary to target (using source)
    moves.extend(tower_of_hanoi(n - 1, auxiliary, source, target))
    
    return moves


# Verification function
def verify_tower_of_hanoi(moves, n):
    """Verify solution is correct"""
    rods = {'A': list(range(n, 0, -1)), 'B': [], 'C': []}
    
    for from_rod, to_rod in moves:
        if not rods[from_rod]:
            return False
        disk = rods[from_rod].pop()
        if rods[to_rod] and rods[to_rod][-1] < disk:
            return False  # Larger disk on smaller
        rods[to_rod].append(disk)
    
    # All disks should be on rod C
    return rods['C'] == list(range(n, 0, -1))


# Time Complexity: O(2^n) - exponential (exactly 2^n - 1 moves)
# Space Complexity: O(n) - call stack depth`,
    timeComplexity: 'O(2^n)',
    spaceComplexity: 'O(n)',
    followUp: [
      'Why does this problem require 2^n - 1 moves?',
      'Can it be solved iteratively?',
      'What if you have 4 rods instead of 3?',
    ],
  },
  {
    id: 'recursion-permutations',
    title: 'Generate All Permutations',
    difficulty: 'Medium',
    topic: 'Recursion',
    description: `Generate all possible permutations of a list of distinct integers using recursion.

A permutation is an arrangement of all elements where order matters.

**Example:**
- Input: [1, 2, 3]
- Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

For n elements, there are n! (factorial) permutations.

This is a classic backtracking problem that demonstrates recursive exploration of all possibilities.`,
    examples: [
      {
        input: 'nums = [1, 2, 3]',
        output: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
      },
      { input: 'nums = [1, 2]', output: '[[1,2],[2,1]]' },
      { input: 'nums = [1]', output: '[[1]]' },
    ],
    constraints: [
      '1 <= nums.length <= 6',
      'All integers are distinct',
      '-10 <= nums[i] <= 10',
    ],
    hints: [
      'Base case: if current permutation has all elements, add to result',
      'Try adding each unused number to current permutation',
      'Recursively build rest of permutation',
      'Backtrack by removing number after exploring',
      'Use a set or list to track which numbers are used',
    ],
    starterCode: `def permutations(nums):
    """
    Generate all permutations of nums.
    
    Args:
        nums: List of distinct integers
        
    Returns:
        List of all permutations
        
    Examples:
        >>> permutations([1, 2, 3])
        [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    """
    pass


# Test cases
result = permutations([1, 2, 3])
print(f"Permutations: {result}")
print(f"Count: {len(result)}")  # Should be 3! = 6
`,
    testCases: [
      {
        input: [[1, 2]],
        expected: [
          [1, 2],
          [2, 1],
        ],
      },
      {
        input: [[1]],
        expected: [[1]],
      },
      {
        input: [[1, 2, 3]],
        expected: [
          [1, 2, 3],
          [1, 3, 2],
          [2, 1, 3],
          [2, 3, 1],
          [3, 1, 2],
          [3, 2, 1],
        ],
      },
    ],
    solution: `def permutations(nums):
    """Generate all permutations using backtracking"""
    result = []
    
    def backtrack(current, remaining):
        # Base case: no more numbers to add
        if not remaining:
            result.append(current[:])  # Make a copy!
            return
        
        # Try each remaining number
        for i in range(len(remaining)):
            # Choose: add remaining[i] to current permutation
            current.append(remaining[i])
            
            # Explore: recurse with remaining numbers
            new_remaining = remaining[:i] + remaining[i+1:]
            backtrack(current, new_remaining)
            
            # Unchoose: backtrack
            current.pop()
    
    backtrack([], nums)
    return result


# Alternative approach using sets for tracking:
def permutations_alt(nums):
    """Generate permutations using set for tracking"""
    result = []
    used = set()
    
    def backtrack(current):
        # Base case: permutation is complete
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        # Try each unused number
        for num in nums:
            if num not in used:
                # Choose
                current.append(num)
                used.add(num)
                
                # Explore
                backtrack(current)
                
                # Unchoose
                current.pop()
                used.remove(num)
    
    backtrack([])
    return result


# Time Complexity: O(n! * n) - n! permutations, each takes O(n) to build
# Space Complexity: O(n) - recursion depth`,
    timeComplexity: 'O(n! * n)',
    spaceComplexity: 'O(n)',
    followUp: [
      'How would you handle duplicate elements?',
      'Can you generate permutations in lexicographic order?',
      'What if you only want permutations of length k?',
    ],
  },
];
