/**
 * Python Fundamentals - Batch 1 (Problems 31-40)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch1: Problem[] = [
  {
    id: 'fundamentals-char-frequency',
    title: 'Character Frequency Map',
    difficulty: 'Easy',
    description: `Create a dictionary that maps each character in a string to its frequency.

- Count all characters (including spaces and punctuation)
- Return a dictionary with character counts
- Case-sensitive counting

**Example:** "hello" → {'h': 1, 'e': 1, 'l': 2, 'o': 1}

This tests:
- Dictionary creation
- String iteration
- Frequency counting`,
    examples: [
      {
        input: 's = "hello"',
        output: "{'h': 1, 'e': 1, 'l': 2, 'o': 1}",
      },
      {
        input: 's = "aaa"',
        output: "{'a': 3}",
      },
    ],
    constraints: ['0 <= len(s) <= 10^4', 'ASCII characters only'],
    hints: [
      'Use dictionary to store counts',
      'Iterate through each character',
      'Increment count for each occurrence',
    ],
    starterCode: `def char_frequency(s):
    """
    Create frequency map of characters.
    
    Args:
        s: Input string
        
    Returns:
        Dictionary mapping characters to their counts
        
    Examples:
        >>> char_frequency("hello")
        {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    """
    pass


# Test
print(char_frequency("hello world"))
`,
    testCases: [
      {
        input: ['hello'],
        expected: { h: 1, e: 1, l: 2, o: 1 },
      },
      {
        input: ['aaa'],
        expected: { a: 3 },
      },
    ],
    solution: `def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq


# Alternative using Counter
from collections import Counter

def char_frequency_counter(s):
    return dict(Counter(s))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(k) where k is unique characters',
    order: 31,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-list-chunk',
    title: 'Chunk List into Groups',
    difficulty: 'Easy',
    description: `Split a list into chunks of specified size.

- Create sublists of size n
- Last chunk may be smaller if elements don't divide evenly
- Maintain original order

**Example:** [1,2,3,4,5], size=2 → [[1,2], [3,4], [5]]

This tests:
- List slicing
- Loop iteration
- List comprehension`,
    examples: [
      {
        input: 'arr = [1,2,3,4,5], size = 2',
        output: '[[1,2], [3,4], [5]]',
      },
      {
        input: 'arr = [1,2,3,4,5,6], size = 3',
        output: '[[1,2,3], [4,5,6]]',
      },
    ],
    constraints: ['1 <= len(arr) <= 1000', '1 <= size <= len(arr)'],
    hints: [
      'Use list slicing with step',
      'Iterate with range and step size',
      'arr[i:i+size] gets each chunk',
    ],
    starterCode: `def chunk_list(arr, size):
    """
    Split list into chunks.
    
    Args:
        arr: List to chunk
        size: Size of each chunk
        
    Returns:
        List of chunks
        
    Examples:
        >>> chunk_list([1,2,3,4,5], 2)
        [[1, 2], [3, 4], [5]]
    """
    pass


# Test
print(chunk_list([1,2,3,4,5,6,7], 3))
`,
    testCases: [
      {
        input: [[1, 2, 3, 4, 5], 2],
        expected: [[1, 2], [3, 4], [5]],
      },
      {
        input: [[1, 2, 3, 4, 5, 6], 3],
        expected: [
          [1, 2, 3],
          [4, 5, 6],
        ],
      },
    ],
    solution: `def chunk_list(arr, size):
    chunks = []
    for i in range(0, len(arr), size):
        chunks.append(arr[i:i + size])
    return chunks


# Alternative using list comprehension
def chunk_list_compact(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 32,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-gcd',
    title: 'Greatest Common Divisor',
    difficulty: 'Easy',
    description: `Find the greatest common divisor (GCD) of two numbers.

The GCD is the largest positive integer that divides both numbers evenly.

**Example:** gcd(48, 18) = 6

Use the Euclidean algorithm:
- gcd(a, b) = gcd(b, a % b)
- Base case: gcd(a, 0) = a

This tests:
- Recursion or iteration
- Modulo operations
- Mathematical algorithms`,
    examples: [
      {
        input: 'a = 48, b = 18',
        output: '6',
      },
      {
        input: 'a = 100, b = 50',
        output: '50',
      },
    ],
    constraints: ['1 <= a, b <= 10^9', 'Both numbers positive'],
    hints: [
      'Use Euclidean algorithm',
      'gcd(a, b) = gcd(b, a % b)',
      'Recursion or while loop',
    ],
    starterCode: `def gcd(a, b):
    """
    Find greatest common divisor.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        GCD of a and b
        
    Examples:
        >>> gcd(48, 18)
        6
    """
    pass


# Test
print(gcd(48, 18))
`,
    testCases: [
      {
        input: [48, 18],
        expected: 6,
      },
      {
        input: [100, 50],
        expected: 50,
      },
      {
        input: [17, 19],
        expected: 1,
      },
    ],
    solution: `def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Recursive version
def gcd_recursive(a, b):
    if b == 0:
        return a
    return gcd_recursive(b, a % b)`,
    timeComplexity: 'O(log(min(a, b)))',
    spaceComplexity: 'O(1) iterative, O(log n) recursive',
    order: 33,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-lcm',
    title: 'Least Common Multiple',
    difficulty: 'Easy',
    description: `Find the least common multiple (LCM) of two numbers.

The LCM is the smallest positive integer divisible by both numbers.

**Formula:** lcm(a, b) = (a * b) / gcd(a, b)

**Example:** lcm(4, 6) = 12

This tests:
- Using GCD to find LCM
- Mathematical relationships
- Integer division`,
    examples: [
      {
        input: 'a = 4, b = 6',
        output: '12',
      },
      {
        input: 'a = 12, b = 18',
        output: '36',
      },
    ],
    constraints: ['1 <= a, b <= 10^6', 'Both numbers positive'],
    hints: [
      'Use relationship: lcm * gcd = a * b',
      'Find GCD first',
      'Use integer division',
    ],
    starterCode: `def lcm(a, b):
    """
    Find least common multiple.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        LCM of a and b
        
    Examples:
        >>> lcm(4, 6)
        12
    """
    pass


# Test
print(lcm(12, 18))
`,
    testCases: [
      {
        input: [4, 6],
        expected: 12,
      },
      {
        input: [12, 18],
        expected: 36,
      },
      {
        input: [5, 7],
        expected: 35,
      },
    ],
    solution: `def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)


# Alternative using math module
import math

def lcm_math(a, b):
    return (a * b) // math.gcd(a, b)`,
    timeComplexity: 'O(log(min(a, b)))',
    spaceComplexity: 'O(1)',
    order: 34,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-valid-parentheses',
    title: 'Valid Parentheses',
    difficulty: 'Easy',
    description: `Determine if a string of parentheses is valid.

Valid means:
- Every opening bracket has a matching closing bracket
- Brackets are closed in the correct order
- Support: (), [], {}

**Example:** "({[]})" → True, "([)]" → False

This tests:
- Stack data structure
- String parsing
- Matching pairs`,
    examples: [
      {
        input: 's = "()"',
        output: 'True',
      },
      {
        input: 's = "()[]{}"',
        output: 'True',
      },
      {
        input: 's = "([)]"',
        output: 'False',
        explanation: 'Brackets not closed in correct order',
      },
    ],
    constraints: ['0 <= len(s) <= 10^4', 'Only parentheses characters'],
    hints: [
      'Use a stack to track opening brackets',
      'Pop from stack when closing bracket found',
      'Check if brackets match',
    ],
    starterCode: `def is_valid_parentheses(s):
    """
    Check if parentheses string is valid.
    
    Args:
        s: String of parentheses
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> is_valid_parentheses("()")
        True
        >>> is_valid_parentheses("([)]")
        False
    """
    pass


# Test
print(is_valid_parentheses("({[]})"))
`,
    testCases: [
      {
        input: ['()'],
        expected: true,
      },
      {
        input: ['()[]{}'],
        expected: true,
      },
      {
        input: ['([)]'],
        expected: false,
      },
      {
        input: ['{[]}'],
        expected: true,
      },
    ],
    solution: `def is_valid_parentheses(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 35,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-string-compress',
    title: 'String Compression',
    difficulty: 'Medium',
    description: `Compress a string using run-length encoding.

- Replace consecutive repeated characters with character + count
- If compressed string is not shorter, return original
- Only compress if result is shorter

**Example:** "aabcccccaaa" → "a2b1c5a3"

This tests:
- String building
- Counting consecutive characters
- Conditional logic`,
    examples: [
      {
        input: 's = "aabcccccaaa"',
        output: '"a2b1c5a3"',
      },
      {
        input: 's = "abcd"',
        output: '"abcd"',
        explanation: 'Compressed would be longer',
      },
    ],
    constraints: ['1 <= len(s) <= 1000', 'Only letters (a-z, A-Z)'],
    hints: [
      'Count consecutive characters',
      'Build compressed string as you go',
      'Compare lengths at the end',
    ],
    starterCode: `def compress_string(s):
    """
    Compress string using run-length encoding.
    
    Args:
        s: Input string
        
    Returns:
        Compressed string or original if not shorter
        
    Examples:
        >>> compress_string("aabcccccaaa")
        "a2b1c5a3"
    """
    pass


# Test
print(compress_string("aabcccccaaa"))
`,
    testCases: [
      {
        input: ['aabcccccaaa'],
        expected: 'a2b1c5a3',
      },
      {
        input: ['abcd'],
        expected: 'abcd',
      },
      {
        input: ['aaa'],
        expected: 'a3',
      },
    ],
    solution: `def compress_string(s):
    if not s:
        return s
    
    compressed = []
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(s[i - 1] + str(count))
            count = 1
    
    # Add last group
    compressed.append(s[-1] + str(count))
    
    result = ''.join(compressed)
    return result if len(result) < len(s) else s`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 36,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-is-subsequence',
    title: 'Check if Subsequence',
    difficulty: 'Easy',
    description: `Check if string s is a subsequence of string t.

A subsequence is formed by deleting some characters without changing the order of remaining characters.

**Example:** "ace" is a subsequence of "abcde"

This tests:
- Two pointer technique
- String traversal
- Character matching`,
    examples: [
      {
        input: 's = "ace", t = "abcde"',
        output: 'True',
      },
      {
        input: 's = "aec", t = "abcde"',
        output: 'False',
      },
    ],
    constraints: [
      '0 <= len(s), len(t) <= 10^4',
      'Only lowercase English letters',
    ],
    hints: [
      'Use two pointers',
      'Match characters in order',
      'All of s must be found in t',
    ],
    starterCode: `def is_subsequence(s, t):
    """
    Check if s is subsequence of t.
    
    Args:
        s: Potential subsequence
        t: Original string
        
    Returns:
        True if s is subsequence of t
        
    Examples:
        >>> is_subsequence("ace", "abcde")
        True
    """
    pass


# Test
print(is_subsequence("ace", "abcde"))
`,
    testCases: [
      {
        input: ['ace', 'abcde'],
        expected: true,
      },
      {
        input: ['aec', 'abcde'],
        expected: false,
      },
      {
        input: ['', 'abcde'],
        expected: true,
      },
    ],
    solution: `def is_subsequence(s, t):
    i = 0  # Pointer for s
    j = 0  # Pointer for t
    
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    
    return i == len(s)


# Alternative using iterator
def is_subsequence_iter(s, t):
    t_iter = iter(t)
    return all(char in t_iter for char in s)`,
    timeComplexity: 'O(n) where n is length of t',
    spaceComplexity: 'O(1)',
    order: 37,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-power-of-two',
    title: 'Power of Two',
    difficulty: 'Easy',
    description: `Check if a number is a power of two.

A number is a power of two if: n = 2^k for some integer k ≥ 0

**Examples:** 1, 2, 4, 8, 16... are powers of two

**Bit trick:** Powers of two have only one bit set
- n & (n-1) == 0 for powers of two

This tests:
- Bit manipulation
- Mathematical properties
- Edge cases`,
    examples: [
      {
        input: 'n = 16',
        output: 'True',
        explanation: '16 = 2^4',
      },
      {
        input: 'n = 5',
        output: 'False',
      },
    ],
    constraints: ['-2^31 <= n <= 2^31 - 1'],
    hints: [
      'Powers of 2 have only one bit set',
      'Use bit manipulation: n & (n-1)',
      'Handle edge cases: 0 and negative numbers',
    ],
    starterCode: `def is_power_of_two(n):
    """
    Check if number is power of two.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is power of 2
        
    Examples:
        >>> is_power_of_two(16)
        True
        >>> is_power_of_two(5)
        False
    """
    pass


# Test
print(is_power_of_two(16))
`,
    testCases: [
      {
        input: [16],
        expected: true,
      },
      {
        input: [5],
        expected: false,
      },
      {
        input: [1],
        expected: true,
      },
      {
        input: [0],
        expected: false,
      },
    ],
    solution: `def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


# Alternative using division
def is_power_of_two_div(n):
    if n <= 0:
        return False
    while n % 2 == 0:
        n //= 2
    return n == 1`,
    timeComplexity: 'O(1)',
    spaceComplexity: 'O(1)',
    order: 38,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-happy-number',
    title: 'Happy Number',
    difficulty: 'Easy',
    description: `Determine if a number is happy.

A happy number is defined by:
1. Start with any positive integer
2. Replace with sum of squares of its digits
3. Repeat until number equals 1 (happy) or loops endlessly (not happy)

**Example:** 19 is happy:
- 1² + 9² = 82
- 8² + 2² = 68
- 6² + 8² = 100
- 1² + 0² + 0² = 1

This tests:
- Set to detect cycles
- Digit extraction
- Loop detection`,
    examples: [
      {
        input: 'n = 19',
        output: 'True',
      },
      {
        input: 'n = 2',
        output: 'False',
        explanation: 'Enters an infinite loop',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'Use set to detect cycles',
      'Extract digits and square them',
      'Stop when n=1 or cycle detected',
    ],
    starterCode: `def is_happy(n):
    """
    Check if number is happy.
    
    Args:
        n: Positive integer
        
    Returns:
        True if happy number
        
    Examples:
        >>> is_happy(19)
        True
    """
    pass


# Test
print(is_happy(19))
`,
    testCases: [
      {
        input: [19],
        expected: true,
      },
      {
        input: [2],
        expected: false,
      },
      {
        input: [1],
        expected: true,
      },
    ],
    solution: `def is_happy(n):
    def sum_of_squares(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit ** 2
            num //= 10
        return total
    
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum_of_squares(n)
    
    return n == 1`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',
    order: 39,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-add-binary',
    title: 'Add Binary Strings',
    difficulty: 'Easy',
    description: `Add two binary strings and return the sum as a binary string.

Binary strings contain only '0' and '1' characters.

**Example:** "11" + "1" = "100" (3 + 1 = 4 in decimal)

This tests:
- String manipulation
- Binary arithmetic
- Carry handling`,
    examples: [
      {
        input: 'a = "11", b = "1"',
        output: '"100"',
      },
      {
        input: 'a = "1010", b = "1011"',
        output: '"10101"',
      },
    ],
    constraints: ['1 <= len(a), len(b) <= 10^4', "Only '0' and '1' characters"],
    hints: [
      'Process from right to left',
      'Keep track of carry',
      'Handle different lengths',
    ],
    starterCode: `def add_binary(a, b):
    """
    Add two binary strings.
    
    Args:
        a: First binary string
        b: Second binary string
        
    Returns:
        Sum as binary string
        
    Examples:
        >>> add_binary("11", "1")
        "100"
    """
    pass


# Test
print(add_binary("1010", "1011"))
`,
    testCases: [
      {
        input: ['11', '1'],
        expected: '100',
      },
      {
        input: ['1010', '1011'],
        expected: '10101',
      },
      {
        input: ['0', '0'],
        expected: '0',
      },
    ],
    solution: `def add_binary(a, b):
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    
    while i >= 0 or j >= 0 or carry:
        digit_a = int(a[i]) if i >= 0 else 0
        digit_b = int(b[j]) if j >= 0 else 0
        
        total = digit_a + digit_b + carry
        result.append(str(total % 2))
        carry = total // 2
        
        i -= 1
        j -= 1
    
    return ''.join(reversed(result))


# Alternative using Python's int conversion
def add_binary_simple(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]`,
    timeComplexity: 'O(max(len(a), len(b)))',
    spaceComplexity: 'O(max(len(a), len(b)))',
    order: 40,
    topic: 'Python Fundamentals',
  },
];
