/**
 * Python Fundamentals - Batch 7 (Problems 91-100) - FINAL BATCH
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch7: Problem[] = [
  {
    id: 'fundamentals-keyboard-row',
    title: 'Keyboard Row Words',
    difficulty: 'Easy',
    description: `Find words that can be typed using only one row of keyboard.

Keyboard rows:
- Row 1: "qwertyuiop"
- Row 2: "asdfghjkl"
- Row 3: "zxcvbnm"

**Example:** ["Hello", "Alaska", "Dad"] → ["Alaska", "Dad"]

This tests:
- Set operations
- String filtering
- Case handling`,
    examples: [
      {
        input: 'words = ["Hello","Alaska","Dad","Peace"]',
        output: '["Alaska","Dad"]',
      },
    ],
    constraints: ['1 <= len(words) <= 20', '1 <= len(words[i]) <= 100'],
    hints: [
      'Create sets for each row',
      'Check if all letters in one set',
      'Handle case-insensitive',
    ],
    starterCode: `def find_words(words):
    """
    Find words from single keyboard row.
    
    Args:
        words: List of words
        
    Returns:
        List of valid words
        
    Examples:
        >>> find_words(["Hello","Alaska","Dad"])
        ["Alaska", "Dad"]
    """
    pass


# Test
print(find_words(["Hello","Alaska","Dad","Peace"]))
`,
    testCases: [
      {
        input: [['Hello', 'Alaska', 'Dad', 'Peace']],
        expected: ['Alaska', 'Dad'],
      },
    ],
    solution: `def find_words(words):
    rows = [
        set('qwertyuiop'),
        set('asdfghjkl'),
        set('zxcvbnm')
    ]
    
    result = []
    
    for word in words:
        word_lower = word.lower()
        for row in rows:
            if all(char in row for char in word_lower):
                result.append(word)
                break
    
    return result`,
    timeComplexity: 'O(n * m) where m is avg word length',
    spaceComplexity: 'O(1)',
    order: 91,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-base7',
    title: 'Base 7',
    difficulty: 'Easy',
    description: `Convert an integer to its base 7 string representation.

**Example:** 100 in base 10 = "202" in base 7

Handle negative numbers: -7 → "-10"

This tests:
- Number system conversion
- String building
- Sign handling`,
    examples: [
      {
        input: 'num = 100',
        output: '"202"',
      },
      {
        input: 'num = -7',
        output: '"-10"',
      },
    ],
    constraints: ['-10^7 <= num <= 10^7'],
    hints: [
      'Handle sign separately',
      'Repeatedly divide by 7',
      'Build result from remainders',
    ],
    starterCode: `def convert_to_base7(num):
    """
    Convert to base 7.
    
    Args:
        num: Integer to convert
        
    Returns:
        Base 7 string representation
        
    Examples:
        >>> convert_to_base7(100)
        "202"
    """
    pass


# Test
print(convert_to_base7(100))
`,
    testCases: [
      {
        input: [100],
        expected: '202',
      },
      {
        input: [-7],
        expected: '-10',
      },
      {
        input: [0],
        expected: '0',
      },
    ],
    solution: `def convert_to_base7(num):
    if num == 0:
        return "0"
    
    negative = num < 0
    num = abs(num)
    
    result = []
    while num > 0:
        result.append(str(num % 7))
        num //= 7
    
    base7 = ''.join(reversed(result))
    return '-' + base7 if negative else base7`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',
    order: 92,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-relative-ranks',
    title: 'Relative Ranks',
    difficulty: 'Easy',
    description: `Given scores, assign ranks with special medals.

Ranks:
- 1st place: "Gold Medal"
- 2nd place: "Silver Medal"
- 3rd place: "Bronze Medal"
- Others: "4", "5", etc.

**Example:** [5,4,3,2,1] → ["Gold Medal","Silver Medal","Bronze Medal","4","5"]

This tests:
- Sorting with indices
- Ranking
- String formatting`,
    examples: [
      {
        input: 'score = [5,4,3,2,1]',
        output: '["Gold Medal","Silver Medal","Bronze Medal","4","5"]',
      },
    ],
    constraints: ['1 <= len(score) <= 10^4', 'All scores unique'],
    hints: [
      'Sort with original indices',
      'Map ranks to medals/numbers',
      'Restore original order',
    ],
    starterCode: `def find_relative_ranks(score):
    """
    Assign ranks with medals.
    
    Args:
        score: Array of scores
        
    Returns:
        Array of rank strings
        
    Examples:
        >>> find_relative_ranks([5,4,3,2,1])
        ["Gold Medal","Silver Medal","Bronze Medal","4","5"]
    """
    pass


# Test
print(find_relative_ranks([10,3,8,9,4]))
`,
    testCases: [
      {
        input: [[5, 4, 3, 2, 1]],
        expected: ['Gold Medal', 'Silver Medal', 'Bronze Medal', '4', '5'],
      },
      {
        input: [[10, 3, 8, 9, 4]],
        expected: ['Gold Medal', '5', 'Bronze Medal', 'Silver Medal', '4'],
      },
    ],
    solution: `def find_relative_ranks(score):
    n = len(score)
    # Create list of (score, index) and sort by score descending
    sorted_scores = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    result = [""] * n
    
    for rank, (original_idx, _) in enumerate(sorted_scores):
        if rank < 3:
            result[original_idx] = medals[rank]
        else:
            result[original_idx] = str(rank + 1)
    
    return result`,
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    order: 93,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-perfect-number-check',
    title: 'Perfect Number (Efficient)',
    difficulty: 'Easy',
    description: `Check if number is perfect (efficient version).

Perfect number: equals sum of its positive divisors (excluding itself).

**Optimization:** Only check divisors up to √n

**Example:** 28 = 1 + 2 + 4 + 7 + 14

This tests:
- Divisor finding
- Square root optimization
- Efficient checking`,
    examples: [
      {
        input: 'num = 28',
        output: 'True',
      },
      {
        input: 'num = 7',
        output: 'False',
      },
    ],
    constraints: ['1 <= num <= 10^8'],
    hints: [
      'Check divisors up to √num',
      'Add both i and num/i',
      'Handle perfect square case',
    ],
    starterCode: `def check_perfect_number(num):
    """
    Check if perfect number (efficient).
    
    Args:
        num: Positive integer
        
    Returns:
        True if perfect
        
    Examples:
        >>> check_perfect_number(28)
        True
    """
    pass


# Test
print(check_perfect_number(28))
`,
    testCases: [
      {
        input: [28],
        expected: true,
      },
      {
        input: [7],
        expected: false,
      },
      {
        input: [1],
        expected: false,
      },
    ],
    solution: `def check_perfect_number(num):
    if num <= 1:
        return False
    
    divisor_sum = 1  # 1 is always a divisor
    
    # Check divisors up to sqrt(num)
    i = 2
    while i * i <= num:
        if num % i == 0:
            divisor_sum += i
            if i * i != num:
                divisor_sum += num // i
        i += 1
    
    return divisor_sum == num`,
    timeComplexity: 'O(√n)',
    spaceComplexity: 'O(1)',
    order: 94,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-fib-recursive-memoized',
    title: 'Fibonacci with Memoization',
    difficulty: 'Easy',
    description: `Implement Fibonacci using recursion with memoization.

Use a dictionary to cache computed values.

**Example:** fib(10) computes each fib(i) only once

This tests:
- Recursion
- Memoization
- Dictionary usage`,
    examples: [
      {
        input: 'n = 10',
        output: '55',
      },
    ],
    constraints: ['0 <= n <= 100'],
    hints: [
      'Use dictionary to store results',
      'Check cache before computing',
      'Base cases: fib(0)=0, fib(1)=1',
    ],
    starterCode: `def fib_memoized(n, memo=None):
    """
    Fibonacci with memoization.
    
    Args:
        n: Position in sequence
        memo: Memoization dict
        
    Returns:
        Nth Fibonacci number
        
    Examples:
        >>> fib_memoized(10)
        55
    """
    pass


# Test
print(fib_memoized(10))
`,
    testCases: [
      {
        input: [10],
        expected: 55,
      },
      {
        input: [20],
        expected: 6765,
      },
    ],
    solution: `def fib_memoized(n, memo=None):
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fib_memoized(n - 1, memo) + fib_memoized(n - 2, memo)
    return memo[n]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 95,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-detect-capital',
    title: 'Detect Capital Use',
    difficulty: 'Easy',
    description: `Check if word uses capitals correctly.

Valid patterns:
1. All capitals: "USA"
2. All lowercase: "leetcode"
3. Only first capital: "Google"

**Example:** "USA" → true, "FlaG" → false

This tests:
- String case checking
- Pattern matching
- Boolean logic`,
    examples: [
      {
        input: 'word = "USA"',
        output: 'True',
      },
      {
        input: 'word = "FlaG"',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(word) <= 100', 'Only letters'],
    hints: [
      'Use isupper() and islower()',
      'Check all caps, all lower, or first cap',
      'Use string methods',
    ],
    starterCode: `def detect_capital_use(word):
    """
    Check if capitals used correctly.
    
    Args:
        word: Input word
        
    Returns:
        True if valid capital usage
        
    Examples:
        >>> detect_capital_use("USA")
        True
    """
    pass


# Test
print(detect_capital_use("FlaG"))
`,
    testCases: [
      {
        input: ['USA'],
        expected: true,
      },
      {
        input: ['FlaG'],
        expected: false,
      },
      {
        input: ['Google'],
        expected: true,
      },
    ],
    solution: `def detect_capital_use(word):
    return (word.isupper() or 
            word.islower() or 
            (word[0].isupper() and word[1:].islower()))


# Alternative explicit check
def detect_capital_use_explicit(word):
    if len(word) == 1:
        return True
    
    # All uppercase
    if all(c.isupper() for c in word):
        return True
    
    # All lowercase
    if all(c.islower() for c in word):
        return True
    
    # First uppercase, rest lowercase
    if word[0].isupper() and all(c.islower() for c in word[1:]):
        return True
    
    return False`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 96,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-longest-palindrome',
    title: 'Longest Palindrome Length',
    difficulty: 'Easy',
    description: `Find length of longest palindrome that can be built from string characters.

Use each character as many times as it appears.

**Example:** "abccccdd" → 7 ("dccaccd")

This tests:
- Character frequency
- Even/odd counting
- Palindrome properties`,
    examples: [
      {
        input: 's = "abccccdd"',
        output: '7',
      },
      {
        input: 's = "a"',
        output: '1',
      },
    ],
    constraints: ['1 <= len(s) <= 2000', 'Only letters'],
    hints: [
      'Count character frequencies',
      'Use pairs (even counts)',
      'Can use one odd count in middle',
    ],
    starterCode: `def longest_palindrome(s):
    """
    Find longest palindrome length.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest palindrome
        
    Examples:
        >>> longest_palindrome("abccccdd")
        7
    """
    pass


# Test
print(longest_palindrome("abccccdd"))
`,
    testCases: [
      {
        input: ['abccccdd'],
        expected: 7,
      },
      {
        input: ['a'],
        expected: 1,
      },
      {
        input: ['bb'],
        expected: 2,
      },
    ],
    solution: `def longest_palindrome(s):
    from collections import Counter
    
    counts = Counter(s)
    length = 0
    has_odd = False
    
    for count in counts.values():
        length += count // 2 * 2
        if count % 2 == 1:
            has_odd = True
    
    # Add 1 for middle char if any odd count
    return length + (1 if has_odd else 0)`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) - at most 52 letters',
    order: 97,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-fizz-buzz-variant',
    title: 'FizzBuzz Variant (Divisibility)',
    difficulty: 'Easy',
    description: `FizzBuzz variant with custom divisors.

Given n, divisor1, divisor2:
- Multiple of both: "FizzBuzz"
- Multiple of divisor1: "Fizz"
- Multiple of divisor2: "Buzz"
- Otherwise: number as string

**Example:** n=15, d1=3, d2=5 (classic FizzBuzz)

This tests:
- Divisibility checking
- Conditional logic
- Parameter handling`,
    examples: [
      {
        input: 'n = 15, divisor1 = 3, divisor2 = 5',
        output: '["1","2","Fizz",...,"FizzBuzz"]',
      },
    ],
    constraints: ['1 <= n <= 10^4', '1 <= divisor1, divisor2 <= 10^4'],
    hints: [
      'Check both divisors first',
      'Then check individual divisors',
      'Return list of strings',
    ],
    starterCode: `def fizz_buzz_variant(n, divisor1, divisor2):
    """
    FizzBuzz with custom divisors.
    
    Args:
        n: Upper limit
        divisor1: First divisor (Fizz)
        divisor2: Second divisor (Buzz)
        
    Returns:
        FizzBuzz list
        
    Examples:
        >>> fizz_buzz_variant(15, 3, 5)
        ["1","2","Fizz",...,"FizzBuzz"]
    """
    pass


# Test
print(fizz_buzz_variant(15, 3, 5))
`,
    testCases: [
      {
        input: [15, 3, 5],
        expected: [
          '1',
          '2',
          'Fizz',
          '4',
          'Buzz',
          'Fizz',
          '7',
          '8',
          'Fizz',
          'Buzz',
          '11',
          'Fizz',
          '13',
          '14',
          'FizzBuzz',
        ],
      },
    ],
    solution: `def fizz_buzz_variant(n, divisor1, divisor2):
    result = []
    
    for i in range(1, n + 1):
        if i % divisor1 == 0 and i % divisor2 == 0:
            result.append("FizzBuzz")
        elif i % divisor1 == 0:
            result.append("Fizz")
        elif i % divisor2 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 98,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-next-greater-element-simple',
    title: 'Next Greater Element (Simple)',
    difficulty: 'Easy',
    description: `For each element in nums1, find the next greater element in nums2.

nums1 is subset of nums2.
Next greater = first greater element to the right in nums2.

**Example:** nums1=[4,1,2], nums2=[1,3,4,2]
→ [-1,3,-1] (4: none, 1→3, 2: none)

This tests:
- Array traversal
- Finding next greater
- Hash map usage`,
    examples: [
      {
        input: 'nums1 = [4,1,2], nums2 = [1,3,4,2]',
        output: '[-1,3,-1]',
      },
    ],
    constraints: ['1 <= len(nums1), len(nums2) <= 1000'],
    hints: [
      'Build map of next greater in nums2',
      'Use stack for efficient next greater',
      'Look up each nums1 element',
    ],
    starterCode: `def next_greater_element(nums1, nums2):
    """
    Find next greater elements.
    
    Args:
        nums1: Query array
        nums2: Search array
        
    Returns:
        Array of next greater elements
        
    Examples:
        >>> next_greater_element([4,1,2], [1,3,4,2])
        [-1, 3, -1]
    """
    pass


# Test
print(next_greater_element([4,1,2], [1,3,4,2]))
`,
    testCases: [
      {
        input: [
          [4, 1, 2],
          [1, 3, 4, 2],
        ],
        expected: [-1, 3, -1],
      },
      {
        input: [
          [2, 4],
          [1, 2, 3, 4],
        ],
        expected: [3, -1],
      },
    ],
    solution: `def next_greater_element(nums1, nums2):
    # Build next greater map for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # Query for nums1
    return [next_greater.get(num, -1) for num in nums1]


# Brute force O(n*m)
def next_greater_element_brute(nums1, nums2):
    result = []
    
    for num in nums1:
        idx = nums2.index(num)
        found = -1
        for i in range(idx + 1, len(nums2)):
            if nums2[i] > num:
                found = nums2[i]
                break
        result.append(found)
    
    return result`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(n)',
    order: 99,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-teemo-attacking',
    title: 'Teemo Attacking',
    difficulty: 'Easy',
    description: `Calculate total time target is poisoned.

Each attack poisons for duration seconds.
If attacked again before poison expires, timer resets (doesn't stack).

**Example:** timeSeries=[1,4], duration=2
→ 4 (poisoned: 1-2, 4-5)

This tests:
- Interval merging
- Time calculation
- Overlap handling`,
    examples: [
      {
        input: 'timeSeries = [1,4], duration = 2',
        output: '4',
      },
      {
        input: 'timeSeries = [1,2], duration = 2',
        output: '3',
        explanation: 'Overlap at time 2',
      },
    ],
    constraints: ['1 <= len(timeSeries) <= 10^4', '1 <= duration <= 10^9'],
    hints: [
      'Compare attack time with previous end time',
      'Add min(duration, gap) for each attack',
      'Last attack always adds full duration',
    ],
    starterCode: `def find_poisoned_duration(time_series, duration):
    """
    Calculate total poisoned time.
    
    Args:
        time_series: Attack times (sorted)
        duration: Poison duration
        
    Returns:
        Total poisoned time
        
    Examples:
        >>> find_poisoned_duration([1,4], 2)
        4
    """
    pass


# Test
print(find_poisoned_duration([1,2,3,4,5], 5))
`,
    testCases: [
      {
        input: [[1, 4], 2],
        expected: 4,
      },
      {
        input: [[1, 2], 2],
        expected: 3,
      },
      {
        input: [[1, 2, 3, 4, 5], 5],
        expected: 9,
      },
    ],
    solution: `def find_poisoned_duration(time_series, duration):
    if not time_series:
        return 0
    
    total = 0
    
    for i in range(len(time_series) - 1):
        # Add either full duration or time until next attack
        total += min(duration, time_series[i + 1] - time_series[i])
    
    # Last attack always adds full duration
    total += duration
    
    return total`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 100,
    topic: 'Python Fundamentals',
  },
];
