/**
 * Python Fundamentals - Batch 2 (Problems 41-50)
 * New unique problems for Python fundamentals
 */

import { Problem } from '../types';

export const pythonFundamentalsBatch2: Problem[] = [
  {
    id: 'fundamentals-roman-to-integer',
    title: 'Roman to Integer',
    difficulty: 'Easy',
    description: `Convert a Roman numeral to an integer.

Roman numerals use these symbols:
- I=1, V=5, X=10, L=50, C=100, D=500, M=1000

Subtraction rules:
- I before V or X: IV=4, IX=9
- X before L or C: XL=40, XC=90
- C before D or M: CD=400, CM=900

**Example:** "MCMXCIV" = 1994

This tests:
- Dictionary lookup
- String parsing
- Conditional logic`,
    examples: [
      {
        input: 's = "III"',
        output: '3',
      },
      {
        input: 's = "LVIII"',
        output: '58',
        explanation: 'L=50, V=5, III=3',
      },
      {
        input: 's = "MCMXCIV"',
        output: '1994',
        explanation: 'M=1000, CM=900, XC=90, IV=4',
      },
    ],
    constraints: ['1 <= len(s) <= 15', 'Valid roman numeral'],
    hints: [
      'Map each symbol to value',
      'If current < next, subtract',
      'Otherwise add',
    ],
    starterCode: `def roman_to_int(s):
    """
    Convert Roman numeral to integer.
    
    Args:
        s: Roman numeral string
        
    Returns:
        Integer value
        
    Examples:
        >>> roman_to_int("III")
        3
        >>> roman_to_int("LVIII")
        58
    """
    pass


# Test
print(roman_to_int("MCMXCIV"))
`,
    testCases: [
      {
        input: ['III'],
        expected: 3,
      },
      {
        input: ['LVIII'],
        expected: 58,
      },
      {
        input: ['MCMXCIV'],
        expected: 1994,
      },
    ],
    solution: `def roman_to_int(s):
    values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    total = 0
    for i in range(len(s)):
        # If current value < next value, subtract
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            total -= values[s[i]]
        else:
            total += values[s[i]]
    
    return total`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 41,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-integer-to-roman',
    title: 'Integer to Roman',
    difficulty: 'Medium',
    description: `Convert an integer to a Roman numeral.

Use these values in descending order:
- 1000='M', 900='CM', 500='D', 400='CD'
- 100='C', 90='XC', 50='L', 40='XL'
- 10='X', 9='IX', 5='V', 4='IV', 1='I'

**Example:** 1994 = "MCMXCIV"

This tests:
- Greedy algorithm
- String building
- Value mapping`,
    examples: [
      {
        input: 'num = 3',
        output: '"III"',
      },
      {
        input: 'num = 58',
        output: '"LVIII"',
      },
      {
        input: 'num = 1994',
        output: '"MCMXCIV"',
      },
    ],
    constraints: ['1 <= num <= 3999'],
    hints: [
      'Use values in descending order',
      'Subtract largest value repeatedly',
      'Build string as you go',
    ],
    starterCode: `def int_to_roman(num):
    """
    Convert integer to Roman numeral.
    
    Args:
        num: Integer to convert
        
    Returns:
        Roman numeral string
        
    Examples:
        >>> int_to_roman(3)
        "III"
        >>> int_to_roman(1994)
        "MCMXCIV"
    """
    pass


# Test
print(int_to_roman(1994))
`,
    testCases: [
      {
        input: [3],
        expected: 'III',
      },
      {
        input: [58],
        expected: 'LVIII',
      },
      {
        input: [1994],
        expected: 'MCMXCIV',
      },
    ],
    solution: `def int_to_roman(num):
    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    
    result = []
    for value, symbol in values:
        count = num // value
        if count:
            result.append(symbol * count)
            num -= value * count
    
    return ''.join(result)`,
    timeComplexity: 'O(1) - fixed number of iterations',
    spaceComplexity: 'O(1)',
    order: 42,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-array-intersection',
    title: 'Intersection of Two Arrays',
    difficulty: 'Easy',
    description: `Find the intersection of two arrays.

Return elements that appear in both arrays.
- Each element should appear as many times as it shows in both arrays
- Result can be in any order

**Example:** [1,2,2,1] ∩ [2,2] = [2,2]

This tests:
- Set operations
- Counter usage
- Array manipulation`,
    examples: [
      {
        input: 'nums1 = [1,2,2,1], nums2 = [2,2]',
        output: '[2,2]',
      },
      {
        input: 'nums1 = [4,9,5], nums2 = [9,4,9,8,4]',
        output: '[4,9] or [9,4]',
      },
    ],
    constraints: ['1 <= len(nums1), len(nums2) <= 1000'],
    hints: [
      'Use Counter or dictionary',
      'Count occurrences in both arrays',
      'Take minimum count for each element',
    ],
    starterCode: `def array_intersection(nums1, nums2):
    """
    Find intersection of two arrays.
    
    Args:
        nums1: First array
        nums2: Second array
        
    Returns:
        Array of intersecting elements
        
    Examples:
        >>> array_intersection([1,2,2,1], [2,2])
        [2, 2]
    """
    pass


# Test
print(array_intersection([4,9,5], [9,4,9,8,4]))
`,
    testCases: [
      {
        input: [
          [1, 2, 2, 1],
          [2, 2],
        ],
        expected: [2, 2],
      },
      {
        input: [
          [4, 9, 5],
          [9, 4, 9, 8, 4],
        ],
        expected: [9, 4],
      },
    ],
    solution: `def array_intersection(nums1, nums2):
    from collections import Counter
    
    count1 = Counter(nums1)
    count2 = Counter(nums2)
    
    result = []
    for num in count1:
        if num in count2:
            result.extend([num] * min(count1[num], count2[num]))
    
    return result


# Alternative using set and list
def array_intersection_simple(nums1, nums2):
    result = []
    nums2_copy = nums2.copy()
    
    for num in nums1:
        if num in nums2_copy:
            result.append(num)
            nums2_copy.remove(num)
    
    return result`,
    timeComplexity: 'O(n + m)',
    spaceComplexity: 'O(min(n, m))',
    order: 43,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-plus-one',
    title: 'Plus One',
    difficulty: 'Easy',
    description: `Given a non-empty array of digits representing a non-negative integer, add one to the integer.

Digits are stored such that most significant digit is at head of list.

**Example:** [1,2,3] represents 123, so +1 = [1,2,4]

Handle carry: [9,9,9] + 1 = [1,0,0,0]

This tests:
- Array manipulation
- Carry propagation
- Edge cases`,
    examples: [
      {
        input: 'digits = [1,2,3]',
        output: '[1,2,4]',
      },
      {
        input: 'digits = [9,9,9]',
        output: '[1,0,0,0]',
      },
    ],
    constraints: ['1 <= len(digits) <= 100', '0 <= digits[i] <= 9'],
    hints: [
      'Start from the end',
      'Handle carry propagation',
      'Add new digit at front if needed',
    ],
    starterCode: `def plus_one(digits):
    """
    Add one to number represented as array.
    
    Args:
        digits: Array of digits
        
    Returns:
        Array representing digits + 1
        
    Examples:
        >>> plus_one([1,2,3])
        [1, 2, 4]
        >>> plus_one([9,9,9])
        [1, 0, 0, 0]
    """
    pass


# Test
print(plus_one([9,9,9]))
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [1, 2, 4],
      },
      {
        input: [[9, 9, 9]],
        expected: [1, 0, 0, 0],
      },
      {
        input: [[0]],
        expected: [1],
      },
    ],
    solution: `def plus_one(digits):
    n = len(digits)
    
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits
        digits[i] = 0
    
    # If we're here, all digits were 9
    return [1] + digits


# Alternative converting to/from int
def plus_one_simple(digits):
    num = int(''.join(map(str, digits))) + 1
    return [int(d) for d in str(num)]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) or O(n) if all 9s',
    order: 44,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-sqrt',
    title: 'Square Root (Integer)',
    difficulty: 'Easy',
    description: `Find the integer square root of a number.

Return the largest integer x such that x² ≤ n.

**Example:** sqrt(8) = 2 (not 3, because 3² = 9 > 8)

Use binary search for efficient solution.

This tests:
- Binary search
- Integer arithmetic
- Boundary conditions`,
    examples: [
      {
        input: 'n = 4',
        output: '2',
      },
      {
        input: 'n = 8',
        output: '2',
        explanation: '2² = 4, 3² = 9',
      },
    ],
    constraints: ['0 <= n <= 2^31 - 1'],
    hints: [
      'Use binary search',
      'Search range: 0 to n',
      'Check if mid * mid <= n',
    ],
    starterCode: `def sqrt(n):
    """
    Find integer square root.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Largest integer x where x² ≤ n
        
    Examples:
        >>> sqrt(4)
        2
        >>> sqrt(8)
        2
    """
    pass


# Test
print(sqrt(8))
`,
    testCases: [
      {
        input: [4],
        expected: 2,
      },
      {
        input: [8],
        expected: 2,
      },
      {
        input: [0],
        expected: 0,
      },
      {
        input: [1],
        expected: 1,
      },
    ],
    solution: `def sqrt(n):
    if n < 2:
        return n
    
    left, right = 1, n // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == n:
            return mid
        elif square < n:
            left = mid + 1
        else:
            right = mid - 1
    
    return right


# Alternative using Newton's method
def sqrt_newton(n):
    if n < 2:
        return n
    
    x = n
    while x * x > n:
        x = (x + n // x) // 2
    
    return x`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    order: 45,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-climbing-stairs',
    title: 'Climbing Stairs',
    difficulty: 'Easy',
    description: `You're climbing a staircase with n steps. You can climb 1 or 2 steps at a time.

How many distinct ways can you climb to the top?

**Pattern:** This follows Fibonacci sequence!
- ways(n) = ways(n-1) + ways(n-2)
- Base cases: ways(1)=1, ways(2)=2

**Example:** n=3 → 3 ways: (1+1+1), (1+2), (2+1)

This tests:
- Dynamic programming
- Pattern recognition
- Memoization`,
    examples: [
      {
        input: 'n = 2',
        output: '2',
        explanation: '1+1 or 2',
      },
      {
        input: 'n = 3',
        output: '3',
        explanation: '1+1+1, 1+2, or 2+1',
      },
    ],
    constraints: ['1 <= n <= 45'],
    hints: [
      "It's like Fibonacci!",
      'ways(n) = ways(n-1) + ways(n-2)',
      'Use dynamic programming',
    ],
    starterCode: `def climb_stairs(n):
    """
    Count ways to climb n stairs.
    
    Args:
        n: Number of stairs
        
    Returns:
        Number of distinct ways
        
    Examples:
        >>> climb_stairs(2)
        2
        >>> climb_stairs(3)
        3
    """
    pass


# Test
print(climb_stairs(5))
`,
    testCases: [
      {
        input: [2],
        expected: 2,
      },
      {
        input: [3],
        expected: 3,
      },
      {
        input: [5],
        expected: 8,
      },
    ],
    solution: `def climb_stairs(n):
    if n <= 2:
        return n
    
    prev2 = 1  # ways(1)
    prev1 = 2  # ways(2)
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


# Recursive with memoization
def climb_stairs_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return n
    
    memo[n] = climb_stairs_memo(n - 1, memo) + climb_stairs_memo(n - 2, memo)
    return memo[n]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 46,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-pascals-triangle',
    title: "Pascal's Triangle",
    difficulty: 'Easy',
    description: `Generate the first n rows of Pascal's Triangle.

Each number is the sum of the two numbers directly above it.

**Pattern:**
\`\`\`
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
\`\`\`

This tests:
- 2D array generation
- Pattern recognition
- List manipulation`,
    examples: [
      {
        input: 'n = 5',
        output: '[[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]',
      },
    ],
    constraints: ['1 <= n <= 30'],
    hints: [
      'Start each row with 1',
      'Each middle element = prev[i-1] + prev[i]',
      'End each row with 1',
    ],
    starterCode: `def generate_pascals_triangle(n):
    """
    Generate first n rows of Pascal's Triangle.
    
    Args:
        n: Number of rows
        
    Returns:
        2D list representing triangle
        
    Examples:
        >>> generate_pascals_triangle(5)
        [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]
    """
    pass


# Test
print(generate_pascals_triangle(5))
`,
    testCases: [
      {
        input: [5],
        expected: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]],
      },
      {
        input: [1],
        expected: [[1]],
      },
    ],
    solution: `def generate_pascals_triangle(n):
    triangle = []
    
    for i in range(n):
        row = [1] * (i + 1)
        
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        
        triangle.append(row)
    
    return triangle


# Alternative more compact
def generate_pascals_triangle_compact(n):
    result = [[1]]
    
    for i in range(1, n):
        prev = result[-1]
        new_row = [1]
        for j in range(len(prev) - 1):
            new_row.append(prev[j] + prev[j + 1])
        new_row.append(1)
        result.append(new_row)
    
    return result`,
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(n²)',
    order: 47,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-excel-column-number',
    title: 'Excel Column Number',
    difficulty: 'Easy',
    description: `Convert an Excel column title to its column number.

Excel columns: A=1, B=2, ..., Z=26, AA=27, AB=28, ...

This is base-26 number system where:
- A-Z represent 1-26 (not 0-25!)

**Example:** "AB" = 1*26 + 2 = 28

This tests:
- Number system conversion
- String processing
- Mathematical calculation`,
    examples: [
      {
        input: 'column = "A"',
        output: '1',
      },
      {
        input: 'column = "AB"',
        output: '28',
      },
      {
        input: 'column = "ZY"',
        output: '701',
      },
    ],
    constraints: ['1 <= len(column) <= 7', 'Only uppercase letters'],
    hints: [
      'Similar to base conversion',
      'A=1, not A=0',
      'Process left to right, multiply by 26',
    ],
    starterCode: `def title_to_number(column):
    """
    Convert Excel column title to number.
    
    Args:
        column: Column title (e.g., "AB")
        
    Returns:
        Column number
        
    Examples:
        >>> title_to_number("A")
        1
        >>> title_to_number("AB")
        28
    """
    pass


# Test
print(title_to_number("ZY"))
`,
    testCases: [
      {
        input: ['A'],
        expected: 1,
      },
      {
        input: ['AB'],
        expected: 28,
      },
      {
        input: ['ZY'],
        expected: 701,
      },
    ],
    solution: `def title_to_number(column):
    result = 0
    
    for char in column:
        result = result * 26 + (ord(char) - ord('A') + 1)
    
    return result


# Alternative more explicit
def title_to_number_explicit(column):
    result = 0
    power = 0
    
    for i in range(len(column) - 1, -1, -1):
        digit = ord(column[i]) - ord('A') + 1
        result += digit * (26 ** power)
        power += 1
    
    return result`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    order: 48,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-majority-element',
    title: 'Majority Element',
    difficulty: 'Easy',
    description: `Find the majority element in an array.

The majority element appears more than ⌊n/2⌋ times.

**Guaranteed:** The majority element always exists in the array.

**Boyer-Moore Algorithm:** Efficient O(1) space solution using voting.

This tests:
- Array traversal
- Counter or voting algorithm
- Majority logic`,
    examples: [
      {
        input: 'nums = [3,2,3]',
        output: '3',
      },
      {
        input: 'nums = [2,2,1,1,1,2,2]',
        output: '2',
      },
    ],
    constraints: ['1 <= len(nums) <= 5*10^4', 'Majority element always exists'],
    hints: [
      'Use Counter for simple solution',
      'Boyer-Moore voting for O(1) space',
      'Candidate changes when count reaches 0',
    ],
    starterCode: `def majority_element(nums):
    """
    Find majority element.
    
    Args:
        nums: Array of integers
        
    Returns:
        The majority element
        
    Examples:
        >>> majority_element([3,2,3])
        3
    """
    pass


# Test
print(majority_element([2,2,1,1,1,2,2]))
`,
    testCases: [
      {
        input: [[3, 2, 3]],
        expected: 3,
      },
      {
        input: [[2, 2, 1, 1, 1, 2, 2]],
        expected: 2,
      },
    ],
    solution: `def majority_element(nums):
    # Boyer-Moore Voting Algorithm
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    
    return candidate


# Alternative using Counter
from collections import Counter

def majority_element_counter(nums):
    counts = Counter(nums)
    return counts.most_common(1)[0][0]`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1) with Boyer-Moore',
    order: 49,
    topic: 'Python Fundamentals',
  },
  {
    id: 'fundamentals-contains-duplicate',
    title: 'Contains Duplicate',
    difficulty: 'Easy',
    description: `Check if an array contains any duplicate values.

Return true if any value appears at least twice.

**Example:** [1,2,3,1] → true, [1,2,3,4] → false

This tests:
- Set operations
- Array traversal
- Duplicate detection`,
    examples: [
      {
        input: 'nums = [1,2,3,1]',
        output: 'True',
      },
      {
        input: 'nums = [1,2,3,4]',
        output: 'False',
      },
    ],
    constraints: ['1 <= len(nums) <= 10^5'],
    hints: [
      'Use set to track seen elements',
      'Or compare len(nums) vs len(set(nums))',
      'Early return when duplicate found',
    ],
    starterCode: `def contains_duplicate(nums):
    """
    Check if array has duplicates.
    
    Args:
        nums: Array of integers
        
    Returns:
        True if duplicate exists
        
    Examples:
        >>> contains_duplicate([1,2,3,1])
        True
        >>> contains_duplicate([1,2,3,4])
        False
    """
    pass


# Test
print(contains_duplicate([1,2,3,1]))
`,
    testCases: [
      {
        input: [[1, 2, 3, 1]],
        expected: true,
      },
      {
        input: [[1, 2, 3, 4]],
        expected: false,
      },
    ],
    solution: `def contains_duplicate(nums):
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False


# Alternative one-liner
def contains_duplicate_simple(nums):
    return len(nums) != len(set(nums))`,
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(n)',
    order: 50,
    topic: 'Python Fundamentals',
  },
];
