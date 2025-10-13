import { Problem } from '../types';

export const mathGeometryProblems: Problem[] = [
  {
    id: 'rotate-image',
    title: 'Rotate Image',
    difficulty: 'Easy',
    description: `You are given an \`n x n\` 2D matrix representing an image, rotate the image by **90 degrees (clockwise)**.

You have to rotate the image **in-place**, which means you have to modify the input 2D matrix directly.


**Approach:**
1. Transpose the matrix (swap rows with columns)
2. Reverse each row

**Key Insight:**
90° clockwise rotation = transpose + reverse each row`,
    examples: [
      {
        input: 'matrix = [[1,2,3],[4,5,6],[7,8,9]]',
        output: '[[7,4,1],[8,5,2],[9,6,3]]',
      },
    ],
    constraints: ['n == matrix.length == matrix[i].length', '1 <= n <= 20'],
    hints: [
      'Transpose: swap matrix[i][j] with matrix[j][i]',
      'Then reverse each row',
    ],
    starterCode: `from typing import List

def rotate(matrix: List[List[int]]) -> None:
    """Rotate matrix 90° clockwise in-place."""
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
        expected: [
          [7, 4, 1],
          [8, 5, 2],
          [9, 6, 3],
        ],
      },
    ],
    solution: `from typing import List

def rotate(matrix: List[List[int]]) -> None:
    """Time: O(n²), Space: O(1)"""
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()
`,
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    order: 1,
    topic: 'Math & Geometry',
    leetcodeUrl: 'https://leetcode.com/problems/rotate-image/',
    youtubeUrl: 'https://www.youtube.com/watch?v=fMSJSS7eO1w',
  },
  {
    id: 'pow-x-n',
    title: 'Pow(x, n)',
    difficulty: 'Medium',
    description: `Implement \`pow(x, n)\`, which calculates \`x\` raised to the power \`n\`.


**Approach:**
Fast exponentiation using divide and conquer:
- If n is even: x^n = (x^(n/2))²
- If n is odd: x^n = x * (x^(n/2))²

**Key Insight:**
Reduces O(n) to O(log n) by halving exponent each step`,
    examples: [
      { input: 'x = 2.0, n = 10', output: '1024.0' },
      { input: 'x = 2.0, n = -2', output: '0.25' },
    ],
    constraints: ['-100 < x < 100', '-2^31 <= n <= 2^31-1'],
    hints: [
      'Use recursion',
      'Halve the exponent each time',
      'Handle negative exponents',
    ],
    starterCode: `def my_pow(x: float, n: int) -> float:
    """Calculate x raised to power n."""
    # Write your code here
    pass
`,
    testCases: [
      { input: [2.0, 10], expected: 1024.0 },
      { input: [2.0, -2], expected: 0.25 },
    ],
    solution: `def my_pow(x: float, n: int) -> float:
    """Fast exponentiation. Time: O(log n), Space: O(log n)"""
    if n == 0:
        return 1
    if n < 0:
        return 1 / my_pow(x, -n)
    
    half = my_pow(x, n // 2)
    if n % 2 == 0:
        return half * half
    return half * half * x
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',

    leetcodeUrl: 'https://leetcode.com/problems/powx-n/',
    youtubeUrl: 'https://www.youtube.com/watch?v=g9YQyYi4IQQ',
    order: 2,
    topic: 'Math & Geometry',
    leetcodeUrl: 'https://leetcode.com/problems/powx-n/',
    youtubeUrl: 'https://www.youtube.com/watch?v=g9YQyYi4IQQ',
  },
  {
    id: 'happy-number',
    title: 'Happy Number',
    difficulty: 'Hard',
    description: `A **happy number** is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat until the number equals 1 (happy), or loops endlessly in a cycle (not happy).

Return \`true\` if \`n\` is a happy number, and \`false\` if not.


**Approach:**
Use **Floyd's Cycle Detection** (slow/fast pointers) to detect if we enter a cycle.

**Key Insight:**
Either reaches 1 or enters a cycle. Detect cycle with two pointers.`,
    examples: [
      {
        input: 'n = 19',
        output: 'true',
        explanation: '1² + 9² = 82, 8² + 2² = 68, ... = 1',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'Calculate sum of squares of digits',
      'Use set to detect cycles',
      'Or use Floyd cycle detection',
    ],
    starterCode: `def is_happy(n: int) -> bool:
    """Determine if n is a happy number."""
    # Write your code here
    pass
`,
    testCases: [
      { input: [19], expected: true },
      { input: [2], expected: false },
    ],
    solution: `def is_happy(n: int) -> bool:
    """Time: O(log n), Space: O(log n)"""
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit ** 2
            num //= 10
        return total
    
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)
    
    return n == 1
`,
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(log n)',

    leetcodeUrl: 'https://leetcode.com/problems/happy-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ljz85bxOYJ0',
    order: 3,
    topic: 'Math & Geometry',
    leetcodeUrl: 'https://leetcode.com/problems/happy-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ljz85bxOYJ0',
  },

  // EASY - Plus One
  {
    id: 'plus-one',
    title: 'Plus One',
    difficulty: 'Easy',
    topic: 'Math & Geometry',
    description: `You are given a **large integer** represented as an integer array \`digits\`, where each \`digits[i]\` is the \`i-th\` digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading \`0\`'s.

Increment the large integer by one and return the resulting array of digits.`,
    examples: [
      {
        input: 'digits = [1,2,3]',
        output: '[1,2,4]',
      },
      {
        input: 'digits = [4,3,2,1]',
        output: '[4,3,2,2]',
      },
      {
        input: 'digits = [9]',
        output: '[1,0]',
      },
    ],
    constraints: [
      '1 <= digits.length <= 100',
      '0 <= digits[i] <= 9',
      'digits does not contain any leading 0s',
    ],
    hints: ['Start from the end', 'Handle carry', 'Special case: all 9s'],
    starterCode: `from typing import List

def plus_one(digits: List[int]) -> List[int]:
    """
    Add one to number represented as array.
    
    Args:
        digits: Array representing large integer
        
    Returns:
        Array with 1 added
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [[1, 2, 3]],
        expected: [1, 2, 4],
      },
      {
        input: [[4, 3, 2, 1]],
        expected: [4, 3, 2, 2],
      },
      {
        input: [[9]],
        expected: [1, 0],
      },
    ],
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/plus-one/',
    youtubeUrl: 'https://www.youtube.com/watch?v=jIaA8boiG1s',
  },

  // EASY - Palindrome Number
  {
    id: 'palindrome-number',
    title: 'Palindrome Number',
    difficulty: 'Easy',
    topic: 'Math & Geometry',
    description: `Given an integer \`x\`, return \`true\` if \`x\` is a palindrome, and \`false\` otherwise.`,
    examples: [
      {
        input: 'x = 121',
        output: 'true',
      },
      {
        input: 'x = -121',
        output: 'false',
        explanation:
          'From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.',
      },
      {
        input: 'x = 10',
        output: 'false',
      },
    ],
    constraints: ['-2^31 <= x <= 2^31 - 1'],
    hints: [
      'Reverse the number',
      'Compare with original',
      'Negative numbers are not palindromes',
    ],
    starterCode: `def is_palindrome(x: int) -> bool:
    """
    Check if integer is palindrome.
    
    Args:
        x: Integer to check
        
    Returns:
        True if palindrome
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [121],
        expected: true,
      },
      {
        input: [-121],
        expected: false,
      },
      {
        input: [10],
        expected: false,
      },
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/palindrome-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=yubRKwixN-U',
  },

  // EASY - Excel Sheet Column Number
  {
    id: 'excel-column-number',
    title: 'Excel Sheet Column Number',
    difficulty: 'Easy',
    topic: 'Math & Geometry',
    description: `Given a string \`columnTitle\` that represents the column title as appears in an Excel sheet, return its corresponding column number.

For example:

\`\`\`
A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28
...
\`\`\``,
    examples: [
      {
        input: 'columnTitle = "A"',
        output: '1',
      },
      {
        input: 'columnTitle = "AB"',
        output: '28',
      },
      {
        input: 'columnTitle = "ZY"',
        output: '701',
      },
    ],
    constraints: [
      '1 <= columnTitle.length <= 7',
      'columnTitle consists only of uppercase English letters',
      'columnTitle is in the range ["A", "FXSHRXW"]',
    ],
    hints: ['Think of it as base-26 conversion', 'Each position is 26^i'],
    starterCode: `def title_to_number(column_title: str) -> int:
    """
    Convert Excel column title to number.
    
    Args:
        column_title: Column title (A, B, AA, etc.)
        
    Returns:
        Column number
    """
    # Write your code here
    pass
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
    timeComplexity: 'O(n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/excel-sheet-column-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=g-PaNc8aD-8',
  },

  // MEDIUM - Sqrt(x)
  {
    id: 'sqrt-x',
    title: 'Sqrt(x)',
    difficulty: 'Medium',
    topic: 'Math & Geometry',
    description: `Given a non-negative integer \`x\`, return the square root of \`x\` rounded down to the nearest integer. The returned integer should be **non-negative** as well.

You **must not use** any built-in exponent function or operator.`,
    examples: [
      {
        input: 'x = 4',
        output: '2',
      },
      {
        input: 'x = 8',
        output: '2',
        explanation:
          'The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.',
      },
    ],
    constraints: ['0 <= x <= 2^31 - 1'],
    hints: [
      'Use binary search',
      'Search range [0, x]',
      'Find largest k where k*k <= x',
    ],
    starterCode: `def my_sqrt(x: int) -> int:
    """
    Find square root rounded down.
    
    Args:
        x: Non-negative integer
        
    Returns:
        Floor of square root
    """
    # Write your code here
    pass
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
    ],
    timeComplexity: 'O(log n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/sqrtx/',
    youtubeUrl: 'https://www.youtube.com/watch?v=zdMhGxRWutQ',
  },

  // MEDIUM - Multiply Strings
  {
    id: 'multiply-strings',
    title: 'Multiply Strings',
    difficulty: 'Medium',
    topic: 'Math & Geometry',
    description: `Given two non-negative integers \`num1\` and \`num2\` represented as strings, return the product of \`num1\` and \`num2\`, also represented as a string.

**Note:** You must not use any built-in BigInteger library or convert the inputs to integer directly.`,
    examples: [
      {
        input: 'num1 = "2", num2 = "3"',
        output: '"6"',
      },
      {
        input: 'num1 = "123", num2 = "456"',
        output: '"56088"',
      },
    ],
    constraints: [
      '1 <= num1.length, num2.length <= 200',
      'num1 and num2 consist of digits only',
      'Both num1 and num2 do not contain any leading zero, except the number 0 itself',
    ],
    hints: [
      'Simulate long multiplication',
      'Product of lengths m and n has at most m+n digits',
      'Position i*j contributes to result[i+j] and result[i+j+1]',
    ],
    starterCode: `def multiply(num1: str, num2: str) -> str:
    """
    Multiply two numbers represented as strings.
    
    Args:
        num1: First number as string
        num2: Second number as string
        
    Returns:
        Product as string
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: ['2', '3'],
        expected: '6',
      },
      {
        input: ['123', '456'],
        expected: '56088',
      },
    ],
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(m + n)',
    leetcodeUrl: 'https://leetcode.com/problems/multiply-strings/',
    youtubeUrl: 'https://www.youtube.com/watch?v=1vZswirL8Y8',
  },

  // MEDIUM - Spiral Matrix
  {
    id: 'spiral-matrix',
    title: 'Spiral Matrix',
    difficulty: 'Medium',
    topic: 'Math & Geometry',
    description: `Given an \`m x n\` matrix, return all elements of the matrix in spiral order.`,
    examples: [
      {
        input: 'matrix = [[1,2,3],[4,5,6],[7,8,9]]',
        output: '[1,2,3,6,9,8,7,4,5]',
      },
      {
        input: 'matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]',
        output: '[1,2,3,4,8,12,11,10,9,5,6,7]',
      },
    ],
    constraints: [
      'm == matrix.length',
      'n == matrix[i].length',
      '1 <= m, n <= 10',
      '-100 <= matrix[i][j] <= 100',
    ],
    hints: [
      'Track boundaries: top, bottom, left, right',
      'Move right, down, left, up in sequence',
      'Shrink boundaries after each direction',
    ],
    starterCode: `from typing import List

def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    Traverse matrix in spiral order.
    
    Args:
        matrix: 2D matrix
        
    Returns:
        Elements in spiral order
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ],
        ],
        expected: [1, 2, 3, 6, 9, 8, 7, 4, 5],
      },
      {
        input: [
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
          ],
        ],
        expected: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7],
      },
    ],
    timeComplexity: 'O(m * n)',
    spaceComplexity: 'O(1)',
    leetcodeUrl: 'https://leetcode.com/problems/spiral-matrix/',
    youtubeUrl: 'https://www.youtube.com/watch?v=BJnMZNwUk1M',
  },
];
