import { Problem } from '../types';

export const mathGeometryProblems: Problem[] = [
  {
    id: 'rotate-image',
    title: 'Rotate Image',
    difficulty: 'Easy',
    description: `You are given an \`n x n\` 2D matrix representing an image, rotate the image by **90 degrees (clockwise)**.

You have to rotate the image **in-place**, which means you have to modify the input 2D matrix directly.

**LeetCode:** [48. Rotate Image](https://leetcode.com/problems/rotate-image/)
**YouTube:** [NeetCode - Rotate Image](https://www.youtube.com/watch?v=fMSJSS7eO1w)

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

**LeetCode:** [50. Pow(x, n)](https://leetcode.com/problems/powx-n/)
**YouTube:** [NeetCode - Pow(x, n)](https://www.youtube.com/watch?v=g9YQyYi4IQQ)

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

**LeetCode:** [202. Happy Number](https://leetcode.com/problems/happy-number/)
**YouTube:** [NeetCode - Happy Number](https://www.youtube.com/watch?v=ljz85bxOYJ0)

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
    order: 3,
    topic: 'Math & Geometry',
    leetcodeUrl: 'https://leetcode.com/problems/happy-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=ljz85bxOYJ0',
  },
];
