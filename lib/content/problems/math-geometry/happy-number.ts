/**
 * Happy Number
 * Problem ID: happy-number
 * Order: 3
 */

import { Problem } from '../../../types';

export const happy_numberProblem: Problem = {
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
};
