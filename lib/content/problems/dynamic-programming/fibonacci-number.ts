/**
 * Fibonacci Number
 * Problem ID: fibonacci-number
 * Order: 4
 */

import { Problem } from '../../../types';

export const fibonacci_numberProblem: Problem = {
  id: 'fibonacci-number',
  title: 'Fibonacci Number',
  difficulty: 'Easy',
  topic: 'Dynamic Programming',
  description: `The **Fibonacci numbers**, commonly denoted \`F(n)\` form a sequence, called the **Fibonacci sequence**, such that each number is the sum of the two preceding ones, starting from \`0\` and \`1\`. That is,

\`\`\`
F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
\`\`\`

Given \`n\`, calculate \`F(n)\`.`,
  examples: [
    {
      input: 'n = 2',
      output: '1',
      explanation: 'F(2) = F(1) + F(0) = 1 + 0 = 1.',
    },
    {
      input: 'n = 3',
      output: '2',
      explanation: 'F(3) = F(2) + F(1) = 1 + 1 = 2.',
    },
    {
      input: 'n = 4',
      output: '3',
      explanation: 'F(4) = F(3) + F(2) = 2 + 1 = 3.',
    },
  ],
  constraints: ['0 <= n <= 30'],
  hints: ['Use bottom-up DP', 'Only need last two values', 'Space can be O(1)'],
  starterCode: `def fib(n: int) -> int:
    """
    Calculate nth Fibonacci number.
    
    Args:
        n: Position in sequence
        
    Returns:
        Fibonacci number at position n
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2],
      expected: 1,
    },
    {
      input: [3],
      expected: 2,
    },
    {
      input: [4],
      expected: 3,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/fibonacci-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=tyB0ztf0DNY',
};
