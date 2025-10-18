/**
 * Arranging Coins
 * Problem ID: arranging-coins
 * Order: 9
 */

import { Problem } from '../../../types';

export const arranging_coinsProblem: Problem = {
  id: 'arranging-coins',
  title: 'Arranging Coins',
  difficulty: 'Easy',
  topic: 'Binary Search',
  description: `You have \`n\` coins and you want to build a staircase with these coins. The staircase consists of \`k\` rows where the \`i-th\` row has exactly \`i\` coins. The last row of the staircase may be incomplete.

Given the integer \`n\`, return the number of complete rows of the staircase you will build.`,
  examples: [
    {
      input: 'n = 5',
      output: '2',
      explanation:
        'The coins can form these rows: ¤, ¤ ¤. The 3rd row is incomplete, so return 2.',
    },
    {
      input: 'n = 8',
      output: '3',
      explanation:
        'The coins can form these rows: ¤, ¤ ¤, ¤ ¤ ¤. The 4th row is incomplete, so return 3.',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1'],
  hints: [
    'Use the formula k * (k + 1) / 2 to calculate the total coins needed for k rows',
    'Binary search for the answer',
  ],
  starterCode: `def arrange_coins(n: int) -> int:
    """
    Find complete rows in coin staircase.
    
    Args:
        n: Total number of coins
        
    Returns:
        Number of complete rows
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [5],
      expected: 2,
    },
    {
      input: [8],
      expected: 3,
    },
    {
      input: [1],
      expected: 1,
    },
    {
      input: [10],
      expected: 4,
    },
  ],
  timeComplexity: 'O(log n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/arranging-coins/',
  youtubeUrl: 'https://www.youtube.com/watch?v=C4TkkOuBd44',
};
