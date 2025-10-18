/**
 * Hamming Distance
 * Problem ID: hamming-distance
 * Order: 6
 */

import { Problem } from '../../../types';

export const hamming_distanceProblem: Problem = {
  id: 'hamming-distance',
  title: 'Hamming Distance',
  difficulty: 'Easy',
  topic: 'Bit Manipulation',
  description: `The **Hamming distance** between two integers is the number of positions at which the corresponding bits are different.

Given two integers \`x\` and \`y\`, return the **Hamming distance** between them.`,
  examples: [
    {
      input: 'x = 1, y = 4',
      output: '2',
      explanation: '1 (0 0 0 1) and 4 (0 1 0 0) differ in 2 positions.',
    },
    {
      input: 'x = 3, y = 1',
      output: '1',
    },
  ],
  constraints: ['0 <= x, y <= 2^31 - 1'],
  hints: ['XOR to find differing bits', 'Count 1s in result'],
  starterCode: `def hamming_distance(x: int, y: int) -> int:
    """
    Find Hamming distance between two integers.
    
    Args:
        x: First integer
        y: Second integer
        
    Returns:
        Number of differing bits
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [1, 4],
      expected: 2,
    },
    {
      input: [3, 1],
      expected: 1,
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/hamming-distance/',
  youtubeUrl: 'https://www.youtube.com/watch?v=yHBsShUIejQ',
};
