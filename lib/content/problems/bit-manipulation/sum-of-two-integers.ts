/**
 * Sum of Two Integers
 * Problem ID: sum-of-two-integers
 * Order: 7
 */

import { Problem } from '../../../types';

export const sum_of_two_integersProblem: Problem = {
  id: 'sum-of-two-integers',
  title: 'Sum of Two Integers',
  difficulty: 'Medium',
  topic: 'Bit Manipulation',
  description: `Given two integers \`a\` and \`b\`, return the sum of the two integers **without using the operators** \`+\` **and** \`-\`.`,
  examples: [
    {
      input: 'a = 1, b = 2',
      output: '3',
    },
    {
      input: 'a = 2, b = 3',
      output: '5',
    },
  ],
  constraints: ['-1000 <= a, b <= 1000'],
  hints: [
    'Use XOR for sum without carry',
    'Use AND and left shift for carry',
    'Repeat until no carry',
  ],
  starterCode: `def get_sum(a: int, b: int) -> int:
    """
    Add two integers without + or - operators.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Sum of a and b
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [1, 2],
      expected: 3,
    },
    {
      input: [2, 3],
      expected: 5,
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/sum-of-two-integers/',
  youtubeUrl: 'https://www.youtube.com/watch?v=gVUrDV4tZfY',
};
