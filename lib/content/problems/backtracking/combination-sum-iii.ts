/**
 * Combination Sum III
 * Problem ID: combination-sum-iii
 * Order: 6
 */

import { Problem } from '../../../types';

export const combination_sum_iiiProblem: Problem = {
  id: 'combination-sum-iii',
  title: 'Combination Sum III',
  difficulty: 'Easy',
  topic: 'Backtracking',
  description: `Find all valid combinations of \`k\` numbers that sum up to \`n\` such that the following conditions are true:

- Only numbers \`1\` through \`9\` are used.
- Each number is used **at most once**.

Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.`,
  examples: [
    {
      input: 'k = 3, n = 7',
      output: '[[1,2,4]]',
      explanation: '1 + 2 + 4 = 7. There are no other valid combinations.',
    },
    {
      input: 'k = 3, n = 9',
      output: '[[1,2,6],[1,3,5],[2,3,4]]',
    },
  ],
  constraints: ['2 <= k <= 9', '1 <= n <= 60'],
  hints: [
    'Use backtracking with numbers 1-9',
    'Track remaining sum and count',
    'Prune when sum exceeds target',
  ],
  starterCode: `from typing import List

def combination_sum3(k: int, n: int) -> List[List[int]]:
    """
    Find combinations of k numbers summing to n.
    
    Args:
        k: Number of elements in combination
        n: Target sum
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [3, 7],
      expected: [[1, 2, 4]],
    },
    {
      input: [3, 9],
      expected: [
        [1, 2, 6],
        [1, 3, 5],
        [2, 3, 4],
      ],
    },
    {
      input: [4, 1],
      expected: [],
    },
  ],
  timeComplexity: 'O(9! / (k! * (9-k)!))',
  spaceComplexity: 'O(k)',
  leetcodeUrl: 'https://leetcode.com/problems/combination-sum-iii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=xVGCxTmXRBI',
};
