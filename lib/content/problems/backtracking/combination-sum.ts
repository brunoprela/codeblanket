/**
 * Combination Sum
 * Problem ID: combination-sum
 * Order: 7
 */

import { Problem } from '../../../types';

export const combination_sumProblem: Problem = {
  id: 'combination-sum',
  title: 'Combination Sum',
  difficulty: 'Medium',
  topic: 'Backtracking',
  description: `Given an array of **distinct** integers \`candidates\` and a target integer \`target\`, return a list of all **unique combinations** of \`candidates\` where the chosen numbers sum to \`target\`. You may return the combinations in **any order**.

The **same** number may be chosen from \`candidates\` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to \`target\` is less than \`150\` combinations for the given input.`,
  examples: [
    {
      input: 'candidates = [2,3,6,7], target = 7',
      output: '[[2,2,3],[7]]',
    },
    {
      input: 'candidates = [2,3,5], target = 8',
      output: '[[2,2,2,2],[2,3,3],[3,5]]',
    },
  ],
  constraints: [
    '1 <= candidates.length <= 30',
    '2 <= candidates[i] <= 40',
    'All elements of candidates are distinct',
    '1 <= target <= 40',
  ],
  hints: [
    'Use backtracking',
    'Numbers can be reused',
    'Start from current index to avoid duplicates',
  ],
  starterCode: `from typing import List

def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find combinations that sum to target (with repetition).
    
    Args:
        candidates: Available numbers
        target: Target sum
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 3, 6, 7], 7],
      expected: [[2, 2, 3], [7]],
    },
    {
      input: [[2, 3, 5], 8],
      expected: [
        [2, 2, 2, 2],
        [2, 3, 3],
        [3, 5],
      ],
    },
    {
      input: [[2], 1],
      expected: [],
    },
  ],
  timeComplexity: 'O(2^target)',
  spaceComplexity: 'O(target)',
  leetcodeUrl: 'https://leetcode.com/problems/combination-sum/',
  youtubeUrl: 'https://www.youtube.com/watch?v=GBKI9VSKdGg',
};
