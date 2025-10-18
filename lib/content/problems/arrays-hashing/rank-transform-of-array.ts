/**
 * Rank Transform of an Array
 * Problem ID: rank-transform-of-array
 * Order: 13
 */

import { Problem } from '../../../types';

export const rank_transform_of_arrayProblem: Problem = {
  id: 'rank-transform-of-array',
  title: 'Rank Transform of an Array',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 13,
  description: `Given an array of integers \`arr\`, replace each element with its rank.

The rank represents how large the element is. The rank has the following rules:
- Rank is an integer starting from 1.
- The larger the element, the larger the rank. If two elements are equal, their rank must be the same.
- Rank should be as small as possible.`,
  examples: [
    {
      input: 'arr = [40,10,20,30]',
      output: '[4,1,2,3]',
      explanation:
        '40 is the largest element. 10 is the smallest. 20 is the second smallest. 30 is the third smallest.',
    },
    {
      input: 'arr = [100,100,100]',
      output: '[1,1,1]',
      explanation: 'Same elements share the same rank.',
    },
  ],
  constraints: ['0 <= arr.length <= 10^5', '-10^9 <= arr[i] <= 10^9'],
  hints: [
    'Sort the unique values',
    'Create a mapping from value to rank',
    'Map each element to its rank',
  ],
  starterCode: `from typing import List

def array_rank_transform(arr: List[int]) -> List[int]:
    """
    Transform array to ranks.
    
    Args:
        arr: Array of integers
        
    Returns:
        Array where each element is replaced by its rank
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[40, 10, 20, 30]],
      expected: [4, 1, 2, 3],
    },
    {
      input: [[100, 100, 100]],
      expected: [1, 1, 1],
    },
    {
      input: [[37, 12, 28, 9, 100, 56, 80, 5, 12]],
      expected: [5, 3, 4, 2, 8, 6, 7, 1, 3],
    },
  ],
  solution: `from typing import List

def array_rank_transform(arr: List[int]) -> List[int]:
    """
    Sort and create rank mapping.
    Time: O(n log n), Space: O(n)
    """
    # Create mapping from value to rank
    rank = {}
    for value in sorted(set(arr)):
        rank[value] = len(rank) + 1
    
    # Map each element to its rank
    return [rank[value] for value in arr]
`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/rank-transform-of-an-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=yccJ_V7Q7DA',
};
