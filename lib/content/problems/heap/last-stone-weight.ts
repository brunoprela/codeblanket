/**
 * Last Stone Weight
 * Problem ID: last-stone-weight
 * Order: 1
 */

import { Problem } from '../../../types';

export const last_stone_weightProblem: Problem = {
  id: 'last-stone-weight',
  title: 'Last Stone Weight',
  difficulty: 'Easy',
  topic: 'Heap / Priority Queue',
  description: `You are given an array of integers \`stones\` where \`stones[i]\` is the weight of the \`i-th\` stone.

We are playing a game with the stones. On each turn, we choose the **heaviest two stones** and smash them together. Suppose the heaviest two stones have weights \`x\` and \`y\` with \`x <= y\`. The result of this smash is:

- If \`x == y\`, both stones are destroyed, and
- If \`x != y\`, the stone of weight \`x\` is destroyed, and the stone of weight \`y\` has new weight \`y - x\`.

At the end of the game, there is **at most one** stone left.

Return the weight of the last remaining stone. If there are no stones left, return \`0\`.`,
  examples: [
    {
      input: 'stones = [2,7,4,1,8,1]',
      output: '1',
      explanation:
        'Combine 7 and 8 to get 1, then 4 and 1 to get 3, then 2 and 3 to get 1, then 1 and 1 to get 0, return 1.',
    },
    {
      input: 'stones = [1]',
      output: '1',
    },
  ],
  constraints: ['1 <= stones.length <= 30', '1 <= stones[i] <= 1000'],
  hints: [
    'Use a max heap',
    'Pop two largest stones, push difference if not equal',
  ],
  starterCode: `from typing import List
import heapq

def last_stone_weight(stones: List[int]) -> int:
    """
    Find weight of last remaining stone.
    
    Args:
        stones: Array of stone weights
        
    Returns:
        Weight of last stone or 0
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 7, 4, 1, 8, 1]],
      expected: 1,
    },
    {
      input: [[1]],
      expected: 1,
    },
    {
      input: [[2, 2]],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/last-stone-weight/',
  youtubeUrl: 'https://www.youtube.com/watch?v=B-QCq79-Vfw',
};
