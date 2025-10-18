/**
 * Shuffle the Array
 * Problem ID: shuffle-the-array
 * Order: 25
 */

import { Problem } from '../../../types';

export const shuffle_the_arrayProblem: Problem = {
  id: 'shuffle-the-array',
  title: 'Shuffle the Array',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  description: `Given the array \`nums\` consisting of \`2n\` elements in the form \`[x1,x2,...,xn,y1,y2,...,yn]\`.

Return the array in the form \`[x1,y1,x2,y2,...,xn,yn]\`.`,
  examples: [
    {
      input: 'nums = [2,5,1,3,4,7], n = 3',
      output: '[2,3,5,4,1,7]',
      explanation:
        'Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].',
    },
    {
      input: 'nums = [1,2,3,4,4,3,2,1], n = 4',
      output: '[1,4,2,3,3,2,4,1]',
    },
  ],
  constraints: ['1 <= n <= 500', 'nums.length == 2n', '1 <= nums[i] <= 10^3'],
  hints: [
    'The element at index i in the first half should go to index 2*i',
    'The element at index i in the second half should go to index 2*(i-n)+1',
  ],
  starterCode: `from typing import List

def shuffle(nums: List[int], n: int) -> List[int]:
    """
    Shuffle array according to pattern.
    
    Args:
        nums: Array of 2n elements
        n: Half length of array
        
    Returns:
        Shuffled array
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 5, 1, 3, 4, 7], 3],
      expected: [2, 3, 5, 4, 1, 7],
    },
    {
      input: [[1, 2, 3, 4, 4, 3, 2, 1], 4],
      expected: [1, 4, 2, 3, 3, 2, 4, 1],
    },
    {
      input: [[1, 1, 2, 2], 2],
      expected: [1, 2, 1, 2],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/shuffle-the-array/',
  youtubeUrl: 'https://www.youtube.com/watch?v=IvIKD_EU8BY',
};
