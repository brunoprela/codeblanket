/**
 * Majority Element
 * Problem ID: majority-element
 * Order: 5
 */

import { Problem } from '../../../types';

export const majority_elementProblem: Problem = {
  id: 'majority-element',
  title: 'Majority Element',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 5,
  description: `Given an array \`nums\` of size \`n\`, return the **majority element**.

The majority element is the element that appears **more than ⌊n / 2⌋ times**. You may assume that the majority element always exists in the array.`,
  examples: [
    {
      input: 'nums = [3,2,3]',
      output: '3',
    },
    {
      input: 'nums = [2,2,1,1,1,2,2]',
      output: '2',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= n <= 5 * 10^4',
    '-10^9 <= nums[i] <= 10^9',
  ],
  hints: [
    'Use a hash map to count frequencies',
    'Return the element with count > n/2',
    'Can you solve it in O(1) space? (Boyer-Moore Voting)',
  ],
  starterCode: `from typing import List

def majority_element(nums: List[int]) -> int:
    """
    Find the majority element.
    
    Args:
        nums: Array of integers
        
    Returns:
        The majority element
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 2, 3]],
      expected: 3,
    },
    {
      input: [[2, 2, 1, 1, 1, 2, 2]],
      expected: 2,
    },
    {
      input: [[1]],
      expected: 1,
    },
  ],
  solution: `from typing import List

def majority_element(nums: List[int]) -> int:
    """
    Hash map approach to count frequencies.
    Time: O(n), Space: O(n)
    """
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
        if count[num] > len(nums) // 2:
            return num
    return -1

# Alternative: Boyer-Moore Voting Algorithm
def majority_element_voting(nums: List[int]) -> int:
    """
    Boyer-Moore Voting: O(n) time, O(1) space.
    """
    candidate = None
    count = 0
    
    # Find candidate
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    
    return candidate
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n) for hash map, O(1) for voting',
  leetcodeUrl: 'https://leetcode.com/problems/majority-element/',
  youtubeUrl: 'https://www.youtube.com/watch?v=7pnhv842keE',
};
