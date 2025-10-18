/**
 * Contains Duplicate II
 * Problem ID: contains-duplicate-ii
 * Order: 5
 */

import { Problem } from '../../../types';

export const contains_duplicate_iiProblem: Problem = {
  id: 'contains-duplicate-ii',
  title: 'Contains Duplicate II',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  order: 5,
  description: `Given an integer array \`nums\` and an integer \`k\`, return \`true\` if there are two **distinct indices** \`i\` and \`j\` in the array such that \`nums[i] == nums[j]\` and \`abs(i - j) <= k\`.`,
  examples: [
    {
      input: 'nums = [1,2,3,1], k = 3',
      output: 'true',
    },
    {
      input: 'nums = [1,0,1,1], k = 1',
      output: 'true',
    },
    {
      input: 'nums = [1,2,3,1,2,3], k = 2',
      output: 'false',
    },
  ],
  constraints: [
    '1 <= nums.length <= 10^5',
    '-10^9 <= nums[i] <= 10^9',
    '0 <= k <= 10^5',
  ],
  hints: [
    'Use a sliding window of size k',
    'Maintain a set of elements in the window',
    'Check if current element exists in the set',
  ],
  starterCode: `from typing import List

def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    """
    Check if duplicate exists within distance k.
    
    Args:
        nums: Array of integers
        k: Maximum distance
        
    Returns:
        True if duplicate exists within distance k
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 2, 3, 1], 3],
      expected: true,
    },
    {
      input: [[1, 0, 1, 1], 1],
      expected: true,
    },
    {
      input: [[1, 2, 3, 1, 2, 3], 2],
      expected: false,
    },
  ],
  solution: `from typing import List

def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
    """
    Sliding window with set.
    Time: O(n), Space: O(min(n, k))
    """
    window = set()
    
    for i, num in enumerate(nums):
        # Check if duplicate in window
        if num in window:
            return True
        
        # Add to window
        window.add(num)
        
        # Remove element outside window
        if len(window) > k:
            window.remove(nums[i - k])
    
    return False

# Alternative: Hash map approach
def contains_nearby_duplicate_map(nums: List[int], k: int) -> bool:
    """
    Hash map to track last index.
    Time: O(n), Space: O(n)
    """
    last_index = {}
    
    for i, num in enumerate(nums):
        if num in last_index and i - last_index[num] <= k:
            return True
        last_index[num] = i
    
    return False
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(min(n, k))',
  leetcodeUrl: 'https://leetcode.com/problems/contains-duplicate-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=ypn0aZ0nrL4',
};
