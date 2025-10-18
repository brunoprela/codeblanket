/**
 * Contains Duplicate
 * Problem ID: contains-duplicate
 * Order: 1
 */

import { Problem } from '../../../types';

export const contains_duplicateProblem: Problem = {
  id: 'contains-duplicate',
  title: 'Contains Duplicate',
  difficulty: 'Easy',
  description: `Given an integer array \`nums\`, return \`true\` if any value appears **at least twice** in the array, and return \`false\` if every element is distinct.


**Approach:**
Use a hash set to track seen numbers. If you encounter a number already in the set, you've found a duplicate.`,
  examples: [
    {
      input: 'nums = [1,2,3,1]',
      output: 'true',
      explanation: 'The number 1 appears twice.',
    },
    {
      input: 'nums = [1,2,3,4]',
      output: 'false',
      explanation: 'All elements are distinct.',
    },
    {
      input: 'nums = [1,1,1,3,3,4,3,2,4,2]',
      output: 'true',
      explanation: 'Multiple numbers appear more than once.',
    },
  ],
  constraints: ['1 <= nums.length <= 10^5', '-10^9 <= nums[i] <= 10^9'],
  hints: [
    "Use a hash set to track numbers you've seen",
    'If you find a number already in the set, return true immediately',
    'Time complexity should be O(n), space complexity O(n)',
  ],
  starterCode: `from typing import List

def contains_duplicate(nums: List[int]) -> bool:
    """
    Determine if array contains any duplicates.
    
    Args:
        nums: List of integers
        
    Returns:
        True if any value appears at least twice, False otherwise
    """
    # Your code here
    pass


# Write your own test cases below
# print() statements will appear in the Console Output section
# Example:
# print(contains_duplicate([1, 2, 3]))  # Should print False
# print(contains_duplicate([1, 2, 1]))  # Should print True
`,
  testCases: [
    {
      input: [[1, 2, 3, 1]],
      expected: true,
    },
    {
      input: [[1, 2, 3, 4]],
      expected: false,
    },
    {
      input: [[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]],
      expected: true,
    },
    {
      input: [[1]],
      expected: false,
    },
  ],
  solution: `def contains_duplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Alternative: Use set length comparison
def contains_duplicate_alt(nums: List[int]) -> bool:
    return len(nums) != len(set(nums))`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/contains-duplicate/',
  youtubeUrl: 'https://www.youtube.com/watch?v=3OamzN90kPg',
  order: 1,
  topic: 'Arrays & Hashing',
};
