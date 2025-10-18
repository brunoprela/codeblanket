/**
 * Find the Duplicate Number
 * Problem ID: duplicate-number
 * Order: 4
 */

import { Problem } from '../../../types';

export const duplicate_numberProblem: Problem = {
  id: 'duplicate-number',
  title: 'Find the Duplicate Number',
  difficulty: 'Medium',
  topic: 'Time & Space Complexity',
  order: 4,
  description: `Given an array of integers \`nums\` containing \`n + 1\` integers where each integer is in the range \`[1, n]\` inclusive.

There is only **one repeated number** in \`nums\`, return this repeated number.

You must solve the problem **without** modifying the array \`nums\` and uses only **constant** extra space.

**Challenge:** Can you solve it in O(n) time and O(1) space?`,
  examples: [
    {
      input: 'nums = [1,3,4,2,2]',
      output: '2',
    },
    {
      input: 'nums = [3,1,3,4,2]',
      output: '3',
    },
  ],
  constraints: [
    '1 <= n <= 10^5',
    'nums.length == n + 1',
    '1 <= nums[i] <= n',
    'All the integers in nums appear only once except for precisely one integer which appears two or more times',
  ],
  hints: [
    'Hash set uses O(n) space - violates constraint',
    'Sorting modifies array - violates constraint',
    'Think of it as a linked list cycle detection problem!',
    "Floyd's Tortoise and Hare algorithm: O(n) time, O(1) space",
  ],
  starterCode: `from typing import List

def find_duplicate(nums: List[int]) -> int:
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[1, 3, 4, 2, 2]],
      expected: 2,
    },
    {
      input: [[3, 1, 3, 4, 2]],
      expected: 3,
    },
    {
      input: [[2, 5, 9, 6, 9, 3, 8, 9, 7, 1, 4]],
      expected: 9,
    },
    {
      input: [[1, 1]],
      expected: 1,
    },
    {
      input: [[1, 1, 2]],
      expected: 1,
    },
  ],
  solution: `# Optimal: Floyd's Cycle Detection - O(n) time, O(1) space
def find_duplicate(nums):
    # Phase 1: Find intersection point in the cycle
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Phase 2: Find entrance to the cycle (the duplicate)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow

# Hash Set - O(n) time, O(n) space (violates space constraint)
def find_duplicate_set(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return num
        seen.add(num)
    return -1

# Sorting - O(n log n) time, O(1) space (but modifies array - violates constraint)
def find_duplicate_sort(nums):
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            return nums[i]
    return -1
`,
  timeComplexity: "O(n) optimal with Floyd's vs O(n log n) with sorting",
  spaceComplexity:
    'O(1) optimal - demonstrates clever algorithm avoiding extra space',
  leetcodeUrl: 'https://leetcode.com/problems/find-the-duplicate-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=wjYnzkAhcNk',
};
