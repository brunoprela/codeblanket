/**
 * Single Number
 * Problem ID: single-number
 * Order: 1
 */

import { Problem } from '../../../types';

export const single_numberProblem: Problem = {
  id: 'single-number',
  title: 'Single Number',
  difficulty: 'Easy',
  description: `Given a **non-empty** array of integers \`nums\`, every element appears **twice** except for one. Find that single one.

You must implement a solution with linear runtime complexity and use only constant extra space.


**Approach:**
Use the **XOR** bitwise operator. XOR has special properties:
- \`a ^ a = 0\` (any number XOR itself equals 0)
- \`a ^ 0 = a\` (any number XOR 0 equals itself)
- XOR is commutative and associative

Since every number except one appears twice, XORing all numbers will cancel out all duplicates, leaving only the unique number.

**Key Insight:**
XOR is perfect for finding the unique element because duplicates cancel out to 0.`,
  examples: [
    {
      input: 'nums = [2,2,1]',
      output: '1',
      explanation: '2 ^ 2 ^ 1 = 0 ^ 1 = 1',
    },
    {
      input: 'nums = [4,1,2,1,2]',
      output: '4',
      explanation: '4 ^ 1 ^ 2 ^ 1 ^ 2 = 4 (all pairs cancel)',
    },
    {
      input: 'nums = [1]',
      output: '1',
      explanation: 'Only one element, return it',
    },
  ],
  constraints: [
    '1 <= nums.length <= 3 * 10^4',
    '-3 * 10^4 <= nums[i] <= 3 * 10^4',
    'Each element in the array appears twice except for one element which appears only once',
  ],
  hints: [
    'Think about the XOR operation',
    'XOR has special properties: a ^ a = 0 and a ^ 0 = a',
    'XOR all numbers together',
    'Duplicates will cancel out (become 0)',
    'Only the single number remains',
    'This is O(n) time and O(1) space',
  ],
  starterCode: `from typing import List

def single_number(nums: List[int]) -> int:
    """
    Find the number that appears only once.
    
    Args:
        nums: Array where every element appears twice except one
        
    Returns:
        The single number that appears only once
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[2, 2, 1]],
      expected: 1,
    },
    {
      input: [[4, 1, 2, 1, 2]],
      expected: 4,
    },
    {
      input: [[1]],
      expected: 1,
    },
  ],
  solution: `from typing import List


def single_number(nums: List[int]) -> int:
    """
    XOR approach - all duplicates cancel out.
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result


# Alternative: More explicit
def single_number_verbose(nums: List[int]) -> int:
    """
    Same approach with explanation.
    """
    # XOR all numbers
    # Duplicates will become 0
    # 0 ^ single_number = single_number
    xor_result = 0
    for num in nums:
        xor_result ^= num
    return xor_result


# Alternative: One-liner with reduce
def single_number_oneliner(nums: List[int]) -> int:
    """
    Functional approach using reduce.
    """
    from functools import reduce
    return reduce(lambda x, y: x ^ y, nums)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',

  leetcodeUrl: 'https://leetcode.com/problems/single-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=qMPX1AOa83k',
  order: 1,
  topic: 'Bit Manipulation',
};
