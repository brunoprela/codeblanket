/**
 * Missing Number
 * Problem ID: missing-number
 * Order: 3
 */

import { Problem } from '../../../types';

export const missing_numberProblem: Problem = {
  id: 'missing-number',
  title: 'Missing Number',
  difficulty: 'Hard',
  description: `Given an array \`nums\` containing \`n\` distinct numbers in the range \`[0, n]\`, return the only number in the range that is missing from the array.


**Multiple Approaches:**

**1. XOR Approach (Most Elegant):**
- XOR all indices (0 to n) with all array values
- Duplicates cancel out, leaving only the missing number
- Uses XOR properties: a ^ a = 0 and a ^ 0 = a

**2. Sum Approach:**
- Calculate expected sum: n * (n + 1) / 2
- Subtract actual sum from expected sum
- Result is the missing number

**3. Set Approach:**
- Create set of all numbers 0 to n
- Find which number is not in array
- Simple but uses O(n) space

**Key Insight:**
The XOR approach is most elegant because it uses O(1) space and leverages bit manipulation properties beautifully.`,
  examples: [
    {
      input: 'nums = [3,0,1]',
      output: '2',
      explanation:
        'n = 3 since there are 3 numbers, range is [0,3]. 2 is missing.',
    },
    {
      input: 'nums = [0,1]',
      output: '2',
      explanation:
        'n = 2 since there are 2 numbers, range is [0,2]. 2 is missing.',
    },
    {
      input: 'nums = [9,6,4,2,3,5,7,0,1]',
      output: '8',
      explanation:
        'n = 9 since there are 9 numbers, range is [0,9]. 8 is missing.',
    },
  ],
  constraints: [
    'n == nums.length',
    '1 <= n <= 10^4',
    '0 <= nums[i] <= n',
    'All the numbers of nums are unique',
  ],
  hints: [
    'Think about XOR properties: a ^ a = 0',
    'XOR all indices 0 to n with all array values',
    'Duplicates will cancel out',
    'Alternatively, use sum formula: n*(n+1)/2',
    'XOR approach is O(1) space, sum approach may overflow',
  ],
  starterCode: `from typing import List

def missing_number(nums: List[int]) -> int:
    """
    Find the missing number in range [0, n].
    
    Args:
        nums: Array of n distinct numbers from [0, n]
        
    Returns:
        The missing number
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[3, 0, 1]],
      expected: 2,
    },
    {
      input: [[0, 1]],
      expected: 2,
    },
    {
      input: [[9, 6, 4, 2, 3, 5, 7, 0, 1]],
      expected: 8,
    },
    {
      input: [[0]],
      expected: 1,
    },
  ],
  solution: `from typing import List


def missing_number(nums: List[int]) -> int:
    """
    XOR approach - most elegant.
    Time: O(n), Space: O(1)
    """
    result = len(nums)  # Start with n
    
    # XOR with all indices and values
    for i, num in enumerate(nums):
        result ^= i ^ num
    
    return result


# Alternative: Sum approach
def missing_number_sum(nums: List[int]) -> int:
    """
    Mathematical approach using sum formula.
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# Alternative: Set approach
def missing_number_set(nums: List[int]) -> int:
    """
    Using set for quick lookup.
    Time: O(n), Space: O(n)
    """
    num_set = set(nums)
    n = len(nums)
    
    for i in range(n + 1):
        if i not in num_set:
            return i
    
    return -1  # Should never reach here


# Alternative: Cleaner XOR
def missing_number_xor_clean(nums: List[int]) -> int:
    """
    XOR approach with clear logic.
    """
    # XOR all numbers from 0 to n
    result = 0
    for i in range(len(nums) + 1):
        result ^= i
    
    # XOR all numbers in array
    for num in nums:
        result ^= num
    
    return result


# Alternative: Gauss formula (most readable)
def missing_number_gauss(nums: List[int]) -> int:
    """
    Using Gauss's formula for sum of first n natural numbers.
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    # Sum of 0 to n is n*(n+1)/2
    return n * (n + 1) // 2 - sum(nums)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) with XOR or sum, O(n) with set',

  leetcodeUrl: 'https://leetcode.com/problems/missing-number/',
  youtubeUrl: 'https://www.youtube.com/watch?v=WnPLSRLSANE',
  order: 3,
  topic: 'Bit Manipulation',
};
