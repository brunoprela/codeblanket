import { Problem } from '../types';

export const bitManipulationProblems: Problem[] = [
  {
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
    order: 1,
    topic: 'Bit Manipulation',
    leetcodeUrl: 'https://leetcode.com/problems/single-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=qMPX1AOa83k',
  },
  {
    id: 'number-of-1-bits',
    title: 'Number of 1 Bits',
    difficulty: 'Medium',
    description: `Write a function that takes an unsigned integer and returns the number of \`1\` bits it has (also known as the **Hamming weight**).


**Approach:**
Use **Brian Kernighan's algorithm**: The expression \`n & (n-1)\` removes the rightmost set bit. Count how many times we can do this until n becomes 0.

**Why it works:**
- \`n-1\` flips all bits after the rightmost set bit (including the rightmost set bit)
- \`n & (n-1)\` removes that rightmost set bit
- Repeat until no set bits remain

**Example:**
- n = 12 (binary: 1100)
- n-1 = 11 (binary: 1011)
- n & (n-1) = 8 (binary: 1000) - removed rightmost 1
- Continue until 0

**Key Insight:**
This algorithm is optimal because it only loops as many times as there are set bits, not as many bits total.`,
    examples: [
      {
        input: 'n = 11 (0000000000000000000000000001011)',
        output: '3',
        explanation: 'Binary has three 1 bits',
      },
      {
        input: 'n = 128 (0000000000000000000000010000000)',
        output: '1',
        explanation: 'Binary has one 1 bit',
      },
      {
        input: 'n = 7 (0000000000000000000000000000111)',
        output: '3',
        explanation: 'Binary has three 1 bits',
      },
    ],
    constraints: ['1 <= n <= 2^31 - 1'],
    hints: [
      'n & (n-1) removes the rightmost set bit',
      'Count how many times you can remove a set bit',
      'Loop until n becomes 0',
      "This is called Brian Kernighan's algorithm",
      'More efficient than checking every bit position',
    ],
    starterCode: `def hamming_weight(n: int) -> int:
    """
    Count the number of 1 bits in an integer.
    
    Args:
        n: Unsigned integer
        
    Returns:
        Number of 1 bits (Hamming weight)
    """
    # Write your code here
    pass
`,
    testCases: [
      {
        input: [11],
        expected: 3,
      },
      {
        input: [128],
        expected: 1,
      },
      {
        input: [7],
        expected: 3,
      },
      {
        input: [1],
        expected: 1,
      },
    ],
    solution: `def hamming_weight(n: int) -> int:
    """
    Brian Kernighan's algorithm.
    Time: O(k) where k = number of set bits, Space: O(1)
    """
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost 1 bit
        count += 1
    return count


# Alternative: Check each bit position
def hamming_weight_naive(n: int) -> int:
    """
    Naive approach checking all 32 bits.
    Time: O(32) = O(1), Space: O(1)
    """
    count = 0
    for i in range(32):
        if (n >> i) & 1:  # Check if bit i is set
            count += 1
    return count


# Alternative: Using built-in
def hamming_weight_builtin(n: int) -> int:
    """
    Using Python's built-in bin() function.
    Time: O(1), Space: O(1)
    """
    return bin(n).count('1')


# Alternative: Lookup table (for interview discussion)
def hamming_weight_lookup(n: int) -> int:
    """
    Using lookup table for 8-bit chunks.
    Precompute counts for 0-255, then process in chunks.
    Time: O(1), Space: O(256) for lookup table
    """
    # Precomputed lookup table (for 0-255)
    lookup = [bin(i).count('1') for i in range(256)]
    
    count = 0
    while n:
        count += lookup[n & 0xFF]  # Process lowest 8 bits
        n >>= 8  # Shift right by 8 bits
    return count`,
    timeComplexity: 'O(k) where k is the number of set bits',
    spaceComplexity: 'O(1)',
    order: 2,
    topic: 'Bit Manipulation',
    leetcodeUrl: 'https://leetcode.com/problems/number-of-1-bits/',
    youtubeUrl: 'https://www.youtube.com/watch?v=5Km3utixwZs',
  },
  {
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
    order: 3,
    topic: 'Bit Manipulation',
    leetcodeUrl: 'https://leetcode.com/problems/missing-number/',
    youtubeUrl: 'https://www.youtube.com/watch?v=WnPLSRLSANE',
  },
];
