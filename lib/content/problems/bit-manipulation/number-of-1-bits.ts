/**
 * Number of 1 Bits
 * Problem ID: number-of-1-bits
 * Order: 2
 */

import { Problem } from '../../../types';

export const number_of_1_bitsProblem: Problem = {
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

  leetcodeUrl: 'https://leetcode.com/problems/number-of-1-bits/',
  youtubeUrl: 'https://www.youtube.com/watch?v=5Km3utixwZs',
  order: 2,
  topic: 'Bit Manipulation',
};
