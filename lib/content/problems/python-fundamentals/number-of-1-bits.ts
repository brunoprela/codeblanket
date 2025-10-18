/**
 * Number of 1 Bits (Hamming Weight)
 * Problem ID: fundamentals-number-of-1-bits
 * Order: 67
 */

import { Problem } from '../../../types';

export const number_of_1_bitsProblem: Problem = {
  id: 'fundamentals-number-of-1-bits',
  title: 'Number of 1 Bits (Hamming Weight)',
  difficulty: 'Easy',
  description: `Count the number of 1 bits in a binary representation.

Also known as the Hamming weight.

**Example:** 11 = 1011 → 3 ones

**Trick:** n & (n-1) removes rightmost 1 bit

This tests:
- Bit manipulation
- Counting
- Binary representation`,
  examples: [
    {
      input: 'n = 11',
      output: '3',
      explanation: '1011 has three 1s',
    },
    {
      input: 'n = 128',
      output: '1',
      explanation: '10000000 has one 1',
    },
  ],
  constraints: ['0 <= n <= 2^31 - 1'],
  hints: [
    'Use n & 1 to check last bit',
    'Or use n & (n-1) trick',
    'Or use bin(n).count("1")',
  ],
  starterCode: `def hamming_weight(n):
    """
    Count number of 1 bits.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Count of 1 bits
        
    Examples:
        >>> hamming_weight(11)
        3
    """
    pass


# Test
print(hamming_weight(11))
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
      input: [0],
      expected: 0,
    },
  ],
  solution: `def hamming_weight(n):
    count = 0
    
    while n:
        count += 1
        n &= n - 1  # Remove rightmost 1 bit
    
    return count


# Alternative checking each bit
def hamming_weight_simple(n):
    count = 0
    
    while n:
        count += n & 1
        n >>= 1
    
    return count


# One-liner
def hamming_weight_oneliner(n):
    return bin(n).count('1')`,
  timeComplexity: 'O(1) - at most 32 bits',
  spaceComplexity: 'O(1)',
  order: 67,
  topic: 'Python Fundamentals',
};
