/**
 * Complement of Base 10 Integer
 * Problem ID: fundamentals-complement-base-10
 * Order: 86
 */

import { Problem } from '../../../types';

export const complement_base_10Problem: Problem = {
  id: 'fundamentals-complement-base-10',
  title: 'Complement of Base 10 Integer',
  difficulty: 'Easy',
  description: `Return the complement of a number's binary representation.

Complement: flip all bits (0→1, 1→0).

**Example:** 5 = 101 → complement = 010 = 2

**Note:** No leading zeros in binary representation.

This tests:
- Bit manipulation
- Bit flipping
- Binary representation`,
  examples: [
    {
      input: 'n = 5',
      output: '2',
      explanation: '101 → 010',
    },
    {
      input: 'n = 7',
      output: '0',
      explanation: '111 → 000',
    },
  ],
  constraints: ['0 <= n <= 10^9'],
  hints: [
    'Find bit length of number',
    'Create mask of all 1s for that length',
    'XOR with mask',
  ],
  starterCode: `def bitwise_complement(n):
    """
    Return complement of number.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Bitwise complement
        
    Examples:
        >>> bitwise_complement(5)
        2
    """
    pass


# Test
print(bitwise_complement(5))
`,
  testCases: [
    {
      input: [5],
      expected: 2,
    },
    {
      input: [7],
      expected: 0,
    },
    {
      input: [0],
      expected: 1,
    },
  ],
  solution: `def bitwise_complement(n):
    if n == 0:
        return 1
    
    # Find number of bits
    bit_length = n.bit_length()
    
    # Create mask: 2^bit_length - 1 (all 1s)
    mask = (1 << bit_length) - 1
    
    # XOR with mask to flip bits
    return n ^ mask


# Alternative using string
def bitwise_complement_string(n):
    if n == 0:
        return 1
    
    binary = bin(n)[2:]
    complement = '.join('1' if bit == '0' else '0' for bit in binary)
    return int(complement, 2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 86,
  topic: 'Python Fundamentals',
};
