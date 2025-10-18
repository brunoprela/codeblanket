/**
 * Reverse Bits
 * Problem ID: fundamentals-reverse-bits
 * Order: 66
 */

import { Problem } from '../../../types';

export const reverse_bitsProblem: Problem = {
  id: 'fundamentals-reverse-bits',
  title: 'Reverse Bits',
  difficulty: 'Easy',
  description: `Reverse bits of a 32-bit unsigned integer.

**Example:** 43261596 (00000010100101000001111010011100)
→ 964176192 (00111001011110000010100101000000)

This tests:
- Bit manipulation
- Binary representation
- Bit shifting`,
  examples: [
    {
      input: 'n = 43261596',
      output: '964176192',
    },
  ],
  constraints: ['Input is 32-bit unsigned integer'],
  hints: [
    'Build result bit by bit',
    'Extract rightmost bit with n & 1',
    'Shift n right, result left',
  ],
  starterCode: `def reverse_bits(n):
    """
    Reverse bits of 32-bit integer.
    
    Args:
        n: 32-bit unsigned integer
        
    Returns:
        Integer with reversed bits
        
    Examples:
        >>> reverse_bits(43261596)
        964176192
    """
    pass


# Test
print(reverse_bits(43261596))
`,
  testCases: [
    {
      input: [43261596],
      expected: 964176192,
    },
  ],
  solution: `def reverse_bits(n):
    result = 0
    
    for i in range(32):
        # Extract rightmost bit
        bit = n & 1
        # Shift result left and add bit
        result = (result << 1) | bit
        # Shift n right
        n >>= 1
    
    return result


# Alternative using string conversion
def reverse_bits_string(n):
    binary = bin(n)[2:].zfill(32)
    reversed_binary = binary[::-1]
    return int(reversed_binary, 2)`,
  timeComplexity: 'O(1) - fixed 32 iterations',
  spaceComplexity: 'O(1)',
  order: 66,
  topic: 'Python Fundamentals',
};
