/**
 * Hamming Distance
 * Problem ID: fundamentals-hamming-distance
 * Order: 85
 */

import { Problem } from '../../../types';

export const hamming_distanceProblem: Problem = {
  id: 'fundamentals-hamming-distance',
  title: 'Hamming Distance',
  difficulty: 'Easy',
  description: `Calculate Hamming distance between two integers.

Hamming distance = number of positions with different bits.

**Example:** x=1 (0001), y=4 (0100) → distance = 2

This tests:
- Bit manipulation
- XOR operation
- Bit counting`,
  examples: [
    {
      input: 'x = 1, y = 4',
      output: '2',
    },
  ],
  constraints: ['0 <= x, y <= 2^31 - 1'],
  hints: [
    'XOR gives positions where bits differ',
    'Count 1s in XOR result',
    'Use bit counting techniques',
  ],
  starterCode: `def hamming_distance(x, y):
    """
    Calculate Hamming distance.
    
    Args:
        x: First integer
        y: Second integer
        
    Returns:
        Number of different bit positions
        
    Examples:
        >>> hamming_distance(1, 4)
        2
    """
    pass


# Test
print(hamming_distance(1, 4))
`,
  testCases: [
    {
      input: [1, 4],
      expected: 2,
    },
    {
      input: [3, 1],
      expected: 1,
    },
  ],
  solution: `def hamming_distance(x, y):
    xor = x ^ y
    count = 0
    
    while xor:
        count += xor & 1
        xor >>= 1
    
    return count


# One-liner
def hamming_distance_oneliner(x, y):
    return bin(x ^ y).count('1')`,
  timeComplexity: 'O(1) - max 32 bits',
  spaceComplexity: 'O(1)',
  order: 85,
  topic: 'Python Fundamentals',
};
