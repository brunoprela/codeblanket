/**
 * Reverse Bits
 * Problem ID: reverse-bits
 * Order: 5
 */

import { Problem } from '../../../types';

export const reverse_bitsProblem: Problem = {
  id: 'reverse-bits',
  title: 'Reverse Bits',
  difficulty: 'Easy',
  topic: 'Bit Manipulation',
  description: `Reverse bits of a given 32 bits unsigned integer.

**Note:**

- In some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
- In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in **Example 2** above, the input represents the signed integer \`-3\` and the output represents the signed integer \`-1073741825\`.`,
  examples: [
    {
      input: 'n = 00000010100101000001111010011100',
      output: '00111001011110000010100101000000',
    },
    {
      input: 'n = 11111111111111111111111111111101',
      output: '10111111111111111111111111111111',
    },
  ],
  constraints: ['The input must be a binary string of length 32'],
  hints: [
    'Process each bit from right to left',
    'Build result by shifting left',
  ],
  starterCode: `def reverse_bits(n: int) -> int:
    """
    Reverse bits of 32-bit integer.
    
    Args:
        n: 32-bit unsigned integer
        
    Returns:
        Reversed 32-bit integer
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [43261596],
      expected: 964176192,
    },
    {
      input: [4294967293],
      expected: 3221225471,
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/reverse-bits/',
  youtubeUrl: 'https://www.youtube.com/watch?v=UcoN6UjAI64',
};
