/**
 * Counting Bits
 * Problem ID: counting-bits
 * Order: 4
 */

import { Problem } from '../../../types';

export const counting_bitsProblem: Problem = {
  id: 'counting-bits',
  title: 'Counting Bits',
  difficulty: 'Easy',
  topic: 'Bit Manipulation',
  description: `Given an integer \`n\`, return an array \`ans\` of length \`n + 1\` such that for each \`i\` (\`0 <= i <= n\`), \`ans[i]\` is the **number of** \`1\`**'s** in the binary representation of \`i\`.`,
  examples: [
    {
      input: 'n = 2',
      output: '[0,1,1]',
    },
    {
      input: 'n = 5',
      output: '[0,1,1,2,1,2]',
    },
  ],
  constraints: ['0 <= n <= 10^5'],
  hints: [
    'DP: count[i] = count[i >> 1] + (i & 1)',
    'Right shift removes last bit, check if it was 1',
  ],
  starterCode: `from typing import List

def count_bits(n: int) -> List[int]:
    """
    Count 1s in binary for 0 to n.
    
    Args:
        n: Upper limit
        
    Returns:
        Array of bit counts
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2],
      expected: [0, 1, 1],
    },
    {
      input: [5],
      expected: [0, 1, 1, 2, 1, 2],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) excluding output',
  leetcodeUrl: 'https://leetcode.com/problems/counting-bits/',
  youtubeUrl: 'https://www.youtube.com/watch?v=RyBM56RIWrM',
};
