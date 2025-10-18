/**
 * Decode Ways
 * Problem ID: decode-ways
 * Order: 14
 */

import { Problem } from '../../../types';

export const decode_waysProblem: Problem = {
  id: 'decode-ways',
  title: 'Decode Ways',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"

To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:
- "AAJF" with the grouping (1 1 10 6)
- "KJF" with the grouping (11 10 6)

Given a string \`s\` containing only digits, return the number of ways to decode it.

**Example 1:**
\`\`\`
Input: s = "12"
Output: 2
Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
\`\`\`

**Example 2:**
\`\`\`
Input: s = "226"
Output: 3
Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
\`\`\`

**Example 3:**
\`\`\`
Input: s = "06"
Output: 0
Explanation: "06" cannot be mapped to "F" because "6" is different from "06".
\`\`\``,
  starterCode: `def num_decodings(s):
    """
    Count number of ways to decode the string.
    
    Args:
        s: String of digits
        
    Returns:
        Number of decoding ways
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['12'],
      expected: 2,
    },
    {
      input: ['226'],
      expected: 3,
    },
    {
      input: ['06'],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1) with optimization',
  leetcodeUrl: 'https://leetcode.com/problems/decode-ways/',
  youtubeUrl: 'https://www.youtube.com/watch?v=6aEyTjOwlJU',
};
