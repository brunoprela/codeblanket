/**
 * Interleaving String
 * Problem ID: interleaving-string
 * Order: 16
 */

import { Problem } from '../../../types';

export const interleaving_stringProblem: Problem = {
  id: 'interleaving-string',
  title: 'Interleaving String',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `Given strings \`s1\`, \`s2\`, and \`s3\`, find whether \`s3\` is formed by an interleaving of \`s1\` and \`s2\`.

An interleaving of two strings \`s\` and \`t\` is a configuration where \`s\` and \`t\` are divided into \`n\` and \`m\` substrings respectively, such that:
- s = s1 + s2 + ... + sn
- t = t1 + t2 + ... + tm
- |n - m| <= 1
- The interleaving is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...

**Example 1:**
\`\`\`
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
\`\`\`

**Example 2:**
\`\`\`
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
\`\`\`

**Example 3:**
\`\`\`
Input: s1 = "", s2 = "", s3 = ""
Output: true
\`\`\``,
  starterCode: `def is_interleave(s1, s2, s3):
    """
    Check if s3 is an interleaving of s1 and s2.
    
    Args:
        s1: First string
        s2: Second string
        s3: Target string
        
    Returns:
        True if s3 is interleaving of s1 and s2
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['aabcc', 'dbbca', 'aadbbcbcac'],
      expected: true,
    },
    {
      input: ['aabcc', 'dbbca', 'aadbbbaccc'],
      expected: false,
    },
    {
      input: ['', '', ''],
      expected: true,
    },
  ],
  timeComplexity: 'O(m * n) where m, n are lengths of s1, s2',
  spaceComplexity: 'O(m * n) or O(n) with optimization',
  leetcodeUrl: 'https://leetcode.com/problems/interleaving-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=3Rw3p9LrgvE',
};
