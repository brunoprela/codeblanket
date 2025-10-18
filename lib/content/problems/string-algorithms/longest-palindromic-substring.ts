/**
 * Longest Palindromic Substring
 * Problem ID: longest-palindromic-substring
 * Order: 1
 */

import { Problem } from '../../../types';

export const longest_palindromic_substringProblem: Problem = {
  id: 'longest-palindromic-substring',
  title: 'Longest Palindromic Substring',
  difficulty: 'Medium',
  topic: 'String Algorithms',
  description: `Given a string \`s\`, return the longest palindromic substring in \`s\`.

**Example 1:**
\`\`\`
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
\`\`\`

**Example 2:**
\`\`\`
Input: s = "cbbd"
Output: "bb"
\`\`\`

**Constraints:**
- 1 ≤ s.length ≤ 1000
- s consist of only digits and English letters`,
  starterCode: `def longest_palindrome(s):
    """
    Find longest palindromic substring.
    
    Args:
        s: Input string
        
    Returns:
        Longest palindromic substring
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['"babad"'],
      expected: '"bab"',
    },
    {
      input: ['"cbbd"'],
      expected: ['"bb"'],
    },
    {
      input: ['"a"'],
      expected: '"a"',
    },
  ],
  timeComplexity: 'O(n²)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/longest-palindromic-substring/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XYQecbcd6_c',
};
