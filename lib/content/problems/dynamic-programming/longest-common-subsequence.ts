/**
 * Longest Common Subsequence
 * Problem ID: longest-common-subsequence
 * Order: 10
 */

import { Problem } from '../../../types';

export const longest_common_subsequenceProblem: Problem = {
  id: 'longest-common-subsequence',
  title: 'Longest Common Subsequence',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `Given two strings \`text1\` and \`text2\`, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

**Example 1:**
\`\`\`
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
\`\`\`

**Example 2:**
\`\`\`
Input: text1 = "abc", text2 = "abc"
Output: 3
\`\`\`

**Example 3:**
\`\`\`
Input: text1 = "abc", text2 = "def"
Output: 0
\`\`\``,
  starterCode: `def longest_common_subsequence(text1, text2):
    """
    Find length of longest common subsequence.
    
    Args:
        text1: First string
        text2: Second string
        
    Returns:
        Length of LCS
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['abcde', 'ace'],
      expected: 3,
    },
    {
      input: ['abc', 'abc'],
      expected: 3,
    },
    {
      input: ['abc', 'def'],
      expected: 0,
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(m * n) or O(min(m,n)) with optimization',
  leetcodeUrl: 'https://leetcode.com/problems/longest-common-subsequence/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Ua0GhsJSlWM',
};
