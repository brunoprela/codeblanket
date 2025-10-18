/**
 * Implement strStr()
 * Problem ID: implement-strstr
 * Order: 5
 */

import { Problem } from '../../../types';

export const implement_strstrProblem: Problem = {
  id: 'implement-strstr',
  title: 'Implement strStr()',
  difficulty: 'Easy',
  topic: 'String Algorithms',
  description: `Given two strings \`needle\` and \`haystack\`, return the index of the first occurrence of \`needle\` in \`haystack\`, or \`-1\` if \`needle\` is not part of \`haystack\`.

**Clarification:**
What should we return when \`needle\` is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when \`needle\` is an empty string. This is consistent to C's strstr() and Java's indexOf().

**Example 1:**
\`\`\`
Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
\`\`\`

**Example 2:**
\`\`\`
Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
\`\`\``,
  starterCode: `def str_str(haystack, needle):
    """
    Find first occurrence of needle in haystack.
    
    Args:
        haystack: String to search in
        needle: String to find
        
    Returns:
        Index of first occurrence, or -1
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['"sadbutsad"', '"sad"'],
      expected: 0,
    },
    {
      input: ['"leetcode"', '"leeto"'],
      expected: -1,
    },
    {
      input: ['"hello"', '""'],
      expected: 0,
    },
  ],
  timeComplexity: 'O(n * m) naive, O(n + m) with KMP',
  spaceComplexity: 'O(1) naive, O(m) with KMP',
  leetcodeUrl:
    'https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Gjkhm1gYIMw',
};
