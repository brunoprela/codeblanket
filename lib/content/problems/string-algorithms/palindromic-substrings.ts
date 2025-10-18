/**
 * Palindromic Substrings
 * Problem ID: palindromic-substrings
 * Order: 3
 */

import { Problem } from '../../../types';

export const palindromic_substringsProblem: Problem = {
  id: 'palindromic-substrings',
  title: 'Palindromic Substrings',
  difficulty: 'Medium',
  topic: 'String Algorithms',
  description: `Given a string \`s\`, return the number of **palindromic substrings** in it.

A string is a **palindrome** when it reads the same backward as forward.

A **substring** is a contiguous sequence of characters within the string.

**Example 1:**
\`\`\`
Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
\`\`\`

**Example 2:**
\`\`\`
Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
\`\`\``,
  starterCode: `def count_substrings(s):
    """
    Count palindromic substrings.
    
    Args:
        s: Input string
        
    Returns:
        Count of palindromic substrings
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['"abc"'],
      expected: 3,
    },
    {
      input: ['"aaa"'],
      expected: 6,
    },
    {
      input: ['"racecar"'],
      expected: 10,
    },
  ],
  timeComplexity: 'O(n²)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/palindromic-substrings/',
  youtubeUrl: 'https://www.youtube.com/watch?v=4RACzI5-du8',
};
