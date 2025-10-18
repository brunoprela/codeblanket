/**
 * Word Break
 * Problem ID: word-break
 * Order: 13
 */

import { Problem } from '../../../types';

export const word_breakProblem: Problem = {
  id: 'word-break',
  title: 'Word Break',
  difficulty: 'Medium',
  topic: 'Dynamic Programming',
  description: `Given a string \`s\` and a dictionary of strings \`wordDict\`, return \`true\` if \`s\` can be segmented into a space-separated sequence of one or more dictionary words.

**Note:** The same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**
\`\`\`
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
\`\`\`

**Example 2:**
\`\`\`
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
\`\`\`

**Example 3:**
\`\`\`
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
\`\`\``,
  starterCode: `def word_break(s, word_dict):
    """
    Check if string can be segmented using dictionary words.
    
    Args:
        s: String to segment
        word_dict: List of valid words
        
    Returns:
        True if segmentation possible, False otherwise
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['leetcode', ['leet', 'code']],
      expected: true,
    },
    {
      input: ['applepenapple', ['apple', 'pen']],
      expected: true,
    },
    {
      input: ['catsandog', ['cats', 'dog', 'sand', 'and', 'cat']],
      expected: false,
    },
  ],
  timeComplexity: 'O(n² * m) where n is string length, m is avg word length',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/word-break/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Sx9NNgInc3A',
};
