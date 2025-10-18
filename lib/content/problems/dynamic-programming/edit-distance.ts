/**
 * Edit Distance
 * Problem ID: edit-distance
 * Order: 11
 */

import { Problem } from '../../../types';

export const edit_distanceProblem: Problem = {
  id: 'edit-distance',
  title: 'Edit Distance',
  difficulty: 'Hard',
  topic: 'Dynamic Programming',
  description: `Given two strings \`word1\` and \`word2\`, return the minimum number of operations required to convert \`word1\` to \`word2\`.

You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

**Example 1:**
\`\`\`
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
\`\`\`

**Example 2:**
\`\`\`
Input: word1 = "intention", word2 = "execution"
Output: 5
\`\`\``,
  starterCode: `def min_distance(word1, word2):
    """
    Calculate minimum edit distance (Levenshtein distance).
    
    Args:
        word1: Source string
        word2: Target string
        
    Returns:
        Minimum number of operations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['horse', 'ros'],
      expected: 3,
    },
    {
      input: ['intention', 'execution'],
      expected: 5,
    },
    {
      input: ['', 'abc'],
      expected: 3,
    },
  ],
  timeComplexity: 'O(m * n)',
  spaceComplexity: 'O(m * n) or O(min(m,n)) with optimization',
  leetcodeUrl: 'https://leetcode.com/problems/edit-distance/',
  youtubeUrl: 'https://www.youtube.com/watch?v=XYi2-LPrwm4',
};
