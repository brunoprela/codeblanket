/**
 * Longest Common Prefix
 * Problem ID: longest-common-prefix-trie
 * Order: 4
 */

import { Problem } from '../../../types';

export const longest_common_prefix_trieProblem: Problem = {
  id: 'longest-common-prefix-trie',
  title: 'Longest Common Prefix',
  difficulty: 'Easy',
  topic: 'Tries',
  description: `Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string \`""\`.`,
  examples: [
    {
      input: 'strs = ["flower","flow","flight"]',
      output: '"fl"',
    },
    {
      input: 'strs = ["dog","racecar","car"]',
      output: '""',
      explanation: 'There is no common prefix.',
    },
  ],
  constraints: [
    '1 <= strs.length <= 200',
    '0 <= strs[i].length <= 200',
    'strs[i] consists of only lowercase English letters',
  ],
  hints: [
    'Build trie from all words',
    'Traverse from root until branching or end',
  ],
  starterCode: `from typing import List

def longest_common_prefix(strs: List[str]) -> str:
    """
    Find longest common prefix in array of strings.
    
    Args:
        strs: Array of strings
        
    Returns:
        Longest common prefix
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['flower', 'flow', 'flight']],
      expected: 'fl',
    },
    {
      input: [['dog', 'racecar', 'car']],
      expected: '',
    },
  ],
  timeComplexity: 'O(S) where S is sum of all characters',
  spaceComplexity: 'O(S)',
  leetcodeUrl: 'https://leetcode.com/problems/longest-common-prefix/',
  youtubeUrl: 'https://www.youtube.com/watch?v=0sWShKIJoo4',
};
