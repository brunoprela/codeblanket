/**
 * Group Anagrams
 * Problem ID: group-anagrams-string
 * Order: 2
 */

import { Problem } from '../../../types';

export const group_anagrams_stringProblem: Problem = {
  id: 'group-anagrams-string',
  title: 'Group Anagrams',
  difficulty: 'Medium',
  topic: 'String Algorithms',
  description: `Given an array of strings \`strs\`, group the anagrams together. You can return the answer in any order.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Example 1:**
\`\`\`
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
\`\`\`

**Example 2:**
\`\`\`
Input: strs = [""]
Output: [[""]]
\`\`\`

**Example 3:**
\`\`\`
Input: strs = ["a"]
Output: [["a"]]
\`\`\``,
  starterCode: `def group_anagrams(strs):
    """
    Group strings that are anagrams.
    
    Args:
        strs: List of strings
        
    Returns:
        List of lists, grouped anagrams
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['eat', 'tea', 'tan', 'ate', 'nat', 'bat']],
      expected: '[["bat"],["nat","tan"],["ate","eat","tea"]]',
    },
    {
      input: [['']],
      expected: '[[""]]',
    },
    {
      input: [['a']],
      expected: '[["a"]]',
    },
  ],
  timeComplexity:
    'O(n * k log k) where n is strs length, k is max string length',
  spaceComplexity: 'O(n * k)',
  leetcodeUrl: 'https://leetcode.com/problems/group-anagrams/',
  youtubeUrl: 'https://www.youtube.com/watch?v=vzdNOK2oB2E',
};
