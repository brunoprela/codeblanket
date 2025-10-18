/**
 * Find All Anagrams in a String
 * Problem ID: find-all-anagrams
 * Order: 9
 */

import { Problem } from '../../../types';

export const find_all_anagramsProblem: Problem = {
  id: 'find-all-anagrams',
  title: 'Find All Anagrams in a String',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  description: `Given two strings \`s\` and \`p\`, return an array of all the start indices of \`p\`'s anagrams in \`s\`. You may return the answer in any order.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.`,
  examples: [
    {
      input: 's = "cbaebabacd", p = "abc"',
      output: '[0,6]',
      explanation:
        'The substring with start index = 0 is "cba", which is an anagram of "abc". The substring with start index = 6 is "bac", which is an anagram of "abc".',
    },
    {
      input: 's = "abab", p = "ab"',
      output: '[0,1,2]',
    },
  ],
  constraints: [
    '1 <= s.length, p.length <= 3 * 10^4',
    's and p consist of lowercase English letters',
  ],
  hints: [
    'Use a sliding window of size len(p)',
    'Compare character frequencies',
  ],
  starterCode: `from typing import List

def find_anagrams(s: str, p: str) -> List[int]:
    """
    Find all start indices of anagrams of p in s.
    
    Args:
        s: String to search in
        p: Pattern string
        
    Returns:
        List of start indices
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['cbaebabacd', 'abc'],
      expected: [0, 6],
    },
    {
      input: ['abab', 'ab'],
      expected: [0, 1, 2],
    },
    {
      input: ['baa', 'aa'],
      expected: [1],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/find-all-anagrams-in-a-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=G8xtZy0fDKg',
};
