/**
 * Concatenated Words
 * Problem ID: concatenated-words
 * Order: 9
 */

import { Problem } from '../../../types';

export const concatenated_wordsProblem: Problem = {
  id: 'concatenated-words',
  title: 'Concatenated Words',
  difficulty: 'Medium',
  topic: 'Tries',
  description: `Given an array of strings \`words\` (**without duplicates**), return all the **concatenated words** in the given list of \`words\`.

A **concatenated word** is defined as a string that is comprised entirely of at least two shorter words (not necessarily distinct) in the given array.`,
  examples: [
    {
      input:
        'words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]',
      output: '["catsdogcats","dogcatsdog","ratcatdogcat"]',
    },
    {
      input: 'words = ["cat","dog","catdog"]',
      output: '["catdog"]',
    },
  ],
  constraints: [
    '1 <= words.length <= 10^4',
    '1 <= words[i].length <= 30',
    'words[i] consists of only lowercase English letters',
    'All the strings of words are unique',
    '1 <= sum(words[i].length) <= 10^5',
  ],
  hints: [
    'Build trie from all words',
    'For each word, check if it can be formed by concatenating',
    'Use DP to check all possible splits',
  ],
  starterCode: `from typing import List

def find_all_concatenated_words_in_a_dict(words: List[str]) -> List[str]:
    """
    Find all concatenated words.
    
    Args:
        words: List of words
        
    Returns:
        List of concatenated words
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          'cat',
          'cats',
          'catsdogcats',
          'dog',
          'dogcatsdog',
          'hippopotamuses',
          'rat',
          'ratcatdogcat',
        ],
      ],
      expected: ['catsdogcats', 'dogcatsdog', 'ratcatdogcat'],
    },
    {
      input: [['cat', 'dog', 'catdog']],
      expected: ['catdog'],
    },
  ],
  timeComplexity: 'O(n * m^2) where n = words, m = avg length',
  spaceComplexity: 'O(n * m)',
  leetcodeUrl: 'https://leetcode.com/problems/concatenated-words/',
  youtubeUrl: 'https://www.youtube.com/watch?v=vSc10FaEWmk',
};
