/**
 * Find Common Characters
 * Problem ID: find-common-characters
 * Order: 12
 */

import { Problem } from '../../../types';

export const find_common_charactersProblem: Problem = {
  id: 'find-common-characters',
  title: 'Find Common Characters',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 12,
  description: `Given a string array \`words\`, return an array of all characters that show up in all strings within the \`words\` (including duplicates). You may return the answer in **any order**.`,
  examples: [
    {
      input: 'words = ["bella","label","roller"]',
      output: '["e","l","l"]',
    },
    {
      input: 'words = ["cool","lock","cook"]',
      output: '["c","o"]',
    },
  ],
  constraints: [
    '1 <= words.length <= 100',
    '1 <= words[i].length <= 100',
    'words[i] consists of lowercase English letters',
  ],
  hints: [
    'Count character frequencies for each word',
    'Find the minimum frequency for each character across all words',
    'Add characters to result based on minimum frequency',
  ],
  starterCode: `from typing import List

def common_chars(words: List[str]) -> List[str]:
    """
    Find characters common to all words.
    
    Args:
        words: List of words
        
    Returns:
        List of common characters (with duplicates)
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['bella', 'label', 'roller']],
      expected: ['e', 'l', 'l'],
    },
    {
      input: [['cool', 'lock', 'cook']],
      expected: ['c', 'o'],
    },
  ],
  solution: `from typing import List
from collections import Counter

def common_chars(words: List[str]) -> List[str]:
    """
    Find minimum frequency of each character.
    Time: O(n * k) where n = number of words, k = avg length
    Space: O(1) - at most 26 letters
    """
    # Start with frequency count of first word
    common_count = Counter(words[0])
    
    # For each subsequent word, keep minimum counts
    for word in words[1:]:
        word_count = Counter(word)
        
        # Update common_count to minimum frequencies
        for char in common_count:
            common_count[char] = min(common_count[char], word_count.get(char, 0))
    
    # Build result from common_count
    result = []
    for char, count in common_count.items():
        result.extend([char] * count)
    
    return result
`,
  timeComplexity: 'O(n * k)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/find-common-characters/',
  youtubeUrl: 'https://www.youtube.com/watch?v=kolXfMZ4kZY',
};
