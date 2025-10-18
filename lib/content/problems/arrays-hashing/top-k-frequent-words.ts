/**
 * Top K Frequent Words
 * Problem ID: top-k-frequent-words
 * Order: 28
 */

import { Problem } from '../../../types';

export const top_k_frequent_wordsProblem: Problem = {
  id: 'top-k-frequent-words',
  title: 'Top K Frequent Words',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  description: `Given an array of strings \`words\` and an integer \`k\`, return the \`k\` most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.`,
  examples: [
    {
      input: 'words = ["i","love","leetcode","i","love","coding"], k = 2',
      output: '["i","love"]',
      explanation:
        '"i" and "love" are the two most frequent words. Note that "i" comes before "love" due to a lower alphabetical order.',
    },
    {
      input:
        'words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4',
      output: '["the","is","sunny","day"]',
    },
  ],
  constraints: [
    '1 <= words.length <= 500',
    '1 <= words[i].length <= 10',
    'words[i] consists of lowercase English letters',
    'k is in the range [1, number of unique words]',
  ],
  hints: [
    'Use a hash map to count frequencies',
    'Use a heap or sort by frequency and lexicographical order',
  ],
  starterCode: `from typing import List

def top_k_frequent(words: List[str], k: int) -> List[str]:
    """
    Find k most frequent words.
    
    Args:
        words: Array of words
        k: Number of top frequent words to return
        
    Returns:
        List of k most frequent words
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['i', 'love', 'leetcode', 'i', 'love', 'coding'], 2],
      expected: ['i', 'love'],
    },
    {
      input: [
        ['the', 'day', 'is', 'sunny', 'the', 'the', 'the', 'sunny', 'is', 'is'],
        4,
      ],
      expected: ['the', 'is', 'sunny', 'day'],
    },
    {
      input: [['a', 'aa', 'aaa'], 1],
      expected: ['a'],
    },
  ],
  timeComplexity: 'O(n log k)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/top-k-frequent-words/',
  youtubeUrl: 'https://www.youtube.com/watch?v=h_lzL-R_MQk',
};
