/**
 * Longest Repeating Character Replacement
 * Problem ID: longest-repeating-character-replacement
 * Order: 13
 */

import { Problem } from '../../../types';

export const longest_repeating_character_replacementProblem: Problem = {
  id: 'longest-repeating-character-replacement',
  title: 'Longest Repeating Character Replacement',
  difficulty: 'Medium',
  topic: 'Sliding Window',
  description: `You are given a string \`s\` and an integer \`k\`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most \`k\` times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.`,
  examples: [
    {
      input: 's = "ABAB", k = 2',
      output: '4',
      explanation: "Replace the two A's with two B's or vice versa.",
    },
    {
      input: 's = "AABABBA", k = 1',
      output: '4',
      explanation:
        'Replace the one A in the middle with B and form "AABBBBA". The substring "BBBB" has the longest repeating letters, which is 4.',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^5',
    's consists of only uppercase English letters',
    '0 <= k <= s.length',
  ],
  hints: [
    'Use sliding window with variable size',
    'Track the most frequent character in the window',
    'If window_size - max_freq > k, shrink window',
  ],
  starterCode: `def character_replacement(s: str, k: int) -> int:
    """
    Find longest substring with same letter after k replacements.
    
    Args:
        s: String of uppercase letters
        k: Maximum number of replacements allowed
        
    Returns:
        Length of longest substring
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['ABAB', 2],
      expected: 4,
    },
    {
      input: ['AABABBA', 1],
      expected: 4,
    },
    {
      input: ['AAAA', 0],
      expected: 4,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl:
    'https://leetcode.com/problems/longest-repeating-character-replacement/',
  youtubeUrl: 'https://www.youtube.com/watch?v=gqXU1UyA8pk',
};
