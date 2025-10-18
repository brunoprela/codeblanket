/**
 * Longest Substring with At Most K Distinct Characters
 * Problem ID: longest-substring-k-distinct
 * Order: 21
 */

import { Problem } from '../../../types';

export const longest_substring_k_distinctProblem: Problem = {
  id: 'longest-substring-k-distinct',
  title: 'Longest Substring with At Most K Distinct Characters',
  difficulty: 'Hard',
  topic: 'Arrays & Hashing',
  order: 21,
  description: `Given a string \`s\` and an integer \`k\`, return the length of the longest substring of \`s\` that contains at most \`k\` distinct characters.`,
  examples: [
    {
      input: 's = "eceba", k = 2',
      output: '3',
      explanation: 'The substring is "ece" with length 3.',
    },
    {
      input: 's = "aa", k = 1',
      output: '2',
      explanation: 'The substring is "aa" with length 2.',
    },
  ],
  constraints: ['1 <= s.length <= 5 * 10^4', '0 <= k <= 50'],
  hints: [
    'Use sliding window with hash map',
    'Track character frequencies',
    'Shrink window when distinct count exceeds k',
  ],
  starterCode: `def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Find longest substring with at most k distinct characters.
    
    Args:
        s: Input string
        k: Maximum number of distinct characters
        
    Returns:
        Length of longest valid substring
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['eceba', 2],
      expected: 3,
    },
    {
      input: ['aa', 1],
      expected: 2,
    },
  ],
  solution: `def length_of_longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Sliding window with frequency map.
    Time: O(n), Space: O(k)
    """
    if k == 0:
        return 0
    
    max_length = 0
    freq = {}
    left = 0
    
    for right in range(len(s)):
        # Add right character
        freq[s[right]] = freq.get(s[right], 0) + 1
        
        # Shrink window if too many distinct characters
        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k)',
  leetcodeUrl:
    'https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/',
  youtubeUrl: 'https://www.youtube.com/watch?v=MK-NZ4hN7rs',
};
