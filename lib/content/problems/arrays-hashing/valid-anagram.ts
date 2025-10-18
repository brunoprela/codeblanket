/**
 * Valid Anagram
 * Problem ID: valid-anagram
 * Order: 4
 */

import { Problem } from '../../../types';

export const valid_anagramProblem: Problem = {
  id: 'valid-anagram',
  title: 'Valid Anagram',
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 4,
  description: `Given two strings \`s\` and \`t\`, return \`true\` if \`t\` is an anagram of \`s\`, and \`false\` otherwise.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.`,
  examples: [
    {
      input: 's = "anagram", t = "nagaram"',
      output: 'true',
    },
    {
      input: 's = "rat", t = "car"',
      output: 'false',
    },
  ],
  constraints: [
    '1 <= s.length, t.length <= 5 * 10^4',
    's and t consist of lowercase English letters',
  ],
  hints: [
    'Sort both strings and compare',
    'Or use a hash map to count character frequencies',
    'Both strings must have same length to be anagrams',
  ],
  starterCode: `def is_anagram(s: str, t: str) -> bool:
    """
    Check if t is an anagram of s.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if t is an anagram of s
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['anagram', 'nagaram'],
      expected: true,
    },
    {
      input: ['rat', 'car'],
      expected: false,
    },
  ],
  solution: `def is_anagram(s: str, t: str) -> bool:
    """
    Hash map approach to count frequencies.
    Time: O(n), Space: O(1) - at most 26 letters
    """
    if len(s) != len(t):
        return False
    
    count = {}
    
    # Count characters in s
    for c in s:
        count[c] = count.get(c, 0) + 1
    
    # Decrement for characters in t
    for c in t:
        if c not in count:
            return False
        count[c] -= 1
        if count[c] < 0:
            return False
    
    return True

# Alternative: Sorting approach
def is_anagram_sort(s: str, t: str) -> bool:
    """
    Sort both strings and compare.
    Time: O(n log n), Space: O(1)
    """
    return sorted(s) == sorted(t)
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/valid-anagram/',
  youtubeUrl: 'https://www.youtube.com/watch?v=9UtInBqnCgA',
};
