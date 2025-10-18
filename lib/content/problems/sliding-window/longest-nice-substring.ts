/**
 * Longest Nice Substring
 * Problem ID: longest-nice-substring
 * Order: 8
 */

import { Problem } from '../../../types';

export const longest_nice_substringProblem: Problem = {
  id: 'longest-nice-substring',
  title: 'Longest Nice Substring',
  difficulty: 'Easy',
  topic: 'Sliding Window',
  order: 8,
  description: `A string \`s\` is **nice** if, for every letter of the alphabet that \`s\` contains, it appears **both** in uppercase and lowercase. For example, \`"abABB"\` is nice because \`'A'\` and \`'a'\` appear, and \`'B'\` and \`'b'\` appear. However, \`"abA"\` is not because \`'b'\` appears, but \`'B'\` does not.

Given a string \`s\`, return the longest **substring** of \`s\` that is **nice**. If there are multiple, return the substring of the **earliest** occurrence. If there are none, return an empty string.`,
  examples: [
    {
      input: 's = "YazaAay"',
      output: '"aAa"',
      explanation:
        '"aAa" is a nice string because \'A/a\' and \'Y/y\' are both present, but "aAa" is longer.',
    },
    {
      input: 's = "Bb"',
      output: '"Bb"',
      explanation: "\"Bb\" is a nice string because both 'B' and 'b' appear.",
    },
    {
      input: 's = "c"',
      output: '""',
      explanation: 'There are no nice substrings.',
    },
  ],
  constraints: [
    '1 <= s.length <= 100',
    's consists of uppercase and lowercase English letters',
  ],
  hints: [
    'Check all substrings',
    "For each substring, verify if it's nice",
    'A string is nice if for every char, both cases exist',
  ],
  starterCode: `def longest_nice_substring(s: str) -> str:
    """
    Find longest nice substring.
    
    Args:
        s: Input string
        
    Returns:
        Longest nice substring
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['YazaAay'],
      expected: 'aAa',
    },
    {
      input: ['Bb'],
      expected: 'Bb',
    },
    {
      input: ['c'],
      expected: '',
    },
  ],
  solution: `def longest_nice_substring(s: str) -> str:
    """
    Check all substrings for nice property.
    Time: O(n^2), Space: O(n)
    """
    def is_nice(substring: str) -> bool:
        """Check if substring is nice"""
        char_set = set(substring)
        for char in char_set:
            if char.swapcase() not in char_set:
                return False
        return True
    
    longest = ""
    n = len(s)
    
    # Try all substrings
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = s[i:j]
            if is_nice(substring) and len(substring) > len(longest):
                longest = substring
    
    return longest

# Alternative: Divide and conquer
def longest_nice_substring_dc(s: str) -> str:
    """
    Divide and conquer approach.
    Time: O(n^2) worst case, Space: O(n)
    """
    if len(s) < 2:
        return ""
    
    char_set = set(s)
    
    for i, char in enumerate(s):
        if char.swapcase() not in char_set:
            # Split at this position
            left = longest_nice_substring_dc(s[:i])
            right = longest_nice_substring_dc(s[i + 1:])
            return left if len(left) >= len(right) else right
    
    # Entire string is nice
    return s
`,
  timeComplexity: 'O(n^2)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/longest-nice-substring/',
  youtubeUrl: 'https://www.youtube.com/watch?v=fS2Rz0_JVVE',
};
