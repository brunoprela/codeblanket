/**
 * Valid Palindrome II
 * Problem ID: valid-palindrome-ii
 * Order: 7
 */

import { Problem } from '../../../types';

export const valid_palindrome_iiProblem: Problem = {
  id: 'valid-palindrome-ii',
  title: 'Valid Palindrome II',
  difficulty: 'Easy',
  topic: 'Two Pointers',
  order: 7,
  description: `Given a string \`s\`, return \`true\` if the \`s\` can be palindrome after deleting **at most one** character from it.`,
  examples: [
    {
      input: 's = "aba"',
      output: 'true',
    },
    {
      input: 's = "abca"',
      output: 'true',
      explanation:
        'You could delete the character "c" or "b" to make it a palindrome.',
    },
    {
      input: 's = "abc"',
      output: 'false',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^5',
    's consists of lowercase English letters',
  ],
  hints: [
    'Use two pointers from both ends',
    'When characters mismatch, try skipping either left or right character',
    'Check if remaining substring is a palindrome',
  ],
  starterCode: `def valid_palindrome(s: str) -> bool:
    """
    Check if string can be palindrome by deleting at most one character.
    
    Args:
        s: Input string
        
    Returns:
        True if can be palindrome with at most one deletion
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['aba'],
      expected: true,
    },
    {
      input: ['abca'],
      expected: true,
    },
    {
      input: ['abc'],
      expected: false,
    },
  ],
  solution: `def valid_palindrome(s: str) -> bool:
    """
    Two pointers with one deletion allowed.
    Time: O(n), Space: O(1)
    """
    def is_palindrome(left: int, right: int) -> bool:
        """Helper to check if substring is palindrome"""
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try skipping either left or right character
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    
    return True
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/valid-palindrome-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=JrxRYBwG6EI',
};
