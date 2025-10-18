/**
 * Longest Substring Without Repeating Characters
 * Problem ID: longest-substring-without-repeating
 * Order: 2
 */

import { Problem } from '../../../types';

export const longest_substring_without_repeatingProblem: Problem = {
  id: 'longest-substring-without-repeating',
  title: 'Longest Substring Without Repeating Characters',
  difficulty: 'Medium',
  description: `Given a string \`s\`, find the length of the **longest substring** without repeating characters.


**Approach:**
Use a variable-size sliding window. Expand the window by moving the right pointer to add new characters. When a duplicate is found, shrink the window from the left until the duplicate is removed. Track the maximum window size seen.`,
  examples: [
    {
      input: 's = "abcabcbb"',
      output: '3',
      explanation: 'The answer is "abc", with the length of 3.',
    },
    {
      input: 's = "bbbbb"',
      output: '1',
      explanation: 'The answer is "b", with the length of 1.',
    },
    {
      input: 's = "pwwkew"',
      output: '3',
      explanation:
        'The answer is "wke", with the length of 3. Note that "pwke" is a subsequence and not a substring.',
    },
  ],
  constraints: [
    '0 <= s.length <= 5 * 10^4',
    's consists of English letters, digits, symbols and spaces',
  ],
  hints: [
    'Use a sliding window with a hash set to track characters in the current window',
    'When you encounter a duplicate, remove characters from the left until the duplicate is gone',
    'Track the maximum window size throughout the process',
    'Alternative: Use a hash map to store the last seen index of each character',
  ],
  starterCode: `def length_of_longest_substring(s: str) -> int:
    """
    Find the length of the longest substring without repeating characters.
    
    Args:
        s: Input string
        
    Returns:
        Length of longest substring without repeating characters
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['abcabcbb'],
      expected: 3,
    },
    {
      input: ['bbbbb'],
      expected: 1,
    },
    {
      input: ['pwwkew'],
      expected: 3,
    },
    {
      input: [''],
      expected: 0,
    },
    {
      input: ['au'],
      expected: 2,
    },
    {
      input: ['dvdf'],
      expected: 3,
    },
  ],
  solution: `def length_of_longest_substring(s: str) -> int:
    """
    Sliding window with hash set.
    Time: O(N), Space: O(min(N, M)) where M is charset size
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Remove characters from left until no duplicate
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length


# Alternative: Hash map with last seen index (more efficient)
def length_of_longest_substring_optimized(s: str) -> int:
    """
    Optimized approach using hash map to store last seen index.
    Allows skipping ahead instead of incrementing left one by one.
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # If character seen before and within current window
        if s[right] in char_index and char_index[s[right]] >= left:
            # Move left pointer past the previous occurrence
            left = char_index[s[right]] + 1
        
        # Update last seen index
        char_index[s[right]] = right
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(min(n, m)) where m is the character set size',

  leetcodeUrl:
    'https://leetcode.com/problems/longest-substring-without-repeating-characters/',
  youtubeUrl: 'https://www.youtube.com/watch?v=wiGpQwVHdE0',
  order: 2,
  topic: 'Sliding Window',
};
