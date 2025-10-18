/**
 * Minimum Window Substring
 * Problem ID: minimum-window-substring
 * Order: 3
 */

import { Problem } from '../../../types';

export const minimum_window_substringProblem: Problem = {
  id: 'minimum-window-substring',
  title: 'Minimum Window Substring',
  difficulty: 'Hard',
  description: `Given two strings \`s\` and \`t\` of lengths \`m\` and \`n\` respectively, return **the minimum window substring** of \`s\` such that every character in \`t\` (including duplicates) is included in the window. If there is no such substring, return the empty string \`""\`.

The testcases will be generated such that the answer is **unique**.


**Approach:**
Use a variable-size sliding window. Expand the window until it contains all characters of \`t\`, then shrink from the left to find the minimum valid window. Use two hash maps to track required characters and current window characters.`,
  examples: [
    {
      input: 's = "ADOBECODEBANC", t = "ABC"',
      output: '"BANC"',
      explanation:
        "The minimum window substring \"BANC\" includes 'A', 'B', and 'C' from string t.",
    },
    {
      input: 's = "a", t = "a"',
      output: '"a"',
      explanation: 'The entire string s is the minimum window.',
    },
    {
      input: 's = "a", t = "aa"',
      output: '""',
      explanation:
        "Both 'a's from t must be included in the window. Since the largest window of s only has one 'a', return empty string.",
    },
  ],
  constraints: [
    'm == s.length',
    'n == t.length',
    '1 <= m, n <= 10^5',
    's and t consist of uppercase and lowercase English letters',
  ],
  hints: [
    'Use two hash maps: one for required characters (from t), one for current window',
    'Expand window by moving right pointer until all characters are included',
    'Shrink window from left while still valid, updating minimum at each step',
    'Track how many required characters have been satisfied',
    'Use Counter from collections for easier frequency comparison',
  ],
  starterCode: `def min_window(s: str, t: str) -> str:
    """
    Find the minimum window substring containing all characters of t.
    
    Args:
        s: Source string
        t: Target string with required characters
        
    Returns:
        Minimum window substring, or empty string if none exists
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['ADOBECODEBANC', 'ABC'],
      expected: 'BANC',
    },
    {
      input: ['a', 'a'],
      expected: 'a',
    },
    {
      input: ['a', 'aa'],
      expected: '',
    },
    {
      input: ['ab', 'b'],
      expected: 'b',
    },
    {
      input: ['abc', 'cba'],
      expected: 'abc',
    },
  ],
  solution: `def min_window(s: str, t: str) -> str:
    """
    Sliding window with two frequency maps.
    Time: O(M + N), Space: O(M + N)
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    from collections import Counter
    
    # Count required characters
    required = Counter(t)
    required_count = len(required)  # Number of unique chars
    
    # Track current window
    window = {}
    formed = 0  # Number of chars with desired frequency
    
    # Result: (window_length, left, right)
    result = float('inf'), 0, 0
    left = 0
    
    for right in range(len(s)):
        # Add character to window
        char = s[right]
        window[char] = window.get(char, 0) + 1
        
        # Check if this character's frequency matches requirement
        if char in required and window[char] == required[char]:
            formed += 1
        
        # Try to shrink window while still valid
        while formed == required_count and left <= right:
            # Update result if this window is smaller
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            # Remove leftmost character
            char = s[left]
            window[char] -= 1
            if char in required and window[char] < required[char]:
                formed -= 1
            
            left += 1
    
    # Return the minimum window, or empty string if none found
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]


# Alternative: More explicit tracking
def min_window_alt(s: str, t: str) -> str:
    if not s or not t:
        return ""
    
    from collections import defaultdict
    
    # Build frequency map for t
    target_freq = defaultdict(int)
    for char in t:
        target_freq[char] += 1
    
    required = len(target_freq)
    formed = 0
    
    window_freq = defaultdict(int)
    left = 0
    min_len = float('inf')
    min_left = 0
    
    for right in range(len(s)):
        char = s[right]
        window_freq[char] += 1
        
        # Check if frequency matches for this character
        if char in target_freq and window_freq[char] == target_freq[char]:
            formed += 1
        
        # Shrink window
        while formed == required:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left
            
            # Remove from window
            left_char = s[left]
            window_freq[left_char] -= 1
            if left_char in target_freq and window_freq[left_char] < target_freq[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_left:min_left + min_len]`,
  timeComplexity: 'O(m + n) where m = len(s), n = len(t)',
  spaceComplexity: 'O(m + n)',

  leetcodeUrl: 'https://leetcode.com/problems/minimum-window-substring/',
  youtubeUrl: 'https://www.youtube.com/watch?v=jSto0O4AJbM',
  order: 3,
  topic: 'Sliding Window',
};
