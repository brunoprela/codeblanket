/**
 * Check Palindrome
 * Problem ID: recursion-palindrome
 * Order: 5
 */

import { Problem } from '../../../types';

export const palindromeProblem: Problem = {
  id: 'recursion-palindrome',
  title: 'Check Palindrome',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Check if a string is a palindrome using recursion.

A palindrome reads the same forward and backward (e.g., "racecar", "madam").

**Approach:**
- Base case: string of length 0 or 1 is a palindrome
- Recursive case: first and last characters must match, AND middle must be palindrome

**Note:** Ignore spaces and case for this problem.`,
  examples: [
    { input: 's = "racecar"', output: 'true' },
    { input: 's = "hello"', output: 'false' },
    { input: 's = "A man a plan a canal Panama"', output: 'true' },
  ],
  constraints: [
    '1 <= s.length <= 1000',
    's consists of printable ASCII characters',
  ],
  hints: [
    'First clean the string: remove spaces and convert to lowercase',
    'Base case: length 0 or 1 is a palindrome',
    'Check if first and last characters match',
    'Recursively check the middle substring',
    'Use helper function with left and right pointers',
  ],
  starterCode: `def is_palindrome(s):
    """
    Check if string is palindrome using recursion.
    
    Args:
        s: String to check (ignore spaces and case)
        
    Returns:
        True if palindrome, False otherwise
        
    Examples:
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
    """
    pass


# Test cases
print(is_palindrome("racecar"))  # Expected: True
print(is_palindrome("hello"))  # Expected: False
`,
  testCases: [
    { input: ['racecar'], expected: true },
    { input: ['hello'], expected: false },
    { input: ['a'], expected: true },
    { input: ['ab'], expected: false },
    { input: ['aba'], expected: true },
  ],
  solution: `def is_palindrome(s):
    """Check if string is palindrome using recursion"""
    # Clean string: remove spaces and lowercase
    s = s.replace(' ', ').lower()
    
    def helper(left, right):
        # Base case: pointers met or crossed
        if left >= right:
            return True
        
        # Check if characters match
        if s[left] != s[right]:
            return False
        
        # Recursively check middle
        return helper(left + 1, right - 1)
    
    return helper(0, len(s) - 1)


# Alternative without helper function:
def is_palindrome_alt(s):
    """Check palindrome - alternative approach"""
    # Clean string
    s = s.replace(' ', ').lower()
    
    # Base case
    if len(s) <= 1:
        return True
    
    # Check first and last, recurse on middle
    if s[0] != s[-1]:
        return False
    
    return is_palindrome_alt(s[1:-1])


# Time Complexity: O(n) - checks each character once
# Space Complexity: O(n) - call stack depth`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  followUp: [
    'How would you handle special characters?',
    'Can you do this with O(1) space using iteration?',
    'What about checking if a linked list is a palindrome?',
  ],
};
