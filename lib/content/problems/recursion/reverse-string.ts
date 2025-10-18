/**
 * Reverse String
 * Problem ID: recursion-reverse-string
 * Order: 4
 */

import { Problem } from '../../../types';

export const reverse_stringProblem: Problem = {
  id: 'recursion-reverse-string',
  title: 'Reverse String',
  difficulty: 'Easy',
  topic: 'Recursion',
  description: `Reverse a string using recursion.

You cannot use built-in reverse functions or slicing [::-1] - must use recursion!

**Approach:**
- Base case: empty or single character returns itself
- Recursive case: last character + reverse of rest`,
  examples: [
    { input: 's = "hello"', output: '"olleh"' },
    { input: 's = "a"', output: '"a"' },
    { input: 's = ""', output: '""' },
  ],
  constraints: [
    '0 <= s.length <= 1000',
    's consists of printable ASCII characters',
  ],
  hints: [
    'Base case: string of length 0 or 1 is already reversed',
    'Recursive case: last char + reverse(all but last char)',
    'Or: reverse(all but first char) + first char',
    'String slicing: s[:-1] is all but last, s[-1] is last char',
  ],
  starterCode: `def reverse_string(s):
    """
    Reverse string using recursion.
    
    Args:
        s: String to reverse
        
    Returns:
        Reversed string
        
    Examples:
        >>> reverse_string("hello")
        "olleh"
        >>> reverse_string("a")
        "a"
    """
    pass


# Test cases
print(reverse_string("hello"))  # Expected: "olleh"
print(reverse_string(""))  # Expected: ""
`,
  testCases: [
    { input: ['hello'], expected: 'olleh' },
    { input: ['a'], expected: 'a' },
    { input: [''], expected: '' },
    { input: ['racecar'], expected: 'racecar' },
    { input: ['Python'], expected: 'nohtyP' },
  ],
  solution: `def reverse_string(s):
    """Reverse string using recursion"""
    # Base case: empty or single character
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])


# Alternative approach (first + rest):
def reverse_string_alt(s):
    """Reverse string using recursion - alternative"""
    if len(s) <= 1:
        return s
    
    # Recursive case: reverse of rest + first char
    return reverse_string_alt(s[1:]) + s[0]


# Time Complexity: O(n²) due to string concatenation
# Space Complexity: O(n²) - call stack + string copies
# Note: In Python, strings are immutable, so each + creates new string`,
  timeComplexity: 'O(n²) due to string immutability',
  spaceComplexity: 'O(n²)',
  followUp: [
    'How can you make this more efficient?',
    'What if you use a list instead of strings?',
    'Can you reverse it in-place (for mutable structures)?',
  ],
};
