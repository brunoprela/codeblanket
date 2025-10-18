/**
 * Valid Parentheses
 * Problem ID: fundamentals-valid-parentheses
 * Order: 35
 */

import { Problem } from '../../../types';

export const valid_parenthesesProblem: Problem = {
  id: 'fundamentals-valid-parentheses',
  title: 'Valid Parentheses',
  difficulty: 'Easy',
  description: `Determine if a string of parentheses is valid.

Valid means:
- Every opening bracket has a matching closing bracket
- Brackets are closed in the correct order
- Support: (), [], {}

**Example:** "({[]})" → True, "([)]" → False

This tests:
- Stack data structure
- String parsing
- Matching pairs`,
  examples: [
    {
      input: 's = "()"',
      output: 'True',
    },
    {
      input: 's = "()[]{}"',
      output: 'True',
    },
    {
      input: 's = "([)]"',
      output: 'False',
      explanation: 'Brackets not closed in correct order',
    },
  ],
  constraints: ['0 <= len(s) <= 10^4', 'Only parentheses characters'],
  hints: [
    'Use a stack to track opening brackets',
    'Pop from stack when closing bracket found',
    'Check if brackets match',
  ],
  starterCode: `def is_valid_parentheses(s):
    """
    Check if parentheses string is valid.
    
    Args:
        s: String of parentheses
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> is_valid_parentheses("()")
        True
        >>> is_valid_parentheses("([)]")
        False
    """
    pass


# Test
print(is_valid_parentheses("({[]})"))
`,
  testCases: [
    {
      input: ['()'],
      expected: true,
    },
    {
      input: ['()[]{}'],
      expected: true,
    },
    {
      input: ['([)]'],
      expected: false,
    },
    {
      input: ['{[]}'],
      expected: true,
    },
  ],
  solution: `def is_valid_parentheses(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    
    for char in s:
        if char in pairs:
            stack.append(char)
        else:
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 35,
  topic: 'Python Fundamentals',
};
