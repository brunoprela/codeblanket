/**
 * Valid Parentheses
 * Problem ID: valid-parentheses
 * Order: 1
 */

import { Problem } from '../../../types';

export const valid_parenthesesProblem: Problem = {
  id: 'valid-parentheses',
  title: 'Valid Parentheses',
  difficulty: 'Easy',
  description: `Given a string \`s\` containing just the characters \`'('\`, \`')'\`, \`'{'\`, \`'}'\`, \`'['\` and \`']'\`, determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.


**Approach:**
Use a stack to track opening brackets. When encountering a closing bracket, check if it matches the most recent opening bracket (top of stack). If all brackets match and the stack is empty at the end, the string is valid.`,
  examples: [
    {
      input: 's = "()"',
      output: 'true',
      explanation: 'The parentheses are balanced.',
    },
    {
      input: 's = "()[]{}"',
      output: 'true',
      explanation: 'All brackets are properly matched.',
    },
    {
      input: 's = "(]"',
      output: 'false',
      explanation: 'Opening parenthesis is closed by a closing bracket.',
    },
    {
      input: 's = "([)]"',
      output: 'false',
      explanation: 'Brackets are not closed in the correct order.',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^4',
    's consists of parentheses only: ()[]{}',
  ],
  hints: [
    'Use a stack to keep track of opening brackets',
    'When you encounter a closing bracket, check if it matches the top of the stack',
    'The string is valid only if the stack is empty after processing all characters',
    'Use a dictionary to map opening brackets to their closing counterparts',
  ],
  starterCode: `def is_valid(s: str) -> bool:
    """
    Determine if the parentheses/brackets are valid.
    
    Args:
        s: String containing only '(', ')', '{', '}', '[', ']'
        
    Returns:
        True if valid, False otherwise
    """
    # Write your code here
    pass
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
      input: ['(]'],
      expected: false,
    },
    {
      input: ['([)]'],
      expected: false,
    },
    {
      input: ['{[]}'],
      expected: true,
    },
    {
      input: ['(('],
      expected: false,
    },
  ],
  solution: `def is_valid(s: str) -> bool:
    stack = []
    pairs = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in pairs:  # Opening bracket
            stack.append(char)
        else:  # Closing bracket
            if not stack or pairs[stack.pop()] != char:
                return False
    
    return len(stack) == 0

# Alternative solution with explicit mapping of closing to opening
def is_valid_alt(s: str) -> bool:
    stack = []
    closing_to_opening = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in closing_to_opening:  # Closing bracket
            if not stack or stack.pop() != closing_to_opening[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',

  leetcodeUrl: 'https://leetcode.com/problems/valid-parentheses/',
  youtubeUrl: 'https://www.youtube.com/watch?v=WTzjTskDFMg',
  order: 1,
  topic: 'Stack',
};
