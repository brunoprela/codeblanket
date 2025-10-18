/**
 * Backspace String Compare
 * Problem ID: backspace-string-compare
 * Order: 6
 */

import { Problem } from '../../../types';

export const backspace_string_compareProblem: Problem = {
  id: 'backspace-string-compare',
  title: 'Backspace String Compare',
  difficulty: 'Easy',
  topic: 'Stack',
  order: 6,
  description: `Given two strings \`s\` and \`t\`, return \`true\` if they are equal when both are typed into empty text editors. \`'#'\` means a backspace character.

Note that after backspacing an empty text, the text will continue empty.`,
  examples: [
    {
      input: 's = "ab#c", t = "ad#c"',
      output: 'true',
      explanation: 'Both s and t become "ac".',
    },
    {
      input: 's = "ab##", t = "c#d#"',
      output: 'true',
      explanation: 'Both s and t become "".',
    },
    {
      input: 's = "a#c", t = "b"',
      output: 'false',
      explanation: 's becomes "c" while t becomes "b".',
    },
  ],
  constraints: [
    '1 <= s.length, t.length <= 200',
    's and t only contain lowercase letters and "#" characters',
  ],
  hints: [
    'Use a stack to process each string',
    'When you see "#", pop from stack if not empty',
    'Otherwise, push the character',
    'Compare the final stacks',
  ],
  starterCode: `def backspace_compare(s: str, t: str) -> bool:
    """
    Compare two strings with backspace characters.
    
    Args:
        s: First string
        t: Second string
        
    Returns:
        True if they are equal after processing backspaces
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['ab#c', 'ad#c'],
      expected: true,
    },
    {
      input: ['ab##', 'c#d#'],
      expected: true,
    },
    {
      input: ['a#c', 'b'],
      expected: false,
    },
  ],
  solution: `def backspace_compare(s: str, t: str) -> bool:
    """
    Build final strings using stacks and compare.
    Time: O(n + m), Space: O(n + m)
    """
    def build_string(string: str) -> str:
        """Helper to process backspaces"""
        stack = []
        for char in string:
            if char == '#':
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)
    
    return build_string(s) == build_string(t)

# Alternative: O(1) space using two pointers from end
def backspace_compare_optimal(s: str, t: str) -> bool:
    """
    Process from end to avoid extra space.
    Time: O(n + m), Space: O(1)
    """
    def next_valid_char_index(string: str, index: int) -> int:
        """Find next valid character index"""
        backspace_count = 0
        while index >= 0:
            if string[index] == '#':
                backspace_count += 1
            elif backspace_count > 0:
                backspace_count -= 1
            else:
                break
            index -= 1
        return index
    
    i, j = len(s) - 1, len(t) - 1
    
    while i >= 0 or j >= 0:
        i = next_valid_char_index(s, i)
        j = next_valid_char_index(t, j)
        
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            return False
        
        i -= 1
        j -= 1
    
    return True
`,
  timeComplexity: 'O(n + m)',
  spaceComplexity: 'O(n + m) for stack, O(1) for two-pointer',
  leetcodeUrl: 'https://leetcode.com/problems/backspace-string-compare/',
  youtubeUrl: 'https://www.youtube.com/watch?v=4YgzB_8dE8Y',
};
