/**
 * Remove All Adjacent Duplicates In String
 * Problem ID: remove-adjacent-duplicates
 * Order: 5
 */

import { Problem } from '../../../types';

export const remove_adjacent_duplicatesProblem: Problem = {
  id: 'remove-adjacent-duplicates',
  title: 'Remove All Adjacent Duplicates In String',
  difficulty: 'Easy',
  topic: 'Stack',
  order: 5,
  description: `You are given a string \`s\` consisting of lowercase English letters. A **duplicate removal** consists of choosing two **adjacent** and **equal** letters and removing them.

We repeatedly make **duplicate removals** on \`s\` until we no longer can.

Return the final string after all such duplicate removals have been made. It can be proven that the answer is **unique**.`,
  examples: [
    {
      input: 's = "abbaca"',
      output: '"ca"',
      explanation: 'Remove "bb" to get "aaca", then remove "aa" to get "ca".',
    },
    {
      input: 's = "azxxzy"',
      output: '"ay"',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^5',
    's consists of lowercase English letters',
  ],
  hints: [
    'Use a stack to process characters',
    'If top of stack equals current char, pop it',
    'Otherwise push current char',
  ],
  starterCode: `def remove_duplicates(s: str) -> str:
    """
    Remove all adjacent duplicate characters.
    
    Args:
        s: Input string
        
    Returns:
        String after removing all adjacent duplicates
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['abbaca'],
      expected: 'ca',
    },
    {
      input: ['azxxzy'],
      expected: 'ay',
    },
    {
      input: ['aa'],
      expected: '',
    },
  ],
  solution: `def remove_duplicates(s: str) -> str:
    """
    Stack to track non-duplicate characters.
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for char in s:
        if stack and stack[-1] == char:
            # Remove adjacent duplicate
            stack.pop()
        else:
            # Add character
            stack.append(char)
    
    return '.join(stack)
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl:
    'https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/',
  youtubeUrl: 'https://www.youtube.com/watch?v=w6LcypDgC4w',
};
