/**
 * Make The String Great
 * Problem ID: make-string-great
 * Order: 9
 */

import { Problem } from '../../../types';

export const make_string_greatProblem: Problem = {
  id: 'make-string-great',
  title: 'Make The String Great',
  difficulty: 'Easy',
  topic: 'Stack',
  description: `Given a string \`s\` of lower and upper case English letters.

A good string is a string which does not have two adjacent characters \`s[i]\` and \`s[i + 1]\` where:
- \`0 <= i <= s.length - 2\`
- \`s[i]\` is a lower-case letter and \`s[i + 1]\` is the same letter but in upper-case or vice-versa.

To make the string good, you can choose two adjacent characters that make the string bad and remove them. You can keep doing this until the string becomes good.

Return the string after making it good. The answer is guaranteed to be unique under the given constraints.`,
  examples: [
    {
      input: 's = "leEeetcode"',
      output: '"leetcode"',
      explanation:
        'In the first step, either you choose i = 1 or i = 2, both will result "leEeetcode" to be reduced to "leetcode".',
    },
    {
      input: 's = "abBAcC"',
      output: '""',
    },
  ],
  constraints: [
    '1 <= s.length <= 100',
    's contains only lower and upper case English letters',
  ],
  hints: [
    'Use a stack to keep track of characters',
    'If top of stack and current character are same letter but different case, pop',
  ],
  starterCode: `def make_good(s: str) -> str:
    """
    Remove adjacent characters that are same letter but different case.
    
    Args:
        s: Input string
        
    Returns:
        Good string
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['leEeetcode'],
      expected: 'leetcode',
    },
    {
      input: ['abBAcC'],
      expected: '',
    },
    {
      input: ['s'],
      expected: 's',
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/make-the-string-great/',
  youtubeUrl: 'https://www.youtube.com/watch?v=D67hXk_ZFQM',
};
