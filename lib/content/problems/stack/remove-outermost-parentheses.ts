/**
 * Remove Outermost Parentheses
 * Problem ID: remove-outermost-parentheses
 * Order: 10
 */

import { Problem } from '../../../types';

export const remove_outermost_parenthesesProblem: Problem = {
  id: 'remove-outermost-parentheses',
  title: 'Remove Outermost Parentheses',
  difficulty: 'Easy',
  topic: 'Stack',
  description: `A valid parentheses string is either empty \`""\`, \`"(" + A + ")"\`, or \`A + B\`, where \`A\` and \`B\` are valid parentheses strings, and \`+\` represents string concatenation.

Given a valid parentheses string \`s\`, consider its primitive decomposition: \`s = P1 + P2 + ... + Pk\`, where \`Pi\` are primitive valid parentheses strings.

Return \`s\` after removing the outermost parentheses of every primitive string in the primitive decomposition of \`s\`.`,
  examples: [
    {
      input: 's = "(()())(())"',
      output: '"()()()"',
      explanation:
        'The input string is "(()())(())", with primitive decomposition "(()())" + "(())". After removing outer parentheses of each part, this is "()()" + "()" = "()()()".',
    },
    {
      input: 's = "()()"',
      output: '""',
    },
  ],
  constraints: [
    '1 <= s.length <= 10^5',
    's[i] is either ( or )',
    's is a valid parentheses string',
  ],
  hints: [
    'Keep track of the depth of parentheses',
    'Only include characters when depth > 1',
  ],
  starterCode: `def remove_outer_parentheses(s: str) -> str:
    """
    Remove outermost parentheses from primitive decomposition.
    
    Args:
        s: Valid parentheses string
        
    Returns:
        String with outermost parentheses removed
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: ['(()())(())'],
      expected: '()()()',
    },
    {
      input: ['()()'],
      expected: '',
    },
    {
      input: ['(()())(())(()(()))'],
      expected: '()()()()(())',
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/remove-outermost-parentheses/',
  youtubeUrl: 'https://www.youtube.com/watch?v=YTqd04zvkp0',
};
