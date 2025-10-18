/**
 * Generate Parentheses
 * Problem ID: generate-parentheses
 * Order: 8
 */

import { Problem } from '../../../types';

export const generate_parenthesesProblem: Problem = {
  id: 'generate-parentheses',
  title: 'Generate Parentheses',
  difficulty: 'Medium',
  topic: 'Backtracking',
  description: `Given \`n\` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.`,
  examples: [
    {
      input: 'n = 3',
      output: '["((()))","(()())","(())()","()(())","()()()"]',
    },
    {
      input: 'n = 1',
      output: '["()"]',
    },
  ],
  constraints: ['1 <= n <= 8'],
  hints: [
    'Add opening parenthesis if count < n',
    'Add closing parenthesis if close < open',
    'Base case: when both counts equal n',
  ],
  starterCode: `from typing import List

def generate_parenthesis(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    
    Args:
        n: Number of pairs
        
    Returns:
        List of valid combinations
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [3],
      expected: ['((()))', '(()())', '(())()', '()(())', '()()()'],
    },
    {
      input: [1],
      expected: ['()'],
    },
    {
      input: [2],
      expected: ['(())', '()()'],
    },
  ],
  timeComplexity: 'O(4^n / sqrt(n))',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/generate-parentheses/',
  youtubeUrl: 'https://www.youtube.com/watch?v=s9fokUqJ76A',
};
