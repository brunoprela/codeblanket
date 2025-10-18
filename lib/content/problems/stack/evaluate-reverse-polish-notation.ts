/**
 * Evaluate Reverse Polish Notation
 * Problem ID: evaluate-reverse-polish-notation
 * Order: 12
 */

import { Problem } from '../../../types';

export const evaluate_reverse_polish_notationProblem: Problem = {
  id: 'evaluate-reverse-polish-notation',
  title: 'Evaluate Reverse Polish Notation',
  difficulty: 'Medium',
  topic: 'Stack',
  description: `You are given an array of strings \`tokens\` that represents an arithmetic expression in Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

**Note that:**
- The valid operators are \`'+'\`, \`'-'\`, \`'*'\`, and \`'/'\`.
- Each operand may be an integer or another expression.
- The division between two integers always truncates toward zero.
- There will not be any division by zero.
- The input represents a valid arithmetic expression in reverse polish notation.
- The answer and all the intermediate calculations can be represented in a 32-bit integer.`,
  examples: [
    {
      input: 'tokens = ["2","1","+","3","*"]',
      output: '9',
      explanation: '((2 + 1) * 3) = 9',
    },
    {
      input: 'tokens = ["4","13","5","/","+"]',
      output: '6',
      explanation: '(4 + (13 / 5)) = 6',
    },
  ],
  constraints: [
    '1 <= tokens.length <= 10^4',
    'tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200]',
  ],
  hints: [
    'Use a stack to store operands',
    'When you see an operator, pop two operands, apply operator, push result',
  ],
  starterCode: `from typing import List

def eval_rpn(tokens: List[str]) -> int:
    """
    Evaluate Reverse Polish Notation expression.
    
    Args:
        tokens: Array of tokens representing RPN expression
        
    Returns:
        Result of evaluation
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['2', '1', '+', '3', '*']],
      expected: 9,
    },
    {
      input: [['4', '13', '5', '/', '+']],
      expected: 6,
    },
    {
      input: [
        ['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+'],
      ],
      expected: 22,
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl:
    'https://leetcode.com/problems/evaluate-reverse-polish-notation/',
  youtubeUrl: 'https://www.youtube.com/watch?v=iu0082c4HDE',
};
