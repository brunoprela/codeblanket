/**
 * Baseball Game
 * Problem ID: baseball-game
 * Order: 4
 */

import { Problem } from '../../../types';

export const baseball_gameProblem: Problem = {
  id: 'baseball-game',
  title: 'Baseball Game',
  difficulty: 'Easy',
  topic: 'Stack',
  order: 4,
  description: `You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.

You are given a list of strings \`operations\`, where \`operations[i]\` is the \`ith\` operation you must apply to the record and is one of the following:

- An integer \`x\`: Record a new score of \`x\`.
- \`"+"\`: Record a new score that is the sum of the previous two scores.
- \`"D"\`: Record a new score that is the double of the previous score.
- \`"C"\`: Invalidate the previous score, removing it from the record.

Return the sum of all the scores on the record after applying all the operations.`,
  examples: [
    {
      input: 'ops = ["5","2","C","D","+"]',
      output: '30',
      explanation:
        '"5" - Add 5 to record: [5]. "2" - Add 2: [5, 2]. "C" - Remove 2: [5]. "D" - Add 10 (double 5): [5, 10]. "+" - Add 15 (sum of 5 and 10): [5, 10, 15]. Total sum = 30.',
    },
  ],
  constraints: [
    '1 <= operations.length <= 1000',
    'operations[i] is "C", "D", "+", or a string representing an integer',
  ],
  hints: [
    'Use a stack to keep track of valid scores',
    'Process each operation according to the rules',
    'Return the sum of all elements in the stack',
  ],
  starterCode: `from typing import List

def cal_points(operations: List[str]) -> int:
    """
    Calculate final score after all operations.
    
    Args:
        operations: List of score operations
        
    Returns:
        Sum of all valid scores
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['5', '2', 'C', 'D', '+']],
      expected: 30,
    },
    {
      input: [['5', '-2', '4', 'C', 'D', '9', '+', '+']],
      expected: 27,
    },
    {
      input: [['1']],
      expected: 1,
    },
  ],
  solution: `from typing import List

def cal_points(operations: List[str]) -> int:
    """
    Stack to track valid scores.
    Time: O(n), Space: O(n)
    """
    stack = []
    
    for op in operations:
        if op == '+':
            # Sum of last two scores
            stack.append(stack[-1] + stack[-2])
        elif op == 'D':
            # Double the last score
            stack.append(2 * stack[-1])
        elif op == 'C':
            # Remove last score
            stack.pop()
        else:
            # Add integer score
            stack.append(int(op))
    
    return sum(stack)
`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/baseball-game/',
  youtubeUrl: 'https://www.youtube.com/watch?v=Id_tqGdsZQI',
};
