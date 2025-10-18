/**
 * Divisor Game
 * Problem ID: divisor-game
 * Order: 6
 */

import { Problem } from '../../../types';

export const divisor_gameProblem: Problem = {
  id: 'divisor-game',
  title: 'Divisor Game',
  difficulty: 'Easy',
  topic: 'Dynamic Programming',
  description: `Alice and Bob take turns playing a game, with Alice starting first.

Initially, there is a number \`n\` on the chalkboard. On each player's turn, that player makes a move consisting of:

- Choosing any \`x\` with \`0 < x < n\` and \`n % x == 0\`.
- Replacing the number \`n\` on the chalkboard with \`n - x\`.

Also, if a player cannot make a move, they lose the game.

Return \`true\` if and only if Alice wins the game, assuming both players play optimally.`,
  examples: [
    {
      input: 'n = 2',
      output: 'true',
      explanation: 'Alice chooses 1, and Bob receives 1 and loses.',
    },
    {
      input: 'n = 3',
      output: 'false',
      explanation:
        'Alice chooses 1, Bob chooses 1, and Alice has no more valid moves.',
    },
  ],
  constraints: ['1 <= n <= 1000'],
  hints: ['Try a few examples and look for pattern', 'Alice wins if n is even'],
  starterCode: `def divisor_game(n: int) -> bool:
    """
    Check if Alice wins the game.
    
    Args:
        n: Starting number
        
    Returns:
        True if Alice wins
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [2],
      expected: true,
    },
    {
      input: [3],
      expected: false,
    },
    {
      input: [1000],
      expected: true,
    },
  ],
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/divisor-game/',
  youtubeUrl: 'https://www.youtube.com/watch?v=0cP_1Z4uDxo',
};
