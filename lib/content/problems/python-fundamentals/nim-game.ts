/**
 * Nim Game
 * Problem ID: fundamentals-nim-game
 * Order: 63
 */

import { Problem } from '../../../types';

export const nim_gameProblem: Problem = {
  id: 'fundamentals-nim-game',
  title: 'Nim Game',
  difficulty: 'Easy',
  description: `You and your friend play Nim game with n stones.

Rules:
- Players take turns removing 1, 2, or 3 stones
- Player who removes the last stone wins
- You go first

Can you win if both play optimally?

**Key insight:** You lose if n is divisible by 4!

This tests:
- Game theory
- Mathematical pattern
- Modulo operation`,
  examples: [
    {
      input: 'n = 4',
      output: 'False',
      explanation: 'Any move leaves 1-3 stones for opponent to win',
    },
    {
      input: 'n = 5',
      output: 'True',
      explanation: 'Remove 1 stone, leave 4 for opponent',
    },
  ],
  constraints: ['1 <= n <= 2^31 - 1'],
  hints: [
    'Think about losing positions',
    'n=4 is losing, why?',
    'Simple modulo check',
  ],
  starterCode: `def can_win_nim(n):
    """
    Check if you can win Nim game.
    
    Args:
        n: Number of stones
        
    Returns:
        True if you can guarantee win
        
    Examples:
        >>> can_win_nim(4)
        False
        >>> can_win_nim(5)
        True
    """
    pass


# Test
print(can_win_nim(4))
`,
  testCases: [
    {
      input: [4],
      expected: false,
    },
    {
      input: [5],
      expected: true,
    },
    {
      input: [1],
      expected: true,
    },
  ],
  solution: `def can_win_nim(n):
    return n % 4 != 0`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 63,
  topic: 'Python Fundamentals',
};
