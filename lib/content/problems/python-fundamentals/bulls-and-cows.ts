/**
 * Bulls and Cows
 * Problem ID: fundamentals-bulls-and-cows
 * Order: 64
 */

import { Problem } from '../../../types';

export const bulls_and_cowsProblem: Problem = {
  id: 'fundamentals-bulls-and-cows',
  title: 'Bulls and Cows',
  difficulty: 'Medium',
  description: `Calculate hint for Bulls and Cows game.

- Bull: digit in correct position
- Cow: digit exists but wrong position

**Example:** secret="1807", guess="7810"
→ "1A3B" (1 bull: 8, 3 cows: 1,7,0)

This tests:
- Character counting
- Position matching
- String formatting`,
  examples: [
    {
      input: 'secret = "1807", guess = "7810"',
      output: '"1A3B"',
    },
    {
      input: 'secret = "1123", guess = "0111"',
      output: '"1A1B"',
    },
  ],
  constraints: ['1 <= len(secret) <= 1000', 'Only digits'],
  hints: [
    'Count bulls first (exact matches)',
    'Count remaining digit frequencies',
    'Cows = min(secret_count, guess_count)',
  ],
  starterCode: `def get_hint(secret, guess):
    """
    Get Bulls and Cows hint.
    
    Args:
        secret: Secret number string
        guess: Guessed number string
        
    Returns:
        Hint in format "xAyB"
        
    Examples:
        >>> get_hint("1807", "7810")
        "1A3B"
    """
    pass


# Test
print(get_hint("1807", "7810"))
`,
  testCases: [
    {
      input: ['1807', '7810'],
      expected: '1A3B',
    },
    {
      input: ['1123', '0111'],
      expected: '1A1B',
    },
  ],
  solution: `def get_hint(secret, guess):
    bulls = 0
    secret_counts = [0] * 10
    guess_counts = [0] * 10
    
    # Count bulls and non-bull digits
    for i in range(len(secret)):
        if secret[i] == guess[i]:
            bulls += 1
        else:
            secret_counts[int(secret[i])] += 1
            guess_counts[int(guess[i])] += 1
    
    # Count cows
    cows = 0
    for i in range(10):
        cows += min(secret_counts[i], guess_counts[i])
    
    return f"{bulls}A{cows}B"`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 64,
  topic: 'Python Fundamentals',
};
