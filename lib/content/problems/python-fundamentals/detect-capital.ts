/**
 * Detect Capital Use
 * Problem ID: fundamentals-detect-capital
 * Order: 96
 */

import { Problem } from '../../../types';

export const detect_capitalProblem: Problem = {
  id: 'fundamentals-detect-capital',
  title: 'Detect Capital Use',
  difficulty: 'Easy',
  description: `Check if word uses capitals correctly.

Valid patterns:
1. All capitals: "USA"
2. All lowercase: "leetcode"
3. Only first capital: "Google"

**Example:** "USA" → true, "FlaG" → false

This tests:
- String case checking
- Pattern matching
- Boolean logic`,
  examples: [
    {
      input: 'word = "USA"',
      output: 'True',
    },
    {
      input: 'word = "FlaG"',
      output: 'False',
    },
  ],
  constraints: ['1 <= len(word) <= 100', 'Only letters'],
  hints: [
    'Use isupper() and islower()',
    'Check all caps, all lower, or first cap',
    'Use string methods',
  ],
  starterCode: `def detect_capital_use(word):
    """
    Check if capitals used correctly.
    
    Args:
        word: Input word
        
    Returns:
        True if valid capital usage
        
    Examples:
        >>> detect_capital_use("USA")
        True
    """
    pass


# Test
print(detect_capital_use("FlaG"))
`,
  testCases: [
    {
      input: ['USA'],
      expected: true,
    },
    {
      input: ['FlaG'],
      expected: false,
    },
    {
      input: ['Google'],
      expected: true,
    },
  ],
  solution: `def detect_capital_use(word):
    return (word.isupper() or 
            word.islower() or 
            (word[0].isupper() and word[1:].islower()))


# Alternative explicit check
def detect_capital_use_explicit(word):
    if len(word) == 1:
        return True
    
    # All uppercase
    if all(c.isupper() for c in word):
        return True
    
    # All lowercase
    if all(c.islower() for c in word):
        return True
    
    # First uppercase, rest lowercase
    if word[0].isupper() and all(c.islower() for c in word[1:]):
        return True
    
    return False`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 96,
  topic: 'Python Fundamentals',
};
