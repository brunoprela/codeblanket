/**
 * Keyboard Row Words
 * Problem ID: fundamentals-keyboard-row
 * Order: 91
 */

import { Problem } from '../../../types';

export const keyboard_rowProblem: Problem = {
  id: 'fundamentals-keyboard-row',
  title: 'Keyboard Row Words',
  difficulty: 'Easy',
  description: `Find words that can be typed using only one row of keyboard.

Keyboard rows:
- Row 1: "qwertyuiop"
- Row 2: "asdfghjkl"
- Row 3: "zxcvbnm"

**Example:** ["Hello", "Alaska", "Dad"] → ["Alaska", "Dad"]

This tests:
- Set operations
- String filtering
- Case handling`,
  examples: [
    {
      input: 'words = ["Hello","Alaska","Dad","Peace"]',
      output: '["Alaska","Dad"]',
    },
  ],
  constraints: ['1 <= len(words) <= 20', '1 <= len(words[i]) <= 100'],
  hints: [
    'Create sets for each row',
    'Check if all letters in one set',
    'Handle case-insensitive',
  ],
  starterCode: `def find_words(words):
    """
    Find words from single keyboard row.
    
    Args:
        words: List of words
        
    Returns:
        List of valid words
        
    Examples:
        >>> find_words(["Hello","Alaska","Dad"])
        ["Alaska", "Dad"]
    """
    pass


# Test
print(find_words(["Hello","Alaska","Dad","Peace"]))
`,
  testCases: [
    {
      input: [['Hello', 'Alaska', 'Dad', 'Peace']],
      expected: ['Alaska', 'Dad'],
    },
  ],
  solution: `def find_words(words):
    rows = [
        set('qwertyuiop'),
        set('asdfghjkl'),
        set('zxcvbnm')
    ]
    
    result = []
    
    for word in words:
        word_lower = word.lower()
        for row in rows:
            if all(char in row for char in word_lower):
                result.append(word)
                break
    
    return result`,
  timeComplexity: 'O(n * m) where m is avg word length',
  spaceComplexity: 'O(1)',
  order: 91,
  topic: 'Python Fundamentals',
};
