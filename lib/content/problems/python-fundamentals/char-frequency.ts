/**
 * Character Frequency Map
 * Problem ID: fundamentals-char-frequency
 * Order: 31
 */

import { Problem } from '../../../types';

export const char_frequencyProblem: Problem = {
  id: 'fundamentals-char-frequency',
  title: 'Character Frequency Map',
  difficulty: 'Easy',
  description: `Create a dictionary that maps each character in a string to its frequency.

- Count all characters (including spaces and punctuation)
- Return a dictionary with character counts
- Case-sensitive counting

**Example:** "hello" → {'h': 1, 'e': 1, 'l': 2, 'o': 1}

This tests:
- Dictionary creation
- String iteration
- Frequency counting`,
  examples: [
    {
      input: 's = "hello"',
      output: "{'h': 1, 'e': 1, 'l': 2, 'o': 1}",
    },
    {
      input: 's = "aaa"',
      output: "{'a': 3}",
    },
  ],
  constraints: ['0 <= len(s) <= 10^4', 'ASCII characters only'],
  hints: [
    'Use dictionary to store counts',
    'Iterate through each character',
    'Increment count for each occurrence',
  ],
  starterCode: `def char_frequency(s):
    """
    Create frequency map of characters.
    
    Args:
        s: Input string
        
    Returns:
        Dictionary mapping characters to their counts
        
    Examples:
        >>> char_frequency("hello")
        {'h': 1, 'e': 1, 'l': 2, 'o': 1}
    """
    pass


# Test
print(char_frequency("hello world"))
`,
  testCases: [
    {
      input: ['hello'],
      expected: { h: 1, e: 1, l: 2, o: 1 },
    },
    {
      input: ['aaa'],
      expected: { a: 3 },
    },
  ],
  solution: `def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq


# Alternative using Counter
from collections import Counter

def char_frequency_counter(s):
    return dict(Counter(s))`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(k) where k is unique characters',
  order: 31,
  topic: 'Python Fundamentals',
};
