/**
 * reversed() Function
 * Problem ID: intermediate-reversed-builtin
 * Order: 49
 */

import { Problem } from '../../../types';

export const intermediate_reversed_builtinProblem: Problem = {
  id: 'intermediate-reversed-builtin',
  title: 'reversed() Function',
  difficulty: 'Easy',
  description: `Use reversed() to iterate in reverse without copying.

reversed() returns an iterator, not a list.

**Benefits:**
- Memory efficient
- Works with any sequence
- No copying

This tests:
- reversed() function
- Iterator usage
- Reverse iteration`,
  examples: [
    {
      input: 'reversed([1, 2, 3])',
      output: 'Iterator yielding 3, 2, 1',
    },
  ],
  constraints: ['Use reversed()', 'Iterator, not list'],
  hints: [
    'reversed(sequence)',
    'Returns iterator',
    'Convert to list if needed',
  ],
  starterCode: `def reverse_string_words(sentence):
    """
    Reverse order of words in sentence.
    
    Args:
        sentence: String with words
        
    Returns:
        String with reversed word order
        
    Examples:
        >>> reverse_string_words("Hello world Python")
        "Python world Hello"
    """
    words = sentence.split()
    reversed_words = reversed(words)
    return ' '.join(reversed_words)


# Test
print(reverse_string_words("I love Python programming"))
`,
  testCases: [
    {
      input: ['I love Python programming'],
      expected: 'programming Python love I',
    },
    {
      input: ['Hello world'],
      expected: 'world Hello',
    },
  ],
  solution: `def reverse_string_words(sentence):
    words = sentence.split()
    return ' '.join(reversed(words))


# Alternative (more concise)
def reverse_string_words_alt(sentence):
    return ' '.join(sentence.split()[::-1])`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 49,
  topic: 'Python Intermediate',
};
