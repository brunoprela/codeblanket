/**
 * Efficient String Joining
 * Problem ID: intermediate-string-join
 * Order: 43
 */

import { Problem } from '../../../types';

export const intermediate_string_joinProblem: Problem = {
  id: 'intermediate-string-join',
  title: 'Efficient String Joining',
  difficulty: 'Easy',
  description: `Use str.join() for efficient string concatenation.

**Why join():**
- Strings are immutable
- += creates new string each time (O(n²))
- join() is O(n)

This tests:
- String operations
- Performance awareness
- join() method`,
  examples: [
    {
      input: 'words = ["Hello", "World"]',
      output: '" ".join(words) = "Hello World"',
    },
  ],
  constraints: ['Use join()', 'More efficient than +='],
  hints: ['separator.join(list)', 'Works with any iterable', 'Faster than +='],
  starterCode: `def build_sentence(words):
    """
    Join words into sentence.
    
    Args:
        words: List of words
        
    Returns:
        Sentence string
        
    Examples:
        >>> build_sentence(['Hello', 'world', 'today'])
        'Hello world today'
    """
    pass


# Test
print(build_sentence(['Python', 'is', 'awesome']))
`,
  testCases: [
    {
      input: [['Python', 'is', 'awesome']],
      expected: 'Python is awesome',
    },
    {
      input: [['a', 'b', 'c']],
      expected: 'a b c',
    },
  ],
  solution: `def build_sentence(words):
    return ' '.join(words)


# Different separators
def build_csv(values):
    return ','.join(str(v) for v in values)

def build_path(parts):
    return '/'.join(parts)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 43,
  topic: 'Python Intermediate',
};
