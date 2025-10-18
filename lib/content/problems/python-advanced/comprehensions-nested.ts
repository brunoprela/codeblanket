/**
 * Advanced List/Dict/Set Comprehensions
 * Problem ID: advanced-comprehensions-nested
 * Order: 32
 */

import { Problem } from '../../../types';

export const comprehensions_nestedProblem: Problem = {
  id: 'advanced-comprehensions-nested',
  title: 'Advanced List/Dict/Set Comprehensions',
  difficulty: 'Medium',
  description: `Master complex comprehensions with nesting, conditionals, and multiple iterations.

Create comprehensions for:
- Nested list flattening
- Matrix operations
- Dictionary inversions
- Set operations with filtering

**Pattern:** Comprehensions are more Pythonic than loops for data transformations.`,
  examples: [
    {
      input: 'flatten([[1,2], [3,4]])',
      output: '[1,2,3,4]',
    },
  ],
  constraints: [
    'Use comprehensions (not loops)',
    'Handle nested structures',
    'Combine with conditionals',
  ],
  hints: [
    'Nested comprehensions: [x for list in lists for x in list]',
    'Conditional: [x for x in items if condition]',
    'Dict comprehension: {k: v for k, v in items}',
  ],
  starterCode: `def flatten(nested_list):
    """Flatten nested list using comprehension.
    
    Args:
        nested_list: List of lists
        
    Returns:
        Flattened list
    """
    pass


def transpose_matrix(matrix):
    """Transpose matrix using comprehension.
    
    Args:
        matrix: 2D list
        
    Returns:
        Transposed matrix
    """
    pass


def invert_dict(d):
    """Invert dictionary (swap keys and values).
    
    Args:
        d: Dictionary with unique values
        
    Returns:
        Inverted dictionary
    """
    pass


def word_lengths(text):
    """Create dict of word -> length for words > 3 chars.
    
    Args:
        text: String of words
        
    Returns:
        Dict of word -> length
    """
    pass


# Test
print(flatten([[1,2], [3,4], [5]]))
print(transpose_matrix([[1,2,3], [4,5,6]]))
print(invert_dict({'a': 1, 'b': 2, 'c': 3}))
print(word_lengths("the quick brown fox"))
`,
  testCases: [
    {
      input: [[[1, 2], [3, 4], [5]]],
      expected: [1, 2, 3, 4, 5],
    },
  ],
  solution: `def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def transpose_matrix(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def invert_dict(d):
    return {v: k for k, v in d.items()}


def word_lengths(text):
    return {word: len(word) for word in text.split() if len(word) > 3}`,
  timeComplexity: 'O(n) where n is total elements',
  spaceComplexity: 'O(n)',
  order: 32,
  topic: 'Python Advanced',
};
