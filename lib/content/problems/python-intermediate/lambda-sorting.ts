/**
 * Lambda Functions for Sorting
 * Problem ID: intermediate-lambda-sorting
 * Order: 46
 */

import { Problem } from '../../../types';

export const intermediate_lambda_sortingProblem: Problem = {
  id: 'intermediate-lambda-sorting',
  title: 'Lambda Functions for Sorting',
  difficulty: 'Easy',
  description: `Use lambda functions as sort keys.

**Syntax:**
\`\`\`python
sorted(items, key=lambda x: x[1])
\`\`\`

Lambda is useful for simple inline functions.

This tests:
- Lambda expressions
- Sorting with key
- Anonymous functions`,
  examples: [
    {
      input: 'Sort list of tuples by second element',
      output: 'key=lambda x: x[1]',
    },
  ],
  constraints: ['Use lambda', 'Sort by custom key'],
  hints: [
    'lambda args: expression',
    'Use as sort key',
    'Can access tuple/list elements',
  ],
  starterCode: `def sort_by_length_then_alpha(words):
    """
    Sort words by length, then alphabetically.
    
    Args:
        words: List of strings
        
    Returns:
        Sorted list
        
    Examples:
        >>> sort_by_length_then_alpha(['apple', 'pie', 'banana', 'cat'])
        ['cat', 'pie', 'apple', 'banana']
    """
    pass


# Test
print(sort_by_length_then_alpha(['dog', 'cat', 'bird', 'elephant']))
`,
  testCases: [
    {
      input: [['dog', 'cat', 'bird', 'elephant']],
      expected: ['cat', 'dog', 'bird', 'elephant'],
    },
    {
      input: [['xx', 'a', 'z', 'yy']],
      expected: ['a', 'z', 'xx', 'yy'],
    },
  ],
  solution: `def sort_by_length_then_alpha(words):
    return sorted(words, key=lambda w: (len(w), w))


# Other examples
def sort_by_last_char(words):
    return sorted(words, key=lambda w: w[-1])

def sort_tuples_by_second(tuples):
    return sorted(tuples, key=lambda t: t[1])`,
  timeComplexity: 'O(n log n)',
  spaceComplexity: 'O(n)',
  order: 46,
  topic: 'Python Intermediate',
};
