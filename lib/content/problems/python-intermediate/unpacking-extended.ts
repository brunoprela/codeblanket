/**
 * Extended Unpacking with *
 * Problem ID: intermediate-unpacking-extended
 * Order: 48
 */

import { Problem } from '../../../types';

export const intermediate_unpacking_extendedProblem: Problem = {
  id: 'intermediate-unpacking-extended',
  title: 'Extended Unpacking with *',
  difficulty: 'Medium',
  description: `Use * for extended unpacking in assignments.

**Syntax:**
\`\`\`python
first, *middle, last = [1, 2, 3, 4, 5]
# first = 1, middle = [2, 3, 4], last = 5
\`\`\`

This tests:
- Extended unpacking
- * operator
- Variable assignment`,
  examples: [
    {
      input: 'first, *rest = [1, 2, 3, 4]',
      output: 'first = 1, rest = [2, 3, 4]',
    },
  ],
  constraints: ['Use * unpacking', 'Multiple variables'],
  hints: [
    'a, *b, c = list',
    '* captures multiple items',
    'Can be in any position',
  ],
  starterCode: `def split_first_last_middle(items):
    """
    Split list into first, middle, and last.
    
    Args:
        items: List with at least 2 items
        
    Returns:
        Tuple of (first, middle, last)
        
    Examples:
        >>> split_first_last_middle([1, 2, 3, 4, 5])
        (1, [2, 3, 4], 5)
    """
    first, *middle, last = items
    return (first, middle, last)


# Test
print(split_first_last_middle([1, 2, 3, 4, 5, 6]))
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5, 6]],
      expected: [1, [2, 3, 4, 5], 6],
    },
    {
      input: [[10, 20, 30]],
      expected: [10, [20], 30],
    },
  ],
  solution: `def split_first_last_middle(items):
    first, *middle, last = items
    return (first, middle, last)


# Other examples
def get_first_and_rest(items):
    first, *rest = items
    return (first, rest)

def get_all_but_last(items):
    *all_but_last, last = items
    return all_but_last`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(n)',
  order: 48,
  topic: 'Python Intermediate',
};
