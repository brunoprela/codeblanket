/**
 * Nested List Comprehension
 * Problem ID: intermediate-list-comprehension-nested
 * Order: 22
 */

import { Problem } from '../../../types';

export const intermediate_list_comprehension_nestedProblem: Problem = {
  id: 'intermediate-list-comprehension-nested',
  title: 'Nested List Comprehension',
  difficulty: 'Medium',
  description: `Use nested list comprehensions to flatten and transform data.

**Example:**
\`\`\`python
matrix = [[1,2], [3,4]]
flat = [x for row in matrix for x in row]
# Result: [1, 2, 3, 4]
\`\`\`

This tests:
- List comprehension syntax
- Multiple loops in one line
- Conditional filtering`,
  examples: [
    {
      input: 'matrix = [[1,2,3], [4,5,6], [7,8,9]]',
      output: '[1, 2, 3, 4, 5, 6, 7, 8, 9]',
    },
  ],
  constraints: ['Use list comprehension', 'Single line preferred'],
  hints: [
    'Outer loop first, inner loop second',
    'Can add conditions with if',
    '[x for row in matrix for x in row]',
  ],
  starterCode: `def flatten_matrix(matrix):
    """
    Flatten 2D matrix using list comprehension.
    
    Args:
        matrix: 2D list
        
    Returns:
        Flattened list
        
    Examples:
        >>> flatten_matrix([[1,2], [3,4]])
        [1, 2, 3, 4]
    """
    pass


# Test
print(flatten_matrix([[1,2,3], [4,5,6], [7,8,9]]))
`,
  testCases: [
    {
      input: [
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
      ],
      expected: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    {
      input: [
        [
          [1, 2],
          [3, 4],
        ],
      ],
      expected: [1, 2, 3, 4],
    },
  ],
  solution: `def flatten_matrix(matrix):
    return [x for row in matrix for x in row]


# With filtering (only even numbers)
def flatten_matrix_even(matrix):
    return [x for row in matrix for x in row if x % 2 == 0]`,
  timeComplexity: 'O(n * m)',
  spaceComplexity: 'O(n * m)',
  order: 22,
  topic: 'Python Intermediate',
};
