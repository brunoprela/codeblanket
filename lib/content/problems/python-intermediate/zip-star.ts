/**
 * Transpose Matrix with zip(*matrix)
 * Problem ID: intermediate-zip-star
 * Order: 23
 */

import { Problem } from '../../../types';

export const intermediate_zip_starProblem: Problem = {
  id: 'intermediate-zip-star',
  title: 'Transpose Matrix with zip(*matrix)',
  difficulty: 'Easy',
  description: `Use zip(*matrix) to transpose a matrix.

The * operator unpacks the matrix rows as arguments to zip.

**Example:**
\`\`\`python
matrix = [[1,2], [3,4], [5,6]]
transposed = list(zip(*matrix))
# Result: [(1,3,5), (2,4,6)]
\`\`\`

This tests:
- zip() function
- * unpacking operator
- Matrix operations`,
  examples: [
    {
      input: 'matrix = [[1,2,3], [4,5,6]]',
      output: '[(1,4), (2,5), (3,6)]',
    },
  ],
  constraints: ['Use zip()', 'Use * operator'],
  hints: [
    'zip(*matrix) unpacks rows',
    'Returns tuples',
    'Convert to list if needed',
  ],
  starterCode: `def transpose(matrix):
    """
    Transpose matrix using zip.
    
    Args:
        matrix: 2D list
        
    Returns:
        Transposed matrix as list of tuples
        
    Examples:
        >>> transpose([[1,2], [3,4]])
        [(1, 3), (2, 4)]
    """
    pass


# Test
print(transpose([[1,2,3], [4,5,6]]))
`,
  testCases: [
    {
      input: [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ],
      expected: [
        [1, 4],
        [2, 5],
        [3, 6],
      ],
    },
    {
      input: [
        [
          [1, 2],
          [3, 4],
        ],
      ],
      expected: [
        [1, 3],
        [2, 4],
      ],
    },
  ],
  solution: `def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


# Alternative
def transpose_alt(matrix):
    return list(map(list, zip(*matrix)))`,
  timeComplexity: 'O(n * m)',
  spaceComplexity: 'O(n * m)',
  order: 23,
  topic: 'Python Intermediate',
};
