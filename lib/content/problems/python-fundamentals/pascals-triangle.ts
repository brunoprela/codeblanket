/**
 * Pascal
 * Problem ID: fundamentals-pascals-triangle
 * Order: 47
 */

import { Problem } from '../../../types';

export const pascals_triangleProblem: Problem = {
  id: 'fundamentals-pascals-triangle',
  title: "Pascal's Triangle",
  difficulty: 'Easy',
  description: `Generate the first n rows of Pascal's Triangle.

Each number is the sum of the two numbers directly above it.

**Pattern:**
\`\`\`
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
\`\`\`

This tests:
- 2D array generation
- Pattern recognition
- List manipulation`,
  examples: [
    {
      input: 'n = 5',
      output: '[[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]',
    },
  ],
  constraints: ['1 <= n <= 30'],
  hints: [
    'Start each row with 1',
    'Each middle element = prev[i-1] + prev[i]',
    'End each row with 1',
  ],
  starterCode: `def generate_pascals_triangle(n):
    """
    Generate first n rows of Pascal's Triangle.
    
    Args:
        n: Number of rows
        
    Returns:
        2D list representing triangle
        
    Examples:
        >>> generate_pascals_triangle(5)
        [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]
    """
    pass


# Test
print(generate_pascals_triangle(5))
`,
  testCases: [
    {
      input: [5],
      expected: [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]],
    },
    {
      input: [1],
      expected: [[1]],
    },
  ],
  solution: `def generate_pascals_triangle(n):
    triangle = []
    
    for i in range(n):
        row = [1] * (i + 1)
        
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        
        triangle.append(row)
    
    return triangle


# Alternative more compact
def generate_pascals_triangle_compact(n):
    result = [[1]]
    
    for i in range(1, n):
        prev = result[-1]
        new_row = [1]
        for j in range(len(prev) - 1):
            new_row.append(prev[j] + prev[j + 1])
        new_row.append(1)
        result.append(new_row)
    
    return result`,
  timeComplexity: 'O(n²)',
  spaceComplexity: 'O(n²)',
  order: 47,
  topic: 'Python Fundamentals',
};
