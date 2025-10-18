/**
 * Pascal
 * Problem ID: pascals-triangle
 * Order: 10
 */

import { Problem } from '../../../types';

export const pascals_triangleProblem: Problem = {
  id: 'pascals-triangle',
  title: "Pascal's Triangle",
  difficulty: 'Easy',
  topic: 'Arrays & Hashing',
  order: 10,
  description: `Given an integer \`numRows\`, return the first numRows of **Pascal's triangle**.

In Pascal's triangle, each number is the sum of the two numbers directly above it.`,
  examples: [
    {
      input: 'numRows = 5',
      output: '[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]',
    },
    {
      input: 'numRows = 1',
      output: '[[1]]',
    },
  ],
  constraints: ['1 <= numRows <= 30'],
  hints: [
    'First and last elements of each row are always 1',
    'Each middle element is the sum of two elements from the previous row',
    'Build row by row',
  ],
  starterCode: `from typing import List

def generate_pascals_triangle(num_rows: int) -> List[List[int]]:
    """
    Generate Pascal's Triangle.
    
    Args:
        num_rows: Number of rows to generate
        
    Returns:
        Pascal's Triangle as a 2D list
    """
    # Write your code here
    pass
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
    {
      input: [3],
      expected: [[1], [1, 1], [1, 2, 1]],
    },
  ],
  solution: `from typing import List

def generate_pascals_triangle(num_rows: int) -> List[List[int]]:
    """
    Build triangle row by row.
    Time: O(numRows^2), Space: O(1) excluding output
    """
    triangle = []
    
    for row_num in range(num_rows):
        # Start with all 1s
        row = [1] * (row_num + 1)
        
        # Fill in middle values
        for j in range(1, row_num):
            row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]
        
        triangle.append(row)
    
    return triangle
`,
  timeComplexity: 'O(numRows^2)',
  spaceComplexity: 'O(1) excluding output',
  leetcodeUrl: 'https://leetcode.com/problems/pascals-triangle/',
  youtubeUrl: 'https://www.youtube.com/watch?v=nPVEaB3AjUM',
};
