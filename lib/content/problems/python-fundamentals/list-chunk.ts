/**
 * Chunk List into Groups
 * Problem ID: fundamentals-list-chunk
 * Order: 32
 */

import { Problem } from '../../../types';

export const list_chunkProblem: Problem = {
  id: 'fundamentals-list-chunk',
  title: 'Chunk List into Groups',
  difficulty: 'Easy',
  description: `Split a list into chunks of specified size.

- Create sublists of size n
- Last chunk may be smaller if elements don't divide evenly
- Maintain original order

**Example:** [1,2,3,4,5], size=2 → [[1,2], [3,4], [5]]

This tests:
- List slicing
- Loop iteration
- List comprehension`,
  examples: [
    {
      input: 'arr = [1,2,3,4,5], size = 2',
      output: '[[1,2], [3,4], [5]]',
    },
    {
      input: 'arr = [1,2,3,4,5,6], size = 3',
      output: '[[1,2,3], [4,5,6]]',
    },
  ],
  constraints: ['1 <= len(arr) <= 1000', '1 <= size <= len(arr)'],
  hints: [
    'Use list slicing with step',
    'Iterate with range and step size',
    'arr[i:i+size] gets each chunk',
  ],
  starterCode: `def chunk_list(arr, size):
    """
    Split list into chunks.
    
    Args:
        arr: List to chunk
        size: Size of each chunk
        
    Returns:
        List of chunks
        
    Examples:
        >>> chunk_list([1,2,3,4,5], 2)
        [[1, 2], [3, 4], [5]]
    """
    pass


# Test
print(chunk_list([1,2,3,4,5,6,7], 3))
`,
  testCases: [
    {
      input: [[1, 2, 3, 4, 5], 2],
      expected: [[1, 2], [3, 4], [5]],
    },
    {
      input: [[1, 2, 3, 4, 5, 6], 3],
      expected: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    },
  ],
  solution: `def chunk_list(arr, size):
    chunks = []
    for i in range(0, len(arr), size):
        chunks.append(arr[i:i + size])
    return chunks


# Alternative using list comprehension
def chunk_list_compact(arr, size):
    return [arr[i:i + size] for i in range(0, len(arr), size)]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 32,
  topic: 'Python Fundamentals',
};
