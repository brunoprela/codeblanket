/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedQuiz = [
  {
    id: 'fenwick-advanced-1',
    question: 'How do you extend Fenwick Tree to handle 2D range queries?',
    hint: 'Think about nesting the update and query operations.',
    sampleAnswer:
      'A 2D Fenwick Tree uses nested loops. For update (r, c, delta), the outer loop iterates over rows using "r += r & -r", and for each row, the inner loop iterates over columns using "c += c & -c". Query works similarly with subtraction. This gives O(log M × log N) complexity for an M×N matrix.',
    keyPoints: [
      'Nest two Fenwick Trees: one for rows, one for columns',
      'Update/Query: O(log M × log N)',
      'Useful for 2D prefix sum problems',
    ],
  },
  {
    id: 'fenwick-advanced-2',
    question:
      'Explain how Fenwick Tree can be used to count inversions in an array.',
    hint: 'Think about processing elements in reverse and tracking what you have seen.',
    sampleAnswer:
      "To count inversions, iterate the array from right to left. Use coordinate compression to map values to ranks. For each element, query the Fenwick Tree for how many smaller elements you have already seen (prefix_sum of rank-1). Then update the tree by adding 1 at this element's rank. The sum of all queries is the inversion count.",
    keyPoints: [
      'Process array right to left',
      'Use coordinate compression for large values',
      'Query counts smaller elements seen so far',
      'Total inversions = sum of all queries',
    ],
  },
  {
    id: 'fenwick-advanced-3',
    question:
      'What is the binary search on Fenwick Tree technique and when is it useful?',
    hint: 'Think about finding the kth element in a frequency array.',
    sampleAnswer:
      'Binary search on Fenwick Tree finds the smallest index with prefix_sum >= k. You start with a large power of 2 and halve it each step, adding it to your position if it does not exceed k. This is useful for finding the kth smallest element in O(log N) when the tree represents cumulative frequencies.',
    keyPoints: [
      'Finds kth element in O(log N)',
      'Requires monotonic (cumulative) data',
      'Uses powers of 2 to navigate',
      'Useful for frequency/rank queries',
    ],
  },
];
