/**
 * Quiz questions for Array Fundamentals & Patterns section
 */

export const arraysQuiz = [
  {
    id: 'q1',
    question:
      'Explain the prefix sum technique and walk through how it helps solve range query problems efficiently.',
    sampleAnswer:
      'Prefix sum is where I precompute cumulative sums at each position - prefix[i] equals the sum of all elements from 0 to i. This takes O(n) time to build. Then, to find the sum of any range from i to j, I can do prefix[j] minus prefix[i-1] in O(1) time. Without prefix sum, each range query would be O(n) to add up elements. With it, I pay O(n) once to build, then answer unlimited range queries in O(1) each. It is perfect when you have multiple range sum queries on the same array. The insight is that sum (i to j) equals sum(0 to j) minus sum(0 to i-1).',
    keyPoints: [
      'Precompute cumulative sums: O(n) build',
      'Range sum = prefix[j] - prefix[i-1]',
      'Each query: O(1) instead of O(n)',
      'Great for multiple range queries',
      'Trade preprocessing time for query speed',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe Kadane algorithm for maximum subarray. What is the key insight that makes it work in O(n)?',
    sampleAnswer:
      'Kadane algorithm finds the maximum sum subarray in O(n) by making a smart observation: at each position, the maximum subarray ending here is either the current element alone, or the current element plus the maximum subarray ending at the previous position. We track max_current which is the best subarray ending at current position, and max_global which is the best we have seen overall. The key insight is that if max_current becomes negative, we are better off starting fresh from the next element rather than dragging along a negative sum. This way we make one pass and track two values, avoiding checking all possible subarrays which would be O(n²).',
    keyPoints: [
      'Max subarray ending here = current or current + previous max',
      'Track max_current (ending here) and max_global (best overall)',
      'If max_current negative, start fresh',
      'One pass: O(n) instead of O(n²)',
      'Dynamic programming approach',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the sliding window technique for finding maximum sum of subarray of size k.',
    sampleAnswer:
      'For max sum of size k subarray, I first calculate the sum of the first k elements - that is my initial window. Then I slide the window one position at a time: add the new element entering the window and subtract the element leaving the window. Each slide is just two operations, so O(1) per position. I track the maximum sum seen as I slide through. Total is O(n) to slide through the array. The key is avoiding recalculating the entire window sum each time - I incrementally update it by removing the left element and adding the right element. This turns what would be O(n×k) brute force into O(n).',
    keyPoints: [
      'Calculate first window sum',
      'Slide: add right element, subtract left element',
      'Each slide: O(1), total O(n)',
      'Track maximum as we go',
      'Incremental update vs recalculation',
    ],
  },
];
