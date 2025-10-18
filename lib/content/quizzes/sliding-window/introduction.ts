/**
 * Quiz questions for Introduction to Sliding Window section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain the core concept of the sliding window technique. Why is it called a "window" and what makes it "slide"?',
    sampleAnswer:
      'The sliding window technique uses two pointers to define a contiguous subarray or substring - that is the "window". We start with both pointers at the beginning, then expand the window by moving the right pointer to include more elements. When certain conditions are met, we shrink the window by moving the left pointer. This creates a sliding motion - the window slides across the array or string. It is powerful because instead of checking all possible subarrays O(n²), we maintain one window and adjust it as we go in O(n) time. The key is that we process each element at most twice - once when entering the window, once when leaving.',
    keyPoints: [
      'Two pointers define a contiguous subarray/substring',
      'Expand: move right pointer',
      'Shrink: move left pointer',
      'Sliding motion across data',
      'O(n) vs O(n²) for all subarrays',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare fixed-size vs variable-size windows. When would you use each approach?',
    sampleAnswer:
      'Fixed-size windows have a predetermined size k - like max sum of k consecutive elements. I expand until reaching size k, then slide by adding right and removing left simultaneously. This is straightforward and great when the problem specifies exact window size. Variable-size windows adjust based on conditions - like longest substring without repeating characters. I expand right to include elements while condition holds, then shrink left when condition breaks. This is more complex but handles optimization problems where we seek maximum or minimum window satisfying constraints. Fixed-size: problem gives exact size. Variable-size: we optimize to find best size.',
    keyPoints: [
      'Fixed: predetermined size k',
      'Fixed: slide by add right, remove left',
      'Variable: adjust based on conditions',
      'Variable: expand and shrink as needed',
      'Fixed: exact size given, Variable: optimize size',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through a simple example of sliding window solving a problem in O(n) that would be O(n²) with brute force.',
    sampleAnswer:
      'Take max sum of k=3 consecutive elements in [2, 1, 5, 1, 3, 2]. Brute force computes sum of every 3-element window: [2,1,5]=8, [1,5,1]=7, [5,1,3]=9, [1,3,2]=6. That is 4 windows each taking 3 adds, total 12 operations. Sliding window: compute first window [2,1,5]=8. Then slide: remove 2, add 1 → [1,5,1]=7. Remove 1, add 3 → [5,1,3]=9. Remove 5, add 2 → [1,3,2]=6. Each slide is 2 operations, total 3+6=9 operations. For large n, brute force is O(n×k) vs sliding window O(n). We reuse previous sum instead of recalculating from scratch each time.',
    keyPoints: [
      'Brute force: recalculate each window',
      'Sliding window: reuse previous calculation',
      'Remove leaving element, add entering element',
      'Brute force: O(n×k), Sliding window: O(n)',
      'Efficiency from incremental updates',
    ],
  },
];
