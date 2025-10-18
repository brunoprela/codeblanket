/**
 * Quiz questions for When NOT to Use Two Pointers section
 */

export const whennottouseQuiz = [
  {
    id: 'q1',
    question:
      'Why does two pointers fail for the classic "Two Sum" problem (return indices)? What should you use instead?',
    hint: 'Think about what happens to indices when you sort.',
    sampleAnswer:
      'Two pointers typically requires sorting the array, but Two Sum asks for the ORIGINAL indices of the two numbers. Once you sort, the indices change and you lose track of original positions. Example: nums = [3,2,4], target = 6. Original indices of 2 and 4 are [1,2]. After sorting to [2,3,4], two pointers would return sorted indices [0,2], which is wrong. Solution: Use hash map. As you iterate, store value->original_index. When you find target-current, lookup gives original index. This is O(n) time, O(n) space, and preserves indices.',
    keyPoints: [
      'Sorting destroys original index information',
      'Two Sum requires original indices, not sorted positions',
      'Hash map preserves value-to-index mapping',
      'Trade space O(n) for ability to keep indices',
      'Still O(n) time like two pointers, but maintains info',
    ],
  },
  {
    id: 'q2',
    question:
      'When would you NOT use sliding window (two-pointer variant) and what would you use instead? Give a specific example.',
    hint: 'Think about problems with multiple windows or non-contiguous elements.',
    sampleAnswer:
      'Do NOT use sliding window when you need multiple non-overlapping subarrays or when elements do not need to be contiguous. Example: "Maximum sum of two non-overlapping subarrays of length L and M." Sliding window tracks ONE contiguous window, but this needs TWO separate windows. If you try, you cannot simultaneously track both windows positions. Solution: Use dynamic programming or prefix sums. Track maximum sum of L-length subarray ending before current position, then for each M-length window, add the best previous L-length sum. This requires DP state tracking, not just two pointers.',
    keyPoints: [
      'Sliding window = one contiguous subarray',
      'Cannot track multiple independent windows',
      'Example: max sum of two non-overlapping subarrays',
      'Use DP to track best previous subarray',
      'Prefix sums help compute window sums efficiently',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain why two pointers does not work for permutation problems. What technique should you use?',
    hint: 'Think about the nature of exploration and the ability to undo choices.',
    sampleAnswer:
      'Two pointers moves linearly forward through the array, possibly backward, but always in a single pass. Permutations require exploring ALL possible orderings, which means making a choice, exploring that path, then UNDOING the choice and trying alternatives. Example: permutations of [1,2,3] include [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]. Two pointers cannot systematically generate these by just moving pointers - you need to try placing each element in each position. Solution: Use backtracking with recursion. At each step, try placing each remaining element, recurse, then remove it (backtrack) to try the next option.',
    keyPoints: [
      'Two pointers is linear, one-pass technique',
      'Permutations need exploration of decision tree',
      'Must undo choices and try alternatives',
      'Backtracking: choose, explore, unchoose',
      'Two pointers cannot backtrack through choices',
    ],
  },
];
