/**
 * Quiz questions for Two-Sum Patterns Family section
 */

export const twosumpatternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain why we use a hash table for Two Sum instead of nested loops. Walk through the optimization and why it works.',
    sampleAnswer:
      'The brute force approach checks all pairs with nested loops: for each element, check every other element to see if they sum to target. This is O(n²) because we check n elements against n-1 others. The hash table optimization works by storing complements: as we iterate once through the array, we calculate complement = target - current, then check if that complement was already seen in O(1) time using a hash table. If yes, we found our pair. If no, we store current for future lookups. This reduces time from O(n²) to O(n) by trading O(n) space for O(1) lookups. The key insight: instead of searching for complement each time (O(n)), we remember all previous elements and look them up instantly (O(1)).',
    keyPoints: [
      'Brute force: nested loops check all pairs O(n²)',
      'Hash table stores seen elements for O(1) lookup',
      'Check if complement exists instead of searching',
      'Single pass through array: O(n) time',
      'Trade O(n) space for O(n²) → O(n) time improvement',
    ],
  },
  {
    id: 'q2',
    question:
      'For 3Sum, why do we sort first? Could we use a hash table like in Two Sum? Explain the trade-offs.',
    sampleAnswer:
      'We sort for 3Sum because it enables two key optimizations: 1) We can use two-pointer technique which gives O(n²) total time (n iterations × n two-pointer search), and 2) Sorting makes duplicate handling easy - we can skip consecutive duplicates to ensure unique triplets without needing a set. Could we use hash table? Yes, but it is harder: we would need nested loops to fix two elements, then hash lookup for the third - still O(n²) time but more complex duplicate handling, and O(n) extra space. The sorted approach is cleaner: O(n log n) sort + O(n²) search, versus O(n²) with hash table but messier code. The sort time does not dominate since O(n²) is larger. The two-pointer technique on sorted array is the standard approach because it is elegant, space-efficient, and handles duplicates naturally.',
    keyPoints: [
      'Sorting enables two-pointer technique: O(n²) total',
      'Easy duplicate handling with sorted array',
      'Hash table possible but more complex',
      'Sort time O(n log n) does not dominate O(n²)',
      'Two-pointer on sorted array is standard, cleanest approach',
    ],
  },
  {
    id: 'q3',
    question:
      'In Two Sum II (sorted array), how do you decide whether to move the left or right pointer? Why does this guarantee we find the answer?',
    sampleAnswer:
      'The decision rule is: if current_sum < target, move left pointer right (increase sum); if current_sum > target, move right pointer left (decrease sum). This works because the array is sorted. Moving left pointer right means we pick a larger number (since sorted), increasing the sum. Moving right pointer left means we pick a smaller number, decreasing the sum. This guarantees finding the answer because: 1) If the answer exists and current sum is too small, increasing left is the only way to potentially reach target (right cannot go further right without skipping). 2) If current sum is too large, decreasing right is needed. 3) The pointers converge, checking all viable pairs exactly once. We cannot miss the answer because for any pair (i, j), we either check it directly or eliminate one of its elements by proving it cannot be part of the solution.',
    keyPoints: [
      'Sorted property: left pointer = smaller values, right = larger',
      'Sum too small → move left right (increase)',
      'Sum too large → move right left (decrease)',
      'Pointers converge, checking all viable pairs once',
      'Cannot miss answer: every pair considered or eliminated',
    ],
  },
];
