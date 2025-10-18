/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain why sliding window is O(n) even though there is a nested loop. How does amortized analysis apply here?',
    sampleAnswer:
      'Sliding window looks like O(n²) with nested loops - outer loop moves right pointer n times, inner loop moves left pointer. But amortized analysis shows it is O(n). The key insight: left pointer moves at most n times total across the entire algorithm, not n times per right pointer iteration. Each element enters the window once (right pointer) and leaves at most once (left pointer), giving 2n operations total. So outer loop contributes O(n), inner loop across all iterations contributes O(n), total is O(2n) = O(n). The inner loop iterations are amortized - distributed across all outer iterations, not concentrated.',
    keyPoints: [
      'Looks like nested loops: O(n²)?',
      'Left pointer moves n times TOTAL',
      'Each element enters once, leaves once',
      '2n operations total: O(n)',
      'Amortized: inner loop cost distributed',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the space complexity of different sliding window problems. When do you need extra space and when can you solve in O(1)?',
    sampleAnswer:
      'Space complexity depends on what you track in the window. For simple sum or count, O(1) space - just variables for left, right, sum. For character or element frequency, O(k) space where k is alphabet size or unique elements - need hash map to count. For subarray problems tracking indices or elements, potentially O(n) if storing all elements in window. Fixed-size numeric windows can be O(1). Variable-size windows with character constraints need O(k) for hash map. The question to ask: what information must I maintain about window contents? Minimal tracking enables O(1), frequency tracking needs O(k).',
    keyPoints: [
      'Simple sum/count: O(1) space',
      'Character frequency: O(k) for hash map',
      'Tracking elements: O(n) worst case',
      'Fixed numeric windows: O(1)',
      'Depends on what you track in window',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is it critical to process the right pointer before checking conditions? Walk me through what happens if you do it wrong.',
    sampleAnswer:
      'Processing right pointer first means adding the element to the window before checking if conditions break. This is correct because we want to include the element, then determine if we need to shrink. If I check conditions before processing right, I am checking the window without the new element, which is the previous window state. For example, in longest substring without repeating, if I check for duplicate before adding current character, I miss that current character might be the duplicate. Then I add it anyway, leaving duplicates in window. Correct order: add to window, update state, check conditions, shrink if needed. This maintains the invariant that we process each position exactly once.',
    keyPoints: [
      'Add element first, then check conditions',
      'Checking before adding tests previous window',
      'Wrong order: might miss violations',
      'Correct: add, update, check, shrink',
      'Maintains invariant: each position processed once',
    ],
  },
];
