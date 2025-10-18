/**
 * Quiz questions for Common Pitfalls section
 */

export const commonpitfallsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the off-by-one error with window size calculation. Why is it right - left + 1 and not right - left?',
    sampleAnswer:
      'Window size is right - left + 1 because indices are inclusive. If left = 2 and right = 5, the window contains indices 2, 3, 4, 5 which is 4 elements. If I just do right - left, I get 5 - 2 = 3, missing one element. The +1 accounts for the element at the left index itself. Another way to think: if left equals right, window has one element, so size should be 1, not 0. Common mistake is writing right - left and wondering why all answers are off by one. This is similar to calculating array length from indices: end - start + 1 when both ends inclusive.',
    keyPoints: [
      'Indices are inclusive: need +1',
      'Example: left=2, right=5 → 4 elements',
      'right - left gives 3, wrong',
      'right - left + 1 gives 4, correct',
      'When left equals right: size is 1, not 0',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the difference between checking conditions before vs after processing right pointer? Why does order matter?',
    sampleAnswer:
      'Processing right pointer means adding arr[right] to window state. I should do this before checking conditions because I want to check the window that includes the new element. If I check conditions first, I am testing the old window without the new element, which is wrong. For example, in longest substring without repeating, if I check for duplicates before adding current character, I miss detecting if current character itself is the duplicate. The correct sequence: add arr[right] to window, update state, check if conditions violated, shrink left if needed. This ensures each element is properly included in the window before evaluation.',
    keyPoints: [
      'Process right: add to window first',
      'Then check conditions on updated window',
      'Checking before: tests old window',
      'Wrong order: miss detecting violations',
      'Correct: add, update, check, shrink',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the logic for maximum vs minimum window problems. Where do you update the answer and why?',
    sampleAnswer:
      'For maximum window problems, I shrink while window is invalid, then update answer outside the while loop with right - left + 1. This captures the maximum valid window at each position. For minimum window problems, I shrink while window is still valid, updating answer inside the while loop before each shrink. This captures progressively smaller valid windows. The key difference: maximum wants largest valid so update after restoring validity. Minimum wants smallest valid so update while shrinking valid window. If I use wrong placement, I either miss the optimal or include invalid windows. The while condition and update placement must match the optimization goal.',
    keyPoints: [
      'Maximum: shrink while invalid, update outside while',
      'Minimum: shrink while valid, update inside while',
      'Maximum: largest valid window',
      'Minimum: smallest valid window',
      'Condition and update placement must match goal',
    ],
  },
];
