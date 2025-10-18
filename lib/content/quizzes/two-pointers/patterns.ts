/**
 * Quiz questions for The Three Main Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the three main two-pointer patterns. For each one, describe when you would use it and give me an example problem.',
    hint: 'Think about opposite direction, same direction, and sliding window.',
    sampleAnswer:
      'The first pattern is opposite direction where pointers start at both ends and move toward each other. I use this for sorted array problems like two sum - if the sum is too high, move the right pointer left, if too low, move left pointer right. Second is same direction where both pointers start at the beginning but move at different speeds. This is perfect for removing duplicates - slow pointer marks where to write next unique element, fast pointer scans ahead to find it. Third is sliding window where two pointers define a window that expands and shrinks. Use this for subarray problems like finding maximum sum of size k - expand by moving right, shrink by moving left, maintain the window constraint as you go.',
    keyPoints: [
      'Opposite direction: start at ends, meet in middle (two sum)',
      'Same direction: both start at beginning, different speeds (remove duplicates)',
      'Sliding window: define expanding/shrinking window (max sum subarray)',
      'Choose pattern based on problem structure',
    ],
  },
  {
    id: 'q2',
    question:
      'For the opposite direction pattern, explain the decision-making process. How do you decide which pointer to move?',
    sampleAnswer:
      'In opposite direction, the decision is based on comparing your current result with what you want. For two sum, if the current sum is too large, I know I need a smaller number, so I move the right pointer left to get smaller values. If the sum is too small, I need a larger number, so I move the left pointer right. The key insight is that moving the wrong pointer makes things worse - if sum is already too big and I move left pointer right, I am adding an even larger number. The sorted property guarantees that moving a pointer in one direction consistently changes the result in a predictable way.',
    keyPoints: [
      'Compare current result with target',
      'Too high? Move right pointer left for smaller values',
      'Too low? Move left pointer right for larger values',
      'Sorted property makes decisions reliable',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how the fast and slow pointer approach works for removing duplicates. Why do we need two pointers instead of one?',
    sampleAnswer:
      'With fast and slow pointers for removing duplicates, the slow pointer marks the position where the next unique element should go, while the fast pointer searches for that next unique element. We need two because we are modifying the array in place - we cannot use one pointer to both read ahead and mark where to write. Fast pointer scans through the array looking for elements different from what slow is pointing to. When fast finds something new, we copy it to slow plus one position and move slow forward. This way we build up the deduplicated portion at the start of the array while fast explores the rest.',
    keyPoints: [
      'Slow marks write position for next unique',
      'Fast scans ahead to find different element',
      'Need two for in-place modification',
      'Build deduplicated section at array start',
    ],
  },
];
