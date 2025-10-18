/**
 * Quiz questions for Sliding Window Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Describe the shrinkable window pattern. When do you expand and when do you shrink?',
    sampleAnswer:
      'In the shrinkable window pattern, I expand by moving right pointer to include more elements, making the window larger. I shrink by moving left pointer when the window violates some condition. For longest substring without repeating characters, I expand right to add characters, but when I hit a duplicate, I shrink left until removing the previous occurrence of that character. The key pattern: expand right in the outer loop unconditionally, shrink left in an inner while loop when condition breaks. This maintains the invariant that the window always satisfies our constraints. The answer is typically the maximum window size seen during expansion.',
    keyPoints: [
      'Expand right: add elements unconditionally',
      'Shrink left: while condition violated',
      'Outer loop: right pointer',
      'Inner while: left pointer',
      'Answer: maximum window size',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the non-shrinkable window pattern and how it differs from shrinkable. When would you use each?',
    sampleAnswer:
      'Non-shrinkable window maintains maximum window size once achieved and never shrinks below it. Instead of a while loop to shrink, I use an if statement to move left just once. This keeps window size at least as large as the best seen so far. For example, in longest substring with at most k distinct characters, once I have found a window of size 5, I never let it shrink below 5 - I just slide it forward looking for larger valid windows. Use non-shrinkable when you want maximum window satisfying constraints and do not need to track all valid windows. Use shrinkable when you need minimum window or need to process all valid windows.',
    keyPoints: [
      'Maintains maximum window size achieved',
      'If statement instead of while to move left',
      'Never shrinks below best size',
      'Slides forward at fixed size',
      'Use for maximum window problems',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk through the minimum window substring problem. How does the sliding window with character counting work?',
    sampleAnswer:
      'For minimum window containing all characters of a target string, I use two hash maps: one for target character counts and one for current window counts. I expand right, adding characters to window count. When window contains all target characters (use a counter to check), I shrink left to minimize window while maintaining validity. For each valid window, I track the minimum. The trick is efficiently checking validity - I maintain a "formed" counter that tracks how many unique characters have reached their required count. When formed equals number of unique chars in target, window is valid. This avoids checking all characters each time.',
    keyPoints: [
      'Two hash maps: target counts and window counts',
      'Expand right: add to window',
      'Shrink left: while window valid',
      '"formed" counter for efficient validity check',
      'Track minimum valid window',
    ],
  },
];
