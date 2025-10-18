/**
 * Quiz questions for Advanced Techniques section
 */

export const advancedQuiz = [
  {
    id: 'q1',
    question:
      'Explain the monotonic deque technique for sliding window maximum. Why use a deque instead of just tracking the max?',
    sampleAnswer:
      'For sliding window maximum, just tracking current max fails when the max element leaves the window - we do not know the next maximum. Monotonic deque solves this by maintaining indices in decreasing order of their values. When a new element enters, we remove elements from the back that are smaller than it (they can never be maximum while the new element is in window). When elements leave the window, we remove from front if their index is outside window. The front always has the maximum for current window. This gives O(n) total time because each element is added and removed from deque at most once. Deque enables both ends operations in O(1).',
    keyPoints: [
      'Need to track next maximum when current leaves',
      'Deque maintains indices in decreasing value order',
      'Remove smaller elements from back',
      'Remove out-of-window indices from front',
      'O(n) total: each element in/out once',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through sliding window with multiple conditions. How do you track vowels and consonants simultaneously?',
    sampleAnswer:
      'For multiple conditions like k vowels and m consonants in window, I track both counts separately using variables or a hash map. As I expand right, I check if the new character is vowel or consonant and update respective count. When shrinking left, I decrement the appropriate count. The window is valid when both conditions are satisfied simultaneously. I check "if vowels == k and consonants == m" to know validity. The key is maintaining independent counters for each condition and checking all conditions together. This extends to any number of simultaneous constraints - just add more counters and check all conditions in your validity check.',
    keyPoints: [
      'Track each condition with separate counter',
      'Update appropriate counter on add/remove',
      'Check all conditions together for validity',
      'Independent counters for each constraint',
      'Extends to any number of conditions',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the pattern for finding all anagrams using sliding window. What makes this more efficient than checking each substring?',
    sampleAnswer:
      'For finding all anagrams of pattern in string, I use fixed-size window of length equal to pattern length. I maintain two frequency maps: one for pattern (built once), one for current window (updated as window slides). I slide through the string, adding the entering character and removing the leaving character from window map. After each slide, I compare window map with pattern map - if equal, found an anagram. This is O(n) because each comparison is O(26) for alphabet size, constant time. Brute force would generate and sort each substring: O(n × m log m) where m is pattern length. Sliding window avoids repeated sorting by maintaining frequency incrementally.',
    keyPoints: [
      'Fixed window size = pattern length',
      'Two frequency maps: pattern and window',
      'Slide: update window map incrementally',
      'Compare maps: O(26) constant time',
      'O(n) vs O(n × m log m) brute force',
    ],
  },
];
