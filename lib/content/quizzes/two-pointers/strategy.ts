/**
 * Quiz questions for Problem-Solving Strategy & Interview Tips section
 */

export const strategyQuiz = [
  {
    id: 'q1',
    question:
      'How would you recognize in an interview that a problem can be solved with two pointers? Walk me through your thought process.',
    sampleAnswer:
      'First thing I check: is the input sorted or can I sort it? That is a huge signal. Second, I look at the problem description - words like "pair", "two numbers", "remove", "partition", or "in-place" jump out at me. Third, if I start thinking "I need to check every pair" that is when I pause and ask if two pointers could be smarter. I also think about what information I have - if comparing elements at different positions helps make decisions, maybe two pointers. If the problem involves symmetry like palindromes, definitely two pointers. And if I need O(1) space for in-place modification, that is another hint. The key question: can I make progress by moving pointers based on comparisons?',
    keyPoints: [
      'Is data sorted or sortable?',
      'Keywords: pair, two numbers, remove, partition, in-place',
      'Alternative to checking all pairs',
      'Symmetry or palindrome problems',
      'Need O(1) space?',
      'Can I make decisions by comparing?',
    ],
  },
  {
    id: 'q2',
    question:
      'In an interview, how would you explain your choice of which two pointer pattern to use for a given problem?',
    sampleAnswer:
      'I would explain my reasoning out loud. For opposite direction, I would say "since the array is sorted and I need to find a pair with a specific sum, I will use pointers at both ends so I can increase or decrease the sum by moving the appropriate pointer". For same direction, I would say "I need to remove duplicates in place, so I will use slow pointer to mark where to write unique elements and fast pointer to scan ahead and find them". For sliding window, I would say "this is asking for a subarray with constraints, so I will use two pointers to define a window and slide it while maintaining those constraints". The key is explaining why the pattern fits the problem structure.',
    keyPoints: [
      'Explain reasoning out loud',
      'Connect pattern to problem requirements',
      'Opposite: sorted + pairs/relationships',
      'Same: in-place building/removal',
      'Window: subarray with constraints',
      'Show you understand why it works',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through how you would test and debug a two pointer solution during an interview.',
    sampleAnswer:
      'First, I would trace through a small example by hand, writing down the pointer positions at each step. I watch for: do the pointers move as expected? Does the loop terminate? Are edge cases handled? I test with edge cases like empty array, single element, all same values, and target at boundaries. For opposite direction, I make sure pointers do not cross incorrectly. For same direction, I verify slow and fast are doing their jobs. If there is a bug, I add print statements to track pointer values each iteration and check if the logic conditions are right. I also verify I am using correct pointer updates like mid plus one instead of just mid. The key is being systematic and explaining my debugging process out loud.',
    keyPoints: [
      'Trace small example by hand',
      'Check: do pointers move correctly, does loop terminate?',
      'Test edge cases: empty, single element, boundaries',
      'Print pointer values if debugging',
      'Verify loop conditions and pointer updates',
      'Explain process out loud',
    ],
  },
];
