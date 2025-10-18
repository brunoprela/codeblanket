/**
 * Quiz questions for Detailed Algorithm Walkthrough section
 */

export const algorithmQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the two sum sorted array algorithm step by step using an example. Explain your thought process at each decision point.',
    sampleAnswer:
      'Let me use [1, 3, 5, 7, 9] with target 12. I start with left at 1 and right at 9. Sum is 10, which is less than 12, so I need a bigger number - move left right to 3. Now sum is 12, found it! The key decisions are: after each sum calculation, I compare with target. Too small means I need bigger numbers, so move left pointer right into larger values. Too big means I need smaller numbers, so move right pointer left into smaller values. If they ever cross without finding it, no solution exists. Each move eliminates possibilities - like when sum was 10, I knew anything paired with 1 would be too small, so I never need to check those pairs.',
    keyPoints: [
      'Start at opposite ends of sorted array',
      'Compare sum with target after each check',
      'Too small → move left right for larger values',
      'Too big → move right left for smaller values',
      'Each move eliminates a set of pairs',
    ],
  },
  {
    id: 'q2',
    question:
      'In the remove duplicates algorithm, explain what slow and fast represent at any point during execution.',
    sampleAnswer:
      'At any moment during remove duplicates, slow points to the last position that contains a confirmed unique element in our result. Everything from index 0 to slow is the deduplicated portion we have built so far. Fast points to the element we are currently examining to see if it is different from what slow is pointing at. So slow says "this is where my result ends so far" and fast says "let me check if this new element belongs in the result". When fast finds something new, we put it at slow plus 1 and advance slow. The gap between slow and fast contains elements we have already processed and determined to be duplicates.',
    keyPoints: [
      'Slow: last position of confirmed unique element',
      'Everything from 0 to slow is deduplicated result',
      'Fast: currently examining this element',
      'Gap between them: processed duplicates',
    ],
  },
  {
    id: 'q3',
    question:
      'Why is it important that the array is sorted for two sum with two pointers? What breaks if it is not sorted?',
    sampleAnswer:
      'Sorting is critical because it gives us the monotonic property - moving left pointer right consistently increases values, moving right pointer left consistently decreases values. This lets us make reliable decisions. If the array is not sorted, say [5, 1, 9, 3], and sum is too big, which pointer do I move? I cannot tell! Moving left might give me a smaller or larger number. The algorithm depends on knowing that moving a pointer in one direction predictably changes the sum. Without sorting, we lose this guarantee and the algorithm fails. That is why the brute force O(n²) check-all-pairs approach is needed for unsorted arrays.',
    keyPoints: [
      'Sorting gives monotonic property',
      'Moving pointers predictably changes sum',
      'Without sorting, cannot make reliable decisions',
      'Would need brute force O(n²) for unsorted',
    ],
  },
];
