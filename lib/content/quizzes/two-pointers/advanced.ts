/**
 * Quiz questions for Advanced Techniques & Variations section
 */

export const advancedQuiz = [
  {
    id: 'q1',
    question:
      'For the "Container With Most Water" problem, explain the counter-intuitive insight about which pointer to move. Why do we move the pointer at the shorter line?',
    sampleAnswer:
      'The insight is that the area is limited by the shorter of the two heights. If I move the pointer at the taller line, the width decreases and the height can only stay the same or get worse (because it is limited by the shorter side). So moving the tall side can only make things worse. But if I move the pointer at the shorter side, width still decreases, but I have a chance of finding a taller line that could compensate for the lost width. It is the only move that has potential to improve. Think of it like this: the short side is the bottleneck, so we try to fix the bottleneck, not the side that is already good.',
    keyPoints: [
      'Area limited by shorter height',
      'Moving tall side: width down, height same or worse',
      'Moving short side: width down, but height might improve',
      'Short side is the bottleneck',
      'Only move with potential to improve',
    ],
  },
  {
    id: 'q2',
    question:
      'Talk through the 3Sum problem. How does it extend the two pointer technique, and what is the time complexity?',
    sampleAnswer:
      'In 3Sum, I want three numbers that sum to zero. I cannot use just two pointers for three numbers, so I add an outer loop. I fix the first number with the loop, then use two pointers to find the other two numbers that sum to negative of the first number. So it becomes: for each element, solve 2Sum with target equals negative that element. The two pointers part is still O(n), but I do it n times in the outer loop, so overall O(n²). I also need to handle duplicates carefully by skipping over repeated values at all three positions to avoid returning duplicate triplets. Sort the array first to enable the two pointer technique.',
    keyPoints: [
      'Fix first number with loop',
      'Use 2Sum for other two numbers',
      'Target for 2Sum: -(first number)',
      'Time: O(n²) = n iterations × O(n) 2Sum',
      'Handle duplicates by skipping repeats',
      'Sort array first',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe cycle detection in linked lists using fast and slow pointers (Floyd Cycle Detection). How does it work?',
    sampleAnswer:
      'For cycle detection, I use two pointers starting at the head. Slow moves one step at a time, fast moves two steps. If there is no cycle, fast reaches the end and we are done. If there is a cycle, fast will eventually lap slow and they will meet inside the cycle - guaranteed because fast is gaining one step per iteration and the cycle is finite. Once they meet, I know there is a cycle. To find where the cycle starts, I can reset one pointer to head and move both one step at a time - they will meet at the cycle entrance. This works due to mathematical properties of the distances involved.',
    keyPoints: [
      'Slow moves 1 step, fast moves 2 steps',
      'No cycle: fast reaches end',
      'Cycle exists: fast laps slow, they meet',
      'Fast gains 1 step per iteration',
      'To find cycle start: reset one to head, move both 1 step',
    ],
  },
];
