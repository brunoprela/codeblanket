/**
 * Quiz questions for Code Templates & Patterns section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through how you would modify the two sum template to find three numbers that sum to a target (3Sum problem).',
    sampleAnswer:
      'For 3Sum, I would first sort the array, then use one loop plus two pointers. For each element at index i, I treat it as the first number and then use two pointers to find two other numbers that sum to target minus that first number. So outer loop fixes one element, inner two pointers solve "two sum equals target minus first element". The two pointers work just like regular two sum - start at ends, move based on whether sum is too big or small. Key detail: I need to skip duplicates at all three levels to avoid duplicate triplets. This is O(n²) because of the outer loop times the O(n) two pointer search.',
    keyPoints: [
      'Sort array first',
      'Outer loop fixes first element',
      'Two pointers find other two elements',
      'Becomes 2Sum for (target - first element)',
      'Skip duplicates to avoid duplicate triplets',
      'Time: O(n²)',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the partition template where you separate elements into two groups. How do the pointers work differently than in the other patterns?',
    sampleAnswer:
      'In partition, I use two pointers to separate elements based on some condition - like all evens before all odds. Both pointers usually move inward from the ends. When left pointer finds an element that belongs in the right section, and right pointer finds an element that belongs in the left section, I swap them. Then I move both pointers. This is different from other patterns because I am actively swapping elements to reorder them, not just reading or comparing. The pointers converge toward each other, and when they meet, the array is partitioned. Think of it like organizing a bookshelf - left hand grabs books that should be on right, right hand grabs books that should be on left, swap them.',
    keyPoints: [
      'Separate elements based on condition',
      'Both pointers move inward from ends',
      'Swap elements when both find misplaced items',
      'Actively reordering, not just reading',
      'Done when pointers meet',
    ],
  },
  {
    id: 'q3',
    question:
      'When should you choose the same-direction pattern over the opposite-direction pattern? What is the key difference in what they are suited for?',
    sampleAnswer:
      'I choose same direction when I need to build up a result incrementally in place, like removing duplicates or moving zeros. The key is that slow pointer marks where I am writing my result, while fast pointer scans ahead. Opposite direction is for finding relationships between elements at different positions - like pairs that sum to a target. Same direction reads ahead and writes behind. Opposite direction compares ends and works inward. If the problem says "remove" or "move elements" or "in-place modification", I think same direction. If it says "find pair" or "two numbers" or involves symmetric operations, I think opposite direction.',
    keyPoints: [
      'Same direction: in-place incremental building',
      'Slow writes, fast scans ahead',
      'Opposite: finding relationships between positions',
      'Remove/move → same direction',
      'Find pairs/symmetric → opposite direction',
    ],
  },
];
