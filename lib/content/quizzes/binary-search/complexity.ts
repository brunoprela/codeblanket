/**
 * Quiz questions for Time & Space Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Explain in your own words why binary search is O(log n). What is the mathematical reasoning behind that logarithm?',
    hint: 'Think about how many times you can cut something in half.',
    sampleAnswer:
      'Binary search is O(log n) because with each comparison, we cut the problem size in half. If I start with say 1000 elements, after one comparison I have 500 left, then 250, then 125, and so on. The question is: how many times can I divide n by 2 until I get down to 1? That is exactly what a logarithm tells us. If I have n/2^k = 1, solving for k gives me k = log n. So no matter how big the array is, I only need log n comparisons. This is why binary search is so incredibly fast - for a million elements, I only need about 20 comparisons.',
    keyPoints: [
      'Each comparison cuts search space in half',
      'Pattern: n → n/2 → n/4 → n/8 → ... → 1',
      'How many divisions by 2? That is log₂(n)',
      'For 1 million elements, only ~20 comparisons needed',
    ],
  },
  {
    id: 'q2',
    question:
      'If I asked you to implement binary search, would you do it iteratively or recursively? Talk through the space complexity implications of each approach.',
    sampleAnswer:
      'I would go with the iterative approach. The iterative version is O(1) space - I only need three variables: left, right, and mid. These variables do not grow with input size. If I did it recursively, each recursive call adds a frame to the call stack, and since I make about log n calls (one for each level going down the tree), that is O(log n) extra space. So iterative is more space efficient. That said, recursive can be cleaner to read, but in production I would usually prefer iterative to avoid any stack overflow issues with very deep recursion.',
    keyPoints: [
      'Iterative: O(1) space - just a few variables',
      'Recursive: O(log n) space - call stack grows',
      'Each recursive call adds stack frame',
      'Iterative preferred for production code',
    ],
  },
  {
    id: 'q3',
    question:
      'Unlike linear search which can vary a lot, binary search is pretty consistent. Why is that? Talk about best, average, and worst case.',
    sampleAnswer:
      'Binary search is consistent because it does not matter where the target is - we always do roughly the same amount of work. Whether the target is at the beginning, end, or somewhere in the middle, we still divide the array in half each time and keep going. The worst case and average case are both O(log n). The only exception is if we get really lucky and the target is exactly in the middle on the first try - that is O(1), but that is rare. Compare this to linear search where if the target is at the end, you check every element, but if it is at the start, you check just one. Binary search is way more predictable.',
    keyPoints: [
      'Always divides in half regardless of target location',
      'Worst and average cases both O(log n)',
      'Only best case O(1) - lucky first guess',
      'Much more predictable than linear search',
    ],
  },
];
