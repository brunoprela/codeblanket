/**
 * Quiz questions for What is the Two Pointers Technique? section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain why the two pointers technique can reduce O(n²) time complexity to O(n). What makes this possible?',
    hint: 'Think about what nested loops do versus what two pointers accomplish.',
    sampleAnswer:
      'The two pointers technique avoids the need for nested loops by making smart decisions about which pointer to move based on the data structure properties. In a nested loop approach, you check every possible pair which is n × n operations. With two pointers, you traverse the array just once - each pointer moves through the array at most n times total, giving you O(n). The key is that we can eliminate checking certain pairs because we have information - like if the array is sorted, or if we have found duplicates. We make progress with every pointer movement instead of checking all combinations.',
    keyPoints: [
      'Nested loops check all pairs: O(n²)',
      'Two pointers traverse once: O(n)',
      'Each pointer moves at most n times total',
      'Smart decisions eliminate need to check all pairs',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the two-people-searching-prices analogy in your own words and explain how it relates to the two pointers algorithm.',
    sampleAnswer:
      'Imagine two people looking for items that add up to exactly your budget in a price list that goes from cheap to expensive. One person starts at the cheapest item, the other at the most expensive. If the total is too much, the person at the expensive end moves to a cheaper item. If too little, the person at the cheap end moves to a more expensive item. They work together, meeting in the middle when they find the right combination. This is exactly how two pointers work on a sorted array - start at ends, move toward center based on whether you need a bigger or smaller value. It is efficient because you never backtrack or check the same pair twice.',
    keyPoints: [
      'One starts cheap, one starts expensive',
      'Move based on whether sum is too high or low',
      'Meet in middle when found',
      'Never check same pair twice',
    ],
  },
  {
    id: 'q3',
    question:
      'When would you consider using two pointers over other techniques? Give me some key signals.',
    sampleAnswer:
      'I look for a few signals. First, if the array is sorted or can be sorted, that is a huge hint. Second, if the problem asks about pairs, triplets, or finding elements that satisfy some relationship. Third, palindrome problems are classic two pointer problems. Fourth, if I need to do something in-place like removing duplicates or partitioning. And finally, if I catch myself thinking "I need nested loops to check all combinations" - that is when I pause and ask if two pointers could do it smarter. The key question is: can I make progress by moving pointers based on comparisons rather than checking everything?',
    keyPoints: [
      'Sorted or sortable data',
      'Finding pairs/triplets with properties',
      'Palindrome or symmetry problems',
      'In-place operations needed',
      'Alternative to nested loops',
    ],
  },
];
