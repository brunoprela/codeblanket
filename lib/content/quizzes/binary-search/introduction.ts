/**
 * Quiz questions for What is Binary Search? section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Talk through what makes binary search work. What do you absolutely need before you can use this algorithm?',
    hint: 'Think about the array properties that enable the algorithm.',
    sampleAnswer:
      'Binary search requires the array to be sorted, either in ascending or descending order. This is fundamental because the entire algorithm is built on making decisions by comparing the target with the middle element. If the middle is smaller than the target, we know the target must be in the right half (in a sorted array). Without sorting, this decision-making breaks down - we would have no idea which half to search next, and the algorithm falls apart.',
    keyPoints: [
      'The array must be sorted',
      'Sorting enables the comparison-based elimination',
      'Each comparison reliably tells us which half contains the target',
      'Without sorting, we cannot determine direction',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through why this algorithm is called "binary" search. What is binary about it?',
    sampleAnswer:
      'The name "binary" comes from the fact that at every step, we are making a binary decision - a two-way choice. When we compare our target with the middle element, we are essentially asking: is the target in the left half or the right half? There are only two possibilities, and we choose one. This happens at every level, creating a binary decision tree as we go deeper. This binary choice at each step is what gives the algorithm its name and also why it is so efficient - we are repeatedly cutting the problem in half.',
    keyPoints: [
      'Binary means two-way decision at each step',
      'Left half or right half - only two choices',
      'Creates a binary decision tree structure',
      'Binary decisions lead to logarithmic efficiency',
    ],
  },
  {
    id: 'q3',
    question:
      'If I give you a sorted linked list, can you use binary search on it? Why or why not?',
    hint: 'Think about what you need to do to find the middle element.',
    sampleAnswer:
      'Even though the linked list is sorted, you cannot efficiently use binary search on it. The problem is random access - to find the middle element in a linked list, you have to start at the head and traverse node by node, which takes O(n) time. Binary search is only fast because it can jump directly to the middle element in O(1) time with arrays. If finding the middle takes O(n), and we do this at every step, we lose all the efficiency gains of binary search. So while technically you could implement it, it would be pointless because the time complexity would still be O(n).',
    keyPoints: [
      'Linked lists lack O(1) random access',
      'Finding middle requires O(n) traversal from head',
      'Binary search needs instant access to middle element',
      'With linked lists, efficiency advantage is lost',
    ],
  },
];
