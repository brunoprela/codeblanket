/**
 * Multiple choice questions for Introduction to Design Problems section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const introductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'Why is LRU Cache the most commonly asked design problem?',
    options: [
      'It is the easiest design problem',
      'It requires combining HashMap and LinkedList, testing data structure knowledge',
      'It only needs one data structure',
      'It has no edge cases',
    ],
    correctAnswer: 1,
    explanation:
      'LRU Cache is popular because it tests the ability to combine data structures optimally. It requires HashMap for O(1) access AND doubly LinkedList for O(1) order updates - neither alone suffices. This tests deep understanding of data structure trade-offs, which is exactly what distinguishes strong candidates.',
  },
  {
    id: 'mc2',
    question: 'What is the main challenge in design problems?',
    options: [
      'Writing syntactically correct code',
      'Choosing and combining data structures to meet time complexity requirements',
      'Memorizing algorithms',
      'Using the fewest lines of code',
    ],
    correctAnswer: 1,
    explanation:
      'The core challenge is selecting the right data structures to satisfy ALL requirements. LRU Cache needs O(1) for get AND put with eviction. Min Stack needs O(1) for push, pop, AND getMin. This requires understanding trade-offs and often combining multiple structures creatively.',
  },
  {
    id: 'mc3',
    question:
      'In design problems, when should you use a doubly linked list over a singly linked list?',
    options: [
      'Always, it is always better',
      'When you need to remove arbitrary nodes in O(1) time',
      'Never, singly linked lists are always sufficient',
      'Only for small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Doubly linked lists allow O(1) removal of nodes when you have a reference to that node, because you can access node.prev directly. Singly linked lists require O(N) to find the previous node. LRU Cache uses doubly linked list because we need to remove arbitrary nodes (when accessing them) in O(1).',
  },
  {
    id: 'mc4',
    question: 'What does "amortized O(1)" mean?',
    options: [
      'Always O(1) for every single operation',
      'Average O(1) over a sequence of operations, though some individual operations may be O(N)',
      'Worse than O(1)',
      'Only works for small inputs',
    ],
    correctAnswer: 1,
    explanation:
      'Amortized O(1) means that while some operations may take O(N) time occasionally, the average time over many operations is O(1). Example: Queue using two stacks - dequeue might occasionally transfer N items (O(N)), but each item is transferred at most once, so average is O(1).',
  },
  {
    id: 'mc5',
    question:
      'Why do many rate limiting problems use a deque (double-ended queue)?',
    options: [
      'Deques use less memory',
      'Deques allow O(1) removal from front (old timestamps) and O(1) addition to back (new timestamps)',
      'Deques automatically sort timestamps',
      'Deques prevent duplicates',
    ],
    correctAnswer: 1,
    explanation:
      'Rate limiters need to remove old timestamps (from front) and add new timestamps (to back), both in O(1). Deque supports O(1) operations at both ends, making it perfect for sliding window patterns. Regular lists have O(N) removal from front.',
  },
];
