/**
 * Quiz questions for Best, Average, and Worst Case Analysis section
 */

export const bestaverageworstQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between worst case and average case complexity. When would you use each in your analysis?',
    sampleAnswer:
      'Worst case is the maximum time or space an algorithm could take under the most pessimal conditions, while average case is what you would expect on typical inputs. Worst case gives you a guarantee - you know performance will never be worse than this. Average case is more realistic but harder to analyze because you need to consider the probability distribution of inputs. In interviews and production code, we typically focus on worst case because it is safer - you want to know your algorithm will not blow up even on adversarial input. But average case is useful for understanding real-world performance, like knowing that hash table lookups are O(1) on average even though they are O(n) worst case.',
    keyPoints: [
      'Worst case: maximum time under pessimal conditions',
      'Average case: expected time on typical inputs',
      'Worst case provides guarantees',
      'Average case more realistic but harder to analyze',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is quicksort O(n²) in the worst case, and when does that worst case actually happen?',
    hint: 'Think about what makes a bad pivot choice.',
    sampleAnswer:
      'Quicksort is O(n²) worst case when the pivot choices are consistently terrible - specifically when the pivot is always the smallest or largest element. This happens when you try to sort an already sorted array using the first or last element as the pivot. In this case, each partition step only removes one element, so you get n levels of recursion instead of log n, and each level does O(n) work. That is n × n = O(n²). You can avoid this by using random pivots or the median-of-three method. This is why quicksort performs poorly on already sorted data unless you add randomization or smarter pivot selection.',
    keyPoints: [
      'Worst case when pivot is always min or max',
      'Happens on sorted/reverse-sorted arrays',
      'Gets n levels of recursion instead of log n',
      'Total: n levels × O(n) work per level = O(n²)',
    ],
  },
  {
    id: 'q3',
    question:
      'What is amortized analysis and how is it different from average case analysis? Use dynamic array appending as an example.',
    sampleAnswer:
      'Amortized analysis looks at the average cost per operation over a sequence of operations, while average case looks at expected cost for a single operation over all possible inputs. For dynamic arrays, when you append an element, most of the time it is O(1) - just add to the end. But occasionally the array is full and you need to resize, which means allocating new memory and copying all n elements - that single operation is O(n). However, this expensive resize happens rarely - roughly every time you double the size. So if you do n appends, you pay O(n) total for resizes spread across n operations, giving O(1) amortized cost per append. It is not average case because we are analyzing a sequence, not averaging over random inputs.',
    keyPoints: [
      'Amortized: average cost over sequence of operations',
      'Average case: expected cost for single operation',
      'Dynamic array: occasional O(n) resize spread over many O(1) appends',
      'Total cost O(n) for n appends → O(1) amortized per append',
    ],
  },
];
