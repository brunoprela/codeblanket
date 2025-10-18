/**
 * Quiz questions for Common Sorting Interview Patterns section
 */

export const interviewproblemsQuiz = [
  {
    id: 'q1',
    question:
      'When would sorting actually make a problem easier to solve? Give me a concrete example.',
    sampleAnswer:
      'Sorting can turn an O(n²) problem into O(n log n) + O(n) = O(n log n). A great example is finding duplicate numbers. Without sorting, you need nested loops to compare every pair - O(n²). But if you sort first, duplicates end up adjacent. Then you just scan once checking if arr[i] == arr[i+1] - that is O(n). Total is O(n log n) for sort plus O(n) for scan, which is O(n log n) overall - much better than O(n²). Another example is the two-sum problem on sorted arrays - you can use two pointers to find a pair in O(n) after sorting. The pattern is: if you can solve it in linear time on sorted data, and sorting costs O(n log n), that is often better than the naive approach.',
    keyPoints: [
      'Sorting can reduce O(n²) to O(n log n)',
      'Example: find duplicates via adjacent elements',
      'Sorted data enables two-pointer technique',
      'Pattern: O(n log n) sort + O(n) scan < O(n²) naive',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the quick select algorithm. How is it related to quicksort, and what is its advantage?',
    hint: 'Think about finding the kth largest element.',
    sampleAnswer:
      "Quick select finds the kth largest (or smallest) element without fully sorting the array. It is based on quicksort's partition step. You partition the array around a pivot, which puts the pivot in its final sorted position. If the pivot is at position k, you found the kth element. If k is less than pivot position, recurse on the left partition. If k is greater, recurse on the right partition. Unlike quicksort which recurses on both sides, quick select only recurses on one side. This gives O(n) average time instead of O(n log n), because n + n/2 + n/4 + ... converges to 2n = O(n). The advantage over sorting is that you get O(n) average time versus O(n log n) - significant for large arrays when you only need one element.",
    keyPoints: [
      'Based on quicksort partition step',
      'Only recurse on one partition (where k is)',
      "Average case: O(n) vs quicksort's O(n log n)",
      'Used for finding kth largest/smallest element',
      'Much faster than full sort when you need one element',
    ],
  },
  {
    id: 'q3',
    question:
      'You need to merge k sorted linked lists. Walk through your approach and complexity.',
    sampleAnswer:
      'The optimal approach is to use a min heap. I would put the first node from each of the k lists into a min heap. Then repeatedly: 1) Extract the minimum from the heap (smallest overall), 2) Add it to the result list, 3) If that node had a next node, add the next node to the heap. The heap always contains at most k elements. Each of the n total nodes goes into the heap once and comes out once, and each heap operation is O(log k), giving O(n log k) total time. Space is O(k) for the heap. This is better than merging lists pairwise which would be O(n × k). The key insight is that I only need to compare the k current candidate nodes, not all n nodes, so a heap of size k is perfect.',
    keyPoints: [
      'Use min heap of size k',
      'Heap contains first unmerged node from each list',
      'Extract min, add to result, add its next to heap',
      'Time: O(n log k) - each of n nodes does O(log k) heap ops',
      'Space: O(k) for heap',
      'Better than pairwise merge: O(n log k) vs O(n × k)',
    ],
  },
];
