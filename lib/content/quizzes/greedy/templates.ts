/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the greedy interval template. What are the key steps?',
    sampleAnswer:
      'Greedy interval template for maximizing count has four steps. First, sort intervals by end time. Second, initialize count and last_end (first interval end). Third, iterate from second interval: if current.start >= last_end (non-overlapping), increment count and update last_end to current.end. Fourth, return count. The key: sorting by end time ensures earliest ending is always processed first, which is provably optimal. For example, [[1,3], [2,5], [4,6]]: sort by end gives [[1,3], [2,5], [4,6]]. Pick [1,3], skip [2,5] (overlaps), pick [4,6] (4 >= 3). Count is 2. This template works for activity selection, non-overlapping intervals, minimum arrows to burst balloons.',
    keyPoints: [
      'Step 1: sort by end time',
      'Step 2: track count and last_end',
      'Step 3: if non-overlap, increment count',
      'Earliest ending is provably optimal',
      'Works: activity, non-overlap, arrows',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the two-pointer greedy template. When is this pattern applicable?',
    sampleAnswer:
      'Two-pointer greedy processes from both ends toward middle, making greedy choice at each step. Applicable when: array is sorted (or sortable), decision involves comparing extremes, want to match/pair/distribute elements. Template: sort array, left at start, right at end. While left < right: check condition, move pointer based on greedy choice. For example, assign cookies: sort children and cookies, try to satisfy smallest child with smallest adequate cookie. Two pointers enable greedy matching. Container with most water: try largest width first, move pointer with smaller height (greedy choice). The pattern: sorted + extremes + pairing → two pointers. Enables O(n) greedy after O(n log n) sort.',
    keyPoints: [
      'Process from both ends toward middle',
      'Applicable: sorted, compare extremes, matching',
      'Template: sort, left/right pointers, move based on choice',
      'Example: assign cookies, container water',
      'O(n) greedy after O(n log n) sort',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the priority queue greedy template. What problems benefit from this?',
    sampleAnswer:
      'Priority queue greedy processes items by priority, using heap for efficient access to next best choice. Use when: need to repeatedly pick best available item, items arrive over time, resource allocation with priorities. Template: create max or min heap, add items, repeatedly pop best and process. For example, Huffman coding: repeatedly merge two lowest frequency nodes (min heap). Meeting rooms II: track earliest ending meeting (min heap). Task scheduler: process most frequent task first (max heap). The heap maintains best choice in O(log n) rather than O(n) scan. This is crucial when many greedy choices need to be made. Total complexity: O(n log n) for n heap operations.',
    keyPoints: [
      'Use heap for next best choice',
      'Applicable: repeated picks, arrivals, priorities',
      'Template: heap, add items, pop best',
      'Examples: Huffman, meeting rooms, task scheduler',
      'O(log n) per choice vs O(n) scan',
    ],
  },
];
