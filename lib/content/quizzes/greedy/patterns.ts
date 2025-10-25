/**
 * Quiz questions for Common Greedy Patterns section
 */

export const patternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the activity selection pattern. Why does sorting by end time give optimal solution?',
    sampleAnswer:
      'Activity selection maximizes non-overlapping activities. Greedy: sort by end time, pick activity if it starts after last picked ends. Why optimal? Finishing early leaves maximum room for future activities. Proof by exchange argument: suppose optimal solution differs from greedy. Replace first activity in optimal with greedy choice (earliest ending). This cannot make solution worse because earliest ending leaves more room. Continue replacing - greedy matches optimal. For example, activities with ends [3,5,6,8]: picking 3 leaves room for 5,6,8. Picking 5 first blocks 3. Earliest end is provably best choice. This pattern extends to interval scheduling, meeting rooms.',
    keyPoints: [
      'Maximize non-overlapping activities',
      'Sort by end time, greedy select',
      'Earliest end leaves most room',
      'Proof: exchange argument shows optimality',
      'Pattern: interval problems with count',
    ],
  },
  {
    id: 'q2',
    question:
      'Describe the fractional knapsack pattern. How does it differ from 0/1 knapsack?',
    sampleAnswer:
      'Fractional knapsack: can take partial items, maximize value with weight limit. Greedy: sort by value/weight ratio, take items in order, take fraction of last if needed. This is optimal for fractional. 0/1 knapsack: must take whole items, greedy fails. For example, capacity 50, items: [60kg,$100], [10kg,$20], [20kg,$30]. Fractional: take all of 10kg ($20), all of 20kg ($30), 20kg of 60kg ($33.33) = $83.33 optimal. Greedy by ratio works. 0/1: cannot take fraction of 60kg item, need DP. Key difference: fractional allows splitting, making greedy safe. 0/1 needs considering combinations. Fractional is O(n log n), 0/1 is O(n×W).',
    keyPoints: [
      'Fractional: can split items, greedy optimal',
      'Sort by value/weight ratio',
      '0/1: must take whole, greedy fails, need DP',
      'Splitting makes greedy safe',
      'Fractional O(n log n), 0/1 O(n×W)',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through the jump game greedy pattern. Why does tracking farthest reachable work?',
    sampleAnswer:
      'Jump game: reach end of array, each element is max jump length. Greedy: iterate, track farthest reachable from current position. At each index i, farthest = max (farthest, i + nums[i]). If current position exceeds farthest, cannot proceed. If farthest >= end, can reach. For minimum jumps: track current range end, when reach end of range, increment jumps and update range to farthest. Works because if we can reach index i, we can reach all indices before i. So tracking farthest from all reachable positions guarantees we find if end is reachable. For minimum jumps, greedily extending range as far as possible minimizes jump count. This is provably optimal.',
    keyPoints: [
      'Track farthest reachable position',
      'At i: farthest = max (farthest, i + nums[i])',
      'Can reach end if farthest >= end',
      'Min jumps: greedily extend range',
      'Provably optimal: can reach implies can reach all before',
    ],
  },
];
