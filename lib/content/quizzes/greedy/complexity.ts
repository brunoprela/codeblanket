/**
 * Quiz questions for Complexity Analysis section
 */

export const complexityQuiz = [
  {
    id: 'q1',
    question:
      'Compare greedy time complexity vs other approaches. Why is greedy usually faster?',
    sampleAnswer:
      'Greedy is typically O(n log n) from sorting + O(n) for one pass = O(n log n) total. DP is O(n²) or O(n×W) for knapsack. Backtracking is exponential O(2^n). Greedy is faster because: makes one irreversible choice per step, no backtracking or trying all options. For example, activity selection: greedy is O(n log n), brute force trying all combinations is O(2^n). Coin change: greedy is O(n) for standard coins, DP is O(n×amount). Greedy trades generality for speed - only works for specific problems but when it works, it is fastest. The one-pass nature after sorting makes it efficient. No memoization overhead, no recursion depth.',
    keyPoints: [
      'Greedy: O(n log n) sort + O(n) pass',
      'DP: O(n²), Backtracking: O(2^n)',
      'Faster: one choice, no backtracking',
      'Example: activity O(n log n) vs brute O(2^n)',
      'Trades generality for speed',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain space complexity of greedy algorithms. Why is it usually O(1) or O(n)?',
    sampleAnswer:
      'Greedy usually uses O(1) extra space: just a few variables to track current choice, best value, etc. Sometimes O(n) for: sorting requires space (though often in-place), storing result array, using priority queue. For example, activity selection: O(1) extra (just track last end time). Jump game: O(1) (track farthest reachable). Huffman coding: O(n) for priority queue. Meeting rooms: O(n) for heap. Compared to DP which is O(n) or O(n²) for table, greedy is more space efficient. In-place sorting and single-pass processing keep space low. If problem allows modifying input or returning indices instead of copying data, can achieve O(1).',
    keyPoints: [
      'Usually O(1): few variables',
      'Sometimes O(n): result array, priority queue',
      'Examples: activity O(1), Huffman O(n)',
      'vs DP: O(n) or O(n²) for table',
      'In-place processing keeps space low',
    ],
  },
  {
    id: 'q3',
    question:
      'When is greedy not the right approach? What signals should make you reconsider?',
    sampleAnswer:
      'Reconsider greedy when: local optimum clearly does not lead to global (0/1 knapsack), problem asks for "all possible" not "optimal" (need backtracking), constraints prevent greedy choice (dependencies between choices), optimization involves combinations not sequences. Signals: cannot find greedy choice property proof, simple greedy gives wrong answer on examples, problem is known NP-hard (usually need approximation or DP). For example, if problem asks "count all ways", that is DP or backtracking not greedy. If simple greedy fails on small example, likely need DP. Test greedy on examples first - if fails, abandon greedy. Good practice: always test greedy candidate on edge cases before full implementation.',
    keyPoints: [
      'Reconsider: local ≠ global, need "all solutions"',
      'Signals: cannot prove greedy, wrong on examples',
      'Known NP-hard usually not greedy',
      '"Count all ways" → DP or backtracking',
      'Test on examples first before implementing',
    ],
  },
];
