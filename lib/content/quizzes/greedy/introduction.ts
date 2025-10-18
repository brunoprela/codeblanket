/**
 * Quiz questions for Introduction to Greedy Algorithms section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Explain what greedy algorithms are and when they produce optimal solutions. What makes a problem suitable for greedy?',
    sampleAnswer:
      'Greedy algorithms make locally optimal choices at each step hoping to find global optimum. At each decision point, choose what looks best right now without considering future consequences. Greedy works when problem has greedy choice property (local optimum leads to global) and optimal substructure (optimal solution contains optimal solutions to subproblems). For example, making change with coins: always take largest coin possible. Activity selection: always pick earliest ending activity. Greedy fails when local optimum does not guarantee global optimum. For example, 0/1 knapsack needs DP because taking highest value item first might leave no room for better combination. Key: greedy is fast O(n log n) but only works for specific problems. Must prove greedy choice is safe.',
    keyPoints: [
      'Make locally optimal choice at each step',
      'Works when: greedy choice property + optimal substructure',
      'Local optimum must lead to global optimum',
      'Fast O(n log n) but limited applicability',
      'Must prove correctness, does not always work',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare greedy vs dynamic programming. When would you choose each approach?',
    sampleAnswer:
      'Greedy makes irreversible choice at each step, no backtracking. DP considers all options, stores results. Greedy is O(n) or O(n log n), DP is O(n²) or higher. Choose greedy when: local optimum provably leads to global (activity selection, Huffman coding), problem has greedy choice property. Choose DP when: need to consider all possibilities (knapsack, LCS), local choice might not be optimal. For example, coin change for making value: greedy (always largest coin) works for standard coins but fails for arbitrary denominations - need DP. Activity selection: greedy (earliest end) is provably optimal. The key: greedy is preferred when it works (faster, simpler) but DP is more general. Try to prove greedy works; if cannot, use DP.',
    keyPoints: [
      'Greedy: one choice per step, no backtrack',
      'DP: try all options, memoize results',
      'Greedy: O(n log n), DP: O(n²) or higher',
      'Greedy when: local → global provably',
      'DP when: need all possibilities',
    ],
  },
  {
    id: 'q3',
    question:
      'Walk me through common greedy patterns. How do you recognize which pattern to apply?',
    sampleAnswer:
      'Common patterns: Interval scheduling (earliest end time), knapsack fractional (best ratio), two-pointer (extremes first), sorting (process in order). Recognize by problem structure. Intervals + maximize count → sort by end time. Maximize/minimize with constraints → try sorting. Array with extremes → two pointers. Resource allocation → priority queue. For example, "maximum units on truck with capacity" → greedy by unit density (sort by units per box). "Minimum arrows to burst balloons" → sort intervals by end, count groups. "Assign cookies to children" → sort both, match smallest. The pattern: identify what to sort by and what greedy choice to make. Drawing examples often reveals the pattern.',
    keyPoints: [
      'Patterns: interval (end time), fractional (ratio), extremes (two pointers)',
      'Intervals + count → sort by end',
      'Maximize/minimize → try sorting',
      'Resource allocation → priority queue',
      'Draw examples to reveal pattern',
    ],
  },
];
