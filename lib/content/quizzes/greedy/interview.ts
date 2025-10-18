/**
 * Quiz questions for Interview Strategy section
 */

export const interviewQuiz = [
  {
    id: 'q1',
    question:
      'How do you recognize a greedy problem? What keywords or patterns signal this?',
    sampleAnswer:
      'Keywords: "maximize", "minimize", "optimal", "scheduling", "selection". Patterns: intervals with counting, resource allocation, matching/pairing, sequence optimization. Recognize by: can you make irreversible choice that is locally best? Does sorting reveal structure? For example, "maximize number of meetings" suggests interval greedy. "Minimum cost" with simple constraints suggests greedy. "Assign tasks optimally" suggests sorting and greedy matching. Contrast with: "count all ways" (DP), "find all combinations" (backtracking), "with capacity constraint and indivisible items" (DP knapsack). If problem feels like optimization with obvious best choice at each step, try greedy. But always verify with examples and proof.',
    keyPoints: [
      'Keywords: maximize, minimize, optimal, scheduling',
      'Patterns: intervals, allocation, matching',
      'Can make irreversible local best choice?',
      'Does sorting help?',
      'Verify with examples and proof',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through your greedy interview approach from recognition to verification.',
    sampleAnswer:
      'First, recognize greedy from keywords and structure. Second, identify greedy choice: what should I pick at each step? Should I sort? Third, explain approach: sort by X, iterate, make greedy choice Y. Fourth, verify on examples: does it give correct answer? Fifth, attempt to prove: use exchange argument or stays ahead. Sixth, state complexity: O(n log n) sort + O(n) pass. Seventh, code clearly showing greedy choice. Eighth, test edge cases. Ninth, if greedy fails verification, discuss why (like 0/1 knapsack) and mention alternative (DP). This demonstrates: understanding greedy, ability to verify, knowing when it fails. Even failed greedy attempt followed by correct alternative shows depth.',
    keyPoints: [
      'Recognize, identify greedy choice',
      'Explain: sort, iterate, choose',
      'Verify on examples, attempt proof',
      'State complexity, code clearly',
      'If fails, explain why and give alternative',
    ],
  },
  {
    id: 'q3',
    question: 'What are common greedy mistakes and how do you avoid them?',
    sampleAnswer:
      'First: assuming greedy works without proof (test examples first). Second: sorting by wrong criteria (e.g., start instead of end for activity selection). Third: making wrong greedy choice (not identifying true local optimum). Fourth: forgetting to update state after greedy choice. Fifth: applying greedy to problem that needs DP (0/1 knapsack). Sixth: not handling ties properly in sorting. My strategy: always test on 2-3 examples before implementing, try to prove or find counterexample, know classic failures (0/1 knapsack), sort by the right key (draw examples to verify), clearly identify and document greedy choice. Most mistakes come from wrong greedy choice or applying to wrong problem.',
    keyPoints: [
      'Assuming works without testing',
      'Sorting by wrong key',
      'Wrong greedy choice',
      'Applying to problem needing DP',
      'Test examples, attempt proof, know classic failures',
    ],
  },
];
