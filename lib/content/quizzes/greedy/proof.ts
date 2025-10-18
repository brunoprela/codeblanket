/**
 * Quiz questions for Proving Greedy Correctness section
 */

export const proofQuiz = [
  {
    id: 'q1',
    question:
      'How do you prove a greedy algorithm is correct? What techniques can you use?',
    sampleAnswer:
      'Three main proof techniques. First, greedy stays ahead: show greedy is always at least as good as optimal after each step. Second, exchange argument: suppose optimal differs from greedy, exchange optimal choice with greedy choice, show this does not worsen solution. Third, induction: prove greedy works for base case, then show if greedy works for k steps, it works for k+1. For activity selection: exchange argument - replacing earliest end in optimal with greedy choice leaves more room. For Huffman: induction - merging two lowest frequency nodes is optimal at each step. Without proof, greedy might be wrong. For example, greedy fails for 0/1 knapsack but works for fractional. Must prove correctness.',
    keyPoints: [
      'Three techniques: stays ahead, exchange, induction',
      'Stays ahead: greedy ≥ optimal at each step',
      'Exchange: swap optimal choice with greedy, no worse',
      'Induction: prove base, then k → k+1',
      'Without proof, greedy might be wrong',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why greedy fails for 0/1 knapsack. What property is missing?',
    sampleAnswer:
      'Greedy fails because local choice (highest ratio) does not guarantee global optimum when items are indivisible. Example: capacity 50, items: [10kg,$60 ratio=6], [20kg,$100 ratio=5], [30kg,$120 ratio=4]. Greedy picks 10kg then 20kg for $160. Optimal picks 20kg+30kg for $220. The problem: after picking 10kg item, cannot fit 30kg item, but 30kg item is part of optimal. Greedy lacks look-ahead. Missing property: greedy choice property does not hold - locally best choice (highest ratio) blocks better global combination. For fractional knapsack, can take partial items, so greedy works (take fractions to fill capacity). For 0/1, need DP to consider all item combinations.',
    keyPoints: [
      'Local best does not guarantee global when indivisible',
      'Example: highest ratio blocks better combination',
      'Lacks look-ahead to see better combos',
      'Greedy choice property does not hold',
      'Fractional works (can split), 0/1 needs DP',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe when sorting enables greedy solutions. What should you sort by?',
    sampleAnswer:
      'Sorting reveals structure that makes greedy safe. Sort by: end time for intervals (activity selection), ratio for fractional knapsack, difficulty or deadline for scheduling, size for matching problems. Sorting orders choices so greedy can process best-first. For example, activity selection: after sorting by end time, earliest ending is provably best choice - picking it leaves most room. Without sorting, would need to search for earliest ending each time or might pick wrong activity. Sorting is O(n log n) but enables O(n) greedy processing. Pattern: if greedy should pick "best" choice but best is not obvious, sorting often reveals it. Sort puts related items together, enables greedy invariants.',
    keyPoints: [
      'Sorting reveals structure for greedy',
      'Sort by: end time, ratio, deadline, size',
      'Orders choices for best-first processing',
      'Example: sort by end → earliest is provably best',
      'O(n log n) sort enables O(n) greedy',
    ],
  },
];
