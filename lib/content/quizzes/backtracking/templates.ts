/**
 * Quiz questions for Code Templates section
 */

export const templatesQuiz = [
  {
    id: 'q1',
    question:
      'Walk me through the basic backtracking template. What does each part accomplish?',
    sampleAnswer:
      'The basic template has: result list, helper function, and base case. Helper takes current state (path, used elements, etc.). Base case: if complete solution (path length equals target), add copy of path to result and return. Otherwise, iterate through choices, for each valid choice: add to path (choose), recurse (explore), remove from path (unchoose). The iteration represents branching at each node. Choose-explore-unchoose is the core pattern - ensures state is independent for each branch. Return result after helper explores all paths. This template adapts to subsets, combinations, permutations by changing what choices are and what complete means.',
    keyPoints: [
      'Result list, helper function, base case',
      'Base: complete solution → add to result',
      'Loop through choices',
      'Choose, recurse, unchoose pattern',
      'Adapts to different problem types',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the subset template with start index. Why is start index important for avoiding duplicates?',
    sampleAnswer:
      'Subset template recursively decides include or exclude for each element, using start index to track progress. At each recursion, we have two branches: include nums[start] then recurse with start+1, or skip nums[start] and recurse with start+1. Start index ensures we only consider elements at or after current position, preventing duplicates. Without start index, we would reconsider earlier elements and generate [1,2] and [2,1] as separate subsets - but they are the same subset. Start index maintains order: once we pass an element, we never go back. This gives us exactly 2^n unique subsets. Alternative: pass remaining elements as parameter.',
    keyPoints: [
      'Start index tracks progress through array',
      'Two branches: include or exclude current',
      'Recurse with start+1 in both branches',
      'Prevents reconsidering earlier elements',
      'Avoids duplicates by maintaining order',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe the permutation template with used array. Why swap and track used array?',
    sampleAnswer:
      'Permutation template generates all arrangements. Two approaches: used array or swapping. Used array: maintain boolean array tracking which elements are used. At each position, try each unused element: mark used, add to path, recurse, remove from path, mark unused. This explores all arrangements by trying each element at each position. Swapping approach: swap current position with each position from current to end, recurse, swap back. This implicitly tracks used elements via array partitioning. Used array is clearer but needs O(n) extra space. Swapping is in-place but harder to understand. Both generate n! permutations by trying all orderings.',
    keyPoints: [
      'Two approaches: used array or swapping',
      'Used array: track which elements used',
      'Try each unused at each position',
      'Swapping: swap, recurse, swap back',
      'Both: O(n!) permutations',
    ],
  },
];
