/**
 * Multiple choice questions for Multi-Agent Workflows section
 */

export const multiagentworkflowsMultipleChoice = [
  {
    id: 'maas-workflow-mc-1',
    question:
      'A linear workflow has 5 agents running sequentially (A → B → C → D → E). Agent C fails on step 3. Which workflow recovery strategy is MOST efficient?',
    options: [
      'Restart from Agent A (full restart)',
      'Checkpoint after each agent, restart from Agent C',
      'Skip Agent C and continue with D and E',
      'Run all agents in parallel to avoid sequential failures',
    ],
    correctAnswer: 1,
    explanation:
      "Checkpointing enables resumption from the failure point. Agents A and B succeeded, so save their outputs and restart from Agent C. Option A wastes work. Option C is incorrect—skipping a failed agent likely breaks dependencies. Option D changes the workflow fundamentally and doesn't address the failure.",
  },
  {
    id: 'maas-workflow-mc-2',
    question:
      'A branching workflow splits after Agent A into two parallel paths (B1 → C1) and (B2 → C2), then merges at Agent D. Path 1 takes 5 seconds, Path 2 takes 15 seconds. How long does the workflow take?',
    options: [
      '5 seconds (fastest path)',
      '15 seconds (slowest path determines completion)',
      '20 seconds (sum of both paths)',
      '10 seconds (average of both paths)',
    ],
    correctAnswer: 1,
    explanation:
      'Branching workflows wait for ALL branches to complete before merging. The slowest path (15 seconds) determines when Agent D can start. Option A would be true if we only needed the fastest result. Options C and D reflect misunderstandings of parallel execution.',
  },
  {
    id: 'maas-workflow-mc-3',
    question:
      'In a conditional workflow, Agent A outputs a classification: "urgent" or "normal". If "urgent", go to Agent B (fast path, 2 minutes). If "normal", go to Agent C → D (thorough path, 10 minutes). Agent A misclassifies 20% of cases. What is the MAIN risk?',
    options: [
      '"Urgent" cases going through slow path (missed deadlines)',
      '"Normal" cases going through fast path (incomplete processing)',
      'The workflow becomes too complex to maintain',
      'Conditional branching is slower than always using the same path',
    ],
    correctAnswer: 1,
    explanation:
      'The main risk is "normal" cases being misclassified as "urgent" and getting incomplete processing (fast path when they need thorough path). This could miss important issues. Option A is also a problem but typically less severe (urgency is delayed, not quality compromised). Options C and D are not inherent risks of the misclassification.',
  },
  {
    id: 'maas-workflow-mc-4',
    question:
      'A loop workflow has Agent A → Agent B → Agent A (iterative refinement). The loop should continue until "quality > 0.9" or "iterations > 5". What is the PRIMARY pitfall of loop workflows?',
    options: [
      'Loop workflows are always slower than non-loop workflows',
      'Infinite loops if the exit condition is never met',
      'Agents forget context from previous iterations',
      'Loop workflows cannot handle failures mid-loop',
    ],
    correctAnswer: 1,
    explanation:
      'Infinite loops are the key risk. If quality never reaches 0.9, the loop could run forever. Always include a maximum iteration count as a safety valve. Option A is too absolute—loops are sometimes necessary. Option C is a design issue (context management), not inherent. Option D is incorrect—failures can be handled like any workflow step.',
  },
  {
    id: 'maas-workflow-mc-5',
    question:
      'A workflow uses checkpoints after each agent to enable resumption on failure. What is the MAIN tradeoff of frequent checkpointing?',
    options: [
      'Checkpoints reduce the quality of agent outputs',
      'Checkpoints increase workflow execution time and storage costs',
      'Checkpoints make workflows non-deterministic',
      'Checkpoints prevent parallel execution of agents',
    ],
    correctAnswer: 1,
    explanation:
      "Checkpointing adds overhead: (1) Time to serialize and save state after each agent, (2) Storage costs for saving intermediate outputs. The tradeoff is robustness (can resume on failure) vs. performance. Option A is incorrect—checkpoints don't affect quality. Option C is incorrect—checkpoints actually improve determinism (reproducible state). Option D is incorrect—checkpointing and parallelism are independent.",
  },
];
