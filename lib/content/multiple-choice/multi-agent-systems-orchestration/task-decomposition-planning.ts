/**
 * Multiple choice questions for Task Decomposition & Planning section
 */

export const taskdecompositionplanningMultipleChoice = [
  {
    id: 'maas-decomp-mc-1',
    question:
      'An LLM-based planning agent generates a task decomposition plan with 15 subtasks. Task 7 fails. What is the BEST strategy for replanning?',
    options: [
      'Restart from scratch and regenerate all 15 tasks',
      'Mark tasks 1-6 as complete, regenerate tasks 7-15',
      'Ask the LLM to generate a "fix plan" for task 7 only, keep 8-15 unchanged',
      'Continue with tasks 8-15 and retry task 7 at the end',
    ],
    correctAnswer: 1,
    explanation:
      "Incremental replanning from the failure point is most efficient. Tasks 1-6 succeeded and don't need regeneration. Regenerate 7-15 because downstream tasks might depend on task 7's output. Option A wastes work. Option C is risky—tasks 8-15 might depend on task 7, so they need reconsideration. Option D (skip and retry later) breaks dependencies.",
  },
  {
    id: 'maas-decomp-mc-2',
    question:
      'A task decomposition agent generates a plan where Task B depends on Task A, and Task C depends on Task B. Task A takes 2 minutes, Task B takes 5 minutes, and Task C takes 3 minutes. A new Task D (2 minutes) is added that depends on Task A but not Task B. What is the critical path?',
    options: [
      'A → B → C (10 minutes)',
      'A → D (4 minutes)',
      'A → B → C → D (12 minutes)',
      'Both A → B → C and A → D can run in parallel (5 minutes total)',
    ],
    correctAnswer: 0,
    explanation:
      "The critical path is the longest sequence: A → B → C (2+5+3 = 10 minutes). Task D (A → D = 4 minutes) can run in parallel with B and C after A finishes, but it doesn't extend the critical path. Option C is incorrect—D doesn't depend on C. Option D misunderstands critical path (it's the longest path, not the shortest).",
  },
  {
    id: 'maas-decomp-mc-3',
    question:
      'What is the PRIMARY advantage of using hierarchical task decomposition (breaking tasks into subtasks recursively) over flat decomposition (all tasks at the same level)?',
    options: [
      'Hierarchical decomposition always executes faster',
      'Hierarchical decomposition reduces the number of tasks',
      'Hierarchical decomposition provides better abstraction and modularity',
      'Flat decomposition cannot handle dependencies between tasks',
    ],
    correctAnswer: 2,
    explanation:
      'Hierarchical decomposition provides abstraction—you can reason about high-level tasks without worrying about low-level details. It also enables modularity (subtasks can be reused). Option A is incorrect—execution speed depends on parallelism, not hierarchy. Option B is incorrect—total task count is the same or higher. Option D is incorrect—flat decomposition can handle dependencies via a dependency graph.',
  },
  {
    id: 'maas-decomp-mc-4',
    question:
      'A planning agent decomposes "Build a REST API" into tasks. Which decomposition is MOST appropriate?',
    options: [
      '["Build API",]—keep it simple, one task',
      '["Design schema", "Implement endpoints", "Write tests", "Deploy",]—sequential stages',
      '["Write POST /users", "Write GET /users", "Write DELETE /users", ...]—one task per endpoint',
      '["Install Python", "Open VSCode", "Create file", "Write code", ...]—very granular steps',
    ],
    correctAnswer: 1,
    explanation:
      'Option B provides balanced granularity—high-level stages that can be further decomposed. Option A is too coarse (no actionable subtasks). Option C is too fine-grained for initial decomposition (endpoint-level tasks are subtasks of "Implement endpoints"). Option D is overly granular (tool-specific steps that should be implicit).',
  },
  {
    id: 'maas-decomp-mc-5',
    question:
      'Dynamic replanning is triggered when an agent detects that the current plan cannot be completed. What is a common PITFALL of dynamic replanning?',
    options: [
      'Replanning is too slow, delaying the workflow',
      'Replanning can enter an infinite loop if the plan keeps failing',
      'Replanning requires human approval for every change',
      'Replanning discards all progress and starts from scratch',
    ],
    correctAnswer: 1,
    explanation:
      "Infinite loops are a key risk: Task fails → Replan → New task fails → Replan → ... This happens when the underlying issue (e.g., missing API key) isn't fixed. Mitigation: Limit replanning attempts (e.g., max 3 replans) and require human intervention if exceeded. Option A can be a concern but is secondary. Options C and D are design choices, not inherent pitfalls.",
  },
];
