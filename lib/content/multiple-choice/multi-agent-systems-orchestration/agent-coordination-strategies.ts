/**
 * Multiple choice questions for Agent Coordination Strategies section
 */

export const agentcoordinationstrategiesMultipleChoice = [
  {
    id: 'maas-coord-mc-1',
    question:
      'In a sequential coordination strategy, Agent A completes its task and passes results to Agent B. Agent B fails. What is the MAIN drawback of sequential execution in this scenario?',
    options: [
      "Agent A's work is wasted because Agent B failed",
      'Sequential execution is slower than parallel execution',
      'Agent B cannot be retried without re-running Agent A',
      "Sequential execution doesn't scale to more than 2 agents",
    ],
    correctAnswer: 1,
    explanation:
      "The main drawback is speed—sequential execution means agents run one after another, with no parallelism. If Agent B fails, you can retry just Agent B (Agent A's work is saved). Option A is incorrect—Agent A's output is preserved. Option C is incorrect—you can retry Agent B with Agent A's previous output. Option D is incorrect—sequential scales to any number of agents, just slowly.",
  },
  {
    id: 'maas-coord-mc-2',
    question:
      'A parallel coordination strategy runs 4 agents simultaneously. One agent finishes in 2 seconds, three agents take 10 seconds. What is the total execution time?',
    options: [
      '2 seconds (fastest agent)',
      '10 seconds (slowest agent)',
      '24 seconds (sum of all agents)',
      '6 seconds (average of all agents)',
    ],
    correctAnswer: 1,
    explanation:
      'Parallel execution time is determined by the slowest agent. All agents run simultaneously, so the workflow completes when the last agent finishes (10 seconds). Options A, C, and D represent common misconceptions about parallel execution.',
  },
  {
    id: 'maas-coord-mc-3',
    question:
      'You implement a voting mechanism where 5 agents vote on the best solution. 3 agents vote for Solution A, 2 for Solution B. Later you discover Solution B was actually correct. What is the PRIMARY issue with simple majority voting?',
    options: [
      'Majority voting is too slow because all agents must finish before deciding',
      'Majority voting is not always correct—popular choice ≠ correct choice',
      'Majority voting cannot handle ties (e.g., 2 vs 2 vs 1)',
      'Majority voting requires an odd number of agents to work properly',
    ],
    correctAnswer: 1,
    explanation:
      'Simple majority voting can be wrong—if three agents make the same mistake, the incorrect answer wins. This is especially problematic if agents are similar (same model, same prompt) and make correlated errors. Option A is a design tradeoff, not an inherent issue. Option C can be handled with tiebreaker rules. Option D is a implementation detail, not a fundamental flaw.',
  },
  {
    id: 'maas-coord-mc-4',
    question:
      'A hierarchical coordination has a Manager agent and 3 Worker agents. The Manager assigns tasks but one Worker is overloaded (5 tasks) while others are idle. What is the BEST solution?',
    options: [
      'Add more Worker agents to handle the load',
      'Implement work-stealing where idle Workers take tasks from the overloaded Worker',
      'The Manager should dynamically balance the workload across Workers',
      'Increase the timeout for the overloaded Worker',
    ],
    correctAnswer: 2,
    explanation:
      "The Manager should balance workload intelligently—assign tasks based on Worker availability and capacity. This is the Manager\'s responsibility in hierarchical coordination. Option B (work-stealing) is a peer-to-peer pattern, not hierarchical. Option A is premature—first optimize task distribution. Option D doesn't solve the imbalance, just delays failure.",
  },
  {
    id: 'maas-coord-mc-5',
    question:
      'In a consensus-based coordination, agents must agree on a decision before proceeding. 4 out of 5 agents agree, but the 5th agent is unresponsive. What is a common strategy to handle this?',
    options: [
      'Wait indefinitely for the 5th agent to respond',
      'Require unanimous agreement and abort if any agent is unresponsive',
      'Use a quorum (e.g., 3/5 agents) instead of unanimous consensus',
      'Exclude the 5th agent permanently from future decisions',
    ],
    correctAnswer: 2,
    explanation:
      "Quorum-based consensus (e.g., 3/5 or 4/5) is more practical than unanimous agreement. It tolerates unresponsive agents while still requiring strong agreement. Option A blocks the system indefinitely. Option B is too strict—one agent shouldn't block the entire workflow. Option D is too harsh—the agent might recover; temporary exclusion or retry is better.",
  },
];
