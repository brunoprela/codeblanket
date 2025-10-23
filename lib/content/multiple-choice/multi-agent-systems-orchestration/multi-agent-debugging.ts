/**
 * Multiple choice questions for Multi-Agent Debugging section
 */

export const multiagentdebuggingMultipleChoice = [
  {
    id: 'maas-debug-mc-1',
    question:
      'A multi-agent workflow fails at Agent C. The logs show Agent A and B completed successfully. What is the FIRST debugging step?',
    options: [
      "Re-run the entire workflow from Agent A to see if it's a transient error",
      "Examine Agent C's input (output from Agent B) to verify it's correct",
      "Check if Agent C's LLM API key is valid",
      "Immediately modify Agent C's prompt and retry",
    ],
    correctAnswer: 1,
    explanation:
      "Start by validating inputs—Agent C might be failing because Agent B's output is malformed or unexpected. This isolates the problem. Option A can mask the issue if it succeeds on retry. Option C is too specific (could be the issue, but check inputs first). Option D is premature—understand the problem before making changes.",
  },
  {
    id: 'maas-debug-mc-2',
    question:
      'You implement distributed tracing with trace IDs. A workflow spans 5 agents across 3 servers. What is the MAIN benefit of trace IDs?',
    options: [
      'Trace IDs improve performance by caching agent outputs',
      'Trace IDs allow you to correlate logs from all agents involved in one workflow',
      'Trace IDs prevent agents from executing the same task twice',
      'Trace IDs automatically rollback failed workflows',
    ],
    correctAnswer: 1,
    explanation:
      "Trace IDs enable log correlation—you can filter logs by trace ID to see the entire workflow's execution across all agents and servers. This is crucial for debugging distributed systems. Options A, C, and D are unrelated to the purpose of trace IDs.",
  },
  {
    id: 'maas-debug-mc-3',
    question:
      'An agent intermittently fails with "timeout error" (10% of executions). Other agents succeed. What is the MOST likely cause?',
    options: [
      "The agent's code has a bug that causes random failures",
      "The agent's LLM API is experiencing intermittent slowness",
      'The agent is running out of memory',
      "The agent's prompt is too long",
    ],
    correctAnswer: 1,
    explanation:
      'Intermittent timeout errors typically indicate external dependency issues—here, the LLM API is sometimes slow. Option A (bug) would likely be deterministic, not intermittent. Option C (memory) would cause OOM errors, not timeouts. Option D (long prompt) would fail consistently, not intermittently.',
  },
  {
    id: 'maas-debug-mc-4',
    question:
      'You add verbose logging to an agent: Every LLM call, input, output, and intermediate step is logged. What is the PRIMARY tradeoff?',
    options: [
      'Verbose logging slows down agent execution significantly',
      'Verbose logging increases log storage costs and makes logs harder to search',
      'Verbose logging exposes sensitive data in logs',
      'Verbose logging reduces the accuracy of agent outputs',
    ],
    correctAnswer: 1,
    explanation:
      "The main tradeoff is log volume—verbose logging generates massive logs, increasing storage costs and making it harder to find relevant information. Option A is a concern but usually minor (logging is fast). Option C is important (PII) but not the primary tradeoff. Option D is incorrect—logging doesn't affect agent accuracy.",
  },
  {
    id: 'maas-debug-mc-5',
    question:
      'A workflow visualization shows Agent A → Agent B → Agent C, but Agent B is skipped in execution. What is the MOST likely cause?',
    options: [
      'Agent B crashed before execution started',
      "A conditional edge bypassed Agent B based on Agent A's output",
      'Agent B is disabled in the configuration',
      'The workflow visualization is incorrect',
    ],
    correctAnswer: 1,
    explanation:
      'Conditional routing can skip nodes based on state. For example, if Agent A outputs "quality=high", a conditional edge might skip Agent B (refinement) and go directly to Agent C. Option A would show an error. Option C is possible but less common. Option D is unlikely—visualizations usually reflect the graph structure.',
  },
];
