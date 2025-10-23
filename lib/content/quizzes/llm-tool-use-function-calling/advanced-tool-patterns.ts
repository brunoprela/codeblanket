export const advancedToolPatternsQuiz = [
  {
    id: 'q1',
    question:
      'Explain how tool chaining differs from traditional sequential execution. Design a system that allows LLMs to chain tools intelligently based on intermediate results.',
    sampleAnswer: `Tool chaining enables dynamic workflows where tool selection depends on previous results, unlike fixed sequential execution. The system should maintain shared state across tool calls, allow tools to access previous results, support conditional execution based on outcomes, and enable parallel execution of independent tools. Implement using a workflow engine that tracks dependencies, manages state, and coordinates execution. The LLM should reason about which tool to call next based on current context and goals, not follow a predetermined sequence. Include error recovery and alternative paths when tools fail.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q2',
    question:
      'Discuss the security implications of meta-tools that can create or modify other tools. How would you implement this safely in production?',
    sampleAnswer: `Meta-tools pose significant security risks including arbitrary code execution, privilege escalation, and system compromise. Safe implementation requires multiple layers: strict permission systems (only admins can create tools), code validation (AST parsing for dangerous patterns), sandboxed execution (Docker or restricted environments), mandatory code review for all generated tools, audit logging of all tool modifications, and rate limiting on tool creation. Never execute generated code directly; always validate and sandbox. Implement tool versioning and rollback capabilities. Consider read-only execution for generated tools initially. Have incident response plans for compromised tools.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a tool composition system that allows building complex tools from simpler primitives. How would you handle error propagation, state management, and testing?',
    sampleAnswer: `Tool composition should follow functional programming principles where tools are composable units. Implement using decorators or wrapper patterns that add functionality (retry logic, caching, logging) to base tools. Handle errors at each level with clear propagation rules and recovery strategies. Maintain immutable state or use copy-on-write to prevent side effects. For testing, unit test each primitive tool independently, integration test composed tools, and property test composition rules. Include rollback mechanisms for failures. Support both sequential and parallel composition. Document composition patterns and provide examples.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
];
