export const buildingAgenticSystemQuiz = [
  {
    id: 'q1',
    question:
      'Design the architecture for a production-ready agentic system that can handle complex multi-step tasks. What components are essential, and how would they interact?',
    sampleAnswer: `A production agent architecture requires several key components: planning module (breaks goals into steps), execution engine (runs tools in appropriate order), memory system (maintains context and state), tool registry (manages available tools), observability layer (tracks execution and costs), and error handling system (recovers from failures). Components interact through well-defined interfaces with the planning module creating execution plans, the execution engine orchestrating tool calls, memory providing context for decisions, and observability monitoring everything. Include feedback loops for learning and adaptation, human-in-the-loop capabilities for critical decisions, and safety guardrails to prevent harmful actions.`,
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
      'Explain how you would implement human-in-the-loop approval for critical agent actions. What actions should require approval, and how would you handle timeout and async approval workflows?',
    sampleAnswer: `Human-in-the-loop requires identifying high-risk actions (data deletion, financial transactions, external communications, system changes), implementing approval workflows (synchronous for urgent, asynchronous for batch), and handling timeouts gracefully. Design approval UI showing full context of proposed action, expected impact, and alternatives. Support multi-level approvals for different risk levels. Handle timeout scenarios with default actions or cancellation. Implement approval queuing for batch operations. Log all approvals and rejections for audit. Allow approval delegation and escalation. Provide approval analytics to optimize which actions need human review. Consider progressive automation where frequently approved actions become automatic over time.`,
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
      'Compare single-agent vs multi-agent architectures. When would you choose each approach, and how would you coordinate multiple specialized agents?',
    sampleAnswer: `Single-agent systems are simpler to build, easier to debug, have clear decision chains, and work well for focused domains with moderate complexity. Multi-agent systems excel at complex domains requiring specialization, parallel task execution, separation of concerns, and simulating team collaboration. Choose single-agent for straightforward workflows and when context fits in one conversation. Choose multi-agent when tasks require distinct expertise, parallel execution improves performance, or different agents need different capabilities/models. Coordinate agents through a coordinator/orchestrator that routes tasks, manages communication, resolves conflicts, and aggregates results. Use message passing or shared state for communication. Implement consensus mechanisms for conflicting outputs.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
];
