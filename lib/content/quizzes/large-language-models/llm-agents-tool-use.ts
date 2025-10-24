export const llmAgentsToolUseQuiz = {
  title: 'LLM Agents & Tool Use Discussion',
  id: 'llm-agents-tool-use-quiz',
  sectionId: 'llm-agents-tool-use',
  questions: [
    {
      id: 1,
      question:
        'The ReAct (Reasoning + Acting) pattern enables LLMs to iteratively reason and take actions using tools. Explain how ReAct works and why interleaving reasoning with actions improves performance compared to separate reasoning or action-only approaches. What are the failure modes, and how do you make ReAct agents more reliable?',
      expectedAnswer:
        'Should cover: thought-action-observation loop, reasoning guiding tool selection, observations informing next steps, comparison to chain-of-thought (pure reasoning) and function calling (pure actions), error recovery through reasoning, hallucinated actions problem, maximum iteration limits to prevent loops, action validation before execution, early stopping on goal achievement, prompt engineering for better reasoning traces, and monitoring agent behavior for debugging.',
    },
    {
      id: 2,
      question:
        'Discuss the challenges in building production LLM agents: reliability, cost, latency, security, and observability. LLM agents can make unpredictable decisions—how do you constrain behavior while maintaining capability? What guardrails and safety mechanisms are necessary?',
      expectedAnswer:
        'Should discuss: non-determinism in agent behavior, cost multiplication from multiple LLM calls, latency from sequential tool use, security risks from executing arbitrary actions, need for action allowlists, input validation on tool calls, rate limiting to prevent runaway costs, human-in-the-loop for critical actions, observability and tracing of agent decisions, structured outputs to reduce hallucinated tool calls, evaluation and testing strategies, rollback mechanisms, and balancing autonomy with safety.',
    },
    {
      id: 3,
      question:
        'Compare different agent frameworks: LangChain, LlamaIndex, AutoGPT, and custom implementations. What abstractions do these frameworks provide, and what are their limitations? When would you build a custom agent system versus using a framework? Discuss the tradeoffs in flexibility, complexity, and vendor lock-in.',
      expectedAnswer:
        "Should analyze: LangChain's comprehensive tooling but heavyweight abstractions, LlamaIndex's focus on data ingestion and indexing, AutoGPT's autonomous goal- driven approach and limitations, frameworks hiding important details, debugging difficulties with abstractions, version instability in fast- moving ecosystem, custom implementations offering control and simplicity, when frameworks add value (quick prototyping, standard patterns), when custom is better(production systems, specific requirements), and maintaining agent systems over time.",
    },
  ],
};
