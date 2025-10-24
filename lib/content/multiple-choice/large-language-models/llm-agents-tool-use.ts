export const llmAgentsToolUseMC = {
  title: 'LLM Agents & Tool Use Quiz',
  id: 'llm-agents-tool-use-mc',
  sectionId: 'llm-agents-tool-use',
  questions: [
    {
      id: 1,
      question: 'What does "ReAct" stand for in the context of LLM agents?',
      options: [
        'React.js integration',
        'Reasoning and Acting',
        'Reactive agents',
        'Recurrent action',
      ],
      correctAnswer: 1,
      explanation:
        'ReAct (Reasoning + Acting) is a prompting pattern where the model alternates between reasoning (thinking about what to do) and acting (calling tools). This interleaving improves performance compared to reasoning-only or acting-only approaches.',
    },
    {
      id: 2,
      question:
        'What is a major risk of giving LLM agents unrestricted tool access?',
      options: [
        'Slower response times',
        'Higher costs due to multiple LLM calls',
        'Executing harmful actions or runaway behavior',
        'Reduced model accuracy',
      ],
      correctAnswer: 2,
      explanation:
        'Unrestricted tool access is dangerous—agents could make destructive API calls, infinite loops, or expensive operations. Production agents need: allowlisted tools, input validation, rate limits, human-in-the-loop for critical actions, and careful security design.',
    },
    {
      id: 3,
      question: 'In LLM agents, what is "hallucinated tool calling"?',
      options: [
        "Tools that don't exist in the codebase",
        'When the agent invokes a non-existent tool or uses incorrect parameters',
        'When tools return imaginary data',
        'When the agent creates new tools',
      ],
      correctAnswer: 1,
      explanation:
        "LLMs can hallucinate tool names or parameters that don't exist, just like they hallucinate facts. Using structured outputs (function calling APIs with schemas) and validation helps catch these errors before execution.",
    },
    {
      id: 4,
      question:
        'What is the typical architecture for an LLM agent with tool use?',
      options: [
        'Single LLM call with all tools embedded',
        'Loop: LLM decides action → Execute tool → Feed result back → Repeat until goal achieved',
        'Parallel execution of all possible tools',
        'Pre-computed tool selection before LLM call',
      ],
      correctAnswer: 1,
      explanation:
        'Agent loop: (1) LLM receives task and tool descriptions, (2) LLM outputs next action and tool to use, (3) Tool executes and returns observation, (4) Observation fed back to LLM, (5) Repeat until task complete. This is the ReAct pattern.',
    },
    {
      id: 5,
      question:
        'What is a key challenge in debugging LLM agents compared to traditional software?',
      options: [
        'Agents run too fast to debug',
        'Non-deterministic behavior makes issues hard to reproduce',
        'Too many tools to understand',
        'Lack of logging capabilities',
      ],
      correctAnswer: 1,
      explanation:
        'LLM non-determinism (even at temperature=0, models can change behavior across versions) makes agent bugs hard to reproduce. Solution: comprehensive logging/tracing of all decisions, prompts, and tool calls; deterministic testing modes; and extensive observability.',
    },
  ],
};
