/**
 * Multiple choice questions for Multi-Agent Architecture Fundamentals section
 */

export const multiagentarchitecturefundamentalsMultipleChoice = [
  {
    id: 'maas-arch-mc-1',
    question:
      'In a hierarchical multi-agent system with a Coordinator agent and three Worker agents (Research, Code, Test), what is the PRIMARY advantage of this architecture over a flat peer-to-peer design?',
    options: [
      'The Coordinator can handle all LLM calls, reducing costs by centralizing API usage',
      'Worker agents never need to communicate with each other, simplifying the system',
      'The Coordinator provides centralized task decomposition and orchestration, preventing conflicts',
      'Hierarchical systems always execute faster because tasks run in parallel',
    ],
    correctAnswer: 2,
    explanation:
      'The primary advantage of hierarchical architecture is centralized orchestration. The Coordinator decomposes tasks, assigns work to specialized Workers, and manages dependencies. This prevents conflicts (e.g., two agents trying to write the same file) and ensures coherent workflows. Option A is incorrect—Worker agents still make their own LLM calls. Option B is incorrect—Workers often do need to communicate (Coordinator facilitates this). Option D is incorrect—parallelism is possible in both architectures; hierarchy provides structure, not necessarily speed.',
  },
  {
    id: 'maas-arch-mc-2',
    question:
      "You're building a multi-agent system where agents share a conversation history that grows to 50K tokens after 20 turns. Which strategy is MOST effective for managing context window limits while preserving critical information?",
    options: [
      'Truncate to most recent 10K tokens and discard older messages',
      'Compress older messages using summarization, keep recent messages full',
      'Store everything in a vector database and retrieve relevant chunks per agent',
      'Restart with empty context every 10 turns to stay within limits',
    ],
    correctAnswer: 1,
    explanation:
      'Compressing older messages via summarization is most effective. Recent context is kept detailed (for immediate coherence), while older messages are condensed to key facts. This balances context preservation with token limits. Option A loses important early context (e.g., requirements). Option C (RAG) can work but adds latency and complexity; summarization is simpler for conversation history. Option D loses continuity—agents would "forget" everything mid-conversation.',
  },
  {
    id: 'maas-arch-mc-3',
    question:
      'In a multi-agent system, Agent A passes a "user_preferences" object to Agent B, but Agent B expects a "config" object with different field names. What is the BEST design pattern to handle this mismatch?',
    options: [
      'Modify Agent A to output "config" format that Agent B expects',
      'Modify Agent B to accept "user_preferences" format from Agent A',
      'Introduce a Translator/Adapter that converts between formats',
      'Store both formats in shared state and let each agent read its preferred format',
    ],
    correctAnswer: 2,
    explanation:
      "The Adapter/Translator pattern is best for decoupling agents. This allows Agent A and Agent B to maintain their own interfaces without depending on each other. Option A and B tightly couple the agents—changing one breaks the other. Option D duplicates data and doesn't solve the core problem (agents still need to know about each other's formats). The Adapter pattern enables independent evolution of agents.",
  },
  {
    id: 'maas-arch-mc-4',
    question:
      'What is the MAIN reason why agent-to-agent communication should use structured schemas (like Pydantic models or JSON Schema) rather than free-form natural language?',
    options: [
      'Structured schemas reduce token usage, lowering LLM API costs',
      'LLMs cannot reliably parse natural language from other LLMs',
      'Structured schemas enable validation, versioning, and type safety',
      'Natural language communication is slower than structured data',
    ],
    correctAnswer: 2,
    explanation:
      "Structured schemas enable validation (catch errors early), versioning (handle format changes gracefully), and type safety (prevent runtime errors). Option A is minor—token savings are not the main reason. Option B is incorrect—LLMs can parse natural language, but it's unreliable (ambiguity, formatting inconsistencies). Option D conflates speed with reliability; the issue is correctness, not latency.",
  },
  {
    id: 'maas-arch-mc-5',
    question:
      'A multi-agent system has 5 agents that can run in parallel, but Agent E depends on outputs from Agents A, B, and C. Using a dependency graph, what is the MINIMUM number of sequential execution stages needed?',
    options: [
      '1 stage (all agents run in parallel)',
      '2 stages (A,B,C,D in stage 1; E in stage 2)',
      '3 stages (one per dependent agent)',
      '5 stages (one per agent)',
    ],
    correctAnswer: 1,
    explanation:
      'The minimum is 2 stages. Stage 1: Run A, B, C, D in parallel (D has no dependencies). Stage 2: Run E after A, B, C complete. The dependency graph shows E cannot start until its dependencies finish, requiring at least 2 stages. Options C and D are inefficient—no need for more stages than necessary. Option A is impossible—E cannot run until A, B, C finish.',
  },
];
