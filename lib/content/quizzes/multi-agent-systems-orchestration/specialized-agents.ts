/**
 * Quiz questions for Specialized Agents section
 */

export const specializedagentsQuiz = [
  {
    id: 'maas-spec-q-1',
    question:
      'Design specialized agents for a software development workflow: requirements gathering, architecture design, implementation, and testing. For each agent, define its system prompt, tools it needs, and how it should hand off to the next agent.',
    hint: 'Think about what makes each role distinct and what information each needs from previous steps.',
    sampleAnswer:
      '**1. Requirements Analyst Agent:** System Prompt: "You are an expert requirements analyst. Given a product idea, you extract clear, testable requirements and user stories. Format requirements as: FR-## for functional, NFR-## for non-functional." Tools: [web_search (research similar products), document_reader (read existing docs)]. Output: Structured requirements document {functional_requirements: [...], non_functional_requirements: [...], user_stories: [...]}. Handoff: Pass full requirements doc to Architect. **2. Architect Agent:** System Prompt: "You are a senior software architect. Given requirements, design system architecture including: components, data flow, technology stack, and key design decisions. Focus on scalability and maintainability." Tools: [diagram_generator, technology_analyzer]. Input: Requirements from previous agent. Output: Architecture document {components: [...], data_model: {...}, tech_stack: {...}, design_decisions: [...]}. Handoff: Pass architecture + requirements to Engineer. **3. Engineer Agent:** System Prompt: "You are an experienced software engineer. Given architecture and requirements, implement clean, well-tested code following best practices." Tools: [code_generator, file_writer, dependency_installer]. Input: Architecture + requirements. Output: Source code files {files: [...], dependencies: [...]}. Handoff: Pass code + requirements to Tester. **4. Tester Agent:** System Prompt: "You are a QA engineer. Given code and requirements, create comprehensive tests covering normal cases, edge cases, and error conditions." Tools: [test_generator, test_runner, code_analyzer]. Input: Code + requirements. Output: Test suite {tests: [...], coverage_report: {...}, issues: [...]}. Final Output: Complete tested application. **Key Design Principles:** Each agent has specialized knowledge, clear input/output contracts, tools specific to their role. Handoff includes all context needed (tester gets original requirements to verify against, not just code).',
    keyPoints: [
      'Each agent needs specialized system prompt defining its role',
      "Tools should match the agent's specific responsibilities",
      'Handoff must include all necessary context from previous steps',
      'Clear output format enables next agent to proceed confidently',
    ],
  },
  {
    id: 'maas-spec-q-2',
    question:
      'Your Reviewer agent consistently approves low-quality outputs. The problem is that its evaluation criteria are too lenient. How would you redesign the agent to be more critical? Consider prompt engineering, example-based learning, and multi-dimensional evaluation.',
    hint: 'Think about how to make evaluation more structured and comprehensive.',
    sampleAnswer:
      '**Problem Analysis:** Generic prompt like "Review this" is too vague. Agent defaults to being positive. No clear criteria means subjective evaluation. **Solution 1 - Explicit Criteria in Prompt:** Define specific dimensions to evaluate. Prompt: "Review this code on these dimensions: (1) Correctness: Does it work? Score 1-10. (2) Code Quality: Clean, readable, maintainable? Score 1-10. (3) Security: Any vulnerabilities? Score 1-10. (4) Performance: Efficient algorithms? Score 1-10. (5) Testing: Well tested? Score 1-10. Provide score for each dimension and overall score (average). Approve only if overall ≥ 8.0." **Solution 2 - Examples-Based:** Provide few-shot examples: "Good code example: [shows clean code] - Score: 9/10. Bad code example: [shows code with SQL injection] - Score: 3/10 (security issues). Now review: [new code]". **Solution 3 - Multi-Step Review:** Break review into multiple focused passes: Pass 1 (Correctness agent): Does it solve the problem? Pass 2 (Security agent): Any vulnerabilities? Pass 3 (Quality agent): Is it maintainable? Pass 4 (Synthesis agent): Combine findings. Each specialized agent is more critical in its domain. **Solution 4 - Checklist Approach:** Provide explicit checklist: "Review must check: □ No security vulnerabilities, □ Error handling present, □ Type hints on all functions, □ Docstrings present, □ No code duplication, □ Tests included. Approve only if ALL boxes checked." **Solution 5 - Negative Examples:** Train on failure cases: "This code FAILED because: no input validation. This code FAILED because: SQL injection. Be thorough in finding such issues." **Best Approach:** Combine #1 and #3 - Multi-dimensional evaluation with specialized agents. Have 3-4 focused agents each expert in one area, then synthesis agent combines scores. Require minimum score on each dimension (e.g., security must be ≥8, can\'t be compensated by high readability score).',
    keyPoints: [
      'Generic review prompts lead to lenient evaluations',
      'Define explicit evaluation criteria and scoring system',
      'Use multi-pass review with specialized focus per pass',
      'Provide examples of both good and bad outputs',
    ],
  },
  {
    id: 'maas-spec-q-3',
    question:
      'Compare the trade-offs of having one generalist agent that can handle multiple tasks versus having many specialized agents, one per task. Use concrete metrics: latency, cost, accuracy, and maintainability.',
    hint: 'Consider prompt size, model selection, and debugging difficulty.',
    sampleAnswer:
      '**Generalist Agent:** One agent handles research, writing, coding, testing. **Latency:** Slower per task. Large prompt (describes all capabilities) means more tokens to process. If tasks are sequential, still sum of all task times. If one generalist does all steps: 10s + 12s + 8s + 6s = 36s. **Cost:** Lower total (fewer API calls, one model instance), but each call more expensive due to large system prompt. If system prompt is 1000 tokens for all capabilities: 1000 + input per task. **Accuracy:** Lower. Generalist is "jack of all trades, master of none". Doesn\'t excel at any specific task. Harder to optimize prompt for all tasks simultaneously. **Maintainability:** Easier in one sense (one agent to update) but harder in another (complex prompt handles many cases, hard to debug). Changes to improve one capability might hurt another. **Specialized Agents:** Separate agents: Researcher, Writer, Coder, Tester. **Latency:** Can be much faster with parallelization. Independent tasks run in parallel. Even if sequential, each agent is faster (smaller, focused prompt). Specialized agents: 10s, 8s (parallel with research), 6s, 4s = 10s + 6s + 4s = 20s (if research and writing can overlap). **Cost:** Could be higher (more API calls) but each call is cheaper (smaller prompts). Focused system prompts: 200 tokens each. 4 agents × 200 = 800 < 1000. Can use cheaper models for simpler tasks (use GPT-3.5 for testing, GPT-4 only for architecture). **Accuracy:** Higher. Each agent optimized for its specific task. Researcher has examples of good research. Tester has testing best practices. Easier to improve one agent without affecting others. **Maintainability:** Better. Each agent is independent. Can debug one without affecting others. Can version agents separately. Can replace one agent (upgrade Tester without touching others). Clear boundaries. **Recommendation:** Specialized agents win on accuracy and maintainability. Win on latency if parallelization possible. Cost is roughly equal but specialized allows optimization (cheap model for simple tasks). Choose specialized for production systems needing high quality. Choose generalist only for prototypes or very simple workflows.',
    keyPoints: [
      'Generalist: simpler setup but lower accuracy and harder to optimize',
      'Specialized: better accuracy, easier to maintain, enables parallelization',
      'Specialized agents can use different models per task (cost optimization)',
      'Specialized agents are independently debuggable and updatable',
    ],
  },
];
