/**
 * Quiz questions for Multi-Agent Architecture Fundamentals section
 */

export const multiagentarchitecturefundamentalsQuiz = [
  {
    id: 'maas-arch-q-1',
    question:
      'Explain the fundamental trade-offs between using a single powerful agent versus multiple specialized agents for a complex task like building a web application. When would each approach be preferable?',
    hint: 'Consider factors like maintainability, debugging, cost, and failure modes.',
    sampleAnswer:
      "**Single Agent Approach:** Pros: Simpler coordination, single point of debugging, lower communication overhead, easier state management. Cons: More complex prompts, harder to optimize for specific tasks, single point of failure, difficult to parallelize. **Multi-Agent Approach:** Pros: Specialization (each agent optimized for its task), easier parallelization, better fault isolation (one agent failing doesn't break everything), more modular and maintainable. Cons: Coordination complexity, potential for communication failures, distributed state management, harder to debug interactions. **When to use Single Agent:** Simple sequential tasks, limited context requirements, prototype/proof-of-concept, budget constraints (fewer API calls). **When to use Multiple Agents:** Complex problem requiring different expertise (research + coding + testing), tasks that can be parallelized, need fault tolerance, long-running system that needs maintenance. **Real Example:** Building a web app - Single agent works for simple CRUD app, but multi-agent (designer, backend dev, frontend dev, tester) excels for complex applications with multiple concerns.",
    keyPoints: [
      'Single agent: simpler but monolithic, harder to optimize',
      'Multi-agent: specialized and parallel but complex coordination',
      'Choose based on task complexity, need for parallelization, fault tolerance',
      'Multi-agent scales better for production systems',
    ],
  },
  {
    id: 'maas-arch-q-2',
    question:
      'Design a multi-agent architecture for a content creation pipeline (research → write → review → publish). Compare sequential, parallel, and hierarchical patterns for this use case. Which would you choose and why?',
    hint: 'Think about dependencies between tasks and efficiency.',
    sampleAnswer:
      "**Sequential Pattern:** Research → Write → Review → Publish. Pros: Natural flow matches task dependencies, simple to implement, clear error handling. Cons: Slow (total time = sum of all steps), no parallelization. **Parallel Pattern:** All agents start simultaneously. Pros: Fastest possible. Cons: Doesn't work - writer needs research first, reviewer needs article. Can't parallelize tasks with dependencies. **Hierarchical Pattern:** Manager agent coordinates workers (researcher, writer, reviewer, publisher). Pros: Manager optimizes order, can add meta-tasks (quality checks), handles exceptions. Cons: Manager is bottleneck, added complexity. **Hybrid (Best Choice):** Sequential for dependent tasks (research → write → review) but parallel within stages. For example: (1) Research phase: multiple researchers work on different aspects in parallel, (2) Writing phase: writer uses all research, (3) Review phase: multiple reviewers check different aspects in parallel (grammar, facts, style), (4) Publish. **Why Hybrid:** Respects dependencies while maximizing parallelism. Research can be split into topics (history, current state, future trends) done in parallel. Review can split into concerns (accuracy, readability, SEO) done in parallel. This reduces total time while maintaining correctness.",
    keyPoints: [
      'Sequential works but is slow for independent sub-tasks',
      "Pure parallel doesn't work when dependencies exist",
      'Hierarchical adds flexibility but increases complexity',
      'Hybrid approach: sequential for dependencies, parallel within stages',
    ],
  },
  {
    id: 'maas-arch-q-3',
    question:
      "You're building a multi-agent system that frequently fails because agents share mutable state. Redesign the architecture using the principles of loose coupling and clear interfaces. What specific changes would you make?",
    hint: 'Consider how agents communicate and share data.',
    sampleAnswer:
      '**Problems with Shared Mutable State:** Race conditions (agents modifying same data simultaneously), hard to debug (which agent caused issue?), tight coupling (agents depend on internal state of others), difficult to test in isolation. **Redesign Principles:** (1) **Immutable Messages:** Agents communicate via immutable messages, not shared state. Each message is a snapshot of data. (2) **Explicit Interfaces:** Define clear input/output contracts for each agent. Agent exposes execute(input) -> output method. (3) **Message Passing:** Replace shared memory with message queues. Agent A sends message to Agent B\'s inbox. (4) **Private State:** Each agent maintains its own private state. No direct access to other agents\' state. (5) **State Snapshots:** If agents need to share state, they share immutable snapshots at specific points. **Example Redesign:** Before: All agents read/write to shared {tasks: [], results: {}}. After: Manager agent maintains state, workers receive task via message, return result via message. Workers have no shared state. **Specific Changes:** (1) Replace: shared_state["tasks",].append(task) with: message_queue.send(agent_name, Task(data)). (2) Add message bus for communication. (3) Each agent has private memory: AgentMemory class per agent. (4) Create AgentInterface protocol that all agents implement: async execute(input: AgentInput) -> AgentOutput. (5) Add state synchronization at checkpoints if needed: StateManager.snapshot() creates immutable copy. **Benefits:** No race conditions, easier debugging (trace messages), agents can be tested independently, can replace agents without affecting others.',
    keyPoints: [
      'Shared mutable state causes race conditions and tight coupling',
      'Use message passing instead of shared state',
      'Define explicit interfaces for all agents',
      'Each agent maintains private state, shares via immutable messages',
    ],
  },
];
