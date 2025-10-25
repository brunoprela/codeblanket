/**
 * Quiz questions for Task Decomposition & Planning section
 */

export const taskdecompositionplanningQuiz = [
  {
    id: 'maas-decomp-q-1',
    question:
      'You need to break down the task "Build an e-commerce website" for a multi-agent system. Create a hierarchical task decomposition with at least 3 levels, identify dependencies between tasks, and determine which tasks can run in parallel.',
    hint: 'Think about what must happen first (requirements) vs what can happen simultaneously (frontend and backend).',
    sampleAnswer:
      '**Level 1 - High Level Tasks:** (1) Requirements & Planning, (2) Design, (3) Development, (4) Testing, (5) Deployment. **Level 2 - Break Down Each:** (1.1) Requirements: Gather user stories, (1.2) Requirements: Define technical requirements, (1.3) Requirements: Create feature list. (2.1) Design: Database schema, (2.2) Design: API design, (2.3) Design: UI/UX mockups. (3.1) Development: Backend implementation, (3.2) Development: Frontend implementation, (3.3) Development: Payment integration. (4.1) Testing: Unit tests, (4.2) Testing: Integration tests, (4.3) Testing: User acceptance testing. (5.1) Deployment: Setup infrastructure, (5.2) Deployment: Deploy application, (5.3) Deployment: Configure monitoring. **Level 3 - Backend Detail:** (3.1.1) Backend: User authentication, (3.1.2) Backend: Product catalog, (3.1.3) Backend: Shopping cart, (3.1.4) Backend: Order processing. **Dependencies:** (1) → (2): Design needs requirements. (2) → (3): Development needs design. (3) → (4): Testing needs implementation. (4) → (5): Deployment needs passing tests. Within level 2: (2.1) Database must be designed before (3.1) Backend can start. (2.2) API design must be done before frontend and backend connect. **Parallelization Opportunities:** Within Requirements: 1.1, 1.2, 1.3 can happen in parallel (different people gathering different info). Within Design: 2.1, 2.3 can be parallel (DB and UI design). 2.2 needs 2.1 first. Within Development: 3.1 and 3.2 can be highly parallel after API design done. 3.1.1, 3.1.2, 3.1.3, 3.1.4 can be parallel (different backend features). Within Testing: 4.1 and 4.2 can overlap. **Optimal Execution Plan:** Stage 1: 1.1, 1.2, 1.3 (parallel) → combine results. Stage 2: 2.1, 2.3 (parallel) → then 2.2. Stage 3: 3.1.x tasks (parallel backend features), 3.2 (frontend), 3.3 (payment). Stage 4: 4.1 (parallel with development), then 4.2, then 4.3. Stage 5: 5.1, then 5.2, then 5.3. **Why This Structure:** Top-down decomposition is natural. Three levels gives enough detail without overwhelming. Clear dependencies prevent blocked agents. Parallelization opportunities identified cut total time by ~50%.',
    keyPoints: [
      'Hierarchical decomposition: general → specific across 3+ levels',
      'Identify strict dependencies (design before implementation)',
      'Find parallelization opportunities within each stage',
      'Balance granularity: too coarse wastes parallelism, too fine adds overhead',
    ],
  },
  {
    id: 'maas-decomp-q-2',
    question:
      'Your task planning system keeps creating circular dependencies (Task A depends on B, B depends on C, C depends on A). Design an algorithm to detect cycles and explain how you would automatically fix them.',
    hint: 'Think about graph traversal and how to identify problematic dependency chains.',
    sampleAnswer:
      '**Cycle Detection Algorithm (DFS-based):** Use Depth-First Search with visited set and recursion stack. (1) For each task: Mark as "visiting" (in current DFS path). Visit all dependencies. If dependency is "visiting" → cycle found! After visiting all dependencies, mark as "visited". (2) Code: visited = set(), rec_stack = set(). For each task: if not in visited: if detect_cycle (task, visited, rec_stack): return "Cycle found". **Example:** Tasks: A→B, B→C, C→A. Start at A: Visit A (rec_stack: {A}). Visit B (rec_stack: {A,B}). Visit C (rec_stack: {A,B,C}). Visit A - already in rec_stack! Cycle: C→A→B→C. **Automatic Fix Strategies:** (1) **Break Weakest Link:** Analyze each dependency in cycle, identify weakest (least critical). Remove it. Example: If C→A is "nice to have" but A→B and B→C are essential, remove C→A. How to determine: Ask LLM to rank dependency criticality. (2) **Introduce Intermediate Task:** Split one task to break cycle. Example: Split C into C1 (doesn\'t need A) and C2 (needs A output). New: A→B, B→C1, C1→C2, C2→A. No longer cyclic if C2→A dependency is removed or C2 happens after A. (3) **Reverse Dependency:** Sometimes dependency is backwards. Example: C depends on A, but really A should depend on C. Ask LLM: "Does C need A\'s output, or does A need C\'s output?" Flip if wrong. (4) **Merge Tasks:** If A and B are tightly coupled in cycle, merge them into one task AB. AB internally handles what was A and B. (5) **Temporal Separation:** Allow task to run with partial input. Example: C starts with assumption about A, runs, then A refines C\'s input, C re-runs. **Best Approach:** (1) Detect cycle with DFS. (2) Extract cycle path (e.g., [A, B, C, A]). (3) Use LLM to analyze: "These tasks have circular dependencies: A→B→C→A. Which dependency is least essential?" (4) Remove weakest or use Intermediate Task strategy. (5) Re-run cycle detection to verify fix. **Prevention:** When adding new dependency, immediately run cycle check. Reject dependency if it creates cycle. Prompt agents to avoid circular planning: "Do not create dependencies where task A needs B and B needs A."',
    keyPoints: [
      'Use DFS with recursion stack to detect cycles',
      'Break cycles by removing weakest dependency',
      'Consider splitting tasks or reversing incorrect dependencies',
      'Prevent cycles by checking before adding each dependency',
    ],
  },
  {
    id: 'maas-decomp-q-3',
    question:
      'Design an adaptive task decomposition system that adjusts granularity based on agent expertise. How would you decompose "implement user authentication" differently for a beginner agent vs an expert agent?',
    hint: 'Consider what level of detail and guidance each agent needs.',
    sampleAnswer:
      "**Beginner Agent (Fine-Grained):** Needs detailed, step-by-step tasks. **Decomposition:** (1.1) Research authentication methods (OAuth, JWT, sessions) - 30 min. (1.2) Choose authentication method based on requirements - 15 min. (1.3) Set up database user table (id, username, password_hash, email) - 20 min. (1.4) Install bcrypt library for password hashing - 5 min. (1.5) Implement user registration endpoint: (1.5.1) Validate input (email format, password strength), (1.5.2) Hash password using bcrypt, (1.5.3) Store in database, (1.5.4) Return success/error. (1.6) Implement login endpoint: (1.6.1) Look up user by email, (1.6.2) Verify password with bcrypt, (1.6.3) Generate JWT token, (1.6.4) Return token. (1.7) Implement middleware to verify JWT on protected routes. (1.8) Write tests for registration. (1.9) Write tests for login. (1.10) Test full authentication flow. **Why:** Beginner needs each step spelled out. Prevents getting stuck. Includes specific technologies (bcrypt, JWT). Time estimates help. ~10 small tasks, each 5-30 minutes. **Expert Agent (Coarse-Grained):** Trusts expert to know details. **Decomposition:** (1) Design authentication system architecture - 30 min. (2) Implement user authentication (registration, login, JWT, password hashing) - 2 hours. (3) Add authentication middleware for protected routes - 30 min. (4) Implement comprehensive test suite - 1 hour. **Why:** Expert knows how to hash passwords, doesn't need step-by-step. Can make technology choices. Larger tasks (hours vs minutes). Only 4 high-level tasks. **Adaptive System Design:** (1) **Assess Expertise:** Track agent's history. Calculate success rate on similar tasks. Beginner: <70% success. Intermediate: 70-85%. Expert: >85%. (2) **Choose Granularity:** Beginner → fine (5-30 min tasks). Intermediate → medium (30-60 min tasks). Expert → coarse (1-3 hour tasks). (3) **Adjust Over Time:** Start new agent at beginner level. As success rate improves, graduate to coarser tasks. Example: After 10 successful fine-grained auth tasks, try medium granularity. (4) **Provide Appropriate Context:** Beginner: Include technology choices, code examples, common pitfalls. Expert: Just requirements, let them design. (5) **Monitoring:** If agent at expert level starts failing, automatically increase granularity. If beginner succeeds consistently, suggest advancement. **Implementation:** Store agent expertise in AgentProfile {expertise_level, success_history}. Decomposition function: decompose (task, agent_profile) → if beginner: fine_grain (task) elif expert: coarse_grain (task). **Benefits:** Beginners get guidance they need. Experts aren't micromanaged. System adapts as agents improve. Optimal use of agent capabilities.",
    keyPoints: [
      'Beginners need fine-grained tasks with explicit instructions',
      'Experts can handle coarse-grained tasks with autonomy',
      'Track success rate to determine current expertise level',
      'Dynamically adjust granularity as agent improves over time',
    ],
  },
];
