/**
 * Multiple choice questions for LangGraph for Agent Orchestration section
 */

export const langgraphagentorchestrationMultipleChoice = [
  {
    id: 'maas-langgraph-mc-1',
    question:
      'In LangGraph, what is the PRIMARY purpose of defining edges between nodes in the state graph?',
    options: [
      'Edges determine the visual layout of the graph in debugging tools',
      'Edges define the flow of execution and data between agent steps',
      'Edges specify which agents can communicate with each other',
      "Edges are optional decorations that don't affect execution",
    ],
    correctAnswer: 1,
    explanation:
      'Edges define the workflow—how execution flows from one node (agent/step) to another, and how data is passed. Option A is incorrect—visualization is a side benefit. Option C is partially true but too narrow. Option D is completely wrong—edges are fundamental to graph execution.',
  },
  {
    id: 'maas-langgraph-mc-2',
    question:
      'In LangGraph, a conditional edge checks if "code_quality > 0.8". If true, go to "deploy" node; if false, go to "refactor" node. The check is implemented as:',
    options: [
      'A node that runs before the conditional edge',
      'A function passed to `add_conditional_edges()` that returns the next node name',
      'A separate LLM call that decides the routing',
      'An edge label that LangGraph automatically evaluates',
    ],
    correctAnswer: 1,
    explanation:
      "Conditional edges in LangGraph use a routing function passed to `add_conditional_edges()`. This function receives the current state and returns the name of the next node. Options A, C, and D reflect misunderstandings of LangGraph's conditional routing API.",
  },
  {
    id: 'maas-langgraph-mc-3',
    question:
      'In LangGraph, what is the difference between a node and an edge?',
    options: [
      'Nodes are for human tasks, edges are for agent tasks',
      'Nodes represent actions/agents (where work happens), edges represent transitions (control flow)',
      'Nodes store data, edges process data',
      'Nodes and edges are interchangeable terms in LangGraph',
    ],
    correctAnswer: 1,
    explanation:
      'Nodes are where computation happens (agent execution, function calls), edges control flow between nodes. Option A is incorrect—both nodes and edges are agent-related. Option C confuses nodes with state. Option D is wrong—they have distinct meanings.',
  },
  {
    id: 'maas-langgraph-mc-4',
    question:
      'You implement a LangGraph workflow with a loop: Node A → Node B → Node A (repeat until condition met). What is the BEST way to prevent infinite loops?',
    options: [
      'LangGraph automatically prevents infinite loops',
      'Add a maximum iteration count in the conditional edge function',
      'Use a time-based timeout for the entire graph',
      'Loop workflows are not supported in LangGraph',
    ],
    correctAnswer: 1,
    explanation:
      'The developer must implement safeguards—typically a counter in the state that the conditional edge checks: `if state["iterations"] >= 5: return "end"`. Option A is incorrect—LangGraph doesn\'t have built-in infinite loop prevention. Option C (timeout) is a backup, not the primary solution. Option D is wrong—loops are supported.',
  },
  {
    id: 'maas-langgraph-mc-5',
    question:
      'In LangGraph, state is passed between nodes. If Node A outputs `{"result": "success"}` and Node B needs both Node A\'s result and the original user query, how is this typically handled?',
    options: [
      "Node B cannot access the original query—only the previous node's output",
      'The state object accumulates data: Node B receives `{"query": "...", "result": "success"}`',
      'Node B must make a separate API call to retrieve the original query',
      'LangGraph automatically stores all previous outputs in a history array',
    ],
    correctAnswer: 1,
    explanation:
      "LangGraph state is accumulated by default—each node can add/modify state fields, and all fields are passed forward. This allows Node B to access both the original query and Node A's result. Option A would be true for simple pipelines but not LangGraph. Options C and D are incorrect.",
  },
];
