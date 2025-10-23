/**
 * Multiple choice questions for CrewAI & Agent Frameworks section
 */

export const crewaiagentframeworksMultipleChoice = [
  {
    id: 'maas-crewai-mc-1',
    question: 'In CrewAI, what is the PRIMARY role of a "Crew" object?',
    options: [
      'A Crew is a single agent with multiple tools',
      'A Crew coordinates multiple agents working together on tasks',
      'A Crew is a database for storing agent state',
      'A Crew is a monitoring tool for debugging agents',
    ],
    correctAnswer: 1,
    explanation:
      "A Crew in CrewAI is a collection of agents working together with defined roles, tasks, and coordination logic. Option A describes a single agent, not a crew. Options C and D are unrelated to CrewAI's design.",
  },
  {
    id: 'maas-crewai-mc-2',
    question:
      'In CrewAI, you define an agent with role="researcher", goal="find latest AI papers", and backstory="You are an expert researcher...". What is the PURPOSE of the backstory field?',
    options: [
      "Backstory is only for documentation and doesn't affect agent behavior",
      'Backstory provides context that influences how the agent approaches tasks (via its system prompt)',
      'Backstory is used to generate a unique ID for the agent',
      'Backstory determines which tools the agent can access',
    ],
    correctAnswer: 1,
    explanation:
      "The backstory is injected into the agent's system prompt, giving it personality and context that shapes its behavior. Option A underestimates its impact. Options C and D are incorrect—backstory doesn't affect IDs or tool access.",
  },
  {
    id: 'maas-crewai-mc-3',
    question:
      'In CrewAI, tasks are assigned to agents. A task has `async_execution=True`. What does this mean?',
    options: [
      'The task will use asynchronous Python (async/await)',
      'The task can execute in parallel with other async tasks',
      'The task requires human approval before execution',
      'The task uses non-blocking LLM API calls',
    ],
    correctAnswer: 1,
    explanation:
      "`async_execution=True` allows the task to run in parallel with other async tasks, rather than waiting for previous tasks to complete. Option A is about Python syntax, not CrewAI's execution model. Options C and D are unrelated to the async flag.",
  },
  {
    id: 'maas-crewai-mc-4',
    question:
      'You create a CrewAI agent with tools=[search_tool, calculator_tool]. The agent needs to multiply 2 numbers but calls search_tool instead. What is the MOST likely cause?',
    options: [
      'The calculator_tool is not properly registered with the agent',
      "The agent's LLM is not capable of using tools",
      "The calculator_tool's description is unclear or missing",
      'CrewAI randomly selects tools regardless of task',
    ],
    correctAnswer: 2,
    explanation:
      "Tool selection depends on tool descriptions—the LLM chooses based on the description's relevance to the task. If calculator_tool has a poor description, the LLM might choose the wrong tool. Option A would cause an error. Option B would mean no tools work. Option D is incorrect—CrewAI uses LLM-based tool selection.",
  },
  {
    id: 'maas-crewai-mc-5',
    question:
      'In CrewAI, what is the difference between `sequential` and `hierarchical` process types for a Crew?',
    options: [
      'Sequential runs tasks one at a time; hierarchical runs them in parallel',
      'Sequential has no manager; hierarchical has a manager agent that delegates tasks',
      'Sequential is faster; hierarchical is more reliable',
      'Sequential is for simple tasks; hierarchical is for complex tasks',
    ],
    correctAnswer: 1,
    explanation:
      'In sequential process, tasks execute in order without a manager. In hierarchical process, a manager agent orchestrates, delegates, and coordinates worker agents. Option A is incorrect—both can run tasks sequentially; the difference is management. Options C and D are too vague.',
  },
];
