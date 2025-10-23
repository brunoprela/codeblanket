import { MultipleChoiceQuestion } from '../../../types';

export const functionCallingWorkflowsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fcfc-mc-1',
      question:
        'What is the main difference between sequential and parallel function calling workflows?',
      options: [
        'Sequential is slower but more reliable',
        'Parallel executes independent function calls simultaneously',
        'Sequential uses less memory',
        'Parallel only works with GPT-4 Turbo',
      ],
      correctAnswer: 1,
      explanation:
        "Parallel workflows execute multiple independent function calls at the same time, significantly reducing total execution time when calls don't depend on each other.",
    },
    {
      id: 'fcfc-mc-2',
      question: 'In the ReAct pattern, what does "ReAct" stand for?',
      options: [
        'Retrieve and Act',
        'Reasoning and Acting',
        'React to errors',
        'Recursive Action',
      ],
      correctAnswer: 1,
      explanation:
        'ReAct stands for Reasoning and Acting - the pattern alternates between reasoning about what to do next and taking actions (calling functions).',
    },
    {
      id: 'fcfc-mc-3',
      question:
        'What is a key benefit of explicit state management in workflow systems?',
      options: [
        'It uses less memory',
        'It enables rollback and debugging',
        'It makes workflows execute faster',
        'It reduces API costs',
      ],
      correctAnswer: 1,
      explanation:
        'Explicit state management allows you to save checkpoints, rollback to previous states, and debug issues by inspecting state at any point in the workflow.',
    },
    {
      id: 'fcfc-mc-4',
      question: 'How should you prevent infinite loops in agentic systems?',
      options: [
        'Use shorter prompts',
        'Set max iterations and detect repetitive patterns',
        'Only use simple functions',
        'Disable error recovery',
      ],
      correctAnswer: 1,
      explanation:
        'Prevent loops by setting maximum iteration limits and detecting when the agent is calling the same functions repeatedly with the same arguments.',
    },
    {
      id: 'fcfc-mc-5',
      question: 'What is the purpose of a workflow orchestrator?',
      options: [
        'To execute functions faster',
        'To coordinate multiple function calls and manage state',
        'To reduce API costs',
        'To validate function schemas',
      ],
      correctAnswer: 1,
      explanation:
        'A workflow orchestrator manages the execution flow of multiple function calls, handles state between steps, and coordinates complex workflows with conditional logic and error handling.',
    },
  ];
