import { MultipleChoiceQuestion } from '../../../types';

export const buildingAgenticSystemMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question: 'What are the core components of a production agentic system?',
    options: [
      'Just an LLM and some tools',
      'Planning module, execution engine, memory system, and observability',
      'Only a database',
      'Just function calling',
    ],
    correctAnswer: 1,
    explanation:
      'A complete agent needs planning (breaking down goals), execution (running tools), memory (maintaining context), and observability (monitoring performance).',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What is the purpose of human-in-the-loop in agentic systems?',
    options: [
      'To slow down execution',
      'To require approval for critical actions before execution',
      'To reduce costs',
      'To improve accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Human-in-the-loop provides oversight for critical actions (deletions, financial transactions) by requiring human approval before execution.',
  },
  {
    id: 'fcfc-mc-3',
    question: 'Why is memory important in agentic systems?',
    options: [
      'To reduce API calls',
      'To maintain context across interactions and learn from experience',
      'To make agents run faster',
      'To reduce costs',
    ],
    correctAnswer: 1,
    explanation:
      'Memory allows agents to maintain context across conversations, remember learned facts, and provide better, more contextual responses.',
  },
  {
    id: 'fcfc-mc-4',
    question:
      'What is the main advantage of multi-agent systems over single agents?',
    options: [
      'They are easier to build',
      'Specialization, parallel execution, and separation of concerns',
      'They use less memory',
      'They are cheaper',
    ],
    correctAnswer: 1,
    explanation:
      'Multi-agent systems allow specialization (agents with specific expertise), parallel task execution, and clear separation of concerns.',
  },
  {
    id: 'fcfc-mc-5',
    question: 'How should production agents handle maximum iteration limits?',
    options: [
      'Never set limits',
      'Set reasonable limits and provide useful feedback when reached',
      'Always fail silently',
      'Restart from the beginning',
    ],
    correctAnswer: 1,
    explanation:
      "Setting max iterations prevents infinite loops, and providing clear feedback when reached helps users understand why the task couldn't complete.",
  },
];
