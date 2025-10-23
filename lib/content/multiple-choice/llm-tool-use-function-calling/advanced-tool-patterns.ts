import { MultipleChoiceQuestion } from '../../../types';

export const advancedToolPatternsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question: 'What is tool chaining?',
    options: [
      'Calling tools in a fixed sequence',
      'Dynamically using output of one tool as input to another',
      'Calling multiple tools simultaneously',
      'Caching tool results',
    ],
    correctAnswer: 1,
    explanation:
      'Tool chaining involves dynamically using the output of one tool as input to subsequent tools, enabling complex workflows.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What is a composite tool?',
    options: [
      'A tool that executes faster',
      'A higher-level tool built from multiple simpler tools',
      'A tool with many parameters',
      'A cached tool',
    ],
    correctAnswer: 1,
    explanation:
      'Composite tools combine multiple simpler tools into a higher-level abstraction that accomplishes more complex tasks.',
  },
  {
    id: 'fcfc-mc-3',
    question: 'What is the main risk of meta-tools that create other tools?',
    options: [
      'They are slow',
      'Security risks from executing generated code',
      'They use too much memory',
      'They are difficult to test',
    ],
    correctAnswer: 1,
    explanation:
      'Meta-tools that execute generated code pose significant security risks and require careful sandboxing, validation, and access control.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'What is a tool factory?',
    options: [
      'A place where tools are stored',
      'A pattern for systematically generating related tools',
      'A tool that runs in production',
      'A tool testing framework',
    ],
    correctAnswer: 1,
    explanation:
      'Tool factories are patterns for systematically generating families of related tools, such as creating API tools from OpenAPI specifications.',
  },
  {
    id: 'fcfc-mc-5',
    question: 'When should tool availability be context-dependent?',
    options: [
      'Never',
      'Based on user permissions, budget limits, or system state',
      'Always show all tools',
      'Only for admin users',
    ],
    correctAnswer: 1,
    explanation:
      'Context-dependent tool availability ensures users only see tools they can use, respecting permissions, budgets, and system state.',
  },
];
