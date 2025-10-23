import { MultipleChoiceQuestion } from '../../../types';

export const buildingToolLibrariesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question: 'What is the main purpose of a tool registry?',
    options: [
      'To store tool API keys',
      'To centrally manage and discover available tools',
      'To execute tools faster',
      'To reduce memory usage',
    ],
    correctAnswer: 1,
    explanation:
      'A tool registry serves as a central catalog of all available tools, enabling discovery, management, and organized access to tools throughout the application.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What is auto-registration of tools?',
    options: [
      'Tools that register themselves automatically when created',
      'Automatic discovery and registration of tools from modules',
      "Tools that don't need schemas",
      'Automatic tool versioning',
    ],
    correctAnswer: 1,
    explanation:
      'Auto-registration automatically discovers tools (often marked with decorators) from Python modules and registers them in the tool registry without manual intervention.',
  },
  {
    id: 'fcfc-mc-3',
    question: 'Why is tool categorization important?',
    options: [
      'It makes tools execute faster',
      'It helps with organization, discovery, and conditional availability',
      'It is required by LLM providers',
      'It reduces API costs',
    ],
    correctAnswer: 1,
    explanation:
      'Categorizing tools aids in organization, makes discovery easier, and enables conditional tool availability based on context or user permissions.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'What should a tool decorator do?',
    options: [
      'Make functions run faster',
      'Automatically generate tool metadata and schemas',
      'Remove the need for function parameters',
      'Cache tool results automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Tool decorators automatically generate metadata like schemas from function signatures and type hints, reducing boilerplate code.',
  },
  {
    id: 'fcfc-mc-5',
    question: 'What is the benefit of tool versioning?',
    options: [
      'It reduces memory usage',
      'It allows multiple versions to coexist and enables gradual migrations',
      'It makes tools execute faster',
      'It is required for compliance',
    ],
    correctAnswer: 1,
    explanation:
      'Tool versioning allows multiple versions of a tool to coexist, enabling gradual migrations, A/B testing, and backward compatibility.',
  },
];
