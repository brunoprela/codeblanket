import { MultipleChoiceQuestion } from '../../../types';

export const definingFunctionsToolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question:
      'What is the purpose of the "description" field in a function schema?',
    options: [
      'It is only for human documentation',
      'It helps the LLM decide when and how to use the function',
      'It is used for API versioning',
      'It has no functional purpose',
    ],
    correctAnswer: 1,
    explanation:
      'The description is crucial - the LLM uses it to understand when the function should be called and what it does. A good description significantly improves function selection accuracy.',
  },
  {
    id: 'fcfc-mc-2',
    question:
      'In JSON Schema for function parameters, what does the "required" array specify?',
    options: [
      'Parameters that must have default values',
      'Parameters that must be provided by the LLM',
      'Parameters that are validated',
      'Parameters that are most important',
    ],
    correctAnswer: 1,
    explanation:
      'The "required" array lists parameters that must be provided when calling the function. Parameters not in this array are optional and should have reasonable defaults in your implementation.',
  },
  {
    id: 'fcfc-mc-3',
    question:
      'What is the benefit of using enum types for function parameters?',
    options: [
      'It makes the function execute faster',
      'It reduces token usage',
      'It constrains the LLM to only valid values',
      'It is required by the OpenAI API',
    ],
    correctAnswer: 2,
    explanation:
      'Enums restrict the LLM to only choosing from predefined valid values, preventing invalid inputs and reducing errors.',
  },
  {
    id: 'fcfc-mc-4',
    question:
      'When using Pydantic to generate function schemas, what is the main advantage?',
    options: [
      'Pydantic schemas are smaller',
      'Automatic validation and type safety',
      'Pydantic works with all LLM providers',
      'It makes functions execute asynchronously',
    ],
    correctAnswer: 1,
    explanation:
      'Pydantic provides automatic validation of function arguments against the schema and ensures type safety, catching errors before execution.',
  },
  {
    id: 'fcfc-mc-5',
    question:
      'What should you include in parameter descriptions to help LLMs provide correct arguments?',
    options: [
      'Only the parameter type',
      'Just the parameter name',
      'Type, format, examples, and constraints',
      'Only examples',
    ],
    correctAnswer: 2,
    explanation:
      'Comprehensive descriptions including type, expected format, concrete examples, and any constraints help the LLM generate correct arguments consistently.',
  },
];
