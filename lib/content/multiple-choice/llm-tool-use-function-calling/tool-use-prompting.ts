import { MultipleChoiceQuestion } from '../../../types';

export const toolUsePromptingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question:
      'What should a system prompt include to guide tool use effectively?',
    options: [
      'Only tool names',
      'Tool descriptions, usage guidelines, and examples',
      'Just error messages',
      'Only tool parameters',
    ],
    correctAnswer: 1,
    explanation:
      'Effective system prompts include comprehensive tool descriptions, clear guidelines on when to use each tool, and concrete examples of correct usage.',
  },
  {
    id: 'fcfc-mc-2',
    question: 'What is the purpose of few-shot examples in tool use prompts?',
    options: [
      'To reduce token usage',
      'To show the LLM correct patterns of tool usage',
      'To make responses faster',
      'To reduce API costs',
    ],
    correctAnswer: 1,
    explanation:
      'Few-shot examples demonstrate correct tool usage patterns, helping the LLM understand when and how to use tools in similar situations.',
  },
  {
    id: 'fcfc-mc-3',
    question:
      'How should you teach an LLM when to use tools versus answering directly?',
    options: [
      'Always force tool use',
      'Provide explicit guidelines distinguishing real-time/action needs from general knowledge',
      'Let the LLM guess',
      'Only allow tool use',
    ],
    correctAnswer: 1,
    explanation:
      'Explicit guidelines help the LLM understand when tools are needed (real-time data, actions) versus when it can answer from its training knowledge.',
  },
  {
    id: 'fcfc-mc-4',
    question: 'What is chain-of-thought prompting for tool use?',
    options: [
      'Calling tools in a sequence',
      'Prompting the LLM to reason explicitly before choosing tools',
      'Using multiple LLMs',
      'Caching tool results',
    ],
    correctAnswer: 1,
    explanation:
      'Chain-of-thought prompting asks the LLM to explicitly reason about what it needs to do before selecting tools, improving decision quality.',
  },
  {
    id: 'fcfc-mc-5',
    question:
      'How do you prevent an LLM from hallucinating function arguments?',
    options: [
      'Use shorter prompts',
      'Provide clear parameter descriptions with examples and constraints',
      'Only use simple functions',
      'Disable error messages',
    ],
    correctAnswer: 1,
    explanation:
      'Clear parameter descriptions with concrete examples and constraints help the LLM generate correct arguments instead of hallucinating values.',
  },
];
