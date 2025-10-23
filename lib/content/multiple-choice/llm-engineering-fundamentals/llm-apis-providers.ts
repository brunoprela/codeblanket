/**
 * Multiple choice questions for LLM APIs & Providers section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const llmapisprovidersMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which LLM provider offers the largest context window as of 2024?',
    options: [
      'OpenAI GPT-4 Turbo (128K tokens)',
      'Google Gemini Pro (32K tokens)',
      'Anthropic Claude 3 (200K tokens)',
      'Meta Llama 3 (8K tokens)',
    ],
    correctAnswer: 2,
    explanation:
      'Anthropic Claude 3 offers the largest context window at 200K tokens (~150K words), significantly larger than GPT-4 Turbo (128K), Gemini Pro (32K), or Llama 3 (8K). This makes Claude ideal for processing very long documents.',
  },
  {
    id: 'mc2',
    question:
      'What is the approximate cost difference between GPT-4 and GPT-3.5 Turbo per 1M tokens?',
    options: [
      'GPT-4 is 2x more expensive',
      'GPT-4 is 10x more expensive',
      'GPT-4 is 20-60x more expensive',
      'They cost the same',
    ],
    correctAnswer: 2,
    explanation:
      'GPT-4 costs $30-60 per 1M tokens while GPT-3.5 Turbo costs $0.50-1.50 per 1M tokens, making GPT-4 approximately 20-60x more expensive depending on input vs output tokens. This huge difference means GPT-3.5 should be preferred for tasks where quality difference is minimal.',
  },
  {
    id: 'mc3',
    question:
      'When making your first OpenAI API call, which of these is the MINIMUM required?',
    options: [
      'API key, model name, and messages array',
      'API key, model name, messages array, and temperature',
      'Only API key and prompt text',
      'API key, model name, temperature, and max_tokens',
    ],
    correctAnswer: 0,
    explanation:
      'The minimum required for an OpenAI chat completion is: API key (for authentication), model name (e.g., "gpt-3.5-turbo"), and messages array with at least one message. Temperature, max_tokens, and other parameters have defaults and are optional.',
  },
  {
    id: 'mc4',
    question:
      'What is the main advantage of using a unified client that supports multiple LLM providers?',
    options: [
      'It makes API calls faster',
      'It reduces token costs',
      'It allows easy switching between providers without code changes',
      'It improves model quality',
    ],
    correctAnswer: 2,
    explanation:
      "A unified client's main advantage is vendor flexibility - you can switch between OpenAI, Anthropic, Google, etc. with configuration changes rather than code rewrites. This protects against vendor lock-in and enables cost optimization by routing to different providers.",
  },
  {
    id: 'mc5',
    question:
      'Which model would be most cost-effective for simple data extraction from 1000s of documents daily?',
    options: [
      'GPT-4 Turbo for maximum accuracy',
      'GPT-3.5 Turbo for good quality at low cost',
      'Claude 3 Opus for best instruction following',
      'Use multiple models for redundancy',
    ],
    correctAnswer: 1,
    explanation:
      'For simple extraction tasks at high volume, GPT-3.5 Turbo is most cost-effective. It provides 90%+ quality of GPT-4 for extraction tasks at 1/20th the cost. At 1000s of requests daily, this cost difference is substantial. GPT-4 should be reserved for complex tasks where quality justifies the expense.',
  },
];
