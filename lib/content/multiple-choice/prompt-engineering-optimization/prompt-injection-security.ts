/**
 * Multiple choice questions for Prompt Injection & Security section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const promptinjectionsecurityMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'peo-security-mc-1',
    question: 'What is prompt injection?',
    options: [
      'Adding prompts to a database',
      'User input that manipulates prompts to bypass instructions',
      'Using multiple prompts at once',
      'Injecting variables into templates',
    ],
    correctAnswer: 1,
    explanation:
      "Prompt injection is when users craft inputs to manipulate the AI into bypassing instructions, revealing system prompts, or performing unintended actions. It's the SQL injection equivalent for AI systems.",
  },
  {
    id: 'peo-security-mc-2',
    question:
      'What is the purpose of delimiters like <<<BEGIN_INPUT>>> in prompts?',
    options: [
      'Make prompts look professional',
      'Separate instructions from user data to prevent injection',
      'Count tokens accurately',
      'Improve model performance',
    ],
    correctAnswer: 1,
    explanation:
      'Delimiters clearly separate system instructions from user input, marking user data as "data not commands". This makes injection harder by establishing clear boundaries that the model should respect.',
  },
  {
    id: 'peo-security-mc-3',
    question: 'What does input sanitization do?',
    options: [
      'Makes input look cleaner',
      'Removes or escapes dangerous patterns before processing',
      'Compresses input to save tokens',
      'Translates input to English',
    ],
    correctAnswer: 1,
    explanation:
      'Input sanitization removes or escapes dangerous patterns like "ignore previous instructions", special tokens, and injection attempts before sending to the LLM. It\'s a critical defense layer against attacks.',
  },
  {
    id: 'peo-security-mc-4',
    question: 'What is output validation checking for?',
    options: [
      'Grammar errors',
      'Sensitive information leakage (system prompts, API keys, PII)',
      'Response length',
      'Token count',
    ],
    correctAnswer: 1,
    explanation:
      "Output validation scans LLM responses for sensitive information before showing to users. It checks for system prompt leakage, API keys, PII, and other confidential data that shouldn't be exposed.",
  },
  {
    id: 'peo-security-mc-5',
    question: 'What is defense in depth for prompt security?',
    options: [
      'Using very long prompts',
      'Layering multiple security measures (delimiters, sanitization, detection, validation)',
      'Using multiple models',
      'Encrypting prompts',
    ],
    correctAnswer: 1,
    explanation:
      'Defense in depth means layering multiple security measures so if one fails, others still protect. Combine delimiters, input sanitization, injection detection, output validation, and monitoring for robust security.',
  },
];
