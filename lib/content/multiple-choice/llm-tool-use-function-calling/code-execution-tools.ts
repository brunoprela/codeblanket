import { MultipleChoiceQuestion } from '../../../types';

export const codeExecutionToolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fcfc-mc-1',
    question:
      'What is the most secure approach for executing LLM-generated code?',
    options: [
      "Use Python\'s eval() function",
      'Run in Docker containers with resource limits and no network access',
      'Use threading for isolation',
      'Validate syntax before execution',
    ],
    correctAnswer: 1,
    explanation:
      'Docker containers provide OS-level isolation with configurable resource limits and network restrictions, offering the strongest security for code execution.',
  },
  {
    id: 'fcfc-mc-2',
    question:
      'Why should network access be disabled in code execution sandboxes?',
    options: [
      'To improve execution speed',
      'To prevent data exfiltration and external attacks',
      'To reduce memory usage',
      'To save API costs',
    ],
    correctAnswer: 1,
    explanation:
      'Disabling network access prevents malicious code from exfiltrating data, downloading additional malware, or attacking external systems.',
  },
  {
    id: 'fcfc-mc-3',
    question:
      'What is the purpose of setting CPU and memory limits in code execution?',
    options: [
      'To improve code quality',
      'To prevent resource exhaustion attacks like infinite loops or memory bombs',
      'To make code run faster',
      'To reduce costs',
    ],
    correctAnswer: 1,
    explanation:
      'Resource limits prevent malicious or buggy code from consuming excessive resources, protecting the host system and other users.',
  },
  {
    id: 'fcfc-mc-4',
    question:
      'What should you do with code execution results before showing them to users?',
    options: [
      'Nothing, return them directly',
      'Sanitize output and check for sensitive information',
      'Encrypt the output',
      'Compress the output',
    ],
    correctAnswer: 1,
    explanation:
      "Output should be sanitized to remove potentially sensitive information and ensure it's safe to display, preventing information leakage.",
  },
  {
    id: 'fcfc-mc-5',
    question: 'What is RestrictedPython used for?',
    options: [
      'Compiling Python faster',
      'Running Python code with restricted capabilities for sandboxing',
      'Reducing memory usage',
      'Improving code quality',
    ],
    correctAnswer: 1,
    explanation:
      'RestrictedPython provides a restricted execution environment for Python code, limiting access to dangerous operations while remaining lightweight.',
  },
];
