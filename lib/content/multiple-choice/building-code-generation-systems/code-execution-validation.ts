/**
 * Multiple choice questions for Code Execution & Validation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codeexecutionvalidationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcgs-execution-mc-1',
    question:
      'What is the most important security consideration when executing generated code?',
    options: [
      'Code must run fast',
      'Execute in isolated sandbox with resource limits',
      'Trust all generated code',
      'Only check syntax',
    ],
    correctAnswer: 1,
    explanation:
      'Execute in isolated sandbox (Docker, VM, WebAssembly) with strict resource limits (CPU, memory, network, filesystem). Generated code could be malicious or buggy. Never execute untrusted code directly.',
  },
  {
    id: 'bcgs-execution-mc-2',
    question: 'What resource limits should be applied to code execution?',
    options: [
      'No limits needed',
      'Only time limits',
      'Time, memory, CPU, network, and filesystem access limits',
      'Just memory limits',
    ],
    correctAnswer: 2,
    explanation:
      'Apply comprehensive limits: time (30s timeout), memory (512MB max), CPU (prevent crypto mining), network (block most external calls), filesystem (read-only or restricted). Prevents infinite loops, memory bombs, and abuse.',
  },
  {
    id: 'bcgs-execution-mc-3',
    question: 'How should execution results be validated?',
    options: [
      'Trust exit code only',
      'Check exit code, stdout, stderr, return value, and side effects',
      'Only check stdout',
      'No validation needed',
    ],
    correctAnswer: 1,
    explanation:
      'Validate comprehensively: exit code (0 = success), stdout (expected output), stderr (errors/warnings), return value, and side effects (files created, DB changes). Partial validation misses errors.',
  },
  {
    id: 'bcgs-execution-mc-4',
    question: 'What should happen when executed code times out?',
    options: [
      'Wait forever',
      'Kill process, report timeout error, show partial output',
      'Let it continue running',
      'Restart it',
    ],
    correctAnswer: 1,
    explanation:
      'Kill process after timeout (likely infinite loop or performance issue), report timeout error clearly, show partial output if any. This feedback helps LLM fix the issue (e.g., "optimize algorithm").',
  },
  {
    id: 'bcgs-execution-mc-5',
    question: 'How should dependencies be handled in code execution sandbox?',
    options: [
      'Allow any package installation',
      'Pre-install common packages, block arbitrary installs',
      'No packages allowed',
      'Let users install anything',
    ],
    correctAnswer: 1,
    explanation:
      'Pre-install common packages (numpy, pandas, requests) in sandbox. Block arbitrary installs (security risk, time consuming). For custom packages, use vetted allowlist. Balance functionality with security.',
  },
];
