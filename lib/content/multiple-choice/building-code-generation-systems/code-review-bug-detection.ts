/**
 * Multiple choice questions for Code Review & Bug Detection section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codereviewbugdetectionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcgs-review-mc-1',
    question: 'What is the most effective way to use LLMs for code review?',
    options: [
      'Review entire codebase at once',
      'Focus on diffs/changes with surrounding context',
      'Only review comments',
      'Review file names only',
    ],
    correctAnswer: 1,
    explanation:
      'Focus on diffs (changed lines) with surrounding context. This targets review to actual changes while providing context. Reviewing unchanged code wastes time and tokens.',
  },
  {
    id: 'bcgs-review-mc-2',
    question: 'What categories should automated code review check?',
    options: [
      'Only syntax errors',
      'Security, performance, bugs, style, and maintainability',
      'Just formatting',
      'Only comments',
    ],
    correctAnswer: 1,
    explanation:
      'Comprehensive review checks: security (SQL injection, XSS), performance (n² loops, unnecessary queries), bugs (off-by-one, null handling), style (naming), maintainability (complexity).',
  },
  {
    id: 'bcgs-review-mc-3',
    question: 'How should LLM-detected issues be prioritized?',
    options: [
      'All issues are equal',
      'By severity: critical (security, data loss), high (bugs), medium (performance), low (style)',
      'Random order',
      'Alphabetically',
    ],
    correctAnswer: 1,
    explanation:
      'Prioritize by severity: critical (security vulnerabilities, data loss) > high (bugs, correctness) > medium (performance, maintainability) > low (style, formatting). Fix critical issues first.',
  },
  {
    id: 'bcgs-review-mc-4',
    question: 'What makes a good code review comment from an LLM?',
    options: [
      'Vague statements like "this could be better"',
      'Specific issue, explanation, and suggested fix with code example',
      'Just saying "bad code"',
      'Only emojis',
    ],
    correctAnswer: 1,
    explanation:
      'Good comments: 1) Specific issue location/description, 2) Why it\'s problematic, 3) Suggested fix with code example. "Line 42: SQL injection risk. User input unsanitized. Use: cursor.execute(query, (user_input,))".',
  },
  {
    id: 'bcgs-review-mc-5',
    question: 'Should LLM code review replace human review?',
    options: [
      'Yes, LLMs catch everything',
      'No, use both: LLM catches common issues, humans focus on design/architecture',
      'Yes, humans are too slow',
      'No, LLMs are useless for review',
    ],
    correctAnswer: 1,
    explanation:
      'Use both: LLMs excel at catching common patterns (missing error handling, security issues). Humans excel at design/architecture decisions, domain knowledge, and context-specific concerns. Complementary strengths.',
  },
];
