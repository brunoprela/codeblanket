/**
 * Multiple choice questions for Prompt Templates & Variables section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const prompttemplatesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of using prompt templates over hard-coded prompts?',
    options: [
      'Templates make API calls faster',
      'Templates enable centralized updates and version control',
      'Templates improve model quality',
      'Templates reduce token costs',
    ],
    correctAnswer: 1,
    explanation:
      'Prompt templates enable centralization - update once, affect all uses. This is critical for maintainability at scale. Hard-coding prompts throughout code creates maintenance nightmares when you need to improve or fix prompts. Templates also enable versioning and A/B testing.',
  },
  {
    id: 'mc2',
    question:
      'When should you use Jinja2 templates instead of simple f-strings?',
    options: [
      'Always use Jinja2',
      'Never, f-strings are always sufficient',
      'When you need conditional logic, loops, or template inheritance',
      'Only for web applications',
    ],
    correctAnswer: 2,
    explanation:
      'Use Jinja2 when you need logic (conditionals, loops), template inheritance, or filters. For simple variable substitution, f-strings are simpler and faster. Example: If premium users get extra instructions, Jinja2\'s {% if premium %} is perfect; for basic "{name} likes {hobby}", f-strings work fine.',
  },
  {
    id: 'mc3',
    question: 'What should template variables be validated for?',
    options: [
      'Presence (required variables exist)',
      'Type correctness',
      'Value constraints',
      'All of the above',
    ],
    correctAnswer: 3,
    explanation:
      'Templates should validate that required variables exist, are the correct type (string vs int), and meet constraints (e.g., max length, allowed values). This prevents runtime errors and ensures prompts are well-formed. Validation catches issues before expensive API calls.',
  },
  {
    id: 'mc4',
    question:
      'Why is template versioning important for production LLM applications?',
    options: [
      'To comply with regulations',
      'To enable A/B testing and safe rollback',
      'To reduce storage costs',
      'To improve response speed',
    ],
    correctAnswer: 1,
    explanation:
      'Template versioning enables A/B testing (compare v1.0 vs v2.0 quality), safe rollback (revert if new version degrades quality), and gradual rollout (deploy to 5%, then 25%, then 100%). Without versioning, prompt changes are risky all-or-nothing deployments.',
  },
  {
    id: 'mc5',
    question:
      'What is the recommended storage for prompt templates in production?',
    options: [
      'Only in code files',
      'Database for persistence, Redis for caching',
      'Environment variables',
      'Hard-coded in functions',
    ],
    correctAnswer: 1,
    explanation:
      'Store templates in a database for persistence, versioning, and centralized management. Cache active templates in Redis for fast access. This allows runtime template updates without code deployment, supports versioning, and provides performance. Never hard-code prompts in production.',
  },
];
