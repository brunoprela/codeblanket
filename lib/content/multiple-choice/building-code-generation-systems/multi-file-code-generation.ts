/**
 * Multiple choice questions for Multi-File Code Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const multifilecodegenerationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'bcgs-multifile-mc-1',
    question:
      'What is the most critical step when applying changes across multiple files?',
    options: [
      'Edit files alphabetically',
      'Create backups and apply changes transactionally',
      'Edit the largest file first',
      'Ask user to approve each file individually',
    ],
    correctAnswer: 1,
    explanation:
      'Creating backups and applying transactionally ensures you can rollback if any change fails. Files depend on each other, so partial application leaves codebase inconsistent.',
  },
  {
    id: 'bcgs-multifile-mc-2',
    question:
      'How should import statements be managed when creating multi-file changes?',
    options: [
      'Ignore imports, user will fix them',
      'Only add imports, never remove',
      'Automatically add required imports and remove unused ones',
      'Always regenerate all imports',
    ],
    correctAnswer: 2,
    explanation:
      'Automatically detect required imports (parse for undefined names), add them properly grouped (stdlib/third-party/local), and remove unused imports to keep code clean.',
  },
  {
    id: 'bcgs-multifile-mc-3',
    question:
      'What analysis technique helps determine which files need changes?',
    options: [
      'Random sampling',
      'Dependency graph analysis',
      'File size analysis',
      'Alphabetical listing',
    ],
    correctAnswer: 1,
    explanation:
      'Dependency graph analysis shows which files import which, allowing you to trace impact of changes. If file A changes interface, all files importing A need updates.',
  },
  {
    id: 'bcgs-multifile-mc-4',
    question: 'When should changes be applied - all at once or one at a time?',
    options: [
      'One at a time for safety',
      'All at once, then validate as unit',
      'Largest files first',
      'User decides each time',
    ],
    correctAnswer: 1,
    explanation:
      'Apply all changes at once (transactionally), then validate the entire system together. Files depend on each other, so partial application might break imports or type consistency.',
  },
  {
    id: 'bcgs-multifile-mc-5',
    question:
      'What should happen if one file in a multi-file change fails validation?',
    options: [
      'Apply successful changes, skip failed',
      'Rollback all changes',
      'Ask user what to do',
      'Retry failed file only',
    ],
    correctAnswer: 1,
    explanation:
      "Rollback all changes if any file fails. Partial application leaves codebase in inconsistent state. It's better to have all old code than mix of old and new.",
  },
];
