/**
 * Multiple choice questions for Single File Code Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const singlefilecodegenerationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcgs-singlefile-mc-1',
      question:
        'What is the correct order for validating a newly generated file?',
      options: [
        'Style → Syntax → Imports → Execution',
        'Syntax → Imports → Security → Style',
        'Execution → Syntax → Imports → Style',
        'Imports → Syntax → Execution → Security',
      ],
      correctAnswer: 1,
      explanation:
        "Correct order: Syntax (must parse), Imports (must exist), Security (must be safe), Style (nice to have). Each level depends on previous passing. Can't check imports if syntax is broken.",
    },
    {
      id: 'bcgs-singlefile-mc-2',
      question:
        'When generating boilerplate code, what should be templated vs. generated fresh?',
      options: [
        'Template everything for consistency',
        'Generate everything fresh for flexibility',
        'Template structure/patterns, generate business logic',
        'Template business logic, generate structure',
      ],
      correctAnswer: 2,
      explanation:
        'Template the structure and common patterns (error handling, imports, setup) for consistency. Generate the business logic fresh (unique per use case). This balances consistency with customization.',
    },
    {
      id: 'bcgs-singlefile-mc-3',
      question:
        'What is the main purpose of extracting style patterns from existing code?',
      options: [
        'To make generated code harder to distinguish from human code',
        'To ensure generated code matches project conventions and looks consistent',
        'To reduce generation time',
        'To avoid copyright issues',
      ],
      correctAnswer: 1,
      explanation:
        'Extracting style patterns ensures generated code matches project conventions (naming, error handling, documentation style). This makes it consistent with the codebase and easier for teams to maintain.',
    },
    {
      id: 'bcgs-singlefile-mc-4',
      question:
        'When should you use a FunctionGenerator vs. a complete FileGenerator?',
      options: [
        'Always use FileGenerator for consistency',
        'FunctionGenerator for adding single functions, FileGenerator for new files',
        'FunctionGenerator is deprecated, always use FileGenerator',
        'They are interchangeable',
      ],
      correctAnswer: 1,
      explanation:
        'Use FunctionGenerator when adding a single function to existing file (more precise). Use FileGenerator when creating entirely new files. Choosing the right tool gives better results for the specific task.',
    },
    {
      id: 'bcgs-singlefile-mc-5',
      question:
        'What validation step catches LLM hallucination of non-existent modules?',
      options: [
        'Syntax validation',
        'Import validation',
        'Style validation',
        'Security validation',
      ],
      correctAnswer: 1,
      explanation:
        'Import validation checks that all imported modules actually exist. This catches when LLMs hallucinate libraries like "sklearn.magic" or "pandas.auto" that sound plausible but don\'t exist.',
    },
  ];
