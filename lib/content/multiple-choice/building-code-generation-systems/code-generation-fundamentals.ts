/**
 * Multiple choice questions for Code Generation Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codegenerationfundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcgs-codegenfund-mc-1',
      question:
        'Which validation check should be performed FIRST when validating generated code?',
      options: [
        'Import validation',
        'Syntax validation',
        'Security scanning',
        'Style checking',
      ],
      correctAnswer: 1,
      explanation:
        "Syntax validation must be first because code that doesn't parse cannot be analyzed further. There\'s no point checking imports, security, or style if the code won't even parse.",
    },
    {
      id: 'bcgs-codegenfund-mc-2',
      question:
        'What is the main reason LLMs frequently hallucinate non-existent imports or functions?',
      options: [
        'They are trained on outdated code',
        'They confuse similar library names',
        "They generate plausible-sounding names that don't exist",
        'They only know popular libraries',
      ],
      correctAnswer: 2,
      explanation:
        "LLMs generate plausible-sounding names based on patterns they've seen. They don't have a database of what actually exists, so they can generate \"sklearn.magic.AutoML\" which sounds reasonable but doesn't exist.",
    },
    {
      id: 'bcgs-codegenfund-mc-3',
      question: 'When should you regenerate code vs. applying a targeted fix?',
      options: [
        'Always regenerate for consistency',
        'Regenerate for fundamental logic errors, fix for localized issues',
        'Always apply targeted fixes to save cost',
        'Regenerate only if syntax errors occur',
      ],
      correctAnswer: 1,
      explanation:
        'Regenerate when the fundamental approach is wrong (>50% changes needed). Apply targeted fixes for localized issues like syntax errors, import errors, or missing edge cases (<20% changes). This balances cost with correctness.',
    },
    {
      id: 'bcgs-codegenfund-mc-4',
      question:
        'What is the most critical resource limit to set when executing untrusted generated code?',
      options: [
        'Memory limit',
        'Execution timeout',
        'Network access',
        'File system access',
      ],
      correctAnswer: 1,
      explanation:
        'Execution timeout is most critical because infinite loops or exponential algorithms can run forever, consuming resources indefinitely. Memory, network, and file access are important but a runaway process can be more immediately damaging.',
    },
    {
      id: 'bcgs-codegenfund-mc-5',
      question:
        'Which temperature setting is recommended for code generation with LLMs?',
      options: [
        'High (0.7-1.0) for creativity',
        'Medium (0.4-0.6) for balance',
        'Low (0.1-0.3) for determinism',
        'Zero (0.0) for exact reproducibility',
      ],
      correctAnswer: 2,
      explanation:
        'Low temperature (0.1-0.3) is recommended for code because you want deterministic, correct output. Code needs to be precise, not creative. High temperature introduces randomness that can cause syntax errors or logic bugs.',
    },
  ];
