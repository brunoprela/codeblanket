/**
 * Multiple choice questions for Code Editing & Diff Generation section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codeeditingdiffgenerationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcgs-codeediting-mc-1',
      question:
        'Why is the SEARCH/REPLACE format preferred over regenerating entire files?',
      options: [
        "It's faster to generate",
        'It preserves unchanged code and has smaller error surface',
        'It uses fewer tokens',
        "It's easier to understand",
      ],
      correctAnswer: 1,
      explanation:
        'SEARCH/REPLACE preserves all unchanged code exactly (comments, formatting, working logic) and only the changed lines can have errors. Regenerating creates 100x more opportunities for errors across the entire file.',
    },
    {
      id: 'bcgs-codeediting-mc-2',
      question:
        'What similarity threshold is recommended for fuzzy matching in SEARCH blocks?',
      options: [
        '50% (very permissive)',
        '70% (moderately permissive)',
        '85% (strict but handles minor differences)',
        '100% (exact match only)',
      ],
      correctAnswer: 2,
      explanation:
        '85% similarity threshold balances permissiveness (handles minor whitespace differences) with accuracy (prevents wrong matches). Too low matches wrong blocks, too high breaks on minor formatting.',
    },
    {
      id: 'bcgs-codeediting-mc-3',
      question:
        'When applying multiple edits sequentially, what ordering strategy prevents line number shifts?',
      options: [
        'Apply in order they appear in code',
        'Apply in random order',
        'Apply from bottom to top (descending line numbers)',
        'Apply largest changes first',
      ],
      correctAnswer: 2,
      explanation:
        "Applying bottom-to-top (descending line numbers) prevents line shifts. Edit at line 100 doesn't affect edit at line 50. Alternatively, track offsets when applying top-to-bottom.",
    },
    {
      id: 'bcgs-codeediting-mc-4',
      question:
        'What should happen if a SEARCH block matches multiple locations in a file?',
      options: [
        'Apply the edit to all matches',
        'Apply to the first match',
        'Reject as ambiguous and ask for more context',
        'Apply to the match closest to cursor',
      ],
      correctAnswer: 2,
      explanation:
        'Multiple matches indicate ambiguity. Reject and ask for more context in the SEARCH block to make it unique. Applying to all or first match could change wrong code.',
    },
    {
      id: 'bcgs-codeediting-mc-5',
      question:
        'When should you regenerate a file instead of using SEARCH/REPLACE edits?',
      options: [
        'Never, always use SEARCH/REPLACE',
        'When changes affect >50% of the file',
        'When there are syntax errors',
        'When the file is >100 lines',
      ],
      correctAnswer: 1,
      explanation:
        'Regenerate when changes affect >50% of the file or require complete restructuring. SEARCH/REPLACE becomes cumbersome with many blocks. For <20% changes, SEARCH/REPLACE is better.',
    },
  ];
