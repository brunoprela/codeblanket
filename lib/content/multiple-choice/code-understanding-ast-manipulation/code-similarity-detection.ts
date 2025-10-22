/**
 * Multiple choice questions for Code Similarity & Clone Detection section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codesimilaritydetectionMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-codesimilaritydetection-mc-1',
        question:
            'What is the difference between Type 1 and Type 2 code clones?',
        options: [
            'Type 1 is faster to detect',
            'Type 1: exact copies; Type 2: identical except names/literals',
            'Type 2 is more common',
            'They are the same',
        ],
        correctAnswer: 1,
        explanation:
            'Type 1 clones are character-for-character identical (except whitespace/comments). Type 2 clones have identical structure but different variable names or literal values.',
    },
    {
        id: 'cuam-codesimilaritydetection-mc-2',
        question:
            'How do you detect Type 2 clones using AST?\n\nCode 1: x = a + b\nCode 2: y = c + d',
        options: [
            'String comparison',
            'Normalize AST (replace identifiers/literals with placeholders), then compare structure',
            'Execute both and compare results',
            'Count lines of code',
        ],
        correctAnswer: 1,
        explanation:
            'Normalize by replacing all names with generic tokens (VAR1, VAR2) and literals with types (INT, STR). Then compare normalized ASTs - structural match indicates Type 2 clone.',
    },
    {
        id: 'cuam-codesimilaritydetection-mc-3',
        question:
            'What is tree edit distance?',
        options: [
            'Number of characters changed',
            'Minimum number of tree operations (insert/delete/rename) to transform one tree to another',
            'Execution time difference',
            'Number of lines changed',
        ],
        correctAnswer: 1,
        explanation:
            'Tree edit distance measures similarity between ASTs by counting minimum edits needed to transform tree A to tree B. Lower distance = more similar. Useful for fuzzy code matching.',
    },
    {
        id: 'cuam-codesimilaritydetection-mc-4',
        question:
            'Why would you hash AST subtrees for clone detection?',
        options: [
            'Hashing makes code secure',
            'Fast lookup - identical subtrees have same hash, enabling efficient duplicate finding',
            'Hashing validates syntax',
            'Hashing improves performance at runtime',
        ],
        correctAnswer: 1,
        explanation:
            'Hash each subtree; identical structures produce same hash. Store hashes in table - collisions indicate potential clones. Much faster than comparing every pair of subtrees (O(n²)).',
    },
    {
        id: 'cuam-codesimilaritydetection-mc-5',
        question:
            'What is semantic clone detection?',
        options: [
            'Detecting identical comments',
            'Finding code with same behavior but different structure',
            'Finding syntax errors',
            'Detecting similar variable names',
        ],
        correctAnswer: 1,
        explanation:
            'Semantic (Type 3/4) clones have same behavior via different implementations. Example: loop vs list comprehension. Requires program analysis/testing, not just AST comparison.',
    },
];

