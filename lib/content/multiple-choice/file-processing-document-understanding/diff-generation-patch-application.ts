/**
 * Multiple choice questions for Diff Generation & Patch Application section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const diffgenerationpatchapplicationMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-diff-mc-1',
        question: 'What does difflib.SequenceMatcher do?',
        options: ['Sorts sequences', 'Finds differences between sequences', 'Merges sequences', 'Validates sequences'],
        correctAnswer: 1,
        explanation: 'SequenceMatcher finds differences between two sequences (like text lines) and returns operations needed to transform one into the other.',
    },
    {
        id: 'fpdu-diff-mc-2',
        question: 'What format does difflib.unified_diff produce?',
        options: ['JSON diff', 'Unified diff format (like git diff)', 'XML diff', 'Binary diff'],
        correctAnswer: 1,
        explanation: 'unified_diff produces the unified diff format with @@ markers, + for additions, and - for deletions, same format used by git and patch tools.',
    },
    {
        id: 'fpdu-diff-mc-3',
        question: 'What are the three types of operations in SequenceMatcher.get_opcodes()?',
        options: ['add, remove, change', 'insert, delete, replace', 'new, old, modified', 'create, update, destroy'],
        correctAnswer: 1,
        explanation: 'SequenceMatcher returns operations: insert (add lines), delete (remove lines), replace (modify lines), and equal (unchanged lines).',
    },
    {
        id: 'fpdu-diff-mc-4',
        question: 'Why are line-based diffs preferred for code over character-based?',
        options: ['They are more accurate', 'They are more readable and match how developers think', 'They are faster to compute', 'They show more detail'],
        correctAnswer: 1,
        explanation: 'Line-based diffs are more readable for code and match how developers think about changes (modified lines). They are also standard in version control.',
    },
    {
        id: 'fpdu-diff-mc-5',
        question: 'What is a three-way merge?',
        options: ['Merging three files', 'Merging with three people', 'Merging using original, version A, and version B', 'A type of conflict'],
        correctAnswer: 2,
        explanation: 'Three-way merge compares the original version with two modified versions to intelligently merge changes and detect conflicts.',
    },
];

