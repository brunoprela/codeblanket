/**
 * Multiple choice questions for Building a Universal File Editor section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const buildinguniversalfileeditorMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fpdu-editor-mc-1',
        question: 'What is the most important safety feature for a file editor?',
        options: ['Fast performance', 'Automatic backups before modifications', 'Syntax highlighting', 'Multi-file support'],
        correctAnswer: 1,
        explanation: 'Automatic backups before modifications ensure data is never lost. If an edit fails or is incorrect, you can always restore the previous version.',
    },
    {
        id: 'fpdu-editor-mc-2',
        question: 'What is an atomic file write operation?',
        options: ['Writing many files at once', 'Writing to temp file then atomic rename to target', 'Writing one byte at a time', 'Writing with multiple threads'],
        correctAnswer: 1,
        explanation: 'Atomic writes involve writing to a temp file first, then atomically renaming it to the target. This ensures you never have a partially-written file if the process is interrupted.',
    },
    {
        id: 'fpdu-editor-mc-3',
        question: 'Why use format-specific processors in a universal file editor?',
        options: ['They are faster', 'Each format has unique structure and requirements', 'They are easier to code', 'They use less memory'],
        correctAnswer: 1,
        explanation: 'Different file formats have unique structures (Python AST, Excel cells, PDF layout) that require specialized processing for accurate reading and modification.',
    },
    {
        id: 'fpdu-editor-mc-4',
        question: 'What should happen if a file edit operation fails?',
        options: ['Try again automatically', 'Restore from backup immediately', 'Ask user what to do', 'Delete the file'],
        correctAnswer: 1,
        explanation: 'On edit failure, immediately restore from backup to ensure file integrity. Log the error and inform the user, but prioritize data safety.',
    },
    {
        id: 'fpdu-editor-mc-5',
        question: 'How should a universal file editor integrate with LLMs?',
        options: ['Send entire file to LLM every time', 'Provide file summary and structure for context, generate diffs', 'Replace the editor with LLM', 'Only use LLM for error messages'],
        correctAnswer: 1,
        explanation: 'Efficient LLM integration provides file structure and context, generates diffs/edits, shows preview, then applies validated changes. This is how tools like Cursor work.',
    },
];

