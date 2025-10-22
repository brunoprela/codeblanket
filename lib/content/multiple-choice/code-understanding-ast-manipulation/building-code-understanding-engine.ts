/**
 * Multiple choice questions for Building a Code Understanding Engine section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const buildingcodeunderstandingengineMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-buildingcodeunderstandingengine-mc-1',
        question:
            'Why is a semantic index critical for large codebases?',
        options: [
            'It makes code run faster',
            'It pre-computes symbols/references for fast queries without re-parsing',
            'It fixes bugs automatically',
            'It reduces file size',
        ],
        correctAnswer: 1,
        explanation:
            'Semantic index stores pre-computed information (symbols, types, references) in a database. Queries (find references) are instant lookups, not full codebase re-parsing. Essential for scale.',
    },
    {
        id: 'cuam-buildingcodeunderstandingengine-mc-2',
        question:
            'What is the purpose of a file watcher in a code understanding engine?',
        options: [
            'Watching user activity',
            'Detecting file changes to trigger incremental re-indexing',
            'Monitoring network traffic',
            'Tracking execution time',
        ],
        correctAnswer: 1,
        explanation:
            'File watchers (using OS APIs like inotify, FSEvents) detect when files change. Triggers incremental re-parse and index update, keeping the semantic index in sync with code.',
    },
    {
        id: 'cuam-buildingcodeunderstandingengine-mc-3',
        question:
            'What is the trade-off between real-time and batch indexing?',
        options: [
            'Real-time is always better',
            'Real-time: accurate but resource-intensive; batch: efficient but can be stale',
            'Batch indexing is faster for queries',
            'Real-time indexing requires compilation',
        ],
        correctAnswer: 1,
        explanation:
            'Real-time indexing updates immediately (accurate but CPU-intensive). Batch indexing defers updates (efficient but index can lag). Best: hybrid approach - real-time for active file, batch for others.',
    },
    {
        id: 'cuam-buildingcodeunderstandingengine-mc-4',
        question:
            'What layers are needed in a production code understanding engine?',
        options: [
            'Only AST parser',
            'Parser, semantic analyzer, indexer, query engine, cache, API',
            'Just a database',
            'Compiler and interpreter',
        ],
        correctAnswer: 1,
        explanation:
            'Complete engine needs: Parser (AST), Analyzer (types/symbols), Indexer (database), Query engine (lookups), Cache (performance), Watch service (updates), API (features). All layers integrated.',
    },
    {
        id: 'cuam-buildingcodeunderstandingengine-mc-5',
        question:
            'How do modern IDEs handle multi-language support?\n\nVS Code, Cursor, etc.',
        options: [
            'Implement each language separately',
            'Use LSP - one server per language, universal client protocol',
            'Only support one language',
            'Compile to a common language',
        ],
        correctAnswer: 1,
        explanation:
            'Modern IDEs implement LSP client once, then connect to language-specific servers (pylsp for Python, rust-analyzer for Rust). This architecture enables universal multi-language support efficiently.',
    },
];

