/**
 * Multiple choice questions for Language Server Protocol section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const languageserverprotocolMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-languageserverprotocol-mc-1',
        question:
            'What problem does LSP solve?',
        options: [
            'Slow code execution',
            'Need to implement language support separately for each editor (N×M problem)',
            'Syntax errors in code',
            'Lack of programming languages',
        ],
        correctAnswer: 1,
        explanation:
            'Before LSP: N editors × M languages = N×M implementations. With LSP: Each language has 1 server, each editor has 1 client = N+M implementations. Massive reduction in effort.',
    },
    {
        id: 'cuam-languageserverprotocol-mc-2',
        question:
            'How does LSP communicate between client (editor) and server?',
        options: [
            'Shared memory',
            'JSON-RPC over stdin/stdout or socket',
            'REST API',
            'File system',
        ],
        correctAnswer: 1,
        explanation:
            'LSP uses JSON-RPC protocol over stdio or TCP/WebSocket. Client sends requests (textDocument/definition), server responds with results. Stateful connection for document sync.',
    },
    {
        id: 'cuam-languageserverprotocol-mc-3',
        question:
            'What is incremental document synchronization in LSP?',
        options: [
            'Parsing one line at a time',
            'Sending only document changes (edits) instead of entire document',
            'Updating documentation comments',
            'Synchronizing between multiple servers',
        ],
        correctAnswer: 1,
        explanation:
            'Instead of sending entire document on each edit, send only changes: "at position X, insert Y". Much more efficient - reduces network traffic and enables incremental parsing.',
    },
    {
        id: 'cuam-languageserverprotocol-mc-4',
        question:
            'What LSP feature provides autocomplete suggestions?',
        options: [
            'textDocument/definition',
            'textDocument/completion',
            'textDocument/hover',
            'textDocument/references',
        ],
        correctAnswer: 1,
        explanation:
            'textDocument/completion request returns completion items at cursor position. Server uses AST + symbol table to suggest valid identifiers, keywords, snippets at that location.',
    },
    {
        id: 'cuam-languageserverprotocol-mc-5',
        question:
            'What is the difference between diagnostics and code actions in LSP?',
        options: [
            'They are the same',
            'Diagnostics report problems; code actions offer fixes/refactorings',
            'Diagnostics are faster',
            'Code actions are deprecated',
        ],
        correctAnswer: 1,
        explanation:
            'Diagnostics (publishDiagnostics) are errors/warnings sent automatically. Code actions (textDocument/codeAction) are user-triggered fixes. Diagnostics identify issues, actions solve them.',
    },
];

