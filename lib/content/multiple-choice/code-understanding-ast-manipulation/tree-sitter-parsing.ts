/**
 * Multiple choice questions for Tree-sitter for Multi-Language Parsing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const treesitterparsingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-treesitterparsing-mc-1',
        question:
            'What is the key advantage of tree-sitter over traditional parsers like ast.parse()?',
        options: [
            'Tree-sitter is Python-specific',
            'Tree-sitter handles syntax errors gracefully and parses incrementally',
            'Tree-sitter is faster for small files',
            'Tree-sitter includes type checking',
        ],
        correctAnswer: 1,
        explanation:
            'Tree-sitter is error-tolerant (produces partial tree even with syntax errors) and incremental (only re-parses changed sections), making it ideal for editors that need real-time parsing.',
    },
    {
        id: 'cuam-treesitterparsing-mc-2',
        question:
            'What does incremental parsing mean in tree-sitter?',
        options: [
            'Parsing one line at a time',
            'Re-using unchanged parts of the tree when code is edited',
            'Parsing multiple languages simultaneously',
            'Parsing without syntax errors',
        ],
        correctAnswer: 1,
        explanation:
            'Incremental parsing means tree-sitter only re-parses the edited regions, reusing unchanged subtrees. This makes re-parsing after edits extremely fast (typically <5ms).',
    },
    {
        id: 'cuam-treesitterparsing-mc-3',
        question:
            'How do you query tree-sitter trees?\n\nquery = language.query("(function_definition name: (identifier) @func-name)")\nmatches = query.matches(tree.root_node)',
        options: [
            'Using Python decorators',
            'Using S-expression patterns with captures',
            'Using regular expressions',
            'Using XPath',
        ],
        correctAnswer: 1,
        explanation:
            'Tree-sitter uses S-expression query syntax inspired by Lisp. Patterns describe tree structures, @captures extract matched nodes. More powerful than regex for structural queries.',
    },
    {
        id: 'cuam-treesitterparsing-mc-4',
        question:
            'Why does tree-sitter include a Concrete Syntax Tree (CST) layer?',
        options: [
            'To execute code',
            'To preserve ALL syntax details including whitespace and comments',
            'To validate code correctness',
            'To optimize parsing speed',
        ],
        correctAnswer: 1,
        explanation:
            'Unlike ASTs which abstract away details, CSTs preserve EVERYTHING - whitespace, comments, punctuation. This enables precise code formatting, diffing, and syntax highlighting.',
    },
    {
        id: 'cuam-treesitterparsing-mc-5',
        question:
            'What is a tree-sitter grammar file used for?',
        options: [
            'Configuring editor syntax highlighting colors',
            'Defining parsing rules for a programming language',
            'Storing parsed trees',
            'Validating code semantics',
        ],
        correctAnswer: 1,
        explanation:
            'Grammar files (grammar.js) define the syntax rules of a language - how to recognize functions, expressions, etc. Tree-sitter compiles these into fast C parsers.',
    },
];

