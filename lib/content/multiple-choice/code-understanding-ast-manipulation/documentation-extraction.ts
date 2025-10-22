/**
 * Multiple choice questions for Documentation & Comment Extraction section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const documentationextractionMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-documentationextraction-mc-1',
        question:
            'How do you extract docstrings from Python functions using AST?',
        options: [
            'Search for triple quotes with regex',
            'Access ast.get_docstring(node) or node.body[0] if it\'s a string',
            'Parse comments',
            'Use ast.unparse()',
        ],
        correctAnswer: 1,
        explanation:
            'ast.get_docstring() extracts the first string literal in a function/class body. Alternatively, check if body[0] is an Expr node containing a Constant string.',
    },
    {
        id: 'cuam-documentationextraction-mc-2',
        question:
            'What is the difference between docstrings and comments in Python?',
        options: [
            'They are the same',
            'Docstrings are string literals (in AST); comments are removed during parsing',
            'Comments are in the AST',
            'Docstrings are deprecated',
        ],
        correctAnswer: 1,
        explanation:
            'Docstrings are actual string literals (part of AST, accessible at runtime via __doc__). Comments are tokenization-level only, stripped during parsing, not in AST.',
    },
    {
        id: 'cuam-documentationextraction-mc-3',
        question:
            'How can you extract type information from docstrings?\n\ndef add(x, y):\n    """Add two numbers.\n    \n    Args:\n        x (int): First number\n        y (int): Second number\n    \n    Returns:\n        int: Sum\n    """',
        options: [
            'Type information is in the AST',
            'Parse docstring text to extract structured type info',
            'Use ast.get_type()',
            'Types cannot be extracted from docstrings',
        ],
        correctAnswer: 1,
        explanation:
            'Docstrings are unstructured text. Must parse them (e.g., Napoleon-style, Google-style) to extract parameter types. Libraries like sphinx and pydoc do this.',
    },
    {
        id: 'cuam-documentationextraction-mc-4',
        question:
            'What does Sphinx use ASTs for when generating documentation?',
        options: [
            'Executing code examples',
            'Extracting signatures, docstrings, and module structure',
            'Formatting code',
            'Type checking',
        ],
        correctAnswer: 1,
        explanation:
            'Sphinx parses Python files to extract module/class/function structures, names, signatures, and docstrings, then generates HTML/PDF documentation from this extracted information.',
    },
    {
        id: 'cuam-documentationextraction-mc-5',
        question:
            'Why is extracting parameter names from AST better than regex?',
        options: [
            'Regex is slower',
            'AST understands syntax (defaults, *args, **kwargs) reliably',
            'AST is easier to write',
            'Regex cannot match function definitions',
        ],
        correctAnswer: 1,
        explanation:
            'AST parsing handles complex cases: def f(a, b=5, *args, **kwargs). Regex struggles with nested parentheses, defaults, annotations. AST gives structured, reliable extraction.',
    },
];

