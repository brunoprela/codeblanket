/**
 * Multiple choice questions for Code Modification with AST section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codemodificationastMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'cuam-codemodificationast-mc-1',
        question:
            'What is the difference between ast.NodeVisitor and ast.NodeTransformer?',
        options: [
            'NodeVisitor is faster',
            'NodeVisitor reads; NodeTransformer modifies and returns new/modified nodes',
            'NodeTransformer is deprecated',
            'They are identical',
        ],
        correctAnswer: 1,
        explanation:
            'NodeVisitor is for read-only traversal (analysis). NodeTransformer can return modified/new nodes or None to delete, enabling AST transformations.',
    },
    {
        id: 'cuam-codemodificationast-mc-2',
        question:
            'How do you rename all occurrences of a variable using AST?\n\nOld: x = 5; print(x)\nNew: y = 5; print(y)',
        options: [
            'String replace "x" with "y"',
            'Build symbol table, find Name nodes referring to x, replace with y',
            'Use ast.unparse()',
            'Regex find and replace',
        ],
        correctAnswer: 1,
        explanation:
            'Must use symbol resolution to distinguish between different x\'s in different scopes. String replace would incorrectly rename all "x" strings (even in comments or strings).',
    },
    {
        id: 'cuam-codemodificationast-mc-3',
        question:
            'Why must you call ast.fix_missing_locations() after AST transformation?',
        options: [
            'To validate syntax',
            'To update line/column numbers for new/modified nodes',
            'To execute the code',
            'To format the code',
        ],
        correctAnswer: 1,
        explanation:
            'New or modified nodes lack proper source location info. fix_missing_locations() copies location from parent/nearby nodes, essential for error messages and code generation.',
    },
    {
        id: 'cuam-codemodificationast-mc-4',
        question:
            'What does this NodeTransformer do?\n\nclass RemovePrints(ast.NodeTransformer):\n    def visit_Call(self, node):\n        if isinstance(node.func, ast.Name) and node.func.id == \'print\':\n            return None\n        return node',
        options: [
            'Converts print to log',
            'Removes all print() calls from code',
            'Adds print statements',
            'Validates print usage',
        ],
        correctAnswer: 1,
        explanation:
            'Returning None from a NodeTransformer visit method deletes that node. This transformer removes all print() calls, useful for production code cleanup.',
    },
    {
        id: 'cuam-codemodificationast-mc-5',
        question:
            'What is a codemod?',
        options: [
            'A code editor',
            'An automated large-scale code transformation tool',
            'A debugging tool',
            'A version control system',
        ],
        correctAnswer: 1,
        explanation:
            'Codemods are scripts that automatically refactor code across entire codebases using AST manipulation. Example: Facebook\'s jscodeshift migrates thousands of files to new APIs.',
    },
];

