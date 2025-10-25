/**
 * Multiple choice questions for Python Code Analysis with AST section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pythoncodeanalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cuam-pythoncodeanalysis-mc-1',
    question:
      "What is the difference between ast.parse() and compile()?\n\nast.parse (code)\ncompile (code, '<string>', 'exec')",
    options: [
      'ast.parse() executes code, compile() creates AST',
      'ast.parse() creates AST, compile() creates bytecode',
      'They are identical',
      'compile() is deprecated',
    ],
    correctAnswer: 1,
    explanation:
      'ast.parse() parses code into an AST for analysis/manipulation. compile() takes source code (or AST) and produces bytecode for execution. Different purposes: analysis vs execution.',
  },
  {
    id: 'cuam-pythoncodeanalysis-mc-2',
    question:
      'What does this ast.NodeTransformer do?\\n\\nclass ConstantFolder (ast.NodeTransformer):\\n    def visit_BinOp (self, node):\\n        if isinstance (node.left, ast.Constant) and isinstance (node.right, ast.Constant):\\n            return ast.Constant (eval (compile(...)))',
    options: [
      'Removes all binary operations',
      'Evaluates constant expressions at compile time',
      'Converts all operations to constants',
      'Validates binary operations',
    ],
    correctAnswer: 1,
    explanation:
      'This is constant folding - optimizing expressions like "2 + 3" to "5" at parse time rather than runtime, improving performance by pre-computing constant calculations.',
  },
  {
    id: 'cuam-pythoncodeanalysis-mc-3',
    question: 'What information does ast.get_source_segment() provide?',
    options: [
      'The parent node of a given node',
      'The original source code text for a node',
      'The bytecode for a node',
      'The type annotations for a node',
    ],
    correctAnswer: 1,
    explanation:
      'ast.get_source_segment (source, node) returns the actual source code text that corresponds to a specific AST node, useful for error messages and code extraction.',
  },
  {
    id: 'cuam-pythoncodeanalysis-mc-4',
    question:
      'Why must you call ast.fix_missing_locations() after modifying an AST?',
    options: [
      'To update parent references',
      'To recalculate line numbers and column offsets',
      'To validate syntax',
      'To optimize the tree structure',
    ],
    correctAnswer: 1,
    explanation:
      'When you create or modify AST nodes, they lack proper line/column information. fix_missing_locations() propagates location data, which is needed for error messages and code generation.',
  },
  {
    id: 'cuam-pythoncodeanalysis-mc-5',
    question: 'What is the purpose of ast.unparse() (Python 3.9+)?',
    options: [
      'Parse code into AST',
      'Convert AST back to Python source code',
      'Remove comments from code',
      'Validate AST structure',
    ],
    correctAnswer: 1,
    explanation:
      'ast.unparse() converts an AST back into valid Python source code. This is essential for code generation and refactoring tools that modify ASTs and need to output the result.',
  },
];
