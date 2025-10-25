/**
 * Multiple choice questions for Abstract Syntax Trees (AST) Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const astfundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cuam-astfundamentals-mc-1',
    question:
      'What is the primary purpose of an Abstract Syntax Tree (AST) in code analysis?',
    options: [
      'To execute code faster',
      'To represent code structure in a tree format for analysis',
      'To compress source code',
      'To generate documentation automatically',
    ],
    correctAnswer: 1,
    explanation:
      'An AST represents the syntactic structure of code as a tree, making it easy to analyze, transform, and understand code programmatically.',
  },
  {
    id: 'cuam-astfundamentals-mc-2',
    question:
      'What information is typically NOT preserved in an AST?\n\nif (x > 5) { return true; }',
    options: [
      'The condition x > 5',
      'The return statement',
      'Whitespace and formatting',
      'The function body structure',
    ],
    correctAnswer: 2,
    explanation:
      'ASTs abstract away syntactic details like whitespace, comments, and formatting. They preserve semantic structure (conditions, statements) but not presentation details.',
  },
  {
    id: 'cuam-astfundamentals-mc-3',
    question:
      'Which tree traversal strategy visits the root node AFTER its children?',
    options: [
      'Pre-order (depth-first)',
      'Level-order (breadth-first)',
      'Post-order (depth-first)',
      'In-order (depth-first)',
    ],
    correctAnswer: 2,
    explanation:
      'Post-order traversal visits children first, then the root. This is useful for operations that need to process leaves before their parents (e.g., evaluating expressions bottom-up).',
  },
  {
    id: 'cuam-astfundamentals-mc-4',
    question:
      'What does this Python ast.NodeVisitor pattern enable?\n\nclass MyVisitor (ast.NodeVisitor):\n    def visit_FunctionDef (self, node):\n        # custom logic\n        self.generic_visit (node)',
    options: [
      'Modifying AST nodes in place',
      'Selective handling of specific node types',
      'Converting AST back to source code',
      'Executing the AST',
    ],
    correctAnswer: 1,
    explanation:
      'NodeVisitor allows you to define custom behavior for specific node types (like FunctionDef) while automatically handling traversal of other nodes via generic_visit().',
  },
  {
    id: 'cuam-astfundamentals-mc-5',
    question:
      'Why do linters and static analyzers use ASTs instead of regular expressions?',
    options: [
      'Regular expressions are too slow',
      'ASTs understand code structure and semantics',
      'ASTs are easier to write',
      'Regular expressions cannot handle syntax errors',
    ],
    correctAnswer: 1,
    explanation:
      'ASTs understand the structural and semantic meaning of code (e.g., distinguishing variable declarations from uses), while regex only matches text patterns without understanding context.',
  },
];
