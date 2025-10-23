/**
 * Multiple choice questions for Symbol Resolution & References section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const symbolresolutionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cuam-symbolresolution-mc-1',
    question: 'What is a symbol table used for in code analysis?',
    options: [
      'Storing source code text',
      'Mapping identifiers to their declarations and types',
      'Executing code',
      'Formatting code',
    ],
    correctAnswer: 1,
    explanation:
      'A symbol table maps names (variables, functions, classes) to their declarations, types, and scopes. Essential for resolving references, type checking, and "go to definition".',
  },
  {
    id: 'cuam-symbolresolution-mc-2',
    question:
      'What is lexical scoping?\n\nx = 10\ndef outer():\n    x = 20\n    def inner():\n        print(x)  # Which x?',
    options: [
      'Variables are resolved based on execution order',
      'Variables are resolved based on textual code structure',
      'All variables are global',
      'Variables must be explicitly imported',
    ],
    correctAnswer: 1,
    explanation:
      "Lexical scoping resolves variables based on WHERE code is written (textual structure), not when it executes. Here, inner() sees outer()'s x=20, not global x=10.",
  },
  {
    id: 'cuam-symbolresolution-mc-3',
    question: 'How do you find all references to a variable in Python AST?',
    options: [
      'Search for the variable name with regex',
      'Build scope tree, then find all Name nodes matching the symbol in scope',
      'Use ast.unparse()',
      'Count FunctionDef nodes',
    ],
    correctAnswer: 1,
    explanation:
      'Must distinguish between different variables with the same name in different scopes. Build scope analysis first, then find Name nodes that resolve to the target symbol.',
  },
  {
    id: 'cuam-symbolresolution-mc-4',
    question: 'What is the difference between a Name node and a symbol?',
    options: [
      'They are the same',
      'Name is AST node (syntax); symbol is semantic entity (meaning)',
      'Name is for variables; symbol is for functions',
      'Symbols are only in compiled code',
    ],
    correctAnswer: 1,
    explanation:
      'Name nodes are syntax (AST nodes representing text "x"). Symbols are semantic (the actual variable x refers to). Multiple Name nodes can refer to the same symbol.',
  },
  {
    id: 'cuam-symbolresolution-mc-5',
    question:
      'Why is resolving imported symbols complex in Python?\n\nfrom module import *',
    options: [
      'Imports are slow',
      'You must parse the imported module to know what names are available',
      'Imports are deprecated',
      'Python does not support imports',
    ],
    correctAnswer: 1,
    explanation:
      '"from module import *" requires analyzing module to find exported names. Also must handle relative imports, circular imports, and dynamic imports for complete symbol resolution.',
  },
];
