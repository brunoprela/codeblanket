/**
 * Multiple choice questions for Code Structure Analysis section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const codestructureanalysisMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cuam-codestructureanalysis-mc-1',
    question: 'What is cyclomatic complexity a measure of?',
    options: [
      'Number of lines in a function',
      'Number of independent paths through code',
      'Number of function calls',
      'Number of variables',
    ],
    correctAnswer: 1,
    explanation:
      'Cyclomatic complexity measures the number of linearly independent paths through code (based on branches/decisions). Higher complexity = harder to test and understand.',
  },
  {
    id: 'cuam-codestructureanalysis-mc-2',
    question:
      'How do you calculate cyclomatic complexity from an AST?\n\ndef example(x):\n    if x > 5:\n        for i in range(x):\n            print(i)\n    else:\n        return 0',
    options: [
      'Count all AST nodes',
      'Count decision nodes (if, for, while, and, or) + 1',
      'Count function definitions',
      'Count all statements',
    ],
    correctAnswer: 1,
    explanation:
      'Cyclomatic complexity = decision points + 1. Here: if (1) + for (1) + else (implicit in if) + 1 = 3. Each branch/loop adds a path through the code.',
  },
  {
    id: 'cuam-codestructureanalysis-mc-3',
    question: 'What information does a call graph provide?',
    options: [
      'Variable usage across functions',
      'Which functions call which other functions',
      'Execution time of functions',
      'Memory usage of functions',
    ],
    correctAnswer: 1,
    explanation:
      'A call graph is a directed graph where nodes are functions and edges represent "A calls B". Essential for understanding control flow, finding unused code, and impact analysis.',
  },
  {
    id: 'cuam-codestructureanalysis-mc-4',
    question:
      'How do you extract all function signatures from Python code using AST?',
    options: [
      'Search for "def" keywords with regex',
      'Visit FunctionDef nodes and extract name, parameters, returns',
      'Use ast.unparse() on the entire file',
      'Parse docstrings only',
    ],
    correctAnswer: 1,
    explanation:
      'Visit all ast.FunctionDef nodes, extract node.name (function name), node.args (parameters including annotations), and node.returns (return type hint).',
  },
  {
    id: 'cuam-codestructureanalysis-mc-5',
    question:
      'What is the difference between syntactic and semantic code analysis?',
    options: [
      'Syntactic is faster',
      'Syntactic analyzes structure; semantic analyzes meaning and types',
      'Semantic requires execution',
      'Syntactic handles multiple languages',
    ],
    correctAnswer: 1,
    explanation:
      'Syntactic analysis examines code structure (AST shape) without understanding meaning. Semantic analysis resolves types, symbols, and meaning (e.g., "does this variable exist?").',
  },
];
