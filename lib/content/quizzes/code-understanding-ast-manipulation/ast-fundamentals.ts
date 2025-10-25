/**
 * Quiz questions for AST Fundamentals section
 */

export const astfundamentalsQuiz = [
  {
    id: 'cuam-astfundamentals-q-1',
    question:
      'Explain the difference between tokens, parse trees, and Abstract Syntax Trees (ASTs). Why do modern code analysis tools prefer ASTs over parse trees?',
    hint: 'Consider the level of detail, size, and usefulness for code analysis.',
    sampleAnswer:
      "**Tokens** are the smallest meaningful units (keywords, identifiers, operators) produced by lexical analysis. **Parse trees** (concrete syntax trees) represent the exact grammar derivation including all syntactic elements like parentheses and semicolons - they're verbose and include information not needed for analysis. **ASTs** abstract away syntactic noise and focus on semantic structure - they're smaller, easier to work with, and contain just what's needed for code understanding. Modern tools prefer ASTs because they're more compact (50-70% smaller), easier to traverse, and focus on semantic meaning rather than syntax details. For example, '(x + y)' in a parse tree includes parentheses nodes, but in an AST it's just a BinOp node with Add operator - the parentheses are implicit in the tree structure.",
    keyPoints: [
      'Tokens: flat list of language units',
      'Parse trees: exact grammar representation, verbose',
      'ASTs: abstracted structure, semantic focus',
      'ASTs are more compact and easier to analyze',
    ],
  },
  {
    id: 'cuam-astfundamentals-q-2',
    question:
      'How does Cursor use AST analysis to provide context-aware code suggestions? Walk through what happens when you start typing inside a function.',
    hint: 'Think about symbol tables, scope analysis, and what information the AST provides.',
    sampleAnswer:
      "When you type inside a function, Cursor: 1) Parses the file into an AST to understand structure, 2) Locates your cursor position within a specific function node in the AST, 3) Analyzes the function's AST to extract available symbols (parameters, local variables, class attributes via 'self'), 4) Walks parent scopes (class, module, imports) to find all accessible names, 5) Extracts type information from annotations and assignments, 6) Builds a context window with relevant symbols, types, and documentation, 7) Sends this structured context to the LLM for intelligent suggestions. For example, in a method, it knows 'self' attributes from __init__, can suggest methods from the class, and understands which imported modules are available - all derived from AST analysis, not just text matching.",
    keyPoints: [
      'Parse file to AST, locate cursor position',
      'Extract symbols from function scope and parents',
      'Gather type hints and context information',
      'Build structured context for LLM suggestions',
    ],
  },
  {
    id: 'cuam-astfundamentals-q-3',
    question:
      'What are the critical steps you must take after modifying an AST before converting it back to source code? Why are these steps necessary?',
    hint: 'Consider location information and validation.',
    sampleAnswer:
      "After modifying an AST, you MUST: 1) Call **ast.fix_missing_locations()** to update line numbers and column offsets - transformations often create nodes without location info, which causes issues in error reporting and debugging, 2) **Validate the modified AST** by attempting to compile it - ensures your transformation didn't create invalid syntax, 3) Optionally **preserve formatting** using tools like LibCST if you need to maintain code style. The locations are critical because without them: error messages point to wrong lines, debuggers break, and tools that depend on accurate position information fail. Validation catches structural errors before they become runtime issues. For production code: tree = transform (original); ast.fix_missing_locations (tree); compile (tree, '<string>', 'exec'); code = ast.unparse (tree). Skipping these steps is a common bug that causes mysterious failures.",
    keyPoints: [
      'ast.fix_missing_locations() - update position info',
      'Validate by compiling the modified AST',
      'Consider LibCST for formatting preservation',
      'Missing locations break error reporting and debugging',
    ],
  },
];
