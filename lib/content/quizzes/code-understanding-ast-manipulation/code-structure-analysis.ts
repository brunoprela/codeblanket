/**
 * Quiz questions for Code Structure Analysis section
 */

export const codestructureanalysisQuiz = [
  {
    id: 'cuam-codestructureanalysis-q-1',
    question:
      'Explain how control flow analysis helps Cursor understand code better than simple AST traversal. Give an example of a refactoring that requires control flow understanding.',
    hint: 'Think about execution paths, early returns, and extract method refactoring.',
    sampleAnswer:
      "Control flow analysis tracks **possible execution paths**, not just syntax structure. Simple AST traversal shows what code exists; control flow shows what code **runs** and in what order. Example: ```python\ndef process (x):\n    if x < 0:\n        return None  # Path 1: early return\n    result = expensive_calc (x)\n    return result  # Path 2\n```\nFor 'extract method' refactoring on the last two lines, control flow analysis reveals: 1) Not all paths reach this code (early return exists), 2) `x` is used and must be a parameter, 3) Extracted function should handle the same return type. Without control flow, you might incorrectly extract including the early return, or miss that `x` is required. Cursor uses control flow to: determine variable liveness, identify dead code, suggest guard clause refactorings, validate extract method won't change behavior. It understands not just structure but **runtime behavior**.",
    keyPoints: [
      'Tracks execution paths, not just syntax',
      'Identifies early returns, branches, loops',
      'Required for safe refactoring operations',
      'Helps determine variable requirements',
    ],
  },
  {
    id: 'cuam-codestructureanalysis-q-2',
    question:
      'How does data flow analysis help detect potential bugs that AST analysis alone would miss? Provide an example.',
    hint: 'Consider variable usage before assignment and unused variable detection.',
    sampleAnswer:
      "Data flow analysis tracks **where data comes from and where it goes**, enabling bug detection that AST structure alone misses. Example bugs it catches:\n\n**1. Use before assignment:**\n```python\ndef calc (x):\n    if x > 0:\n        result = x * 2\n    return result  # Bug: result undefined if x <= 0\n```\nAST sees 'result' used; data flow sees it's not assigned on all paths.\n\n**2. Unused variables:**\n```python\ndef process():\n    total = expensive_calc()  # Assigned but never read\n    return 42\n```\nAST sees assignment; data flow sees no subsequent reads.\n\n**3. Uninitialized reads:**\nData flow tracks assign → use relationships, detecting when a variable is read before any assignment. This is how Cursor warns 'variable may be undefined' - it analyzes data flow paths and finds scenarios where use precedes assignment. Critical for safety in refactoring and code generation.",
    keyPoints: [
      'Tracks variable definitions and uses',
      'Detects use-before-assignment bugs',
      'Finds unused variables (dead code)',
      'Validates all paths initialize variables',
    ],
  },
  {
    id: 'cuam-codestructureanalysis-q-3',
    question:
      'Why is building a call graph essential for impact analysis in large codebases? How does it help with "what breaks if I change this function" questions?',
    hint: 'Think about transitive dependencies and cascading changes.',
    sampleAnswer:
      "A call graph maps **which functions call which**, enabling impact analysis through transitive dependencies. When you change a function, you need to know: 1) **Direct callers** - functions calling it immediately, 2) **Indirect callers** - functions that call the direct callers (transitive), 3) **Call chains** - how changes propagate through the system. Example: changing `validateUser()` signature. Call graph shows: `createAccount()` calls it (direct impact), `registerUser()` calls `createAccount()` (indirect impact), `handleSignup()` calls `registerUser()` (further indirect). Without call graph, you might update direct callers but miss indirect ones, causing runtime errors. Cursor uses call graphs to: show 'this change affects N functions', suggest bulk refactoring, generate comprehensive tests, highlight coupling. The graph reveals **architectural dependencies** - if one function has 50 callers, that's a refactoring opportunity. Critical for safe evolution of large codebases.",
    keyPoints: [
      'Maps caller-callee relationships',
      'Enables transitive dependency tracking',
      'Shows cascading impact of changes',
      'Identifies highly coupled code',
    ],
  },
];
