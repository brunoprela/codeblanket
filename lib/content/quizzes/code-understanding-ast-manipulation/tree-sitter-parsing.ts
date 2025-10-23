/**
 * Quiz questions for Tree-sitter for Multi-Language Parsing section
 */

export const treesitterparsingQuiz = [
  {
    id: 'cuam-treesitterparsing-q-1',
    question:
      "Why does Cursor use tree-sitter instead of language-specific parsers like Python's ast module for each language? What are the trade-offs?",
    hint: 'Consider consistency, maintenance, incremental parsing, and error recovery.',
    sampleAnswer:
      'Cursor uses tree-sitter for **universal multi-language support with a consistent API**. Benefits: 1) **One system for 40+ languages** - learning tree-sitter once works everywhere, versus learning 40 different parser APIs, 2) **Incremental parsing** - tree-sitter only re-parses changed code (2-5ms updates), critical for real-time IDE features, 3) **Error recovery** - produces best-effort trees even with syntax errors, so analysis continues, 4) **Consistent query language** - same pattern matching across all languages. Trade-offs: 1) Concrete syntax trees are larger/more detailed than ASTs - need more traversal, 2) No language-specific semantic analysis (type checking, etc.) built-in, 3) Each language grammar needs maintenance. For Cursor, consistency + incremental parsing + error recovery outweigh the larger tree size. They can layer semantic analysis on top when needed.',
    keyPoints: [
      'Single API for all languages',
      'Incremental parsing for real-time editing',
      'Error recovery keeps analysis working',
      'Trade-off: larger trees, less semantic info',
    ],
  },
  {
    id: 'cuam-treesitterparsing-q-2',
    question:
      "Explain how tree-sitter's error recovery allows IDEs to provide features even with broken code. Why is this critical for tools like Cursor?",
    hint: 'Think about what happens as you type mid-line, and incomplete code states.',
    sampleAnswer:
      "Tree-sitter creates a **best-effort parse tree even with syntax errors**, marking problematic regions as ERROR nodes but preserving structure around them. Example: typing 'def func(x' (missing closing paren) - tree-sitter still creates FunctionDef node with parameter, just marks the missing ')' as an error. This is critical for Cursor because: 1) **Code is often invalid while typing** - you're mid-expression, mid-function, etc., 2) **Auto-complete needs partial trees** - to suggest what comes next, you need to know you're inside a function parameter list, 3) **Go-to-definition should work** even if current line is broken, 4) **Real-time diagnostics** can show errors without blocking other features. Without error recovery, every keystroke that creates invalid syntax would crash the parser and lose all IDE features until syntax is valid again - terrible UX. Tree-sitter's graceful degradation means Cursor stays helpful even with incomplete code.",
    keyPoints: [
      'Creates partial trees with ERROR markers',
      'Preserves valid structure around errors',
      'Enables features while typing incomplete code',
      'Critical for real-time IDE assistance',
    ],
  },
  {
    id: 'cuam-treesitterparsing-q-3',
    question:
      'How would you use tree-sitter queries to find all async functions across JavaScript, TypeScript, and Python in a codebase? What makes this powerful?',
    hint: 'Consider the S-expression query syntax and language-agnostic patterns.',
    sampleAnswer:
      'Use tree-sitter queries with language-specific patterns: **JavaScript/TypeScript**: `(function_declaration (async) @async_keyword name: (identifier) @func_name)` + `(arrow_function (async) @async)`. **Python**: `(function_definition (async) @async_keyword name: (identifier) @func_name)`. The power is: 1) **Same query syntax** across languages - learn once, apply everywhere, 2) **Structural matching** not text regex - finds async functions regardless of whitespace/formatting, 3) **Captures** (@func_name) let you extract specific pieces, 4) **Composable** - build library of reusable queries. For Cursor: write one query pattern per language, run across entire codebase, get structured results. Could then: highlight all async functions, suggest await usage, find async/sync mismatches. This is how Cursor performs cross-language refactoring and pattern detection - unified query system over diverse languages.',
    keyPoints: [
      'S-expression queries work across languages',
      'Structural matching more reliable than regex',
      'Captures extract specific information',
      'Enables cross-language code analysis',
    ],
  },
];
