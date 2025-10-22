/**
 * Quiz questions for Code Modification with AST section
 */

export const codemodificationastQuiz = [
    {
        id: 'cuam-codemodificationast-q-1',
        question:
            'Why is ast.fix_missing_locations() critical after AST modifications? What breaks if you skip it?',
        hint: 'Think about line numbers, debugging, and error reporting.',
        sampleAnswer:
            "ast.fix_missing_locations() is critical because **AST transformations create nodes without location info** (lineno, col_offset). Without it: 1) **Error messages** point to wrong/missing lines, 2) **Debuggers** can't set breakpoints or step correctly, 3) **Coverage tools** report incorrect coverage, 4) **Linters** can't locate issues, 5) **Stack traces** become useless. When you create a new AST node (e.g., new function call), it has lineno=None. Compiling this technically works but breaks tooling. fix_missing_locations() walks the tree, propagating parent node locations to children, ensuring every node has valid position data. For production: ALWAYS call it: `new_tree = transform(old_tree); ast.fix_missing_locations(new_tree)`. Skipping it is a common bug that causes mysterious issues - code runs but debugging/error reporting fails. It's like compiling without source maps - technically works but impossible to debug.",
        keyPoints: [
            'Transformations create nodes without positions',
            'Missing locations break debugging and error reporting',
            'Propagates parent locations to children',
            'Always call after ANY AST modification',
        ],
    },
    {
        id: 'cuam-codemodificationast-q-2',
        question:
            'When extracting a method from selected code, how do you determine which variables must become parameters vs which can remain local?',
        hint: 'Consider variable definitions, uses, and scope boundaries.',
        sampleAnswer:
            "Determine parameters using **data flow analysis**: 1) **Find all variables referenced** in extracted code (ast.Name nodes with Load context), 2) **Identify which are defined** in extracted code (Name nodes with Store context), 3) **Parameters = used but not defined** in extraction - these values come from outside and must be passed in, 4) **Local variables = defined in extraction** - these are internal, 5) **Return values = defined in extraction AND used after** - these must be returned. Example:\n```python\n# Extract these lines:\nfiltered = [x for x in items if x > threshold]\ncount = len(filtered)\n```\n`items` and `threshold` are used but not defined → parameters. `filtered` is defined and used locally. `count` is defined and may be used after → return value (or parameter if also used after). The algorithm: params = (reads - writes in extraction), returns = (writes in extraction ∩ reads after extraction). This is how IDEs safely extract methods - analyzing data dependencies to preserve behavior.",
        keyPoints: [
            'Parameters: variables used but not defined in extraction',
            'Local: variables defined and only used internally',
            'Returns: variables defined and used after extraction',
            'Requires data flow analysis to determine',
        ],
    },
    {
        id: 'cuam-codemodificationast-q-3',
        question:
            'Why does production refactoring use LibCST instead of ast for code modifications? What does it preserve that ast loses?',
        hint: 'Think about formatting, comments, and whitespace.',
        sampleAnswer:
            "LibCST uses **Concrete Syntax Trees** preserving: 1) **Formatting** - indentation, spacing, line breaks, 2) **Comments** - inline and block comments (ast discards these!), 3) **Whitespace** - blank lines, alignment, 4) **String quotes** - single vs double quotes, 5) **Syntax style** - trailing commas, parentheses. Python's ast module produces ASTs that lose all this - ast.unparse() generates syntactically correct but differently formatted code. For production: users expect refactoring to preserve their code style. Example: ast converts `x=1` to `x = 1`, removes comments, changes quote style. LibCST maintains exact formatting. Trade-off: LibCST is more complex (concrete trees are larger), but necessary for professional tools. For Cursor: use ast for analysis (fast, simple), LibCST for modifications (preserves style). This is why Cursor's refactorings don't reformat your entire file - LibCST maintains formatting while changing structure.",
        keyPoints: [
            'AST loses comments, formatting, whitespace',
            'LibCST preserves exact source representation',
            'Essential for production tools to maintain style',
            'Trade-off: more complex but better UX',
        ],
    },
];
