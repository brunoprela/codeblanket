/**
 * Quiz questions for Single File Code Generation section
 */

export const singlefilecodegenerationQuiz = [
    {
        id: 'bcgs-singlefile-q-1',
        question:
            'When generating a complete new file from scratch, what are the most critical validation steps before accepting the generated code? Prioritize them and explain your reasoning.',
        hint: 'Think about what can go wrong and the cost of each type of error.',
        sampleAnswer:
            'Critical validation steps in priority order: **1) Syntax Validation (Highest Priority)** - Must parse correctly. No point in other checks if code won\'t even parse. Use ast.parse() for Python. Cost of error: Code is unusable. **2) Import Validation** - Verify all imports exist. LLMs frequently hallucinate non-existent modules/functions. Cost: Runtime ImportError, harder to debug than syntax. **3) Type Consistency** - Check that type hints match actual usage. Prevents subtle bugs. **4) Basic Execution** - Run in sandbox with simple test case. Catches runtime errors like undefined variables. **5) Security Scan** - Check for dangerous patterns (eval, hardcoded secrets). **6) Style Validation** - Run linter, check conventions. Lowest priority as it doesn\'t affect functionality. Reasoning: Fatal errors first (syntax, imports), then correctness (types, execution), then quality (security, style). Each level depends on previous passing. Example: Even perfect style doesn\'t matter if imports don\'t exist. In production, might skip style check for speed but never skip first 3.',
        keyPoints: [
            'Syntax check is non-negotiable - code must parse',
            'Import validation catches LLM hallucinations early',
            'Type and execution checks ensure correctness',
            'Security and style are important but lower priority',
        ],
    },
    {
        id: 'bcgs-singlefile-q-2',
        question:
            'You\'re generating boilerplate code (API routes, database models, test files). How would you design a template system that balances reusability with customization? What information should be templated vs. generated fresh each time?',
        hint: 'Consider what stays constant across similar files vs. what must be unique.',
        sampleAnswer:
            '**Template (Reusable Patterns):** 1) **Structure** - File organization, import order, class/function layout. Example: API route always has: imports, route decorator, request/response models, error handling. 2) **Common Patterns** - Error handling structure, logging setup, validation approach. These are project-wide conventions. 3) **Boilerplate Code** - Standard try-catch blocks, common decorators, setup/teardown for tests. **Generate Fresh (Unique Per File):** 1) **Business Logic** - The actual functionality. No two endpoints are identical. 2) **Entity-Specific Details** - Model field names, validation rules, specific error messages. 3) **Function/Variable Names** - Must match the specific use case. **Hybrid Approach:** Use templates for structure, generate content. Example API route template: ```python\n@app.route("/{endpoint}")\nasync def {handler_name}({params}):\n    try:\n        # {GENERATE: Validation logic}\n        # {GENERATE: Business logic}\n        # {GENERATE: Response formatting}\n    except {TEMPLATE: Common exceptions}:\n        # {TEMPLATE: Error handling}\n``` This gives consistency (all routes structured same) with customization (business logic unique). Benefits: Fast generation, consistent patterns, reduces errors.',
        keyPoints: [
            'Template structure and common patterns for consistency',
            'Generate unique business logic and entity-specific details',
            'Hybrid approach: template structure + generated content',
            'Balance speed (templates) with flexibility (generation)',
        ],
    },
    {
        id: 'bcgs-singlefile-q-3',
        question:
            'Your file generator produces syntactically correct code that passes validation, but users complain it doesn\'t match their project\'s coding style. Design a system to learn and apply project-specific style patterns.',
        hint: 'Think about what constitutes "style" and how to extract it from existing code.',
        sampleAnswer:
            '**System Design for Learning Project Style:** **1) Style Pattern Extraction** - Analyze existing project files: a) **Naming Conventions** - Extract patterns: snake_case vs camelCase, prefixes (get_, set_, is_), function name lengths. b) **Documentation Style** - Docstring format (Google/NumPy/Sphinx), detail level, examples included? c) **Error Handling Patterns** - Exceptions vs error codes, specific exception types used, error message formats. d) **Code Organization** - Imports grouped (stdlib, third-party, local), function ordering, class structure. e) **Complexity Preferences** - List comprehensions vs loops, functional vs imperative, one-liners vs explicit. **2) Style Rules Generation** - Convert patterns to rules: "Use snake_case for functions", "Always include docstrings with examples", "Group imports by type with blank lines". **3) Application in Generation** - Add style rules to prompt: ```python\nProject Style Rules:\n- Function names: snake_case, avg length 15 chars\n- Docstrings: Google style with examples\n- Error handling: Specific exceptions (ValueError, TypeError)\n- Imports: Grouped (stdlib / third-party / local)\n``` **4) Validation** - Check generated code matches style rules. **Example**: Analyzing 50 files shows 95% use list comprehensions for simple filters → add rule "Prefer list comprehensions for simple filters".',
        keyPoints: [
            'Extract patterns from existing code (naming, docs, error handling)',
            'Convert patterns to explicit rules',
            'Include rules in generation prompts',
            'Validate generated code matches learned style',
        ],
    },
];

