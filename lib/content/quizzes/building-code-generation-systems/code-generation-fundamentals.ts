/**
 * Quiz questions for Code Generation Fundamentals section
 */

export const codegenerationfundamentalsQuiz = [
    {
        id: 'bcgs-codegenfund-q-1',
        question:
            'Why is code generation fundamentally more challenging than generating prose? Explain the key differences and give examples of how each challenge manifests in practice.',
        hint: 'Think about precision requirements, executability, syntax rules, and consequences of errors.',
        sampleAnswer:
            "Code generation is more challenging than prose for several critical reasons: 1) **Precision Requirements** - Code must be 100% syntactically correct; even a missing bracket breaks everything, while prose can have minor errors and still be readable. 2) **Executability** - Code must actually run without errors, whereas prose just needs to convey meaning. 3) **Context Dependencies** - Code rarely exists in isolation; it depends on imports, function signatures, and project structure. A function that looks perfect in isolation might fail because of missing dependencies. 4) **Testing Requirements** - Generated code must be validated through execution; you can't just read it and assume it works. For example, a function that looks correct might have edge cases that cause runtime errors (like division by zero with empty lists). These challenges mean code generation requires validation at every step, unlike prose where 'close enough' often suffices.",
        keyPoints: [
            'Code requires 100% syntactic correctness vs ~90% for readable prose',
            'Must be executable and handle edge cases correctly',
            'Strong dependencies on imports, types, and project context',
            'Requires validation through testing, not just human review',
        ],
    },
    {
        id: 'bcgs-codegenfund-q-2',
        question:
            'Describe a comprehensive validation pipeline for generated code. What checks should it include, in what order, and why? How would you handle validation failures at each stage?',
        hint: 'Consider syntax, imports, security, style, and execution. Think about early vs late stage checks.',
        sampleAnswer:
            'A comprehensive validation pipeline should include: 1) **Syntax Check** (first) - Parse with AST to catch syntax errors immediately; no point continuing if code won\'t parse. Handle failure: retry with error context. 2) **Import Validation** - Verify all imports exist; prevents hallucinated dependencies. Handle failure: remove invalid imports or retry generation. 3) **Security Scan** - Check for dangerous functions (eval, exec), SQL injection patterns, hardcoded secrets. Handle failure: reject and regenerate with security constraints. 4) **Style Check** - Run linters (pylint, black) for consistency. Handle failure: auto-fix if possible, warn otherwise. 5) **Static Analysis** - Check types, detect code smells, complexity metrics. Handle failure: suggest refactoring. 6) **Execution in Sandbox** - Run code in Docker container with resource limits. Handle failure: retry with test failure context. This order is crucial: catch cheap issues (syntax) before expensive ones (execution). Each failure should feed back to the generator for improved retry attempts.',
        keyPoints: [
            'Order matters: syntax → imports → security → style → execution',
            'Early checks are cheap, later checks are expensive but comprehensive',
            'Each failure should provide context for retry attempts',
            'Always sandbox execution for safety',
        ],
    },
    {
        id: 'bcgs-codegenfund-q-3',
        question:
            'You\'re building a production code generation system. When should you regenerate code vs. when should you apply targeted fixes? Design a decision framework with specific criteria and examples.',
        hint: 'Consider the type of error, cost, context preservation, and user experience.',
        sampleAnswer:
            'Decision framework for regeneration vs. targeted fixes: **Regenerate When:** 1) Fundamental logic errors - The entire approach is wrong (e.g., wrong algorithm chosen). 2) Multiple cascading errors - Fixing one would expose many others. 3) Missing major requirements - Forgot to handle async, error handling, etc. 4) First generation attempt - No context to preserve yet. Example: Generated sync code but user wanted async. **Apply Targeted Fixes When:** 1) Syntax errors - Single missing bracket, typo in variable name. 2) Import errors - Missing or wrong import statement. 3) Type errors - Wrong type hint or missing Optional. 4) Style violations - Naming convention, formatting issues. 5) Edge case handling - Function works but misses edge cases. Example: Division by zero check missing. **Criteria:** Cost (regeneration is 10x more expensive), Context Preservation (targeted fixes keep good code), User Experience (fixes are faster), Error Isolation (is error localized?). In production, prefer targeted fixes for <20% of code changes, regenerate for >50% changes.',
        keyPoints: [
            'Regenerate for fundamental logic errors or major missing features',
            'Apply targeted fixes for localized issues (syntax, imports, types)',
            'Consider cost: regeneration is ~10x more expensive than fixes',
            'Preserve context and user experience with targeted fixes when possible',
        ],
    },
];

