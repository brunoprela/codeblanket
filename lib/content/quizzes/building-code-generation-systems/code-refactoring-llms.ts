/**
 * Quiz questions for Code Refactoring with LLMs section
 */

export const coderefactoringllmsQuiz = [
  {
    id: 'bcgs-refactoring-q-1',
    question:
      'When performing an automated rename refactoring across a large codebase, what are the risks and how would you mitigate them? Consider both false positives and false negatives.',
    hint: 'Think about scope boundaries, string literals, comments, and similar names.',
    sampleAnswer:
      '**Risks in Automated Rename:** **False Positives (Renaming Wrong Things):** 1) Same name in different scopes (local variable vs parameter vs global), 2) Name in string literals/comments ("user" in docs), 3) Similar names (rename "user" catches "username"), 4) Name in different files with no relation. **False Negatives (Missing Renames):** 1) Dynamic access (getattr (obj, "old_name")), 2) String-based references (route names, config keys), 3) Generated code referencing the name, 4) External files (docs, configs). **Mitigation Strategies:** **1) Scope-Aware Analysis** - Use AST to understand scopes: ```python\\ndef find_in_scope (name, file, scope_type):\\n    # Only rename if in correct scope\\n    if scope_type == "class":\\n        find_in_class_only()\\n    elif scope_type == "function":\\n        find_in_function_only()``` **2) Semantic Understanding** - Don\'t rename in: String literals (unless specifically requested), Comments (update separately), Unrelated files (check imports first). **3) Confirmation** - Show preview of all changes, Group by certainty (100% safe vs needs review), Require user confirmation for ambiguous cases. **4) Test-Driven** - Run tests after rename, Rollback if tests fail, Report which tests broke. **Example:** Renaming "process" function. Risk: "processing" variable shouldn\'t change, "process" string in SQL query might need to change. Solution: AST-based rename of function only, separate pass for string analysis with user confirmation.',
    keyPoints: [
      'Use AST for scope-aware analysis',
      'Exclude strings/comments unless specifically requested',
      'Show preview and require confirmation for ambiguous cases',
      'Run tests and rollback if failures occur',
    ],
  },
  {
    id: 'bcgs-refactoring-q-2',
    question:
      'Explain the "extract function" refactoring. What makes it challenging for an LLM, and how would you design prompts to ensure the extracted function has correct parameters and return values?',
    hint: 'Consider variable scope, dependencies, and side effects.',
    sampleAnswer:
      '**Extract Function Challenges:** 1) **Parameter Detection** - Must identify variables from outer scope used in extracted code. Missing one causes NameError. 2) **Return Value** - Must identify what values the extracted code produces that outer code needs. 3) **Side Effects** - If code modifies external state, need to handle carefully. 4) **Proper Scope** - Variables defined inside extracted block shouldn\'t become parameters. **Prompt Design:** ```python\\nprompt = f"""Extract this code into a function:\\n\\nCode to extract (lines {start}-{end}):\\n{code_block}\\n\\nContext before:\\n{context_before}\\n\\nContext after:\\n{context_after}\\n\\nAnalyze:\\n1) Which variables from context are USED in code block? → parameters\\n2) Which variables are DEFINED in block and USED after? → return values  \\n3) Are there side effects (file I/O, global state)?\\n4) What is appropriate name for this function?\\n\\nOutput:\\n- Function definition with parameters and return type\\n- Function call to replace extracted code\\n- Explanation of parameters and return value"""``` **Validation:** After extraction: Check no NameError (all variables defined), Check return value used correctly, Run original tests (behavior unchanged), Verify side effects preserved. **Example:** ```python\\nresult = []\\nfor item in items:  # Extract this\\n    if item.price > 100:\\n        result.append (item.name)\\n# result used here\\n``` Parameters: items (used from context), Return: result (defined and used after), Function: `def filter_expensive_items (items): ...`',
    keyPoints: [
      'Parameters = variables from outer scope used in block',
      'Return value = variables defined in block and used after',
      'Validate no NameErrors and behavior unchanged',
      'Consider side effects carefully',
    ],
  },
  {
    id: 'bcgs-refactoring-q-3',
    question:
      'You\'re building a "change signature" refactoring that updates a function signature and all its call sites. What are the edge cases you must handle, and how would you ensure no calls are missed?',
    hint: 'Think about different call patterns, dynamic calls, and indirect references.',
    sampleAnswer:
      '**Edge Cases in Change Signature:** **1) Different Call Patterns** - Direct: `function (arg1, arg2)`, Method: `obj.function (arg1)`, Keyword args: `function (arg1=x, arg2=y)`, Unpacking: `function(*args, **kwargs)`, Partial application: `partial (function, arg1)`. **2) Dynamic Calls** - `getattr (obj, "function")()` - String-based, can\'t analyze statically, `globals()["function",]()` - Dynamic lookup, `eval/exec` - Impossible to analyze. **3) Indirect References** - Function assigned to variable: `f = function; f()`, Passed as callback: `map (function, items)`, Decorator chains: `@decorator\\ndef function()`. **4) Cross-File References** - Imports: `from module import function`, Relative imports, `import module; module.function()`. **Ensuring No Calls Missed:** **1) Multi-Pass Analysis** - AST analysis for static calls (99% of cases), Grep for string occurrences (catches some dynamic), Manual review list for user check. **2) Cross-Reference Building** - Find ALL files that import the module, Search each for function name, Build call graph. **3) Validation Strategy** - After changes: Import all modules (catch ImportError), Run type checker (catch signature mismatches), Run all tests (catch logic errors), Check for remaining references: `grep -r "function_name" .` **4) Reporting** - Show all found calls and their updates, Flag suspicious patterns for review, Provide "couldn\'t analyze" list. **Example:** Changing `def process (x, y)` to `def process (x, y, z=0)`. Must update: `process(1, 2)` → `process(1, 2)` (default works), But catch: `map (process, items)` might break (arity changed).',
    keyPoints: [
      'Handle direct calls, methods, keyword args, and unpacking',
      'Use AST for static analysis + grep for dynamic patterns',
      'Build cross-file reference graph',
      'Validate with imports, type checking, and tests',
    ],
  },
];
