/**
 * Quiz questions for Static Analysis & Code Quality section
 */

export const staticanalysisQuiz = [
  {
    id: 'cuam-staticanalysis-q-1',
    question:
      'Explain the difference between cyclomatic complexity and cognitive complexity. Why does cognitive complexity better predict maintainability?',
    hint: 'Consider nesting, human understanding, and real-world code difficulty.',
    sampleAnswer:
      "**Cyclomatic complexity** counts decision points: if/for/while/except each add +1. **Cognitive complexity** adds penalties for **nesting** - deeply nested code is harder to understand. Example:\n```python\n# Both have 3 decisions\n# Cyclomatic: 4, Cognitive: 4\nif a: if b: if c: pass\n\n# Cyclomatic: 4, Cognitive: 3\nif a: pass\nif b: pass\nif c: pass\n```\nThe nested version is harder to understand but has same cyclomatic complexity. Cognitive complexity better predicts maintainability because: 1) **Nested code** requires holding more context in mind, 2) **Linear sequences** of conditions are easier to follow than nested ones, 3) Matches human perception of difficulty. Studies show cognitive complexity correlates better with bugs and time-to-understand. For Cursor: use cognitive complexity to warn about genuinely hard-to-maintain code, not just any code with decisions. It's why flat code with guard clauses scores better than deeply nested code - reflects real maintainability.",
    keyPoints: [
      'Cyclomatic: count decision points',
      'Cognitive: adds nesting penalties',
      'Cognitive matches human difficulty better',
      'Predicts maintainability more accurately',
    ],
  },
  {
    id: 'cuam-staticanalysis-q-2',
    question:
      'How can static analysis detect the "mutable default argument" bug in Python? Why is this bug so common and dangerous?',
    hint: "Think about AST nodes for defaults and Python's behavior with mutable objects.",
    sampleAnswer:
      "Detect by checking FunctionDef.args.defaults for **ast.List, ast.Dict, ast.Set nodes** - these are mutable literals used as defaults. The bug:\n```python\ndef add_item(item, items=[]):  # Bug!\n    items.append(item)\n    return items\n```\nDangerous because: 1) **Defaults are evaluated once** at function definition, not per call, 2) Same list object is shared across all calls, 3) Leads to unexpected state accumulation. It's common because it looks innocent - natural to write. Static analysis saves you: when it sees ast.List in defaults, warn 'Mutable default argument'. Safe pattern: `items=None` then `if items is None: items = []` inside function. This creates new list per call. Cursor flags this immediately because it's a well-known Python gotcha - static analysis can catch it 100% of the time (unlike runtime bugs). Demonstrates value of static analysis: catching language-specific pitfalls before runtime.",
    keyPoints: [
      'Check for ast.List/Dict/Set in defaults',
      'Bug: defaults evaluated once, shared across calls',
      'Leads to unexpected state accumulation',
      'Common pitfall static analysis catches reliably',
    ],
  },
  {
    id: 'cuam-staticanalysis-q-3',
    question:
      'Why is detecting prompt injection or SQL injection harder than detecting other security issues with static analysis? What limitations exist?',
    hint: 'Consider data flow, string construction, and false positives.',
    sampleAnswer:
      "Injection detection is hard because: 1) **Requires data flow tracking** - need to know if untrusted input reaches dangerous function, 2) **String construction is complex** - f-strings, +, format(), join() all build strings differently, 3) **Validation is hard to detect** - static analysis can't know if `sanitize(input)` actually sanitizes, 4) **False positives** - flagging safe cases (hardcoded strings, validated input) annoys users. Example: `cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")` - need to: track user_id source (HTTP request = untrusted), detect string interpolation into execute() call, recognize execute() as dangerous. **Limitations**: 1) Can't track through complex flows, 2) Doesn't know which functions validate input, 3) Indirect construction (variables, functions) breaks simple pattern matching. Solution: flag obvious cases (f-strings in execute()), educate about parameterized queries, accept some false positives/negatives. This is why security tools combine static + dynamic analysis - static catches obvious cases, dynamic catches runtime-specific issues.",
    keyPoints: [
      'Requires complex data flow tracking',
      'String construction has many forms',
      'Hard to distinguish validated from untrusted input',
      'Balance: catch obvious cases, accept limitations',
    ],
  },
];
