/**
 * Quiz questions for Documentation & Comment Extraction section
 */

export const documentationextractionQuiz = [
  {
    id: 'cuam-documentationextraction-q-1',
    question:
      "Why are comments not included in Python's AST? How do you extract them, and why is this important for AI coding tools?",
    hint: 'Think about the tokenize module and the semantic vs syntactic distinction.',
    sampleAnswer:
      "Comments aren't in AST because **AST represents program structure, not documentation**. Comments don't affect execution so they're discarded during parsing. To extract: use Python's **tokenize module** which processes source at token level (before AST), where comments are token.COMMENT tokens. Why important for AI tools: 1) **Human intent** - comments explain WHY, not just WHAT, 2) **TODO/FIXME** - action items that should inform suggestions, 3) **Context** - 'Legacy code, don't modify' prevents bad suggestions, 4) **Examples** - inline examples show usage patterns. For Cursor: comments provide crucial context that pure code analysis misses. Example: `# TODO: Add error handling` tells Cursor what's missing; `# This breaks with negative numbers` warns about limitations. Without comment extraction, Cursor would miss explicit developer guidance. Combine AST (structure) + comments (intent) = complete understanding.",
    keyPoints: [
      'AST focuses on executable structure',
      'Use tokenize module for comment extraction',
      'Comments contain human intent and context',
      'Critical for understanding developer guidance',
    ],
  },
  {
    id: 'cuam-documentationextraction-q-2',
    question:
      'How would you detect which docstring format (Google, NumPy, or Sphinx) a codebase uses and ensure generated documentation matches?',
    hint: 'Consider marker strings, parsing patterns, and consistency.',
    sampleAnswer:
      "Detect format by checking **marker strings** in existing docstrings: 1) **Google**: look for 'Args:', 'Returns:', 'Raises:', 2) **NumPy**: look for 'Parameters' followed by '----------' (underline), 3) **Sphinx**: look for ':param name:', ':return:', ':type:'. Algorithm: scan all docstrings, count format markers, majority wins. Example:\n```python\ndef analyze_codebase():\n    formats = {'google': 0, 'numpy': 0, 'sphinx': 0}\n    for docstring in all_docstrings:\n        if 'Args:' in docstring: formats['google',] += 1\n        elif '----------' in docstring: formats['numpy',] += 1\n        elif ':param' in docstring: formats['sphinx',] += 1\n    return max(formats, key=formats.get)\n```\nEnsure consistency: when generating docstrings, use detected format. For Cursor: respect project conventions. If codebase uses Google style, generate Google style. This maintains consistency, crucial for teams. Store format preference per-project. Also respect line length, indentation style from existing docs.",
    keyPoints: [
      'Detect via marker strings in existing docstrings',
      'Count format occurrences, use majority',
      'Generate new docstrings in detected format',
      'Maintains project consistency',
    ],
  },
  {
    id: 'cuam-documentationextraction-q-3',
    question:
      'Why is documentation coverage (% of functions with docstrings) a useful metric? What are its limitations?',
    hint: "Think about quality vs quantity and what coverage doesn't measure.",
    sampleAnswer:
      "**Documentation coverage** = (documented functions / total functions) × 100. Useful because: 1) **Simple metric** - easy to calculate and track, 2) **Improvement driver** - teams can set goals (80% coverage), 3) **Red flag detector** - 20% coverage indicates problem, 4) **Progress tracking** - watch trend over time. **Limitations**: 1) **Quality blind** - counts 'TODO: add docs' as documented (not useful!), 2) **Context missing** - doesn't measure if docs are accurate or helpful, 3) **Not all functions equal** - public API needs docs more than private helpers, 4) **Doesn't catch outdated docs** - function changed but docs didn't. Better metrics: doc quality score (completeness, parameters documented, examples included), doc freshness (last updated vs code changes). For Cursor: use coverage as first pass (find undocumented code), but also analyze doc quality - check if parameters match, if examples exist, if docs describe behavior. Combine quantity + quality metrics.",
    keyPoints: [
      'Useful: simple, trackable, motivating',
      'Limitations: quality-blind, context-free',
      "Doesn't measure accuracy or helpfulness",
      'Combine with quality metrics for better insight',
    ],
  },
];
