/**
 * Quiz questions for Code Similarity & Clone Detection section
 */

export const codesimilaritydetectionQuiz = [
  {
    id: 'cuam-codesimilaritydetection-q-1',
    question:
      'Explain the four types of code clones (Type I-IV) and why detecting Type III clones is significantly harder than Type I.',
    hint: 'Consider exact matches, renamed variables, structural differences, and semantic equivalence.',
    sampleAnswer:
      "**Type I**: Exact clones (ignore whitespace/comments) - identical code. **Type II**: Renamed clones - same structure, different names (`total` vs `sum`). **Type III**: Near-miss clones - similar with modifications (added statements, different conditions). **Type IV**: Semantic clones - different implementation, same behavior (`for` loop vs `sum()` function). Type III is harder because: 1) **Fuzzy matching** - need similarity threshold (how different is too different?), 2) **Structural variations** - added/removed statements break simple comparison, 3) **False positives** - similar structure doesn't mean clones (common patterns), 4) **Performance** - can't use exact hash matching, need expensive comparisons. Type I: hash code, O(n). Type III: compare all pairs, O(n²), with complex similarity calculation. For Cursor: focus on Type I/II for quick wins, Type III for deep analysis, Type IV is often intentional (different solutions to same problem). Balance: catch meaningful duplications without overwhelming users with noise.",
    keyPoints: [
      'Type I: exact, Type II: renamed, Type III: near-miss, Type IV: semantic',
      'Type III requires fuzzy matching and thresholds',
      'Structural variations complicate comparison',
      'Trade-off: thoroughness vs false positives',
    ],
  },
  {
    id: 'cuam-codesimilaritydetection-q-2',
    question:
      'How does Cursor use clone detection to improve code suggestions? Give examples of how finding similar code patterns helps AI assistance.',
    hint: 'Think about examples, consistency, and refactoring opportunities.',
    sampleAnswer:
      "Cursor uses clone detection for: 1) **Example-based suggestions** - 'You wrote similar code in fileA, here's that pattern', 2) **Consistency enforcement** - if 5 functions use pattern X, suggest X for new similar function, 3) **Refactoring prompts** - 'These 3 functions are clones, extract common logic?', 4) **Learning coding style** - detect your patterns (prefer list comprehensions, guard clauses, etc.) and apply to generated code. Examples: Writing validation function → Cursor: 'Similar to validate_user, validate_product. Pattern: if not x: return False'. Writing API call → Cursor: 'You usually wrap API calls in try/except with logging, add that?'. Writing loop → Cursor: 'Detected accumulator pattern 5 times, use sum()?' Clone detection reveals **implicit coding standards** - what patterns you prefer, how you structure code. This makes AI suggestions feel native to your codebase - not generic, but following YOUR established patterns. Also: bulk refactoring - 'Found 10 clones of this pattern, refactor all?'",
    keyPoints: [
      'Shows similar examples from codebase',
      'Enforces consistency with existing patterns',
      'Suggests refactoring for duplicates',
      'Learns and applies your coding style',
    ],
  },
  {
    id: 'cuam-codesimilaritydetection-q-3',
    question:
      'Why is AST-based similarity better than text-based similarity for code clone detection? What does it handle that text comparison misses?',
    hint: 'Consider formatting, comments, and structural equivalence.',
    sampleAnswer:
      "AST-based similarity is better because it focuses on **semantic structure, not syntax**. What it handles:\n\n**1. Formatting independence:**\n```python\n# Text: different\n# AST: identical\nif x>0:return x\nif x > 0:\n    return x\n```\n\n**2. Comment differences:**\n```python\n# Text: different\n# AST: identical\nresult = x + y  # Calculate sum\nresult = x + y  # Add values\n```\n\n**3. Renamed variables (Type II):**\nAST can normalize names (replace all identifiers with IDENTIFIER token), detecting clones despite different variable names. Text comparison fails.\n\n**4. Structural equivalence:**\nAST compares tree structure - two implementations with different formatting but same control flow are detected. Text needs exact match.\n\n**Limitations**: AST approach requires valid syntax (can't detect clones in broken code), and is slower than text hashing. But catches real clones that text misses due to formatting. For production: use text for fast Type I detection, AST for accurate Type II/III. This is why Cursor's clone detection is smart - doesn't flag reformatted code as 'duplicate'.",
    keyPoints: [
      'Focuses on structure, not formatting',
      'Handles whitespace and comment differences',
      'Can normalize names for Type II detection',
      'Catches semantically similar code',
    ],
  },
];
