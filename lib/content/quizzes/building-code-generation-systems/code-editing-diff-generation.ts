/**
 * Quiz questions for Code Editing & Diff Generation section
 */

export const codeeditingdiffgenerationQuiz = [
  {
    id: 'bcgs-codeediting-q-1',
    question:
      'Explain why the SEARCH/REPLACE format is more reliable for code editing than regenerating entire files. What are the specific failure modes it prevents, and when might regenerating still be preferable?',
    hint: 'Consider precision, context preservation, and error surface area.',
    sampleAnswer:
      "**Why SEARCH/REPLACE is More Reliable:** 1) **Preserves Context** - Keeps all unchanged code exactly as is, including comments, formatting, and subtle logic. Regeneration might lose these. 2) **Smaller Error Surface** - Only the changed lines can have errors. Regenerating entire file creates 100x more opportunities for errors. 3) **Easier Review** - Humans can quickly review specific changes vs scanning entire new file. 4) **Maintains Formatting** - Preserves indentation, line breaks, spacing that might be intentional. 5) **Prevents Regressions** - Won't accidentally change working code. **Failure Modes Prevented:** a) Losing inline comments explaining complex logic, b) Breaking unrelated code due to context misunderstanding, c) Changing variable names consistently throughout file, d) Modifying test data or config accidentally. **When to Regenerate:** 1) Changes affect >50% of file - more efficient to regenerate, 2) Complete restructuring needed - moving functions, changing architecture, 3) First generation attempt - no context to preserve, 4) Multiple cascading changes - too many SEARCH/REPLACE blocks would be confusing. Example: Adding one error handler? SEARCH/REPLACE. Converting sync to async throughout? Regenerate.",
    keyPoints: [
      'SEARCH/REPLACE preserves unchanged code and context',
      'Smaller error surface area (only changed lines)',
      'Prevents accidental regressions in working code',
      'Regenerate when changes affect >50% of file',
    ],
  },
  {
    id: 'bcgs-codeediting-q-2',
    question:
      "Your SEARCH block doesn't match exactly due to whitespace differences. Design a fuzzy matching system that's permissive enough to handle minor differences but strict enough to prevent wrong matches. What's the algorithm?",
    hint: 'Think about what differences are safe to ignore vs. what differences change meaning.',
    sampleAnswer:
      '**Fuzzy Matching Algorithm:** **1) Normalize Whitespace** (Safe to ignore) - Convert all runs of whitespace to single space EXCEPT: Keep indentation differences (they matter in Python), Keep empty lines (structural meaning). **2) Similarity Threshold** - Use difflib.SequenceMatcher, require ratio ≥ 0.85 (85% similar). Too low and we match wrong blocks, too high and minor formatting breaks matching. **3) Unique Context Requirements** - Match must be unique in file. If multiple blocks match at 85%, reject - too ambiguous. **4) Structural Anchors** - Preserve: function signatures (def name), class declarations, decorators. These identify blocks uniquely. **5) Multi-Line Matching** - For multi-line SEARCH, ensure line boundaries match. Don\'t match across function boundaries. **Algorithm:** ```python\ndef fuzzy_match(search, content, threshold=0.85):\n    lines = content.split("\\n")\n    search_lines = search.split("\\n")\n    \n    for i in range(len(lines) - len(search_lines) + 1):\n        candidate = "\\n".join(lines[i:i+len(search_lines)])\n        ratio = SequenceMatcher(None, normalize(search), normalize(candidate)).ratio()\n        \n        if ratio >= threshold:\n            # Check uniqueness\n            if only_one_match_above_threshold(search, content, threshold):\n                return candidate\n``` **Safety Checks:** If >1 match: reject, show options to user. If 0 matches: show closest match (even <85%) for debugging.',
    keyPoints: [
      'Normalize safe differences (extra spaces) but preserve meaningful ones (indentation)',
      'Use similarity threshold (85%) - balance permissiveness vs accuracy',
      'Require unique match - reject if ambiguous',
      'Preserve structural anchors (function signatures, class declarations)',
    ],
  },
  {
    id: 'bcgs-codeediting-q-3',
    question:
      'You need to apply multiple edits to a file sequentially, but later edits depend on earlier ones being applied first. How would you handle edit ordering, dependencies, and validation to ensure all edits apply correctly?',
    hint: 'Consider line number shifts, overlapping changes, and partial failures.',
    sampleAnswer:
      '**Sequential Edit Application System:** **1) Dependency Analysis** - Before applying, analyze edits: Detect overlapping changes (same lines), Identify dependencies (Edit B inserts line that Edit C modifies), Order by line number (ascending or descending). **2) Ordering Strategy** - Apply bottom-to-top (descending line numbers) - prevents line number shifts. Edit at line 100 doesn\'t affect edit at line 50. OR Apply top-to-bottom with offset tracking - track how many lines added/removed, adjust subsequent edit positions. **3) Conflict Detection** - Check if edits overlap: ```python\ndef detect_conflicts(edits):\n    for i, edit1 in enumerate(edits):\n        for edit2 in edits[i+1:]:\n            if ranges_overlap(edit1.lines, edit2.lines):\n                raise ConflictError("Edits overlap")``` **4) Atomic Application** - Apply all edits in transaction. If any fails, rollback all. Keep original content until all succeed. **5) Validation Per Edit** - After each edit: Verify syntax still valid, Check edit actually applied (verify presence of new code), Update line tracking for remaining edits. **6) Partial Failure Handling** - If edit 3 of 5 fails: Show which edits succeeded (1,2), Show which failed (3), Show which weren\'t attempted (4,5), Allow user to: retry failed ones, apply successfully only, rollback all. **Example:** Edits at lines [100, 50, 75]. Apply order: 100 → 75 → 50 (descending) to avoid line shifts. Or: 50 → 75+offset → 100+offset (ascending with tracking).',
    keyPoints: [
      'Apply bottom-to-top to avoid line number shifts',
      'Detect conflicts before applying any edits',
      'Use atomic transactions - all succeed or all rollback',
      'Validate syntax after each edit',
    ],
  },
];
