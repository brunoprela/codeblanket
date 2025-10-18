/**
 * Quiz questions for None and Null Values section
 */

export const nonehandlingQuiz = [
  {
    id: 'pf-none-q-1',
    question:
      'Why should you use "is None" instead of "== None" to check for None? What is the fundamental difference?',
    hint: 'Think about identity vs equality and how None is implemented.',
    sampleAnswer:
      '"is None" checks object identity - whether the variable points to the exact same None object in memory. "== None" checks value equality, which can be overridden by implementing __eq__. Since None is a singleton (only one None object exists in Python), "is" is both more efficient (no method call) and more correct (can\'t be fooled by classes that define __eq__ to return True when compared to None). Additionally, "is None" is the idiomatic Pythonic way and is recommended by PEP 8. While "== None" usually works, edge cases exist where custom __eq__ methods could break it.',
    keyPoints: [
      '"is" checks identity (same object in memory)',
      '"==" checks equality (can be overridden)',
      'None is a singleton',
      '"is" is more efficient and correct',
      'PEP 8 recommends "is None"',
    ],
  },
  {
    id: 'pf-none-q-2',
    question:
      'Explain the mutable default argument problem. Why should you use None as default instead of [] or {}?',
    hint: 'Consider when default arguments are created and how they are shared between function calls.',
    sampleAnswer:
      'Default arguments are created once when the function is defined, not each time the function is called. If you use a mutable default like [] or {}, all calls to the function share the same list/dict object. Example: "def add(item, items=[])" - the first call creates the list, and subsequent calls reuse it, causing items to accumulate across calls. This is almost never the intended behavior. The solution is to use None as default and create a new mutable object inside the function: "def add(item, items=None): if items is None: items = []". This ensures each call gets its own fresh list.',
    keyPoints: [
      'Default arguments created once at function definition',
      'Mutable defaults shared between all calls',
      'Causes unexpected accumulation of data',
      'Use None as default, create mutable inside function',
      'Pattern: if items is None: items = []',
    ],
  },
  {
    id: 'pf-none-q-3',
    question:
      'When should you return None vs an empty collection ([], {}, "")? What are the trade-offs?',
    hint: 'Consider how callers will use the return value and what makes their code simpler.',
    sampleAnswer:
      'Return empty collections when the function is querying/searching and the "not found" case is normal, not exceptional. This lets callers iterate or check membership without None checks: "for item in get_items()" works even if no items. Return None when: 1) distinguishing "no result" from "empty result" matters (None = error/not-found, [] = found but empty), 2) the operation failed or is invalid, 3) the value is truly optional/unset. Example: search_users() returning [] means "no matches" (normal), returning None means "search failed" (error). Generally, prefer empty collections for better API usability unless None conveys important semantic information.',
    keyPoints: [
      'Empty collection: "not found" is normal, allows iteration',
      'None: operation failed or value truly missing',
      'Empty collection makes caller code simpler',
      'None when distinguishing error from empty result',
      'Example: [] for "no matches", None for "search failed"',
    ],
  },
];
