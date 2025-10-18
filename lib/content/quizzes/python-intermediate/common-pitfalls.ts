/**
 * Quiz questions for Common Python Pitfalls section
 */

export const commonpitfallsQuiz = [
  {
    id: 'q1',
    question: 'Why do mutable default arguments cause unexpected behavior?',
    sampleAnswer:
      "Default arguments are evaluated once when the function is defined, not each time it's called. For mutable defaults like [] or {}, the same object is reused across all calls. When you modify it (e.g., append), those changes persist. Example: def f(l=[]): l.append(1); return l → f() returns [1], f() returns [1,1], etc. Fix: Use None and create new object inside: def f(l=None): if l is None: l = []. This creates a fresh list each call.",
    keyPoints: [
      'Default args evaluated at function definition, not call time',
      'Same mutable object reused across calls',
      'Modifications persist between calls',
      'Fix: Use None, create new object inside',
      'Common with [], {}, but also class instances',
    ],
  },
  {
    id: 'q2',
    question: 'What is the difference between "is" and "==" in Python?',
    sampleAnswer:
      '"is" checks identity (same object in memory), "==" checks equality (same value). Example: a = [1,2]; b = [1,2] → a == b is True (same value) but a is b is False (different objects). Use "==" for value comparison (99% of cases). Use "is" only for singletons: None, True, False. Python caches small integers (-5 to 256), so a=100; b=100; a is b might be True, but a=1000; b=1000; a is b is False. Never rely on "is" for numbers/strings.',
    keyPoints: [
      'is: identity check (same object)',
      '==: equality check (same value)',
      'Use == for value comparison',
      'Use is only for None, True, False',
      "Small integers cached, don't rely on identity",
    ],
  },
  {
    id: 'q3',
    question:
      'Why is string concatenation in a loop inefficient and how do you fix it?',
    sampleAnswer:
      'String concatenation in loops is O(n²) because strings are immutable in Python. Each s += "x" creates a new string by copying all existing characters plus the new one. For n iterations: 1st copy=1 char, 2nd=2 chars, ..., nth=n chars. Total: 1+2+3+...+n = O(n²). Fix: Build a list and use join(): parts = []; for x in items: parts.append(str(x)); result = "".join(parts). This is O(n) because join() only copies characters once. Example: 10,000 concatenations take ~50 million operations with +=, but only 10,000 with join().',
    keyPoints: [
      'Strings immutable - each += copies all chars',
      'Loop concatenation: O(n²) time',
      'Solution: Build list, use join()',
      'join() is O(n) - copies once',
      'Huge performance difference for large strings',
    ],
  },
];
