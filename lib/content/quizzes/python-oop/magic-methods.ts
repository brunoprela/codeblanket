/**
 * Quiz questions for Magic Methods (Dunder Methods) section
 */

export const magicmethodsQuiz = [
  {
    id: 'q1',
    question:
      'What is the difference between __str__ and __repr__? When would you use each?',
    sampleAnswer:
      '__str__ provides a user-friendly, human-readable string representation (called by str() and print()), while __repr__ provides an unambiguous, developer-focused representation for debugging (called by repr() and used in containers). __repr__ should ideally allow recreating the object: eval(repr(obj)) == obj. Always implement __repr__ since Python falls back to it if __str__ is missing. Use __str__ when you need different user-facing output. Example: Point.__repr__ might be "Point(3, 4)" while __str__ might be "Point at (3, 4)".',
    keyPoints: [
      '__str__: user-friendly, called by print()',
      '__repr__: developer-friendly, unambiguous',
      '__repr__ used in containers and debugging',
      'Always implement __repr__, __str__ is optional',
      '__repr__ should enable object recreation if possible',
    ],
  },
  {
    id: 'q2',
    question:
      'Why must __hash__ be implemented when __eq__ is customized for hashable objects?',
    sampleAnswer:
      'Python requires that if a == b, then hash(a) == hash(b). The default __hash__ uses object identity (id), but custom __eq__ might consider two different objects equal based on their values. Without a custom __hash__, equal objects could have different hashes, breaking sets and dictionaries. For example, Person("Alice", "123") == Person("Alice", "123") with custom __eq__, but they would have different default hashes. The fix is to hash based on the same attributes used in __eq__: def __hash__(self): return hash(self.ssn). Note: only hash immutable attributes.',
    keyPoints: [
      'Equal objects must have equal hashes (hash invariant)',
      'Default __hash__ uses object identity (id)',
      'Custom __eq__ breaks hash invariant without custom __hash__',
      'Hash only immutable attributes',
      'Required for objects used as dict keys or in sets',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain how __getitem__ enables both indexing and iteration in Python.',
    sampleAnswer:
      '__getitem__(self, key) is called for indexing operations (obj[key]). If you implement __getitem__ with integer indices, Python automatically makes your object iterable—it tries obj[0], obj[1], obj[2] until IndexError is raised. This is called "sequence protocol". However, explicit __iter__ is preferred for iteration because it\'s more efficient and clearer. __getitem__ is essential for: (1) indexing: cart[0], (2) slicing: cart[1:3], (3) fallback iteration if __iter__ is missing. Pro tip: Implement both __getitem__ for indexing and __iter__ for efficient iteration.',
    keyPoints: [
      '__getitem__ enables obj[key] syntax',
      'Implementing __getitem__ with integers enables automatic iteration',
      'Python tries obj[0], obj[1], ... until IndexError',
      'Explicit __iter__ preferred for iteration (more efficient)',
      '__getitem__ also enables slicing support',
    ],
  },
];
