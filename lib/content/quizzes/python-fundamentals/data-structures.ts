/**
 * Quiz questions for Data Structures section
 */

export const datastructuresQuiz = [
  {
    id: 'pf-datastructures-q-1',
    question:
      'Explain when you should use a list, tuple, set, or dictionary. What are the key trade-offs between them?',
    hint: 'Consider mutability, ordering, uniqueness, and access patterns.',
    sampleAnswer:
      "Use **lists** for ordered, mutable sequences when you need to modify elements, add/remove items, or maintain order. Use **tuples** for immutable sequences, like function returns or dictionary keys, when data shouldn't change. Use **sets** when you need unique elements and don't care about order - great for membership testing and removing duplicates. Use **dictionaries** for key-value mappings when you need fast lookup by key. Trade-offs: lists are flexible but slower for membership tests; tuples are faster and memory-efficient but immutable; sets are fast for membership but unordered; dicts provide fast access but use more memory.",
    keyPoints: [
      'Lists: ordered, mutable, allows duplicates',
      'Tuples: ordered, immutable, hashable',
      'Sets: unordered, mutable, unique elements only',
      'Dicts: key-value pairs, fast lookups',
    ],
  },
  {
    id: 'pf-datastructures-q-2',
    question:
      "Why are list comprehensions considered more Pythonic than traditional for loops? Are there cases where you shouldn't use them?",
    hint: 'Think about readability, performance, and complexity.',
    sampleAnswer:
      'List comprehensions are more Pythonic because they\'re concise, readable (for simple operations), and often faster than equivalent for loops. They express the intent "create a list from a transformation" clearly. However, avoid them when: 1) Logic is complex (multiple conditions, nested loops) - they become hard to read, 2) You need side effects during iteration, 3) The expression is very long, 4) You\'re not actually creating a list (use generator expressions instead). Remember: "Explicit is better than implicit" - if a comprehension is confusing, use a traditional loop.',
    keyPoints: [
      'More concise and often faster',
      'Better for simple transformations and filters',
      'Avoid when logic is complex or needs debugging',
      'Generator expressions for memory efficiency',
    ],
  },
  {
    id: 'pf-datastructures-q-3',
    question:
      'What is the difference between dict.get() and dict[key]? When would you use each method?',
    hint: "Consider what happens when a key doesn't exist and when you want different behaviors.",
    sampleAnswer:
      'dict[key] raises a KeyError if the key doesn\'t exist, while dict.get (key, default) returns None (or a specified default) if the key is missing. Use dict[key] when you expect the key to exist and want to catch programming errors - the KeyError signals a bug. Use dict.get() when missing keys are valid, like when checking optional configuration settings, or when you want to provide a default value. For example: config.get("debug", False) is cleaner than checking "if \'debug\' in config" first.',
    keyPoints: [
      'dict[key]: raises KeyError if missing',
      'dict.get (key, default): returns default if missing',
      'Use [key] when absence is an error',
      'Use .get() for optional values',
    ],
  },
];
