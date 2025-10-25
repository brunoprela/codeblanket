/**
 * Quiz questions for Variables and Data Types section
 */

export const variablestypesQuiz = [
  {
    id: 'pf-variables-q-1',
    question:
      'Explain the difference between mutable and immutable types in Python. Why does it matter when assigning variables?',
    hint: 'Think about what happens when you modify a list vs. a string, and what happens with variable assignment.',
    sampleAnswer:
      'In Python, immutable types (like int, float, str, tuple) cannot be changed after creation. When you "modify" them, you actually create a new object. Mutable types (like list, dict, set) can be changed in place. This matters for variable assignment because multiple variables can point to the same mutable object, so changes through one variable affect all references. For example: a = [1, 2]; b = a; b.append(3) will modify the list that both a and b reference.',
    keyPoints: [
      'Immutable types: int, float, str, tuple, frozenset',
      'Mutable types: list, dict, set',
      'Assignment creates references, not copies',
      'Use .copy() or copy.deepcopy() for actual copies',
    ],
  },
  {
    id: 'pf-variables-q-2',
    question:
      'When would you choose to use a float instead of an integer in Python? What are the potential pitfalls of using floats?',
    hint: 'Consider precision requirements, mathematical operations, and how computers represent decimal numbers.',
    sampleAnswer:
      "Use floats when you need decimal precision, like measurements, scientific calculations, or division operations. However, floats have precision limitations due to binary representation. For example, 0.1 + 0.2 doesn't exactly equal 0.3 in Python due to floating-point arithmetic. For financial calculations requiring exact decimal precision, use the Decimal class instead. Use integers when working with counts, indices, or when exact values are critical.",
    keyPoints: [
      'Floats are for decimal/fractional values',
      'Binary representation causes precision issues',
      'Use Decimal class for exact decimal arithmetic',
      'Integers are exact and should be preferred when possible',
    ],
  },
  {
    id: 'pf-variables-q-3',
    question:
      "Explain Python\'s dynamic typing. What are the advantages and disadvantages compared to statically-typed languages?",
    hint: 'Think about flexibility, runtime behavior, debugging, and development speed.',
    sampleAnswer:
      "Python\'s dynamic typing means variables don't have fixed types - the same variable can hold different types at different times. This offers flexibility and faster prototyping since you don't declare types. However, it can lead to runtime type errors that statically-typed languages catch at compile time. Tools like type hints (PEP 484) and mypy can add static type checking while maintaining dynamic runtime behavior, giving you the best of both worlds.",
    keyPoints: [
      'Variables can change types freely',
      'No compile-time type checking',
      'More flexible but can hide bugs until runtime',
      'Type hints (annotations) can add static checking',
    ],
  },
];
