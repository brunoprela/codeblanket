/**
 * Quiz questions for Type Hints & Static Type Checking section
 */

export const typehintsQuiz = [
  {
    id: 'q1',
    question:
      'What is the difference between List[int] and list[int]? When should you use each?',
    hint: 'Consider Python versions and the typing module.',
    sampleAnswer:
      "List[int] is from the typing module and works in Python 3.5+. list[int] uses the built-in list type directly and works only in Python 3.9+. They mean the same thing: a list of integers. Use list[int] (lowercase) if you're on Python 3.9+ as it's simpler and doesn't require imports. Use List[int] (uppercase) if you need to support older Python versions. Python 3.9+ made built-in collections generic, so you can write dict[str, int] instead of Dict[str, int], list[str] instead of List[str], etc.",
    keyPoints: [
      'List[int]: typing module, Python 3.5+',
      'list[int]: built-in type, Python 3.9+',
      'Same meaning, different syntax',
      'Prefer lowercase (list[int]) in Python 3.9+',
      'Python 3.9+ made built-ins generic',
    ],
  },
  {
    id: 'q2',
    question:
      "Explain TypeVar and why it's useful for generic functions. How does it preserve type information?",
    hint: 'Think about what happens when you pass different types to the same function.',
    sampleAnswer:
      'TypeVar creates a generic type placeholder that preserves the actual type used. Without TypeVar, a function returning "any type" would lose type information. For example, def first(items: List) -> object means "returns something", but def first(items: List[T]) -> T (with T = TypeVar("T")) means "returns same type as input". This lets the type checker know that first([1,2,3]) returns an int, not just "some object". This is crucial for maintaining type safety in generic functions—the type checker can verify you\'re using the result correctly.',
    keyPoints: [
      'Creates generic type placeholder',
      'Preserves type information through function',
      'Without it, type information is lost',
      'Example: List[T] -> T preserves element type',
      'Enables type-safe generic functions',
    ],
  },
  {
    id: 'q3',
    question:
      "Why are type hints beneficial even though they don't affect runtime behavior? What problems do they solve?",
    hint: 'Consider development tools, documentation, and catching bugs.',
    sampleAnswer:
      'Type hints are zero-cost at runtime but invaluable during development. Benefits: 1) IDE autocomplete and inline error detection catch typos and type mismatches immediately, 2) Self-documenting code—function signatures show expected types, 3) Refactoring is safer—type checker catches breaking changes, 4) mypy catches bugs in CI/CD before deployment, 5) New developers understand code faster. For example, def process(data: Dict[str, List[int]]) -> Optional[int] immediately tells you what the function expects and returns, without reading documentation or code. The investment in adding types pays off in reduced bugs and development time.',
    keyPoints: [
      'Zero runtime cost, huge development benefits',
      'IDE catches errors as you type',
      'Self-documenting code',
      'Safer refactoring',
      'Catches bugs before runtime with mypy',
    ],
  },
];
