/**
 * Quiz questions for Type System Understanding section
 */

export const typesystemunderstandingQuiz = [
    {
        id: 'cuam-typesystemunderstanding-q-1',
        question:
            'How does type inference help Cursor provide intelligent suggestions even in untyped Python code? Give examples of what can be inferred from usage patterns.',
        hint: 'Think about operations, method calls, and control flow.',
        sampleAnswer:
            "Type inference derives types from **usage patterns** without annotations: 1) **Literals**: `x = 5` → infer int, 2) **Operations**: `result = x + y` → numeric types, 3) **Method calls**: `data.append(item)` → data is list-like, 4) **Subscripting**: `config[\"key\"]` → dict/sequence, 5) **Built-in calls**: `len(items)` → items is sized collection, returns int, 6) **Control flow**: `if not value:` → value is boolean-testable. Example: ```python\ndef process(items):\n    total = 0  # infer int\n    for item in items:  # items is iterable\n        if item > 10:  # item is comparable numeric\n            total += item\n    return total  # returns int\n```\nWithout annotations, Cursor infers: items→list[int], return→int from usage. When you type `total.` it suggests int methods (bit_length, etc). When you type `items.` it suggests list methods. This enables intelligence without requiring full type annotations - crucial since most Python code isn't fully typed. Inference covers ~70% of cases; annotations handle the rest.",
        keyPoints: [
            'Infer from literals, operations, method calls',
            'Track types through assignments and returns',
            'Enable suggestions without annotations',
            'Covers ~70% of real-world code',
        ],
    },
    {
        id: 'cuam-typesystemunderstanding-q-2',
        question:
            'Why is Optional[T] different from T in type checking? What bugs does proper Optional handling prevent?',
        hint: 'Consider None values and attribute access.',
        sampleAnswer:
            "Optional[T] means **'T or None'** - must handle None case. Regular T assumes non-None. Bugs prevented:\n\n**1. None attribute access:**\n```python\ndef get_user(id: int) -> Optional[User]:\n    return None  # user not found\n\nuser = get_user(123)\nprint(user.name)  # Bug! user could be None\n```\nProper handling: `if user: print(user.name)`\n\n**2. Unexpected None propagation:**\nFunction returning Optional[T] forces callers to consider None, preventing cascading failures.\n\n**3. Type narrowing:**\nAfter `if user:` check, type checker narrows Optional[User] to User in that branch - enables safe access. Without Optional: assume everything is present, get runtime AttributeError. With Optional: forced to handle None explicitly. This is how Cursor warns 'possible None value' - tracks Optional types through the codebase. Makes None handling explicit rather than implicit, preventing billion-dollar mistake (null pointer errors).",
        keyPoints: [
            'Optional[T] requires None handling',
            'Prevents None attribute access bugs',
            'Type narrowing after None checks',
            'Makes null safety explicit in types',
        ],
    },
    {
        id: 'cuam-typesystemunderstanding-q-3',
        question:
            'How do type annotations enable better code completion than untyped code? Compare completion for typed vs untyped function parameters.',
        hint: 'Think about what the IDE knows about types and available methods.',
        sampleAnswer:
            "Type annotations give IDE **exact information** about what's available:\n\n**Untyped:**\n```python\ndef process(data):\n    data.  # IDE shows: generic object methods only\n```\nWithout type, IDE can't suggest anything useful - just universal methods like __str__.\n\n**Typed:**\n```python\ndef process(data: pd.DataFrame):\n    data.  # IDE shows: head(), tail(), describe(), 600+ DataFrame methods!\n```\nWith type annotation, IDE knows it's a DataFrame, loads that class's API, suggests relevant methods with documentation.\n\n**Even better with generics:**\n```python\ndef get_first(items: List[str]) -> str:\n    return items[0]  # IDE knows items is List[str]\n    # Suggests: append, extend, etc.\n    # Knows return type is str → suggests str methods for result\n```\nThis is why Cursor with type hints is so powerful - it knows EXACTLY what type each thing is, provides precise suggestions, catches type errors. Without hints, it relies on inference (less accurate) or shows everything (noisy).",
        keyPoints: [
            'Annotations provide exact type information',
            'Enable precise method/attribute suggestions',
            'Generic types specify element types too',
            'Dramatically improves completion accuracy',
        ],
    },
];
