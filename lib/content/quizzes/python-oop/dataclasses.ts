/**
 * Quiz questions for Dataclasses for Structured Data section
 */

export const dataclassesQuiz = [
  {
    id: 'q1',
    question:
      'Why should you use field(default_factory=list) instead of a simple list as a default value? What problem does this solve?',
    hint: 'Think about mutable default arguments and shared state between instances.',
    sampleAnswer:
      'Using a bare list as default (tags: list = []) creates ONE shared list that all instances will reference—if you modify the list in one instance, it affects all instances. This is Python\'s mutable default argument gotcha. field(default_factory=list) calls list() for each new instance, creating a fresh list every time. For example, with bare list: person1.tags.append("vip") would add "vip" to person2.tags too! With default_factory, each person gets their own independent list. This applies to any mutable default: dict, set, or custom objects.',
    keyPoints: [
      'Bare list creates shared mutable object',
      'Changes affect all instances',
      'default_factory creates new object per instance',
      'Applies to all mutable defaults (list, dict, set)',
      'Critical for correctness in dataclasses',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the difference between frozen=True and regular dataclasses? When would you use frozen dataclasses?',
    hint: 'Consider immutability, hashability, and use cases like dict keys or sets.',
    sampleAnswer:
      "frozen=True makes dataclasses immutable—you can't modify attributes after creation, like tuples. This has several benefits: 1) Instances become hashable and can be used as dict keys or in sets, 2) Thread-safe by default (no race conditions), 3) Easier to reason about (values never change), 4) Better for value objects and DTOs. Use frozen dataclasses for: configuration objects, coordinates, API responses, or any data that represents a value rather than an entity. For example, Point(x=10, y=20) should never change—if you need a different point, create a new instance.",
    keyPoints: [
      'frozen=True makes instances immutable',
      'Enables use as dict keys (hashable)',
      'Thread-safe by default',
      'Use for value objects, configs, DTOs',
      'Cannot modify after creation',
    ],
  },
  {
    id: 'q3',
    question:
      'When should you use a dataclass versus a regular class? What are the trade-offs?',
    hint: 'Consider boilerplate, flexibility, and the primary purpose of the class.',
    sampleAnswer:
      'Use dataclasses when the class primarily stores data with minimal logic: API models, DTOs, configuration objects, or data containers. Dataclasses excel at reducing boilerplate—auto-generating __init__, __repr__, __eq__ saves dozens of lines. Use regular classes when: 1) You need complex __init__ logic beyond simple assignment, 2) The class has substantial business logic and few attributes, 3) You need dynamic attributes or metaclasses, 4) Backward compatibility with Python < 3.7. Trade-off: dataclasses sacrifice some flexibility for convenience. For example, a User dataclass is perfect for API responses, but a UserManager class with authentication logic should be a regular class.',
    keyPoints: [
      'Dataclass: primarily stores data, minimal boilerplate',
      'Regular class: complex logic, maximum flexibility',
      'Dataclass auto-generates __init__, __repr__, __eq__',
      'Use dataclass for DTOs, configs, data containers',
      'Use regular class for business logic, complex behavior',
    ],
  },
];
