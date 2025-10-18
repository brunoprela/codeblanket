/**
 * Quiz questions for Metaclasses & Class Creation section
 */

export const metaclassesQuiz = [
  {
    id: 'q1',
    question:
      'What are metaclasses and when should you use them? Why are they considered advanced?',
    sampleAnswer:
      'Metaclasses are classes whose instances are classes. They control how classes are created, just like classes control how objects are created. Use metaclasses for: (1) ORMs like Django models where you need to transform class definitions into database schemas, (2) enforcing class-level constraints or patterns, (3) automatic registration systems, or (4) plugin architectures. They are considered advanced because: they are meta-programming (code that writes code), they are rarely needed in application code, there are usually simpler alternatives, and misuse can make code hard to understand. The Python mantra is "metaclasses are deeper magic than 99% of users should ever worry about."',
    keyPoints: [
      'Classes whose instances are classes',
      'Control how classes are created',
      'Use cases: ORMs, registration, constraints',
      'Rarely needed in application code',
      'Often simpler alternatives exist',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between __new__ and __init__ in a metaclass.',
    sampleAnswer:
      'In a metaclass, __new__ is called before the class is created and receives the class name, bases, and attributes dict. It can modify these before the class is constructed and must return the class object. __init__ is called after the class is created to initialize it. The key difference: __new__ can prevent class creation or modify the class definition, while __init__ can only set attributes on an already-created class. Use __new__ when you need to modify the class structure (add/remove methods, change bases), and __init__ for simple initialization like registering the class or setting metadata.',
    keyPoints: [
      '__new__: called before class creation',
      '__new__: can modify class definition',
      '__init__: called after class created',
      '__init__: initializes the class',
      '__new__ for structure, __init__ for initialization',
    ],
  },
  {
    id: 'q3',
    question:
      'What are some alternatives to metaclasses that are simpler but solve similar problems? When would you use each?',
    sampleAnswer:
      'Modern Python offers simpler alternatives: 1) **Class decorators**: Apply @decorator to a class to modify it after creation. Use for adding methods, wrapping methods, or registering classes. Simpler than metaclasses. 2) **__init_subclass__** (Python 3.6+): A class method called when a class is subclassed. Use for validation, registration, or modifying subclasses. Cleaner than metaclasses for inheritance patterns. 3) **Descriptors**: Control attribute access at the instance level. Use for validation, computed properties. Only use metaclasses when you need to: control how ALL subclasses are created, modify class structure deeply, or implement frameworks like ORMs. Rule of thumb: try decorators first, then __init_subclass__, metaclasses last.',
    keyPoints: [
      'Class decorators: simpler, for post-creation modification',
      '__init_subclass__: cleaner for inheritance patterns',
      'Descriptors: for attribute-level control',
      'Use metaclasses only when alternatives insufficient',
      'Order: decorators → __init_subclass__ → metaclasses',
    ],
  },
];
