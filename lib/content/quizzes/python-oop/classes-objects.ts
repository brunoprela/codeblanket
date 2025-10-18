/**
 * Quiz questions for Classes and Objects section
 */

export const classesobjectsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between instance attributes, class attributes, and properties.',
    sampleAnswer:
      'Instance attributes are unique to each object, defined in __init__ with self.name. Class attributes are shared by all instances, defined at class level. Properties are methods that look like attributes using @property decorator. Instance attribute example: each Person has their own name. Class attribute example: all Employees share the same company name. Property example: Circle.area is computed from radius. Properties allow validation, computed values, and controlled access while maintaining attribute syntax. Use instance attributes for object-specific data, class attributes for shared data, and properties for computed or validated access.',
    keyPoints: [
      'Instance: unique per object (self.name)',
      'Class: shared by all instances',
      'Property: method accessed like attribute',
      'Properties enable validation and computation',
      'Choose based on sharing and access needs',
    ],
  },
  {
    id: 'q2',
    question:
      'What are dunder methods (magic methods) and why are they important? Give examples of common ones.',
    hint: 'Think about operator overloading and special Python behaviors.',
    sampleAnswer:
      'Dunder methods (double underscore) like __init__, __str__, __add__ allow you to customize how your objects behave with built-in Python operations. They enable operator overloading and special behaviors. Common ones: __init__ for initialization, __str__ for print(), __repr__ for debugging, __eq__ for ==, __lt__ for <, __add__ for +, __len__ for len(), __getitem__ for indexing. For example, defining __add__ lets you use + with your objects: point1 + point2. They make your classes feel like built-in Python types and enable natural syntax.',
    keyPoints: [
      'Double underscore methods for special behaviors',
      'Enable operator overloading',
      '__str__ for string representation',
      '__eq__, __add__ for operators',
      'Make custom classes feel built-in',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the difference between @classmethod and @staticmethod. When should you use each?',
    hint: 'Think about access to class vs instance data, and the first parameter.',
    sampleAnswer:
      "@classmethod receives the class (cls) as first parameter and can access/modify class state. Use for factory methods that create instances, or methods that need to access class attributes. @staticmethod receives no special first parameter and can't access class or instance state—it's just a regular function grouped with the class for organization. Use when the function is related to the class but doesn't need access to class/instance data. Example: Pizza.margherita() as classmethod creates a Pizza instance. Pizza.is_valid_topping() as staticmethod just validates a value without needing class state.",
    keyPoints: [
      'classmethod: receives cls, accesses class state',
      'staticmethod: no special parameter, no state access',
      'Use classmethod for factories, alternative constructors',
      'Use staticmethod for utility functions',
      'Example: Date.from_string() vs Date.is_valid_date()',
    ],
  },
];
