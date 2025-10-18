/**
 * Quiz questions for Inheritance and Polymorphism section
 */

export const inheritanceQuiz = [
  {
    id: 'q1',
    question:
      'Explain Method Resolution Order (MRO) and why it matters in multiple inheritance.',
    sampleAnswer:
      'MRO is the order Python searches for methods in the inheritance hierarchy. It matters because in multiple inheritance, a method might exist in multiple parent classes. Python uses C3 linearization algorithm to create a consistent order. You can see it with ClassName.mro(). For example, if D inherits from B and C, which both inherit from A, the MRO is D -> B -> C -> A -> object. Python searches left to right, depth-first, but ensures each class appears only once and parents appear after their children. This prevents the diamond problem and ensures super() works correctly.',
    keyPoints: [
      'Order of method search in inheritance',
      'Matters for multiple inheritance',
      'C3 linearization algorithm',
      'View with ClassName.mro()',
      'Prevents diamond problem',
    ],
  },
  {
    id: 'q2',
    question:
      'When should you use abstract base classes (ABC)? What problem do they solve?',
    hint: 'Think about enforcing interfaces and preventing incomplete implementations.',
    sampleAnswer:
      "Use ABCs to define interfaces that subclasses MUST implement. They prevent instantiation of incomplete classes and enforce a contract. Use when: 1) you have a family of related classes that should share an interface, 2) you want to ensure subclasses implement certain methods, 3) you're building a framework or library. For example, Shape ABC requires area() and perimeter() methods—you can't create a Shape, only concrete subclasses like Circle or Rectangle that implement these methods. This catches errors at instantiation time rather than runtime. ABCs document intent and enable isinstance() checks for interface compliance.",
    keyPoints: [
      'Enforce interface contracts',
      'Prevent instantiation of incomplete classes',
      'Use @abstractmethod decorator',
      'Subclasses must implement abstract methods',
      'Good for frameworks and plugin systems',
    ],
  },
  {
    id: 'q3',
    question:
      "Explain how super() works in multiple inheritance and why it's important.",
    hint: 'Consider cooperative multiple inheritance and method chaining.',
    sampleAnswer:
      'super() follows the Method Resolution Order (MRO) to call the next method in the chain, not just the immediate parent. This enables cooperative multiple inheritance where each class calls super() to ensure all parents get called in the correct order. Without super(), in multiple inheritance you might call parent methods explicitly (Parent1.__init__(self), Parent2.__init__(self)), but this breaks if the hierarchy changes or causes methods to be called multiple times. super() is especially critical for __init__—it ensures proper initialization order. Always use super() instead of calling parent class methods directly for maintainability and correctness.',
    keyPoints: [
      'Follows MRO, not just immediate parent',
      'Enables cooperative multiple inheritance',
      'Ensures all parents called in correct order',
      'Critical for __init__ in multiple inheritance',
      'More maintainable than explicit parent calls',
    ],
  },
];
