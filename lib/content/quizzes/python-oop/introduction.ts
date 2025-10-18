/**
 * Quiz questions for Object-Oriented Programming in Python section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question: 'Explain the four pillars of OOP and how they work together.',
    sampleAnswer:
      'The four pillars are: (1) Encapsulation—bundling data and methods together, hiding internal details. (2) Abstraction—exposing only essential features, hiding complexity. (3) Inheritance—creating new classes from existing ones, promoting code reuse. (4) Polymorphism—using objects of different types through a common interface. They work together: encapsulation hides how something works, abstraction defines what it does, inheritance lets you extend functionality, and polymorphism lets you use different implementations interchangeably. For example, a Car class encapsulates engine details, abstracts a drive() method, inherits from Vehicle, and can be used polymorphically with Truck.',
    keyPoints: [
      'Encapsulation: bundle data and methods',
      'Abstraction: hide complexity, show essentials',
      'Inheritance: reuse code through parent classes',
      'Polymorphism: common interface for different types',
      'Work together to organize complex systems',
    ],
  },
  {
    id: 'q2',
    question:
      'When should you use composition over inheritance? Give a concrete example.',
    sampleAnswer:
      'Use composition when you have a "has-a" relationship rather than "is-a". Composition is more flexible and avoids fragile base class problems. For example, bad inheritance: class Car(Engine, Transmission, Stereo)—a car "is-a" engine? No. Good composition: class Car has self.engine, self.transmission, self.stereo. Composition lets you: (1) change implementations at runtime, (2) avoid deep inheritance hierarchies, (3) have better encapsulation. Use inheritance for true "is-a" relationships like ElectricCar(Car) or Dog(Animal), but prefer composition for behavior delegation.',
    keyPoints: [
      'Composition: "has-a" relationship',
      'Inheritance: "is-a" relationship',
      'Composition more flexible',
      'Example: Car has-a Engine, not is-a Engine',
      'Prefer composition to avoid fragile hierarchies',
    ],
  },
  {
    id: 'q3',
    question:
      'What is the difference between class attributes and instance attributes? When should you use each?',
    hint: 'Think about what is shared vs unique to each object, and memory implications.',
    sampleAnswer:
      'Class attributes are shared by ALL instances of a class and defined directly in the class body. Instance attributes are unique to each object and defined in __init__. Use class attributes for: 1) Constants shared by all instances (like Dog.species = "Canis familiaris"), 2) Default values, 3) Counters tracking total instances. Use instance attributes for: 1) Data unique to each object (like self.name, self.age), 2) State that varies per instance. Be careful: if you modify a mutable class attribute (like a list), it affects ALL instances! For example, if all Dogs share Dog.tricks = [], adding a trick to one dog adds it to all dogs—use instance attributes for per-object data.',
    keyPoints: [
      'Class attributes: shared by all instances',
      'Instance attributes: unique per object',
      'Use class attributes for constants, defaults',
      'Use instance attributes for per-object state',
      'Beware: mutable class attributes are shared!',
    ],
  },
];
