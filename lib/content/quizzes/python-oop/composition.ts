/**
 * Quiz questions for Composition Over Inheritance section
 */

export const compositionQuiz = [
  {
    id: 'q1',
    question:
      'Why is composition often more flexible than inheritance? Give a concrete example where composition solves a problem that inheritance creates.',
    hint: 'Think about changing behavior at runtime and the fragile base class problem.',
    sampleAnswer:
      "Composition is more flexible because you can change behavior at runtime by swapping components, while inheritance is fixed at class definition. Example: With inheritance, if Car extends Engine, you can't switch from V6 to Electric engine at runtime—the engine is part of the class definition. With composition (Car has-a engine), you can swap: car.engine = ElectricEngine(). This also solves the fragile base class problem—if Engine class changes, Car inheritance might break, but with composition, Car only depends on the Engine interface, not implementation. Composition also lets you test components independently and reuse Engine in Boat, Plane, etc. without inheritance chains.",
    keyPoints: [
      'Can change behavior at runtime',
      'Avoids fragile base class problem',
      'Components independently testable',
      'Better code reuse',
      'Example: swapping payment processors dynamically',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the "is-a" vs "has-a" test? How do you decide whether to use inheritance or composition?',
    hint: 'Think about the relationship between objects and how you would describe it in English.',
    sampleAnswer:
      'The "is-a" test: if ClassA IS-A ClassB, use inheritance. The "has-a" test: if ClassA HAS-A ClassB, use composition. Examples: Dog IS-A Animal → inheritance makes sense. Car HAS-AN Engine → use composition. Also consider: can you substitute the child for the parent everywhere (Liskov Substitution)? Does the child need all parent methods? For instance, Penguin IS-A Bird, but if Bird.fly() exists, Penguin breaks this—better to use composition with Flyable/Swimmable components. If you find yourself disabling parent methods in child class, that\'s a sign to use composition instead.',
    keyPoints: [
      'is-a: inheritance (Dog is-a Animal)',
      'has-a: composition (Car has-a Engine)',
      'Check Liskov Substitution Principle',
      'If child breaks parent interface, use composition',
      "Example: Penguin shouldn't inherit fly() from Bird",
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the Strategy Pattern using composition. Why is it better than conditional logic or inheritance?',
    hint: 'Think about adding new strategies and the Open/Closed Principle.',
    sampleAnswer:
      "Strategy Pattern uses composition to encapsulate interchangeable algorithms. Instead of if/elif chains or subclasses for each variant, you compose with a strategy object. Example: ShoppingCart composes with PaymentProcessor—to add Bitcoin payment, create BitcoinPayment class without touching existing code (Open/Closed Principle). With conditionals, you'd modify checkout() every time (error-prone). With inheritance (CreditCardCart, PayPalCart), you'd duplicate cart logic. Composition lets you: 1) add strategies without modifying context, 2) swap strategies at runtime, 3) test strategies independently, 4) reuse strategies across contexts. This is more flexible and maintainable than deeply nested conditionals or inheritance pyramids.",
    keyPoints: [
      'Encapsulates interchangeable algorithms',
      'Add strategies without modifying context',
      'Swap strategies at runtime',
      'Follows Open/Closed Principle',
      'Better than conditionals or inheritance for variants',
    ],
  },
];
