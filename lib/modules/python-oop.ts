/**
 * Python Object-Oriented Programming module content
 */

import { Module } from '@/lib/types';

export const pythonOOPModule: Module = {
  id: 'python-oop',
  title: 'Python Object-Oriented Programming',
  description:
    'Master object-oriented programming in Python including classes, inheritance, polymorphism, and design patterns.',
  icon: '🏗️',
  sections: [
    {
      id: 'introduction',
      title: 'Object-Oriented Programming in Python',
      content: `Object-Oriented Programming (OOP) is a programming paradigm that organizes code around objects that contain both data (attributes) and behavior (methods).

**Why OOP Matters:**
- **Encapsulation:** Bundle data and methods that operate on that data
- **Inheritance:** Reuse code through parent-child class relationships
- **Polymorphism:** Use objects of different types through a common interface
- **Abstraction:** Hide complex implementation details

**Real-World Applications:**
- **Game Development:** Player, Enemy, Weapon classes
- **Web Frameworks:** Django models, Flask views
- **Data Processing:** Custom data structures and transformations
- **API Clients:** Service classes with methods for different endpoints

**Key Insight:**
OOP helps organize complex systems by modeling real-world entities and their relationships, making code more maintainable and easier to reason about.`,
      quiz: [
        {
          id: 'q1',
          question:
            'Explain the four pillars of OOP and how they work together.',
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
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does encapsulation mean in OOP?',
          options: [
            'Using multiple classes',
            'Bundling data and methods that operate on that data',
            'Creating child classes from parent classes',
            'Using the same method name in different classes',
          ],
          correctAnswer: 1,
          explanation:
            'Encapsulation means bundling data (attributes) and methods that operate on that data together in a class, while hiding internal implementation details from outside code.',
        },
        {
          id: 'mc2',
          question: 'Which relationship should use inheritance?',
          options: [
            'A Car has an Engine',
            'A Student has a Backpack',
            'A Dog is an Animal',
            'A House has a Roof',
          ],
          correctAnswer: 2,
          explanation:
            'Inheritance represents "is-a" relationships. A Dog IS AN Animal is a true subtype relationship. The others are "has-a" relationships better modeled with composition.',
        },
        {
          id: 'mc3',
          question: 'What is polymorphism in OOP?',
          options: [
            'Having many classes',
            'Ability to use objects of different types through a common interface',
            'Creating multiple instances',
            'Hiding data from outside access',
          ],
          correctAnswer: 1,
          explanation:
            'Polymorphism allows objects of different types to be used through a common interface, enabling code that works with multiple types.',
        },
        {
          id: 'mc4',
          question: 'What is the main benefit of abstraction?',
          options: [
            'Faster code execution',
            'Hiding complexity and exposing only essential features',
            'Using less memory',
            'Creating more classes',
          ],
          correctAnswer: 1,
          explanation:
            'Abstraction hides complex implementation details and exposes only the essential features, making code easier to understand and use.',
        },
        {
          id: 'mc5',
          question: 'Which is an example of composition?',
          options: [
            'class Dog(Animal)',
            'class Car: self.engine = Engine()',
            'class Student(Person)',
            'class Circle(Shape)',
          ],
          correctAnswer: 1,
          explanation:
            'class Car with self.engine = Engine() is composition—a Car HAS AN Engine. The others show inheritance (is-a relationships).',
        },
      ],
    },
    {
      id: 'classes-objects',
      title: 'Classes and Objects',
      content: `**Defining Classes:**
\`\`\`python
class Dog:
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says woof!"
    
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Creating objects
buddy = Dog("Buddy", 5)
miles = Dog("Miles", 4)

print(buddy.bark())  # "Buddy says woof!"
print(miles)  # "Miles is 4 years old"
\`\`\`

**Instance vs Class Attributes:**
\`\`\`python
class Employee:
    company = "TechCorp"  # Class attribute
    
    def __init__(self, name):
        self.name = name  # Instance attribute

emp1 = Employee("Alice")
emp2 = Employee("Bob")

# Class attribute shared
print(emp1.company)  # "TechCorp"
print(emp2.company)  # "TechCorp"

# Change class attribute affects all
Employee.company = "NewCorp"
print(emp1.company)  # "NewCorp"
\`\`\`

**Special Methods (Dunder Methods):**
\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1 + p2)  # Point(4, 6)
print(p1 == p2)  # False
\`\`\`

**Properties:**
\`\`\`python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

circle = Circle(5)
print(circle.area)  # 78.53975
circle.radius = 10  # Uses setter
\`\`\`

**Class Methods and Static Methods:**
\`\`\`python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients
    
    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])
    
    @staticmethod
    def is_valid_topping(topping):
        return topping in ['mozzarella', 'tomatoes', 'pepperoni']

# Use class method as factory
pizza = Pizza.margherita()

# Use static method
Pizza.is_valid_topping('mozzarella')  # True
\`\`\``,
      quiz: [
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
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the purpose of the __init__ method?',
          options: [
            'To delete an object',
            "To initialize an object's attributes when it is created",
            'To compare two objects',
            'To convert an object to a string',
          ],
          correctAnswer: 1,
          explanation:
            "__init__ is the constructor method called when creating a new instance. It initializes the object's attributes with values passed as arguments.",
        },
        {
          id: 'mc2',
          question: 'What does @property decorator do?',
          options: [
            'Makes a method private',
            'Allows a method to be accessed like an attribute',
            'Creates a class method',
            'Makes an attribute immutable',
          ],
          correctAnswer: 1,
          explanation:
            '@property decorator allows a method to be accessed like an attribute without parentheses, enabling computed values, validation, and controlled access.',
        },
        {
          id: 'mc3',
          question: 'What is the purpose of the __str__ method?',
          options: [
            'To convert the object to an integer',
            'To define how an object is represented as a string',
            'To create a new object',
            'To delete an object',
          ],
          correctAnswer: 1,
          explanation:
            '__str__ defines how an object should be represented as a human-readable string, used by print() and str().',
        },
        {
          id: 'mc4',
          question: 'What does @classmethod receive as its first parameter?',
          options: [
            'self (the instance)',
            'cls (the class)',
            'No parameter',
            'The parent class',
          ],
          correctAnswer: 1,
          explanation:
            '@classmethod receives cls (the class) as the first parameter, allowing it to access and modify class state.',
        },
        {
          id: 'mc5',
          question: 'Which is true about class attributes?',
          options: [
            'Each instance has its own copy',
            'They are shared by all instances of the class',
            'They cannot be modified',
            'They must be private',
          ],
          correctAnswer: 1,
          explanation:
            'Class attributes are shared by all instances of the class. Modifying a class attribute affects all instances.',
        },
      ],
    },
    {
      id: 'inheritance',
      title: 'Inheritance and Polymorphism',
      content: `**Basic Inheritance:**
\`\`\`python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks"

class Cat(Animal):
    def speak(self):
        return f"{self.name} meows"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # "Buddy barks"
print(cat.speak())  # "Whiskers meows"
\`\`\`

**Method Resolution Order (MRO):**
\`\`\`python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # "B"
print(D.mro())  # [D, B, C, A, object]
\`\`\`

**super() Function:**
\`\`\`python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, size):
        # Call parent's __init__
        super().__init__(size, size)

square = Square(5)
print(square.area())  # 25
\`\`\`

**Polymorphism:**
\`\`\`python
def make_animals_speak(animals):
    for animal in animals:
        print(animal.speak())

animals = [Dog("Buddy"), Cat("Whiskers"), Dog("Max")]
make_animals_speak(animals)
# Buddy barks
# Whiskers meows  
# Max barks
\`\`\`

**Abstract Base Classes:**
\`\`\`python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Cannot instantiate abstract class
# shape = Shape()  # TypeError

circle = Circle(5)  # OK
\`\`\`

**Multiple Inheritance:**
\`\`\`python
class Flyable:
    def fly(self):
        return "Flying!"

class Swimmable:
    def swim(self):
        return "Swimming!"

class Duck(Animal, Flyable, Swimmable):
    def speak(self):
        return f"{self.name} quacks"

duck = Duck("Donald")
print(duck.speak())  # "Donald quacks"
print(duck.fly())    # "Flying!"
print(duck.swim())   # "Swimming!"
\`\`\`

**Composition vs Inheritance:**
\`\`\`python
# Inheritance (is-a)
class Employee(Person):
    pass

# Composition (has-a)
class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheels = [Wheel() for _ in range(4)]
    
    def start(self):
        self.engine.start()
\`\`\``,
      quiz: [
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
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What does super() do?',
          options: [
            'Creates a superclass',
            "Calls the parent class's method",
            'Makes a method private',
            'Creates multiple inheritance',
          ],
          correctAnswer: 1,
          explanation:
            'super() returns a temporary object that allows you to call methods of the parent class, enabling proper method chaining in inheritance hierarchies.',
        },
        {
          id: 'mc2',
          question: 'What is polymorphism?',
          options: [
            'Having multiple classes',
            'Using the same interface for different data types',
            'Inheriting from multiple parents',
            'Hiding implementation details',
          ],
          correctAnswer: 1,
          explanation:
            'Polymorphism means using the same interface (method name) for different data types. Different classes can implement the same method in their own way.',
        },
        {
          id: 'mc3',
          question:
            'What happens if you try to instantiate an abstract base class?',
          options: [
            'It works normally',
            'TypeError is raised',
            'Returns None',
            'Creates an empty object',
          ],
          correctAnswer: 1,
          explanation:
            'Python raises a TypeError if you try to instantiate an abstract base class that has abstract methods. You must create a concrete subclass that implements all abstract methods.',
        },
        {
          id: 'mc4',
          question:
            'In class Child(Parent1, Parent2), what is the order of parent class checking?',
          options: [
            'Parent2, then Parent1',
            'Parent1, then Parent2',
            'Random order',
            'Only checks Parent1',
          ],
          correctAnswer: 1,
          explanation:
            'Python checks parent classes from left to right: Parent1, then Parent2. This is part of the Method Resolution Order (MRO).',
        },
        {
          id: 'mc5',
          question: 'What does the @abstractmethod decorator do?',
          options: [
            'Makes a method private',
            'Marks a method that must be implemented by subclasses',
            'Makes a method faster',
            'Converts a method to a class method',
          ],
          correctAnswer: 1,
          explanation:
            '@abstractmethod marks a method that subclasses must implement. Classes with abstract methods cannot be instantiated.',
        },
      ],
    },
  ],
  keyTakeaways: [
    'Classes bundle data and behavior—use __init__ to initialize instance attributes',
    'Inheritance models "is-a" relationships—use super() to call parent methods',
    'Polymorphism allows using objects of different types through common interface',
    'Composition ("has-a") is often better than inheritance for flexibility',
    'Abstract base classes define interfaces that subclasses must implement',
  ],
  relatedProblems: [
    'class-bankaccount',
    'inheritance-shapes',
    'polymorphism-animals',
  ],
};
