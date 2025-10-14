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
    {
      id: 'dataclasses',
      title: 'Dataclasses for Structured Data',
      content: `**What are Dataclasses?**
Dataclasses (Python 3.7+) are a cleaner, more concise way to create classes that primarily store data, automatically generating \`__init__\`, \`__repr__\`, \`__eq__\`, and other methods.

**Basic Dataclass:**
\`\`\`python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str

# Automatically generated __init__
person = Person("Alice", 30, "alice@example.com")

# Automatically generated __repr__
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

# Automatically generated __eq__
person2 = Person("Alice", 30, "alice@example.com")
print(person == person2)  # True
\`\`\`

**Default Values:**
\`\`\`python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Default value
    tags: list = field(default_factory=list)  # Mutable default

# Use default values
product = Product("Widget", 19.99)
print(product.quantity)  # 0

# IMPORTANT: Use field(default_factory) for mutable defaults!
# WRONG: tags: list = []  # Shared between instances!
# CORRECT: tags: list = field(default_factory=list)
\`\`\`

**Post-Init Processing:**
\`\`\`python
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Not in __init__
    
    def __post_init__(self):
        """Called after __init__"""
        self.area = self.width * self.height
        if self.width < 0 or self.height < 0:
            raise ValueError("Dimensions must be positive")

rect = Rectangle(5, 10)
print(rect.area)  # 50
\`\`\`

**Immutable Dataclasses:**
\`\`\`python
@dataclass(frozen=True)
class Point:
    x: float
    y: float

point = Point(10, 20)
# point.x = 30  # FrozenInstanceError!

# Frozen dataclasses are hashable
points_set = {Point(0, 0), Point(1, 1)}
\`\`\`

**Ordering:**
\`\`\`python
@dataclass(order=True)
class Score:
    value: int
    player: str = field(compare=False)  # Don't use in comparisons

scores = [Score(90, "Alice"), Score(85, "Bob"), Score(95, "Charlie")]
print(sorted(scores))  # Sorted by value
# [Score(value=85, player='Bob'), Score(value=90, player='Alice'), ...]
\`\`\`

**Field Options:**
\`\`\`python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    id: int = field(repr=False)  # Don't show in repr
    grades: List[int] = field(default_factory=list)
    gpa: float = field(init=False, repr=False)  # Calculated field
    
    def __post_init__(self):
        if self.grades:
            self.gpa = sum(self.grades) / len(self.grades)
        else:
            self.gpa = 0.0
\`\`\`

**Inheritance with Dataclasses:**
\`\`\`python
@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: int
    department: str

# Inherits fields from Person
emp = Employee("Alice", 30, 12345, "Engineering")
print(emp)  # Employee(name='Alice', age=30, employee_id=12345, ...)
\`\`\`

**Conversion Methods:**
\`\`\`python
from dataclasses import dataclass, asdict, astuple

@dataclass
class Config:
    host: str
    port: int
    debug: bool

config = Config("localhost", 8080, True)

# Convert to dict
print(asdict(config))
# {'host': 'localhost', 'port': 8080, 'debug': True}

# Convert to tuple
print(astuple(config))
# ('localhost', 8080, True)
\`\`\`

**Real-World Example - API Response:**
\`\`\`python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool = True
    roles: List[str] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, data: dict):
        """Create from JSON response"""
        return cls(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True),
            roles=data.get('roles', [])
        )
    
    def has_role(self, role: str) -> bool:
        return role in self.roles

# Usage
user_data = {
    'id': 1,
    'username': 'alice',
    'email': 'alice@example.com',
    'created_at': '2024-01-01T00:00:00',
    'roles': ['admin', 'user']
}
user = User.from_json(user_data)
print(user.has_role('admin'))  # True
\`\`\`

**Dataclass vs Regular Class:**

| Feature | Regular Class | Dataclass |
|---------|--------------|-----------|
| \`__init__\` | Manual | Auto-generated |
| \`__repr__\` | Manual | Auto-generated |
| \`__eq__\` | Manual | Auto-generated |
| Type Hints | Optional | Required |
| Boilerplate | Lots | Minimal |
| Flexibility | Maximum | Good |

**When to Use Dataclasses:**
- ✅ Classes that primarily store data
- ✅ API models, config objects, DTOs
- ✅ When you want immutability (frozen=True)
- ✅ Need auto-generated comparison methods
- ❌ Complex business logic classes (use regular classes)
- ❌ Need dynamic attributes (use regular class or dict)

**Best Practices:**
- Always use \`field(default_factory)\` for mutable defaults
- Use type hints for all fields
- Use \`frozen=True\` for immutable data
- Add \`__post_init__\` for validation or computed fields
- Use \`@classmethod\` for alternative constructors`,
      quiz: [
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
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question:
            'What does the @dataclass decorator automatically generate?',
          options: [
            'Only __init__',
            '__init__, __repr__, and __eq__',
            'Only __str__',
            'All dunder methods',
          ],
          correctAnswer: 1,
          explanation:
            '@dataclass automatically generates __init__, __repr__, __eq__, and optionally __hash__ and __order__ methods based on the fields.',
        },
        {
          id: 'mc2',
          question:
            'Why must you use field(default_factory=list) for mutable defaults?',
          options: [
            "It's faster",
            'To avoid sharing mutable objects between instances',
            'Required by Python syntax',
            'To make the list immutable',
          ],
          correctAnswer: 1,
          explanation:
            'field(default_factory=list) creates a new list for each instance, preventing the shared mutable default argument gotcha where all instances would share the same list.',
        },
        {
          id: 'mc3',
          question: 'What does frozen=True do in a dataclass?',
          options: [
            'Makes the class run faster',
            'Makes instances immutable',
            'Freezes the class at creation time',
            'Prevents inheritance',
          ],
          correctAnswer: 1,
          explanation:
            'frozen=True makes dataclass instances immutable—you cannot modify attributes after creation, similar to tuples. This also makes them hashable.',
        },
        {
          id: 'mc4',
          question: 'When is __post_init__ called?',
          options: [
            'Before __init__',
            'After __init__, for additional processing',
            'When the object is deleted',
            'Only on first access',
          ],
          correctAnswer: 1,
          explanation:
            '__post_init__ is called automatically after __init__ completes, allowing you to perform validation, compute derived values, or other initialization logic.',
        },
        {
          id: 'mc5',
          question:
            'What is the main advantage of dataclasses over regular classes?',
          options: [
            'Faster execution',
            'Less memory usage',
            'Reduced boilerplate code',
            'Better inheritance',
          ],
          correctAnswer: 2,
          explanation:
            'The main advantage is reduced boilerplate—dataclasses automatically generate __init__, __repr__, __eq__ and other methods, saving you from writing repetitive code.',
        },
      ],
    },
    {
      id: 'properties',
      title: 'Property Decorators Deep-Dive',
      content: `**What are Properties?**
Properties allow you to define methods that behave like attributes, enabling controlled access, validation, and computed values while maintaining a clean interface.

**Basic Property:**
\`\`\`python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self, value):
        """Deleter (rarely used)"""
        del self._celsius

# Usage - looks like attribute access
temp = Temperature(25)
print(temp.celsius)  # Calls getter
temp.celsius = 30    # Calls setter
# temp.celsius = -300  # Raises ValueError
\`\`\`

**Computed Properties:**
\`\`\`python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def diameter(self):
        """Computed on-the-fly"""
        return self.radius * 2
    
    @property
    def area(self):
        """No storage, calculated when accessed"""
        return 3.14159 * self.radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self.radius

circle = Circle(5)
print(circle.area)  # 78.53975 (calculated)
circle.radius = 10
print(circle.area)  # 314.159 (recalculated)
\`\`\`

**Property for Validation:**
\`\`\`python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age  # Uses setter
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise TypeError("Age must be an integer")
        if value < 0 or value > 150:
            raise ValueError("Age must be between 0 and 150")
        self._age = value

# Validation happens automatically
person = Person("Alice", 30)
# person.age = -5  # Raises ValueError
# person.age = "30"  # Raises TypeError
\`\`\`

**Read-Only Properties:**
\`\`\`python
class BankAccount:
    def __init__(self, account_number, balance):
        self._account_number = account_number
        self._balance = balance
    
    @property
    def account_number(self):
        """Read-only property (no setter)"""
        return self._account_number
    
    @property
    def balance(self):
        """Read-only balance"""
        return self._balance
    
    def deposit(self, amount):
        """Controlled balance modification"""
        if amount > 0:
            self._balance += amount

account = BankAccount("12345", 1000)
print(account.balance)  # OK
# account.balance = 5000  # AttributeError!
\`\`\`

**Lazy Evaluation with Properties:**
\`\`\`python
class DataProcessor:
    def __init__(self, filename):
        self.filename = filename
        self._data = None  # Not loaded yet
    
    @property
    def data(self):
        """Load data only when first accessed"""
        if self._data is None:
            print(f"Loading {self.filename}...")
            self._data = self._load_data()
        return self._data
    
    def _load_data(self):
        # Expensive operation
        return "Loaded data"

processor = DataProcessor("huge_file.txt")
# Data not loaded yet
print(processor.data)  # Loads now
print(processor.data)  # Uses cached version
\`\`\`

**Property with Type Conversion:**
\`\`\`python
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        """Auto-convert to float and validate"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("Price must be a number")
        
        if value < 0:
            raise ValueError("Price cannot be negative")
        
        self._price = round(value, 2)  # Store as 2 decimals

product = Product("Widget", "19.99")  # String accepted
print(product.price)  # 19.99 (float)
product.price = "25"  # Auto-converts
\`\`\`

**Dependent Properties:**
\`\`\`python
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("Width must be positive")
        self._width = value
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("Height must be positive")
        self._height = value
    
    @property
    def area(self):
        """Depends on width and height"""
        return self._width * self._height
    
    @property
    def aspect_ratio(self):
        """Depends on width and height"""
        return self._width / self._height

rect = Rectangle(10, 5)
print(rect.area)  # 50
rect.width = 20
print(rect.area)  # 100 (automatically recalculated)
\`\`\`

**Property vs Regular Method:**

**Use Property When:**
- Getting/setting feels like attribute access
- No parameters needed
- Fast computation (< 0.1s)
- No side effects beyond validation

**Use Method When:**
- Operation is expensive
- Parameters needed
- Operation has side effects
- Operation might fail frequently

\`\`\`python
class User:
    @property
    def full_name(self):
        """Property: simple, no parameters"""
        return f"{self.first_name} {self.last_name}"
    
    def send_email(self, subject, body):
        """Method: has side effects, needs parameters"""
        # Send email logic
        pass
\`\`\`

**Common Patterns:**

**1. Calculated Field:**
\`\`\`python
@property
def bmi(self):
    return self.weight / (self.height ** 2)
\`\`\`

**2. Cached Property (Python 3.8+):**
\`\`\`python
from functools import cached_property

class DataAnalyzer:
    @cached_property
    def statistics(self):
        """Computed once, then cached"""
        print("Computing statistics...")
        return self._compute_stats()
\`\`\`

**3. Aliasing:**
\`\`\`python
@property
def username(self):
    return self.email.split('@')[0]
\`\`\`

**Best Practices:**
- Use properties for simple attribute access
- Keep property getters fast (no heavy computation)
- Validate in setters, not getters
- Don't change object state in getters
- Use \`_underscore\` for internal attributes
- Document property behavior in docstrings`,
      quiz: [
        {
          id: 'q1',
          question:
            'When should you use a property versus a regular method? What are the design guidelines?',
          hint: 'Consider parameters, speed, side effects, and how the operation feels.',
          sampleAnswer:
            'Use properties when the operation feels like attribute access: no parameters needed, fast execution (< 0.1s), and minimal side effects beyond validation. Properties should act like attributes—getting a value should be cheap and idempotent. Use methods when: 1) the operation is expensive (database query, file I/O), 2) parameters are needed, 3) significant side effects occur (sending email, modifying external state), 4) the operation might fail frequently. For example, user.age is a property (fast, no parameters), but user.send_email(subject, body) must be a method (side effects, parameters). Think: "Would I be surprised if this had a getter/setter?" If yes, use a method.',
          keyPoints: [
            'Property: fast, no parameters, minimal side effects',
            'Method: expensive, needs parameters, side effects',
            'Properties feel like attribute access',
            'Methods feel like actions',
            'Example: user.age (property) vs user.calculate_taxes() (method)',
          ],
        },
        {
          id: 'q2',
          question:
            'Explain lazy evaluation using properties. Why is this useful and what are the trade-offs?',
          hint: 'Consider when expensive operations happen and memory vs computation trade-offs.',
          sampleAnswer:
            "Lazy evaluation with properties defers expensive computation until the value is first accessed. The pattern: check if cached value exists, compute and cache if not, return cached value. This is useful when: 1) not all instances need the value, 2) the computation is expensive, 3) you want fast initialization. Trade-offs: saves computation if never accessed, but first access is slower. Use cached_property (Python 3.8+) for automatic caching. For example, loading a large dataset: don't load in __init__ (slow startup), load on first access (lazy). This is especially useful for optional features or computed statistics that might not be needed in every code path.",
          keyPoints: [
            'Defers expensive computation until first access',
            'Useful when value might not be needed',
            'Trade-off: fast init, slower first access',
            'Use cached_property for automatic caching',
            'Example: lazy loading large datasets',
          ],
        },
        {
          id: 'q3',
          question:
            'Why use properties for validation instead of validating in __init__? What advantage does this provide?',
          hint: 'Think about when validation happens and maintaining invariants over time.',
          sampleAnswer:
            "Properties provide continuous validation—they validate not just during initialization but every time the attribute is modified. This maintains class invariants throughout the object's lifetime. Without properties, you can set invalid values after creation: person._age = -5 bypasses validation. With properties, every assignment goes through the setter: person.age = -5 raises ValueError, whether in __init__ or later. This also provides a single source of truth for validation logic—one setter handles all assignments, not scattered validation. Additionally, properties allow adding validation to existing code without breaking the interface—change direct attribute to property without callers knowing.",
          keyPoints: [
            'Properties validate on every assignment, not just __init__',
            'Maintains invariants throughout lifetime',
            'Single source of truth for validation',
            'Can add validation without breaking interface',
            'Example: person.age = -5 always validated',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is the main purpose of properties?',
          options: [
            'Make code run faster',
            'Allow methods to be accessed like attributes with validation',
            'Create private attributes',
            'Enable multiple inheritance',
          ],
          correctAnswer: 1,
          explanation:
            'Properties allow methods to be accessed like attributes while enabling validation, computed values, and controlled access.',
        },
        {
          id: 'mc2',
          question: 'How do you create a read-only property?',
          options: [
            'Use @readonly decorator',
            'Define only @property getter, no setter',
            'Set attribute to None',
            'Use const keyword',
          ],
          correctAnswer: 1,
          explanation:
            'Defining only the @property getter without a setter makes the property read-only. Attempting to set it will raise AttributeError.',
        },
        {
          id: 'mc3',
          question: 'When should you avoid using properties?',
          options: [
            'For simple attribute access',
            'For expensive or slow computations',
            'For validation',
            'For read-only attributes',
          ],
          correctAnswer: 1,
          explanation:
            'Avoid properties for expensive computations that take significant time. Properties should be fast since they look like attribute access. Use methods for slow operations.',
        },
        {
          id: 'mc4',
          question: 'What does @cached_property do (Python 3.8+)?',
          options: [
            'Makes property faster',
            'Computes property once and caches the result',
            'Makes property read-only',
            'Validates property value',
          ],
          correctAnswer: 1,
          explanation:
            '@cached_property computes the value once on first access and caches it, returning the cached value on subsequent accesses without recomputing.',
        },
        {
          id: 'mc5',
          question: 'What happens if you try to set a read-only property?',
          options: [
            'Value is silently ignored',
            'AttributeError is raised',
            'TypeError is raised',
            'Value is set successfully',
          ],
          correctAnswer: 1,
          explanation:
            'Attempting to set a read-only property (one without a setter) raises an AttributeError.',
        },
      ],
    },
    {
      id: 'composition',
      title: 'Composition Over Inheritance',
      content: `**What is Composition?**
Composition is a design principle where you build complex objects by combining simpler objects (has-a relationships) rather than inheriting from them (is-a relationships).

**Why Composition Matters:**
- More flexible than inheritance
- Avoids deep inheritance hierarchies
- Easier to test and modify
- Components can be reused independently
- Reduces coupling between classes

**Inheritance vs Composition:**

**❌ Bad: Inheritance Abuse**
\`\`\`python
class Engine:
    def start(self):
        return "Engine starting..."

class Wheels:
    def rotate(self):
        return "Wheels rotating..."

class Stereo:
    def play(self):
        return "Music playing..."

# Wrong: Car "is-a" Engine? No!
class Car(Engine, Wheels, Stereo):
    pass

# Problems:
# 1. Car inherits methods it might not need
# 2. Can't easily swap engine types
# 3. Tight coupling
# 4. Multiple inheritance complexity
\`\`\`

**✅ Good: Composition**
\`\`\`python
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower
    
    def start(self):
        return f"{self.horsepower}hp engine starting..."

class Wheels:
    def __init__(self, count=4):
        self.count = count
    
    def rotate(self):
        return f"{self.count} wheels rotating..."

class Stereo:
    def __init__(self, brand):
        self.brand = brand
    
    def play(self):
        return f"{self.brand} stereo playing..."

# Right: Car "has-a" Engine, Wheels, Stereo
class Car:
    def __init__(self, engine, wheels, stereo):
        self.engine = engine  # Composition
        self.wheels = wheels  # Composition
        self.stereo = stereo  # Composition
    
    def start(self):
        """Delegates to components"""
        return f"{self.engine.start()}\\n{self.wheels.rotate()}"
    
    def play_music(self):
        return self.stereo.play()

# Easy to swap components!
v6_engine = Engine(300)
wheels = Wheels(4)
bose_stereo = Stereo("Bose")

car = Car(v6_engine, wheels, bose_stereo)
print(car.start())
print(car.play_music())

# Can easily create different configurations
electric_engine = Engine(400)
sport_car = Car(electric_engine, wheels, bose_stereo)
\`\`\`

**Delegation Pattern:**
\`\`\`python
class Logger:
    """Handles all logging"""
    def log(self, message):
        print(f"[LOG] {message}")

class Database:
    """Handles database operations"""
    def save(self, data):
        print(f"Saving {data} to database")

class UserService:
    """Composes logger and database"""
    def __init__(self):
        self.logger = Logger()  # Composition
        self.db = Database()    # Composition
    
    def create_user(self, name):
        """Delegates to composed objects"""
        self.logger.log(f"Creating user: {name}")
        self.db.save({'user': name})
        self.logger.log("User created successfully")
        return name

service = UserService()
service.create_user("Alice")
\`\`\`

**Strategy Pattern with Composition:**
\`\`\`python
class PaymentProcessor:
    """Different payment strategies"""
    def pay(self, amount):
        raise NotImplementedError

class CreditCardPayment(PaymentProcessor):
    def pay(self, amount):
        return f"Paid \${amount} with credit card"

class PayPalPayment(PaymentProcessor):
    def pay(self, amount):
        return f"Paid \${amount} with PayPal"

class CryptoPayment(PaymentProcessor):
    def pay(self, amount):
        return f"Paid \${amount} with crypto"

class ShoppingCart:
    def __init__(self, payment_processor):
        self.items = []
        self.payment_processor = payment_processor  # Composition
    
    def add_item(self, item, price):
        self.items.append((item, price))
    
    def checkout(self):
        total = sum(price for _, price in self.items)
        return self.payment_processor.pay(total)

# Easy to swap payment methods!
cart = ShoppingCart(CreditCardPayment())
cart.add_item("Book", 25)
cart.add_item("Pen", 5)
print(cart.checkout())  # Credit card payment

# Change payment method
cart.payment_processor = PayPalPayment()
print(cart.checkout())  # PayPal payment
\`\`\`

**Mixin Pattern (Composition via Inheritance):**
\`\`\`python
class JSONSerializableMixin:
    """Adds JSON serialization capability"""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class LoggableMixin:
    """Adds logging capability"""
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class User(JSONSerializableMixin, LoggableMixin):
    """Composes behaviors via mixins"""
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def save(self):
        self.log("Saving user...")
        print(self.to_json())
        self.log("User saved")

user = User("Alice", "alice@example.com")
user.save()
\`\`\`

**When to Use Each:**

**Use Inheritance When:**
- True "is-a" relationship (Dog is an Animal)
- Shared interface and behavior
- Polymorphism needed
- Liskov Substitution Principle holds

**Use Composition When:**
- "Has-a" or "uses-a" relationship
- Need flexibility to swap components
- Want to avoid deep hierarchies
- Multiple capabilities from different sources
- Want easier testing (can mock components)

**Real-World Example - Game Character:**
\`\`\`python
# ❌ Bad: Deep inheritance
class Character: pass
class Warrior(Character): pass
class MagicWarrior(Warrior): pass
class HealingMagicWarrior(MagicWarrior): pass  # Too specific!

# ✅ Good: Composition
class Weapon:
    def __init__(self, damage):
        self.damage = damage
    
    def attack(self):
        return f"Deals {self.damage} damage"

class Armor:
    def __init__(self, defense):
        self.defense = defense
    
    def protect(self):
        return f"Blocks {self.defense} damage"

class Spell:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect
    
    def cast(self):
        return f"{self.name}: {self.effect}"

class Character:
    def __init__(self, name):
        self.name = name
        self.weapon = None
        self.armor = None
        self.spells = []
    
    def equip_weapon(self, weapon):
        self.weapon = weapon
    
    def equip_armor(self, armor):
        self.armor = armor
    
    def learn_spell(self, spell):
        self.spells.append(spell)
    
    def attack(self):
        if self.weapon:
            return self.weapon.attack()
        return "Punches for 1 damage"
    
    def cast_spell(self, spell_index):
        if spell_index < len(self.spells):
            return self.spells[spell_index].cast()
        return "No spell in that slot"

# Flexible character customization!
warrior = Character("Conan")
warrior.equip_weapon(Weapon(50))
warrior.equip_armor(Armor(30))

mage = Character("Gandalf")
mage.learn_spell(Spell("Fireball", "Burns enemy"))
mage.learn_spell(Spell("Heal", "Restores HP"))
mage.equip_weapon(Weapon(10))  # Staff

# Same Character class, different configurations!
\`\`\`

**Best Practices:**
- Prefer composition over inheritance (general rule)
- Use inheritance for true "is-a" relationships only
- Keep inheritance hierarchies shallow (2-3 levels max)
- Components should have single responsibility
- Use dependency injection for flexibility
- Make components replaceable (interfaces/protocols)`,
      quiz: [
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
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'Which relationship should use composition?',
          options: [
            'Dog is an Animal',
            'Car has an Engine',
            'Circle is a Shape',
            'Manager is an Employee',
          ],
          correctAnswer: 1,
          explanation:
            'Car HAS-AN Engine is a "has-a" relationship, perfect for composition. The others are "is-a" relationships suited for inheritance.',
        },
        {
          id: 'mc2',
          question:
            'What is the main advantage of composition over inheritance?',
          options: [
            'Faster execution',
            'Less memory usage',
            'Greater flexibility and easier to change',
            'Simpler syntax',
          ],
          correctAnswer: 2,
          explanation:
            'Composition provides greater flexibility—you can swap components at runtime, avoid fragile base class problems, and more easily modify behavior.',
        },
        {
          id: 'mc3',
          question: 'What is delegation in the context of composition?',
          options: [
            'Creating subclasses',
            'Forwarding method calls to composed objects',
            'Multiple inheritance',
            'Private methods',
          ],
          correctAnswer: 1,
          explanation:
            'Delegation means forwarding method calls from the containing object to its composed objects, like car.start() calling self.engine.start().',
        },
        {
          id: 'mc4',
          question: 'When is inheritance appropriate?',
          options: [
            'When you have a "has-a" relationship',
            'When you want to reuse code',
            'When you have a true "is-a" relationship',
            'Always use inheritance',
          ],
          correctAnswer: 2,
          explanation:
            'Inheritance is appropriate for true "is-a" relationships where the child class can fully substitute for the parent (Liskov Substitution Principle).',
        },
        {
          id: 'mc5',
          question:
            'What problem does the Strategy Pattern (composition-based) solve?',
          options: [
            'Deep inheritance hierarchies',
            'Need for interchangeable algorithms without conditional logic',
            'Memory leaks',
            'Slow performance',
          ],
          correctAnswer: 1,
          explanation:
            'Strategy Pattern uses composition to provide interchangeable algorithms, avoiding conditional logic and making it easy to add new strategies without modifying existing code.',
        },
      ],
    },
    {
      id: 'magic-methods',
      title: 'Magic Methods (Dunder Methods)',
      content: `# Magic Methods (Dunder Methods)

Magic methods (also called **dunder methods** for "double underscore") allow you to define custom behavior for Python's built-in operations. They make your objects feel like native Python types.

## Why Magic Methods Matter

Magic methods let you:
- Use operators like \`+\`, \`<\`, \`==\` with your objects
- Make objects printable, iterable, callable
- Support indexing (\`obj[key]\`) and slicing
- Integrate with built-in functions like \`len()\`, \`str()\`

**They're essential for creating Pythonic, professional code!**

---

## String Representation

### \`__str__\` and \`__repr__\`

\`\`\`python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """Human-readable string (for print, str())"""
        return f"Point at ({self.x}, {self.y})"
    
    def __repr__(self):
        """Developer-friendly representation (for debugging)"""
        return f"Point({self.x}, {self.y})"

p = Point(3, 4)
print(str(p))   # Point at (3, 4)  - uses __str__
print(repr(p))  # Point(3, 4)      - uses __repr__
print(p)        # Point at (3, 4)  - print uses __str__ if available
print([p])      # [Point(3, 4)]    - containers use __repr__
\`\`\`

**Rule of thumb:**
- \`__str__\`: User-friendly output
- \`__repr__\`: Unambiguous, ideally \`eval(repr(obj)) == obj\`
- Always implement \`__repr__\`; \`__str__\` is optional

---

## Comparison Magic Methods

Make objects comparable with \`<\`, \`>\`, \`==\`, etc.

\`\`\`python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        """Check equality (==)"""
        if not isinstance(other, Student):
            return False
        return self.name == other.name and self.grade == other.grade
    
    def __lt__(self, other):
        """Less than (<)"""
        return self.grade < other.grade
    
    def __le__(self, other):
        """Less than or equal (<=)"""
        return self.grade <= other.grade
    
    def __gt__(self, other):
        """Greater than (>)"""
        return self.grade > other.grade
    
    def __ge__(self, other):
        """Greater than or equal (>=)"""
        return self.grade >= other.grade
    
    def __repr__(self):
        return f"Student('{self.name}', {self.grade})"

alice = Student("Alice", 95)
bob = Student("Bob", 87)

print(alice > bob)         # True
print(alice == alice)      # True
print(sorted([alice, bob]))  # [Student('Bob', 87), Student('Alice', 95)]
\`\`\`

**Pro tip:** Use \`@functools.total_ordering\` decorator:
\`\`\`python
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    
    # Python auto-generates <=, >, >= from __eq__ and __lt__!
\`\`\`

---

## Arithmetic Magic Methods

Make objects work with \`+\`, \`-\`, \`*\`, etc.

\`\`\`python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Addition (+)"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtraction (-)"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Multiplication (*)"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        """Division (/)"""
        return Vector(self.x / scalar, self.y / scalar)
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)    # Vector(4, 6)
print(v1 - v2)    # Vector(2, 2)
print(v1 * 2)     # Vector(6, 8)
print(v1 / 2)     # Vector(1.5, 2.0)
\`\`\`

**In-place variants:**
\`\`\`python
class Vector:
    # ... previous code ...
    
    def __iadd__(self, other):
        """In-place addition (+=)"""
        self.x += other.x
        self.y += other.y
        return self
    
    def __imul__(self, scalar):
        """In-place multiplication (*=)"""
        self.x *= scalar
        self.y *= scalar
        return self

v = Vector(3, 4)
v += Vector(1, 1)  # Uses __iadd__
print(v)  # Vector(4, 5)
\`\`\`

---

## Container Magic Methods

Make objects behave like lists/dicts.

\`\`\`python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def __len__(self):
        """Support len()"""
        return len(self.items)
    
    def __getitem__(self, index):
        """Support indexing: cart[0]"""
        return self.items[index]
    
    def __setitem__(self, index, value):
        """Support assignment: cart[0] = 'apple'"""
        self.items[index] = value
    
    def __delitem__(self, index):
        """Support deletion: del cart[0]"""
        del self.items[index]
    
    def __contains__(self, item):
        """Support 'in' operator"""
        return item in self.items
    
    def __iter__(self):
        """Make iterable (for loops)"""
        return iter(self.items)
    
    def add(self, item):
        self.items.append(item)

cart = ShoppingCart()
cart.add("apple")
cart.add("banana")

print(len(cart))         # 2 - uses __len__
print(cart[0])           # 'apple' - uses __getitem__
print("apple" in cart)   # True - uses __contains__

for item in cart:        # Uses __iter__
    print(item)
\`\`\`

---

## Callable Objects

Make instances callable like functions.

\`\`\`python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        """Make object callable"""
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))   # 10 - calls __call__(5)
print(triple(5))   # 15

# Common use: function decorators, closures, functors
\`\`\`

---

## Context Managers

Implement \`with\` statement.

\`\`\`python
class Timer:
    def __enter__(self):
        """Called when entering 'with' block"""
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block"""
        import time
        elapsed = time.time() - self.start
        print(f"Elapsed: {elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

with Timer():
    # Code to time
    sum(range(1000000))
# Output: Elapsed: 0.0234 seconds

# Classic use: file handling, database connections, locks
\`\`\`

---

## Hash and Equality

Make objects hashable (for sets, dict keys).

\`\`\`python
class Person:
    def __init__(self, name, ssn):
        self.name = name
        self.ssn = ssn  # Immutable identifier
    
    def __eq__(self, other):
        """Objects are equal if SSNs match"""
        if not isinstance(other, Person):
            return False
        return self.ssn == other.ssn
    
    def __hash__(self):
        """Hash based on immutable attribute"""
        return hash(self.ssn)
    
    def __repr__(self):
        return f"Person('{self.name}', '{self.ssn}')"

p1 = Person("Alice", "123-45-6789")
p2 = Person("Alice", "123-45-6789")
p3 = Person("Bob", "987-65-4321")

print(p1 == p2)  # True - same SSN
print(p1 is p2)  # False - different objects

# Can use in sets/dicts
people = {p1, p2, p3}
print(len(people))  # 2 - p1 and p2 treated as same

lookup = {p1: "Manager", p3: "Engineer"}
print(lookup[p2])  # "Manager" - p2 treated as same key as p1
\`\`\`

**Important:** If \`__eq__\` is defined, \`__hash__\` must also be defined for hashable objects.

---

## Attribute Access

Control attribute access/assignment.

\`\`\`python
class DynamicAttributes:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        """Called when attribute not found"""
        print(f"Getting {name}")
        return self._data.get(name, f"No attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Called on all attribute assignment"""
        print(f"Setting {name} = {value}")
        if name == '_data':
            super().__setattr__(name, value)  # Avoid recursion
        else:
            self._data[name] = value
    
    def __delattr__(self, name):
        """Called when deleting attribute"""
        print(f"Deleting {name}")
        del self._data[name]

obj = DynamicAttributes()
obj.x = 10     # Setting x = 10
print(obj.x)   # Getting x \n 10
del obj.x      # Deleting x
\`\`\`

---

## Quick Reference Table

| Method | Operator/Function | Description |
|--------|-------------------|-------------|
| \`__init__\` | Constructor | Initialize object |
| \`__str__\` | \`str()\`, \`print()\` | User-friendly string |
| \`__repr__\` | \`repr()\` | Developer string |
| \`__eq__\` | \`==\` | Equality |
| \`__lt__\` | \`<\` | Less than |
| \`__le__\` | \`<=\` | Less or equal |
| \`__gt__\` | \`>\` | Greater than |
| \`__ge__\` | \`>=\` | Greater or equal |
| \`__add__\` | \`+\` | Addition |
| \`__sub__\` | \`-\` | Subtraction |
| \`__mul__\` | \`*\` | Multiplication |
| \`__truediv__\` | \`/\` | Division |
| \`__len__\` | \`len()\` | Length |
| \`__getitem__\` | \`obj[key]\` | Index access |
| \`__setitem__\` | \`obj[key]=val\` | Index assignment |
| \`__contains__\` | \`in\` | Membership test |
| \`__iter__\` | \`for x in obj\` | Iteration |
| \`__call__\` | \`obj()\` | Make callable |
| \`__enter__\` | \`with obj:\` | Enter context |
| \`__exit__\` | \`with obj:\` | Exit context |
| \`__hash__\` | \`hash()\` | Hash value |

---

## Best Practices

1. **Always implement \`__repr__\`** - critical for debugging
2. **\`__eq__\` + \`__hash__\`** - if you define one, define both (for hashable objects)
3. **Use \`@total_ordering\`** - reduces comparison method boilerplate
4. **Check types** - use \`isinstance()\` in comparison/arithmetic methods
5. **Return \`NotImplemented\`** - when operation not supported with given type
\`\`\`python
def __add__(self, other):
    if not isinstance(other, Vector):
        return NotImplemented  # Let Python try other.__radd__(self)
    return Vector(self.x + other.x, self.y + other.y)
\`\`\`
6. **Immutable hash** - \`__hash__\` must be based on immutable attributes

---

## Real-World Examples

**Custom Exception:**
\`\`\`python
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")
\`\`\`

**Singleton Pattern:**
\`\`\`python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
\`\`\`

**Lazy Property:**
\`\`\`python
class Dataset:
    def __init__(self, path):
        self.path = path
        self._data = None
    
    def __len__(self):
        if self._data is None:
            self._data = load_data(self.path)  # Load on first access
        return len(self._data)
\`\`\`

---

## Interview Relevance

Magic methods appear in:
- **Custom data structures** - implement list/dict-like objects
- **Algorithm problems** - comparable objects for sorting/heaps
- **Design problems** - context managers, decorators, iterators
- **LeetCode** - \`__lt__\` for custom priority queues

**Example:** LeetCode's "Merge K Sorted Lists" is easier with:
\`\`\`python
class ListNode:
    def __lt__(self, other):
        return self.val < other.val

# Now can use with heapq!
import heapq
heap = [node1, node2, node3]
heapq.heapify(heap)
\`\`\``,
      quiz: [
        {
          id: 'q1',
          question:
            'What is the difference between __str__ and __repr__? When would you use each?',
          sampleAnswer:
            '__str__ provides a user-friendly, human-readable string representation (called by str() and print()), while __repr__ provides an unambiguous, developer-focused representation for debugging (called by repr() and used in containers). __repr__ should ideally allow recreating the object: eval(repr(obj)) == obj. Always implement __repr__ since Python falls back to it if __str__ is missing. Use __str__ when you need different user-facing output. Example: Point.__repr__ might be "Point(3, 4)" while __str__ might be "Point at (3, 4)".',
          keyPoints: [
            '__str__: user-friendly, called by print()',
            '__repr__: developer-friendly, unambiguous',
            '__repr__ used in containers and debugging',
            'Always implement __repr__, __str__ is optional',
            '__repr__ should enable object recreation if possible',
          ],
        },
        {
          id: 'q2',
          question:
            'Why must __hash__ be implemented when __eq__ is customized for hashable objects?',
          sampleAnswer:
            'Python requires that if a == b, then hash(a) == hash(b). The default __hash__ uses object identity (id), but custom __eq__ might consider two different objects equal based on their values. Without a custom __hash__, equal objects could have different hashes, breaking sets and dictionaries. For example, Person("Alice", "123") == Person("Alice", "123") with custom __eq__, but they would have different default hashes. The fix is to hash based on the same attributes used in __eq__: def __hash__(self): return hash(self.ssn). Note: only hash immutable attributes.',
          keyPoints: [
            'Equal objects must have equal hashes (hash invariant)',
            'Default __hash__ uses object identity (id)',
            'Custom __eq__ breaks hash invariant without custom __hash__',
            'Hash only immutable attributes',
            'Required for objects used as dict keys or in sets',
          ],
        },
        {
          id: 'q3',
          question:
            'Explain how __getitem__ enables both indexing and iteration in Python.',
          sampleAnswer:
            '__getitem__(self, key) is called for indexing operations (obj[key]). If you implement __getitem__ with integer indices, Python automatically makes your object iterable—it tries obj[0], obj[1], obj[2] until IndexError is raised. This is called "sequence protocol". However, explicit __iter__ is preferred for iteration because it\'s more efficient and clearer. __getitem__ is essential for: (1) indexing: cart[0], (2) slicing: cart[1:3], (3) fallback iteration if __iter__ is missing. Pro tip: Implement both __getitem__ for indexing and __iter__ for efficient iteration.',
          keyPoints: [
            '__getitem__ enables obj[key] syntax',
            'Implementing __getitem__ with integers enables automatic iteration',
            'Python tries obj[0], obj[1], ... until IndexError',
            'Explicit __iter__ preferred for iteration (more efficient)',
            '__getitem__ also enables slicing support',
          ],
        },
      ],
      multipleChoice: [
        {
          id: 'mc1',
          question: 'What is __init__?',
          options: [
            'A magic spell',
            'The constructor/initializer method',
            'A destructor',
            'A class variable',
          ],
          correctAnswer: 1,
          explanation:
            '__init__ is the initializer method called when creating a new instance of a class.',
        },
        {
          id: 'mc2',
          question: 'What does __str__ return?',
          options: [
            'A string representation for users',
            'The object type',
            'The object ID',
            'A boolean',
          ],
          correctAnswer: 0,
          explanation:
            '__str__ returns a user-friendly string representation, used by str() and print().',
        },
        {
          id: 'mc3',
          question: 'What is the difference between __str__ and __repr__?',
          options: [
            'No difference',
            '__str__ is user-friendly, __repr__ is developer-friendly/unambiguous',
            '__repr__ is faster',
            '__str__ is deprecated',
          ],
          correctAnswer: 1,
          explanation:
            '__str__ for readable output (users), __repr__ for unambiguous representation (developers/debugging).',
        },
        {
          id: 'mc4',
          question: 'What does __len__ allow?',
          options: [
            'Calling len(obj)',
            'Comparing objects',
            'Adding objects',
            'Iterating over object',
          ],
          correctAnswer: 0,
          explanation:
            '__len__ allows len(obj) to work on custom objects, returning the "length" you define.',
        },
        {
          id: 'mc5',
          question: 'What are __enter__ and __exit__ used for?',
          options: [
            'Entering functions',
            'Context managers (with statement)',
            'Loops',
            'Error handling',
          ],
          correctAnswer: 1,
          explanation:
            '__enter__ and __exit__ enable objects to be used with the "with" statement as context managers.',
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
    'Dataclasses reduce boilerplate for data-focused classes—use field(default_factory) for mutables',
    'Properties provide controlled attribute access with validation and computed values',
    'Prefer composition over inheritance—build complex objects from simple, reusable components',
    'Magic methods make objects Pythonic—implement __repr__, __eq__, __lt__ for comparable objects',
    'Use @total_ordering to auto-generate comparison methods from __eq__ and __lt__',
  ],
  relatedProblems: [
    'class-bankaccount',
    'inheritance-shapes',
    'polymorphism-animals',
  ],
};
