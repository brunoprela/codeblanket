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
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What is the purpose of the __init__ method?',
                    options: [
                        'To delete an object',
                        'To initialize an object\'s attributes when it is created',
                        'To compare two objects',
                        'To convert an object to a string',
                    ],
                    correctAnswer: 1,
                    explanation:
                        '__init__ is the constructor method called when creating a new instance. It initializes the object\'s attributes with values passed as arguments.',
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
            ],
            multipleChoice: [
                {
                    id: 'mc1',
                    question: 'What does super() do?',
                    options: [
                        'Creates a superclass',
                        'Calls the parent class\'s method',
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

